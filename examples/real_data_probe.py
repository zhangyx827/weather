"""Real-data probe for MAZU Saudi multi-hazard early warning.

This script exercises the project chain end to end:

1. CDS-backed ERA5 acquisition for the Saudi extreme-heat window
   2025-06-16 to 2025-06-17.
2. Local spatial trimming and alignment to the required 0.1 degree grid.
3. Layer-2 physical indicator derivation.
4. Layer-4 LightGBM multi-hazard risk scoring.

The probe is strict by design:
- it requires CDS credentials and network access,
- it only processes the real NetCDF download,
- it keeps the runtime dependency surface light, and
- it emits a JSON artifact suitable for downstream validation.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

try:  # Optional dependency in minimal environments.
    import xarray as xr
except Exception:  # pragma: no cover - environment dependent
    xr = None

from mazu_saudi.data import read_netcdf_dataset

TARGET_NETCDF = ROOT / "data" / "raw" / "era5_saudi_20250616.nc"
TARGET_JSON = ROOT / "data" / "output" / "risk_probe_result.json"
DEFAULT_LAYER4_MODEL_DIR = ROOT / "models" / "layer4"

CDS_DATASET = "reanalysis-era5-single-levels"
CDS_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
]
CDS_AREA = [32.0, 34.0, 16.0, 56.0]  # North, West, South, East
CDS_DAYS = ["16", "17"]
CDS_TIMES = [f"{hour:02d}:00" for hour in range(24)]

TARGET_LATITUDE = np.round(np.arange(31.9, 15.9, -0.1), 1)
TARGET_LONGITUDE = np.round(np.arange(34.0, 56.0, 0.1), 1)

CANONICAL_VAR_MAP = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "tp": "total_precipitation",
}
REVERSE_CANONICAL_VAR_MAP = {value: key for key, value in CANONICAL_VAR_MAP.items()}
PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
PROXY_ENV_KEYS_LOWER = tuple(key.lower() for key in PROXY_ENV_KEYS)
PROXY_OVERRIDE_KEYS = {
    "MAZU_CDS_HTTP_PROXY": "HTTP_PROXY",
    "MAZU_CDS_HTTPS_PROXY": "HTTPS_PROXY",
    "MAZU_CDS_ALL_PROXY": "ALL_PROXY",
    "MAZU_CDS_NO_PROXY": "NO_PROXY",
    "MAZU_CDS_PROXY_URL": "ALL_PROXY",
}


def _use_color() -> bool:
    return sys.stdout.isatty() or os.environ.get("FORCE_COLOR") == "1"


def _ansi(code: str, text: str) -> str:
    if not _use_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def red(text: str) -> str:
    return _ansi("31;1", text)


def yellow(text: str) -> str:
    return _ansi("33;1", text)


def green(text: str) -> str:
    return _ansi("32;1", text)


def cyan(text: str) -> str:
    return _ansi("36;1", text)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("real_data_probe")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


LOGGER = setup_logging()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_cds_credentials() -> tuple[str | None, str | None, str | None]:
    """Return ``(url, key, error_message)`` from ``~/.cdsapirc``."""

    rc_path = Path.home() / ".cdsapirc"
    if not rc_path.exists():
        return None, None, f"missing CDS credentials file: {rc_path}"

    try:
        lines = rc_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return None, None, f"unable to read CDS credentials file {rc_path}: {exc}"

    entries: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        entries[key.strip().lower()] = value.strip()

    url = entries.get("url")
    key = entries.get("key")
    if not url or not key:
        return url, key, f"CDS credentials file is missing url/key entries: {rc_path}"
    if not url.startswith("http"):
        return url, key, f"CDS credentials URL is malformed: {url!r}"
    return url, key, None


def _proxy_env_snapshot() -> dict[str, str | None]:
    snapshot = {key: os.environ.get(key) for key in (*PROXY_ENV_KEYS, *PROXY_ENV_KEYS_LOWER)}
    for override_key, target_key in PROXY_OVERRIDE_KEYS.items():
        if override_key in os.environ:
            snapshot[target_key] = os.environ[override_key]
    return snapshot


def _apply_proxy_env() -> dict[str, str | None]:
    """Apply proxy overrides for the CDS client and return the original env."""

    original = _proxy_env_snapshot()

    if os.environ.get("MAZU_CDS_DISABLE_PROXY", "").strip().lower() in {"1", "true", "yes"}:
        for key in (*PROXY_ENV_KEYS, *PROXY_ENV_KEYS_LOWER):
            os.environ.pop(key, None)
        return original

    for override_key, target_key in PROXY_OVERRIDE_KEYS.items():
        value = os.environ.get(override_key)
        if value:
            os.environ[target_key] = value
            os.environ[target_key.lower()] = value

    return original


def _restore_proxy_env(original: dict[str, str | None]) -> None:
    for key, value in original.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _describe_proxy_env() -> str:
    values = []
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY"):
        value = os.environ.get(key) or os.environ.get(key.lower())
        if value:
            values.append(f"{key}=set")
    if "MAZU_CDS_PROXY_URL" in os.environ:
        values.append("MAZU_CDS_PROXY_URL=set")
    if "MAZU_CDS_HTTPS_PROXY" in os.environ:
        values.append("MAZU_CDS_HTTPS_PROXY=set")
    if "MAZU_CDS_HTTP_PROXY" in os.environ:
        values.append("MAZU_CDS_HTTP_PROXY=set")
    if "MAZU_CDS_ALL_PROXY" in os.environ:
        values.append("MAZU_CDS_ALL_PROXY=set")
    if "MAZU_CDS_DISABLE_PROXY" in os.environ:
        values.append(f"MAZU_CDS_DISABLE_PROXY={os.environ['MAZU_CDS_DISABLE_PROXY']}")
    return ", ".join(values) if values else "no proxy env detected"


def build_cds_request() -> dict[str, Any]:
    return {
        "product_type": "reanalysis",
        "variable": CDS_VARIABLES,
        "year": "2025",
        "month": "06",
        "day": CDS_DAYS,
        "time": CDS_TIMES,
        "area": CDS_AREA,
        "format": "netcdf",
    }


def try_download_era5(target_path: Path) -> None:
    """Attempt CDS download and raise on any failure."""

    ensure_parent(target_path)
    _, _, credentials_error = _read_cds_credentials()
    if credentials_error:
        raise RuntimeError(credentials_error)

    try:
        import cdsapi  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"cdsapi import failed: {exc}") from exc

    proxy_snapshot = _apply_proxy_env()
    try:
        LOGGER.info("Proxy configuration: %s", _describe_proxy_env())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client = cdsapi.Client(
                quiet=True,
                progress=False,
                timeout=20,
                retry_max=1,
                sleep_max=1,
                wait_until_complete=True,
            )
            client.retrieve(CDS_DATASET, build_cds_request(), str(target_path))
    except Exception as exc:  # pragma: no cover - depends on runtime/network
        raise RuntimeError(f"CDS retrieval failed: {exc}") from exc
    finally:
        _restore_proxy_env(proxy_snapshot)


def normalize_dataset(dataset: "xr.Dataset") -> "xr.Dataset":
    ds = dataset.copy()

    rename_dims = {}
    if "lat" in ds.dims and "latitude" not in ds.dims:
        rename_dims["lat"] = "latitude"
    if "lon" in ds.dims and "longitude" not in ds.dims:
        rename_dims["lon"] = "longitude"
    if rename_dims:
        ds = ds.rename(rename_dims)

    rename_vars = {}
    for long_name, short_name in REVERSE_CANONICAL_VAR_MAP.items():
        if short_name in ds.data_vars:
            continue
        if long_name in ds.data_vars:
            rename_vars[long_name] = short_name
    if rename_vars:
        ds = ds.rename(rename_vars)

    if "latitude" not in ds.coords and "lat" in ds.coords:
        ds = ds.rename({"lat": "latitude"})
    if "longitude" not in ds.coords and "lon" in ds.coords:
        ds = ds.rename({"lon": "longitude"})

    return ds


def load_dataset(path: Path) -> "xr.Dataset":
    if xr is None:
        raise RuntimeError("xarray is required to load NetCDF datasets")
    return read_netcdf_dataset(path)


def _to_celsius(temperature: Any) -> Any:
    temp = temperature if xr is not None and hasattr(temperature, "dims") else np.asarray(temperature)
    return temp - 273.15


def saturation_vapor_pressure_kpa(temp_c: Any) -> Any:
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def relative_humidity_from_dewpoint(temp_c: Any, dewpoint_c: Any) -> Any:
    es = saturation_vapor_pressure_kpa(temp_c)
    ea = saturation_vapor_pressure_kpa(dewpoint_c)
    rh = 100.0 * ea / np.maximum(es, 1e-6)
    return np.clip(rh, 0.0, 100.0)


def compute_vpd_kpa(temp_c: Any, dewpoint_c: Any) -> Any:
    es = saturation_vapor_pressure_kpa(temp_c)
    ea = saturation_vapor_pressure_kpa(dewpoint_c)
    return np.maximum(es - ea, 0.0)


def compute_heat_index_c(temp_c: Any, rh_percent: Any) -> Any:
    """Vectorized NOAA heat index with a Celsius fallback for cool conditions."""

    temp_c = np.asarray(temp_c)
    rh_percent = np.asarray(rh_percent)
    temp_f = temp_c * 9.0 / 5.0 + 32.0
    hi_f = (
        -42.379
        + 2.04901523 * temp_f
        + 10.14333127 * rh_percent
        - 0.22475541 * temp_f * rh_percent
        - 6.83783e-3 * temp_f**2
        - 5.481717e-2 * rh_percent**2
        + 1.22874e-3 * temp_f**2 * rh_percent
        + 8.5282e-4 * temp_f * rh_percent**2
        - 1.99e-6 * temp_f**2 * rh_percent**2
    )

    cool_mask = temp_f < 80.0
    hi_f = np.where(cool_mask, 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (rh_percent * 0.094)), hi_f)
    sqrt_term = np.sqrt(np.maximum((17.0 - np.abs(temp_f - 95.0)) / 17.0, 0.0))
    hi_f = np.where((rh_percent < 13.0) & (temp_f >= 80.0) & (temp_f <= 112.0), hi_f - ((13.0 - rh_percent) / 4.0) * sqrt_term, hi_f)
    hi_f = np.where((rh_percent > 85.0) & (temp_f >= 80.0) & (temp_f <= 87.0), hi_f + ((rh_percent - 85.0) / 10.0) * ((87.0 - temp_f) / 5.0), hi_f)
    hi_f = np.where(cool_mask, temp_f, hi_f)
    return (hi_f - 32.0) * 5.0 / 9.0


def compute_wind_speed_mps(u10: Any, v10: Any) -> Any:
    return np.sqrt(np.asarray(u10) ** 2 + np.asarray(v10) ** 2)


def _target_grid_dataset(dataset: "xr.Dataset") -> "xr.Dataset":
    if xr is None:
        raise RuntimeError("xarray is required for spatial alignment")

    ds = normalize_dataset(dataset)
    ds = ds.sel(latitude=slice(31.9, 16.0), longitude=slice(34.0, 55.9))
    ds = ds.interp(latitude=TARGET_LATITUDE, longitude=TARGET_LONGITUDE)
    ds = ds.sortby("latitude", ascending=False)
    ds = ds.sortby("longitude", ascending=True)
    return ds


def extract_probe_slice(dataset: "xr.Dataset") -> tuple["xr.Dataset", str | None]:
    ds = _target_grid_dataset(dataset)
    selected_time = None
    if "time" in ds.dims and ds.sizes.get("time", 0) > 0:
        selected_time = str(np.asarray(ds["time"].isel(time=-1).values).item())
        ds = ds.isel(time=-1, drop=True)
    return ds, selected_time


def add_layer2_indicators(dataset: "xr.Dataset") -> "xr.Dataset":
    if xr is None:
        raise RuntimeError("xarray is required for indicator derivation")

    ds = normalize_dataset(dataset)
    if "time" in ds.dims:
        ds = ds.isel(time=-1, drop=True)

    t2m_c = _to_celsius(ds["t2m"])
    d2m_c = _to_celsius(ds["d2m"])
    rh = relative_humidity_from_dewpoint(t2m_c, d2m_c)
    vpd = compute_vpd_kpa(t2m_c, d2m_c)
    hi = compute_heat_index_c(t2m_c, rh)
    wind = compute_wind_speed_mps(ds["u10"], ds["v10"])

    derived = xr.Dataset(
        data_vars={
            "vpd_kpa": (ds["t2m"].dims, np.asarray(vpd), {"units": "kPa"}),
            "heat_index_c": (ds["t2m"].dims, np.asarray(hi), {"units": "degC"}),
            "wind_speed_mps": (ds["t2m"].dims, np.asarray(wind), {"units": "m s-1"}),
            "relative_humidity_percent": (ds["t2m"].dims, np.asarray(rh), {"units": "%"}),
        },
        coords={name: ds.coords[name] for name in ds.coords},
        attrs=dict(ds.attrs),
    )
    merged = xr.merge([ds, derived], compat="override")
    merged.attrs["layer2_indicators"] = "vpd_kpa, heat_index_c, wind_speed_mps, relative_humidity_percent"
    return merged


def _levels_from_probability(probability: Any) -> Any:
    p = np.asarray(probability)
    return np.where(
        p >= 0.70,
        3,
        np.where(
            p >= 0.45,
            2,
            np.where(
                p >= 0.20,
                1,
                0,
            ),
        ),
    ).astype(np.int8)


class LightGBMLayer4Model:
    """Real Layer-4 inference backed by trained LightGBM booster files."""

    feature_names = (
        "temp_c",
        "vpd_kpa",
        "heat_index_c",
        "wind_speed_mps",
        "relative_humidity_percent",
    )

    def __init__(
        self,
        extreme_heat_model_path: str | Path | None = None,
        dry_heat_model_path: str | Path | None = None,
    ) -> None:
        try:
            import lightgbm as lgb
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "lightgbm is required for real Layer-4 inference. "
                "Install it with `pip install lightgbm` and provide trained model files."
            ) from exc

        self.lgb = lgb
        self.extreme_heat_model_path = self._resolve_model_path(
            explicit=extreme_heat_model_path,
            env_key="MAZU_LAYER4_EXTREME_HEAT_MODEL",
            default_name="extreme_heat.txt",
        )
        self.dry_heat_model_path = self._resolve_model_path(
            explicit=dry_heat_model_path,
            env_key="MAZU_LAYER4_DRY_HEAT_MODEL",
            default_name="dry_heat_stress.txt",
        )
        self.extreme_heat_model = self._load_booster(self.extreme_heat_model_path)
        self.dry_heat_model = self._load_booster(self.dry_heat_model_path)

    @staticmethod
    def _resolve_model_path(explicit: str | Path | None, env_key: str, default_name: str) -> Path:
        raw_path = explicit or os.environ.get(env_key) or (DEFAULT_LAYER4_MODEL_DIR / default_name)
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(
                f"Layer-4 LightGBM model file not found: {path}. "
                f"Set {env_key} to a trained LightGBM booster file."
            )
        return path

    def _load_booster(self, path: Path) -> Any:
        try:
            return self.lgb.Booster(model_file=str(path))
        except Exception as exc:
            raise RuntimeError(f"Unable to load LightGBM model from {path}: {exc}") from exc

    @staticmethod
    def _display_path(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    def _feature_matrix(self, ds: "xr.Dataset") -> tuple[np.ndarray, tuple[int, ...]]:
        required = ("t2m", "vpd_kpa", "heat_index_c", "wind_speed_mps", "relative_humidity_percent")
        missing = [name for name in required if name not in ds.data_vars]
        if missing:
            raise ValueError(f"Layer-4 input dataset is missing required variables: {', '.join(missing)}")

        temp_c = np.asarray(_to_celsius(ds["t2m"]))
        shape = temp_c.shape
        fields = {
            "temp_c": temp_c,
            "vpd_kpa": np.asarray(ds["vpd_kpa"]),
            "heat_index_c": np.asarray(ds["heat_index_c"]),
            "wind_speed_mps": np.asarray(ds["wind_speed_mps"]),
            "relative_humidity_percent": np.asarray(ds["relative_humidity_percent"]),
        }
        for name, values in fields.items():
            if values.shape != shape:
                raise ValueError(f"Layer-4 feature {name!r} has shape {values.shape}, expected {shape}")

        matrix = np.column_stack([fields[name].reshape(-1) for name in self.feature_names]).astype(np.float32)
        return matrix, shape

    @staticmethod
    def _predict_probability(model: Any, features: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        prediction = np.asarray(model.predict(features))
        if prediction.ndim == 2:
            if prediction.shape[1] == 2:
                prediction = prediction[:, 1]
            elif prediction.shape[1] == 1:
                prediction = prediction[:, 0]
            else:
                raise ValueError(f"Expected binary LightGBM probabilities, got prediction shape {prediction.shape}")
        if prediction.size != features.shape[0]:
            raise ValueError(
                f"LightGBM prediction size {prediction.size} does not match feature rows {features.shape[0]}"
            )
        return np.clip(prediction.reshape(shape), 0.0, 1.0)

    def predict_fields(self, dataset: "xr.Dataset") -> "xr.Dataset":
        if xr is None:
            raise RuntimeError("xarray is required for Layer-4 prediction fields")

        ds = normalize_dataset(dataset)
        features, shape = self._feature_matrix(ds)
        extreme_heat_prob = self._predict_probability(self.extreme_heat_model, features, shape)
        dry_heat_prob = self._predict_probability(self.dry_heat_model, features, shape)
        extreme_heat_level = _levels_from_probability(extreme_heat_prob)
        dry_heat_level = _levels_from_probability(dry_heat_prob)

        return xr.Dataset(
            data_vars={
                "ExtremeHeat_Risk_Prob": (ds["t2m"].dims, np.asarray(np.clip(extreme_heat_prob, 0.0, 1.0)), {"units": "1"}),
                "ExtremeHeat_Risk_Level": (ds["t2m"].dims, extreme_heat_level, {"units": "class"}),
                "DryHeatStress_Risk_Prob": (ds["t2m"].dims, np.asarray(np.clip(dry_heat_prob, 0.0, 1.0)), {"units": "1"}),
                "DryHeatStress_Risk_Level": (ds["t2m"].dims, dry_heat_level, {"units": "class"}),
            },
            coords={name: ds.coords[name] for name in ds.coords},
            attrs={
                "model_family": "LightGBMLayer4Model",
                "extreme_heat_model": self._display_path(self.extreme_heat_model_path),
                "dry_heat_model": self._display_path(self.dry_heat_model_path),
                "feature_names": ",".join(self.feature_names),
            },
        )


def _summary_stats(field: Any) -> dict[str, Any]:
    arr = np.asarray(field)
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }


def _coords_to_list(dataset: "xr.Dataset", coord_name: str) -> list[float]:
    if coord_name not in dataset.coords:
        return []
    return [float(value) for value in np.asarray(dataset[coord_name].values).tolist()]


def build_json_payload(
    dataset: "xr.Dataset",
    risk_fields: "xr.Dataset",
    download_status: str,
    selected_time: str | None,
) -> dict[str, Any]:
    payload = {
        "status": download_status,
        "selected_time": selected_time,
        "grid": {
            "dimensions": {name: int(size) for name, size in dataset.sizes.items()},
            "latitude": _coords_to_list(dataset, "latitude"),
            "longitude": _coords_to_list(dataset, "longitude"),
        },
        "artifacts": {
            "real_netcdf": str(TARGET_NETCDF.relative_to(ROOT)),
        },
        "variables": {
            "raw": [name for name in ("t2m", "d2m", "u10", "v10", "tp") if name in dataset.data_vars],
            "derived": [name for name in ("vpd_kpa", "heat_index_c", "wind_speed_mps", "relative_humidity_percent") if name in dataset.data_vars],
            "risk": [name for name in risk_fields.data_vars],
        },
        "derived_fields": {
            name: np.asarray(dataset[name]).tolist()
            for name in ("vpd_kpa", "heat_index_c", "wind_speed_mps", "relative_humidity_percent")
            if name in dataset.data_vars
        },
        "risk_fields": {
            name: np.asarray(risk_fields[name]).tolist()
            for name in risk_fields.data_vars
        },
        "summary": {
            "ExtremeHeat_Risk_Prob": _summary_stats(risk_fields["ExtremeHeat_Risk_Prob"]),
            "DryHeatStress_Risk_Prob": _summary_stats(risk_fields["DryHeatStress_Risk_Prob"]),
        },
    }
    return payload


def _format_dimensions(ds: "xr.Dataset") -> str:
    return ", ".join(f"{name}={int(size)}" for name, size in ds.sizes.items())


def _ensure_cds_download() -> tuple["xr.Dataset", str]:
    LOGGER.info("Starting CDS acquisition probe for %s", CDS_DATASET)

    if TARGET_NETCDF.exists():
        LOGGER.info("CDS acquisition succeeded; loading NetCDF from %s", TARGET_NETCDF)
        raw_dataset = load_dataset(TARGET_NETCDF)
        return raw_dataset, "SUCCESS_CACHED"

    try_download_era5(TARGET_NETCDF)
    LOGGER.info("CDS acquisition succeeded; loading NetCDF from %s", TARGET_NETCDF)
    raw_dataset = load_dataset(TARGET_NETCDF)
    # print(raw_dataset)
    return raw_dataset, "SUCCESS"


def main() -> None:
    ensure_parent(TARGET_NETCDF)
    ensure_parent(TARGET_JSON)

    raw_dataset, download_status = _ensure_cds_download()
    aligned_dataset, selected_time = extract_probe_slice(raw_dataset)

    if tuple(int(aligned_dataset.sizes.get(name, 0)) for name in ("latitude", "longitude")) != (160, 220):
        raise RuntimeError(
            "Aligned probe grid must be exactly 160x220; "
            f"got {_format_dimensions(aligned_dataset)}"
        )

    layer2_dataset = add_layer2_indicators(aligned_dataset)
    risk_model = LightGBMLayer4Model()
    risk_fields = risk_model.predict_fields(layer2_dataset)
    payload = build_json_payload(layer2_dataset, risk_fields, download_status, selected_time)

    TARGET_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    derived_names = [name for name in ("vpd_kpa", "heat_index_c", "wind_speed_mps", "relative_humidity_percent") if name in layer2_dataset.data_vars]
    spatial_dims = f"{int(layer2_dataset.sizes.get('latitude', 0))}x{int(layer2_dataset.sizes.get('longitude', 0))}"

    print(green(f"DATA STATUS: {download_status}"))
    print(cyan(f"GRID DIMS: {spatial_dims}"))
    print(cyan(f"DERIVED: {', '.join(derived_names)}"))
    print(cyan(f"RISK JSON: {TARGET_JSON.relative_to(ROOT)}"))
    print(cyan(
        "RISK SUMMARY: "
        f"ExtremeHeat p={payload['summary']['ExtremeHeat_Risk_Prob']}, "
        f"DryHeatStress p={payload['summary']['DryHeatStress_Risk_Prob']}"
    ))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - final safety net
        print(red(f"Unhandled probe error: {exc}"))
        print(yellow("The script stopped before JSON export; inspect the traceback above."))
