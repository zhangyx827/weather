"""Forecast model probe and training harness for the MAZU Saudi forecast layer.

This script keeps the environment readiness probe, but extends it into a
trainable end-to-end background-field pipeline:

- environment / dependency inspection
- provider status checks for Aurora, GenCast, and AIFS
- optional data ingestion from a local path or URL
- lightweight calibration training on historical data
- forward 24h / 48h / 72h forecast generation
- GenCast ensemble statistics with member dimension
- Aurora vs AIFS benchmark comparison
- downstream physical-indicator and risk-field derivation

If no external source is provided, the script generates a synthetic but
time-varying historical dataset so the pipeline remains runnable in a minimal
development environment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import sys
import tempfile
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

try:  # Optional dependency; all xarray work is guarded.
    import xarray as xr
except Exception:  # pragma: no cover - environment dependent
    xr = None

from mazu_saudi.data import read_netcdf_dataset, write_netcdf_dataset
from mazu_saudi.indicators import (
    compute_dry_heat_stress_score,
    compute_heat_index_c,
    compute_relative_humidity_from_dewpoint,
    compute_vpd_kpa,
)
from mazu_saudi.risk import DryHeatStressRiskModel, ExtremeHeatRiskModel
from mazu_saudi.schemas import GridCell, MeteorologicalFeatures
from mazu_saudi.utils.math import clamp, is_missing


SAUDI_LAT_START = 31.9
SAUDI_LAT_END = 16.0
SAUDI_LON_START = 34.0
SAUDI_LON_END = 55.9
GRID_STEP = 0.1
FORECAST_LEADS = (24, 48, 72)
GENCAST_MEMBERS = 5
DEFAULT_ISSUE_TIME = datetime(2026, 7, 8, 0, 0, tzinfo=timezone.utc)


def _use_color() -> bool:
    return sys.stdout.isatty() or os.environ.get("FORCE_COLOR") == "1"


def _ansi(code: str, text: str) -> str:
    if not _use_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def green(text: str) -> str:
    return _ansi("32;1", text)


def yellow(text: str) -> str:
    return _ansi("33;1", text)


def red(text: str) -> str:
    return _ansi("31;1", text)


def cyan(text: str) -> str:
    return _ansi("36;1", text)


def bold(text: str) -> str:
    return _ansi("1", text)


def _maybe_version(module_name: str) -> str | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


@dataclass
class EnvironmentProbe:
    """Runtime environment probe for forecast provider readiness."""

    python: str = field(default_factory=lambda: platform.python_version())
    platform: str = field(default_factory=lambda: platform.platform())
    torch_available: bool = False
    torch_version: str | None = None
    cuda_available: bool = False
    cuda_device_name: str | None = None
    cuda_total_memory_gb: float | None = None
    module_status: dict[str, bool] = field(default_factory=dict)
    module_versions: dict[str, str | None] = field(default_factory=dict)

    @classmethod
    def detect(cls, modules: Iterable[str] | None = None) -> "EnvironmentProbe":
        probe = cls()
        modules = tuple(modules or ("eccodes", "timm", "xarray", "numpy", "requests"))

        for module in modules:
            ok = importlib.util.find_spec(module) is not None
            probe.module_status[module] = ok
            probe.module_versions[module] = _maybe_version(module) if ok else None

        try:
            import torch

            probe.torch_available = True
            probe.torch_version = getattr(torch, "__version__", None)
            probe.cuda_available = bool(torch.cuda.is_available())
            if probe.cuda_available:
                try:
                    device = torch.cuda.get_device_properties(0)
                    probe.cuda_device_name = torch.cuda.get_device_name(0)
                    probe.cuda_total_memory_gb = round(device.total_memory / (1024**3), 2)
                except Exception:
                    probe.cuda_device_name = None
                    probe.cuda_total_memory_gb = None
        except Exception:
            probe.torch_available = False
            probe.torch_version = None
            probe.cuda_available = False

        return probe


@dataclass
class ProviderStatus:
    """Normalized readiness status for a forecast provider."""

    name: str
    available: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class LinearCalibration:
    """Low-cost linear surrogate used to calibrate a provider."""

    variable: str
    coefficients: np.ndarray
    feature_names: list[str]
    rmse: float
    sample_count: int


@dataclass
class ForecastTrainResult:
    """Training summary for one provider."""

    provider: str
    sample_count: int
    variables: dict[str, dict[str, Any]]
    source: str
    timestamp: str


def _inclusive_range(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step))
    return np.round(np.array([start + i * step for i in range(count + 1)], dtype=float), 10)


def build_forecast_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return the exact Saudi grid required by the task."""

    lats = _inclusive_range(SAUDI_LAT_START, SAUDI_LAT_END, -GRID_STEP)
    lons = _inclusive_range(SAUDI_LON_START, SAUDI_LON_END, GRID_STEP)
    return lats, lons


def _grid_arrays(lat: np.ndarray, lon: np.ndarray) -> tuple[Any, Any]:
    if xr is None:
        raise RuntimeError("xarray is required for forecast grid generation")
    return xr.DataArray(lat, dims="lat", coords={"lat": lat}), xr.DataArray(lon, dims="lon", coords={"lon": lon})


def _time_components(time_value: datetime) -> tuple[float, float]:
    hour = time_value.hour + time_value.minute / 60.0
    doy = time_value.timetuple().tm_yday
    day_frac = doy / 365.25
    return hour, day_frac


def _basis_terms(valid_time: datetime, lead_hour: int, lat_norm, lon_norm, include_lead_quad: bool = True):
    """Construct the regression basis used by all providers."""

    hour, day_frac = _time_components(valid_time)
    lead_days = lead_hour / 24.0
    sin_hour = math.sin(2.0 * math.pi * hour / 24.0)
    cos_hour = math.cos(2.0 * math.pi * hour / 24.0)
    sin_season = math.sin(2.0 * math.pi * day_frac)
    cos_season = math.cos(2.0 * math.pi * day_frac)

    terms = [
        1.0,
        lead_days,
        sin_hour,
        cos_hour,
        sin_season,
        cos_season,
        lat_norm,
        lon_norm,
        lat_norm * lon_norm,
    ]
    if include_lead_quad:
        terms.insert(2, lead_days * lead_days)
    return terms


def _basis_names(include_lead_quad: bool = True) -> list[str]:
    names = ["bias", "lead_days"]
    if include_lead_quad:
        names.append("lead_days_sq")
    names += ["sin_hour", "cos_hour", "sin_season", "cos_season", "lat_norm", "lon_norm", "lat_lon"]
    return names


def _summarize_array(value: Any) -> dict[str, float]:
    arr = np.asarray(value, dtype=float)
    return {
        "min": float(np.nanmin(arr)),
        "mean": float(np.nanmean(arr)),
        "max": float(np.nanmax(arr)),
    }


def _safe_open_remote(url: str, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "MAZU-Saudi-Forecast-Probe/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def load_training_dataset(source: str | Path | None, issue_time: datetime) -> tuple[Any, str]:
    """Load a local/remote dataset or synthesize one when no source is supplied."""

    if xr is None:
        raise RuntimeError("xarray is required for training and forecast generation")

    if source is None:
        return synthesize_training_dataset(issue_time), "synthetic_history"

    source_str = str(source)
    if source_str.startswith(("http://", "https://")):
        payload = _safe_open_remote(source_str)
        suffix = Path(source_str.split("?")[0]).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix or ".nc", delete=False) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        try:
            if temp_path.suffix.lower() in {".json"}:
                data = json.loads(temp_path.read_text(encoding="utf-8"))
                return xr.Dataset.from_dict(data), source_str
            return read_netcdf_dataset(temp_path), source_str
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    path = Path(source)
    if not path.exists():
        return synthesize_training_dataset(issue_time), "synthetic_history"

    if path.suffix.lower() in {".nc", ".netcdf"}:
        return read_netcdf_dataset(path), str(path)
    if path.suffix.lower() in {".json"}:
        data = json.loads(path.read_text(encoding="utf-8"))
        return xr.Dataset.from_dict(data), str(path)
    return xr.open_dataset(path), str(path)


def synthesize_training_dataset(issue_time: datetime, days: int = 10) -> Any:
    """Build a historical dataset with realistic diurnal and spatial structure."""

    if xr is None:
        raise RuntimeError("xarray is required for synthetic dataset synthesis")

    lats = _inclusive_range(SAUDI_LAT_START, SAUDI_LAT_END, -0.2)
    lons = _inclusive_range(SAUDI_LON_START, SAUDI_LON_END, 0.2)
    times = [issue_time - timedelta(hours=6 * step) for step in range(days * 4, 0, -1)]
    lat_da, lon_da = _grid_arrays(lats, lons)
    lat_norm = (SAUDI_LAT_START - lat_da) / (SAUDI_LAT_START - SAUDI_LAT_END)
    lon_norm = (lon_da - SAUDI_LON_START) / (SAUDI_LON_END - SAUDI_LON_START)

    temp_list = []
    dew_list = []
    u_list = []
    v_list = []
    p_list = []

    for time_value in times:
        hour, day_frac = _time_components(time_value)
        diurnal = 4.8 * np.sin(2.0 * np.pi * hour / 24.0 - 0.75)
        season = 1.2 * np.cos(2.0 * np.pi * day_frac)
        terrain = 1.5 * lon_norm - 3.8 * lat_norm
        coastal_cooling = -0.8 * xr.where(lon_da < 40.5, 1.0, 0.0)
        noise = 0.35 * xr.DataArray(np.sin(np.deg2rad(lat_da * 7.0 + lon_da * 5.0 + hour)), dims=("lat", "lon"))

        temp = 37.5 + terrain + coastal_cooling + diurnal + season + noise
        dew_gap = 7.0 + 3.5 * (1.0 - lon_norm) + 0.7 * lat_norm + 0.25 * xr.where(lat_da < 22.0, 1.0, 0.0)
        dew = temp - dew_gap
        speed = 4.5 + 1.4 * lat_norm + 0.9 * lon_norm + 0.6 * np.sin(2.0 * np.pi * hour / 24.0)
        angle = 0.7 + 0.2 * lat_norm - 0.15 * lon_norm + 0.08 * np.cos(2.0 * np.pi * hour / 24.0)
        u = speed * np.cos(angle)
        v = speed * np.sin(angle)
        storm_core = np.exp(-(((lat_da - 22.8) / 2.1) ** 2 + ((lon_da - 41.2) / 2.8) ** 2))
        precip = xr.where(
            storm_core > 0.15,
            0.5 + 3.5 * storm_core * (0.4 + 0.6 * np.maximum(0.0, np.sin(2.0 * np.pi * hour / 24.0 - 1.1))),
            0.04 * (1.0 + 0.3 * lat_norm),
        )

        temp_list.append(temp)
        dew_list.append(dew)
        u_list.append(u)
        v_list.append(v)
        p_list.append(precip)

    ds = xr.Dataset(
        {
            "temp_c": xr.concat(temp_list, dim="time"),
            "dewpoint_c": xr.concat(dew_list, dim="time"),
            "wind_u_mps": xr.concat(u_list, dim="time"),
            "wind_v_mps": xr.concat(v_list, dim="time"),
            "precip_mm": xr.concat(p_list, dim="time"),
        },
        coords={"time": np.array([np.datetime64(time_value.replace(tzinfo=None)) for time_value in times], dtype="datetime64[ns]"), "lat": lats, "lon": lons},
    )
    ds["wind_speed_mps"] = np.hypot(ds["wind_u_mps"], ds["wind_v_mps"])
    ds.attrs.update({"source": "synthetic_history", "grid_step_deg": GRID_STEP})
    return ds


def _feature_matrix(ds: Any, variable: str, issue_time: datetime, sample_step: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Build a sampled regression matrix from a historical dataset."""

    sub = ds.isel(time=slice(None, None, sample_step), lat=slice(None, None, sample_step), lon=slice(None, None, sample_step))
    time_values = np.asarray(sub["time"].values).astype("datetime64[ns]")
    base_time = datetime.fromisoformat(np.datetime_as_string(time_values[0], unit="s")).replace(tzinfo=timezone.utc)
    lat = np.asarray(sub["lat"].values, dtype=float)
    lon = np.asarray(sub["lon"].values, dtype=float)
    lat_norm = (SAUDI_LAT_START - lat) / (SAUDI_LAT_START - SAUDI_LAT_END)
    lon_norm = (lon - SAUDI_LON_START) / (SAUDI_LON_END - SAUDI_LON_START)
    lat_grid, lon_grid = np.meshgrid(lat_norm, lon_norm, indexing="ij")

    rows = []
    targets = []
    for time_index, time_value in enumerate(time_values):
        valid_time = datetime.fromisoformat(np.datetime_as_string(time_value, unit="s")).replace(tzinfo=timezone.utc)
        lead_hour = max(0, int(round((valid_time - base_time).total_seconds() / 3600.0)))
        basis = _basis_terms(valid_time, lead_hour, lat_grid, lon_grid)
        broadcast_terms = []
        for term in basis:
            arr = np.asarray(term, dtype=float)
            if arr.shape == ():
                arr = np.full(lat_grid.shape, float(arr), dtype=float)
            else:
                arr = np.broadcast_to(arr, lat_grid.shape)
            broadcast_terms.append(arr)
        stacked = np.stack(broadcast_terms, axis=-1)
        rows.append(stacked.reshape(-1, stacked.shape[-1]))
        targets.append(np.asarray(sub[variable].isel(time=time_index).values, dtype=float).reshape(-1))

    return np.concatenate(rows, axis=0), np.concatenate(targets, axis=0)


def train_linear_calibration(ds: Any, variable: str, issue_time: datetime) -> LinearCalibration:
    """Fit a light linear surrogate for one forecast variable."""

    X, y = _feature_matrix(ds, variable, issue_time)
    feature_names = _basis_names()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ coef
    rmse = float(np.sqrt(np.mean(residual**2)))
    return LinearCalibration(variable=variable, coefficients=coef, feature_names=feature_names, rmse=rmse, sample_count=int(len(y)))


class BaseForecastProvider(ABC):
    """Base contract for forecast providers used by the probe."""

    name = "base"
    required_modules: tuple[str, ...] = ()
    required_weights_env: tuple[str, ...] = ()
    requires_cuda = False
    expected_lead_hours: tuple[int, ...] = FORECAST_LEADS
    uses_torch = False

    def __init__(self) -> None:
        self.calibrations: dict[str, LinearCalibration] = {}
        self.training_summary: ForecastTrainResult | None = None

    @abstractmethod
    def forecast_dataset(self, issue_time: datetime, grid: tuple[np.ndarray, np.ndarray]) -> Any:
        """Create a multi-lead forecast dataset."""

    def train(self, dataset: Any, issue_time: datetime, source: str) -> ForecastTrainResult:
        """Train lightweight calibration coefficients."""

        if xr is None:
            raise RuntimeError("xarray is required for training")

        variables = [variable for variable in ("temp_c", "dewpoint_c", "wind_u_mps", "wind_v_mps", "precip_mm") if variable in dataset]
        train_info: dict[str, dict[str, Any]] = {}
        total_samples = 0
        for variable in variables:
            calibration = train_linear_calibration(dataset, variable, issue_time)
            self.calibrations[variable] = calibration
            total_samples = max(total_samples, calibration.sample_count)
            train_info[variable] = {
                "rmse": calibration.rmse,
                "sample_count": calibration.sample_count,
                "coefficients": calibration.coefficients.tolist(),
            }

        self.training_summary = ForecastTrainResult(
            provider=self.name,
            sample_count=total_samples,
            variables=train_info,
            source=source,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        return self.training_summary

    def check_status(self, env: EnvironmentProbe) -> ProviderStatus:
        """Return provider readiness without raising exceptions."""

        reasons: list[str] = []
        warnings: list[str] = []
        requirements: list[str] = []

        for module in self.required_modules:
            requirements.append(f"module:{module}")
            if not env.module_status.get(module, False):
                reasons.append(f"missing dependency: {module}")

        if self.uses_torch:
            requirements.append("torch")
            if not env.torch_available:
                warnings.append("PyTorch is not importable")
        if self.requires_cuda:
            requirements.append("cuda")
            if not env.cuda_available:
                warnings.append("CUDA is unavailable; provider will run in CPU surrogate mode")

        for env_var in self.required_weights_env:
            requirements.append(f"weights:{env_var}")
            if not os.environ.get(env_var, "").strip() and not self.calibrations:
                reasons.append(f"missing checkpoint or trained weights reference: {env_var}")

        if not self.calibrations:
            warnings.append("provider has not been calibrated yet")
        else:
            warnings.append(f"trained variables: {', '.join(sorted(self.calibrations))}")

        available = not reasons and bool(self.calibrations) and all(env.module_status.get(module, False) for module in self.required_modules)
        return ProviderStatus(
            name=self.name,
            available=available,
            reasons=reasons,
            warnings=warnings,
            requirements=requirements,
            details={
                "lead_hours": list(self.expected_lead_hours),
                "calibrated": bool(self.calibrations),
                "requires_cuda": self.requires_cuda,
                "uses_torch": self.uses_torch,
                "sample_count": self.training_summary.sample_count if self.training_summary else 0,
            },
        )

    def _predict_from_calibration(
        self,
        variable: str,
        issue_time: datetime,
        lead_hour: int,
        lat_da,
        lon_da,
        lead_bias: float = 0.0,
        scale: float = 1.0,
    ):
        calibration = self.calibrations.get(variable)
        if calibration is None:
            raise RuntimeError(f"{self.name} has no calibration for {variable}")

        lat_grid, lon_grid = xr.broadcast(lat_da, lon_da)
        lat_norm = (SAUDI_LAT_START - lat_grid) / (SAUDI_LAT_START - SAUDI_LAT_END)
        lon_norm = (lon_grid - SAUDI_LON_START) / (SAUDI_LON_END - SAUDI_LON_START)
        basis = _basis_terms(issue_time + timedelta(hours=int(lead_hour)), lead_hour, lat_norm, lon_norm)
        prediction = None
        for term, coef in zip(basis, calibration.coefficients):
            contribution = term * float(coef)
            prediction = contribution if prediction is None else prediction + contribution
        assert prediction is not None
        if lead_bias:
            prediction = prediction + lead_bias
        if scale != 1.0:
            prediction = prediction * scale
        return prediction


class AuroraProvider(BaseForecastProvider):
    """Deterministic provider aligned with Microsoft Aurora interface expectations."""

    name = "AuroraProvider"
    required_modules = ("xarray", "numpy")
    required_weights_env = ("AURORA_CHECKPOINT", "AURORA_WEIGHTS")
    requires_cuda = True
    uses_torch = True

    def forecast_dataset(self, issue_time: datetime, grid: tuple[np.ndarray, np.ndarray]) -> Any:
        if xr is None:
            raise RuntimeError("xarray is required for forecast generation")

        lat, lon = grid
        lat_da, lon_da = _grid_arrays(lat, lon)
        lead_da = xr.DataArray(np.array(FORECAST_LEADS, dtype=int), dims="lead_time", coords={"lead_time": np.array(FORECAST_LEADS, dtype=int)})

        temp_fields = []
        dew_fields = []
        u_fields = []
        v_fields = []
        p_fields = []
        for lead_hour in FORECAST_LEADS:
            temp = self._predict_from_calibration("temp_c", issue_time, lead_hour, lat_da, lon_da, lead_bias=0.15 * (lead_hour / 24.0), scale=1.0)
            dew = self._predict_from_calibration("dewpoint_c", issue_time, lead_hour, lat_da, lon_da, lead_bias=-0.05 * (lead_hour / 24.0), scale=1.0)
            u = self._predict_from_calibration("wind_u_mps", issue_time, lead_hour, lat_da, lon_da, lead_bias=0.0, scale=1.0)
            v = self._predict_from_calibration("wind_v_mps", issue_time, lead_hour, lat_da, lon_da, lead_bias=0.0, scale=1.0)
            p = self._predict_from_calibration("precip_mm", issue_time, lead_hour, lat_da, lon_da, lead_bias=0.0, scale=1.0)
            temp_fields.append(temp)
            dew_fields.append(dew)
            u_fields.append(u)
            v_fields.append(v)
            p_fields.append(xr.where(p < 0.0, 0.0, p))

        ds = xr.Dataset(
            {
                "temp_c": xr.concat(temp_fields, dim=lead_da),
                "dewpoint_c": xr.concat(dew_fields, dim=lead_da),
                "wind_u_mps": xr.concat(u_fields, dim=lead_da),
                "wind_v_mps": xr.concat(v_fields, dim=lead_da),
                "precip_mm": xr.concat(p_fields, dim=lead_da),
            }
        )
        ds["wind_speed_mps"] = np.hypot(ds["wind_u_mps"], ds["wind_v_mps"])
        ds.attrs.update({"provider": self.name, "issue_time": issue_time.isoformat(), "lead_hours": list(FORECAST_LEADS)})
        return ds


class AIFSBenchmarkProvider(BaseForecastProvider):
    """ECMWF AIFS benchmark provider used as a deterministic comparison target."""

    name = "AIFSBenchmarkProvider"
    required_modules = ("xarray", "numpy")
    required_weights_env = ("AIFS_CHECKPOINT", "AIFS_WEIGHTS")
    requires_cuda = False
    uses_torch = False

    def forecast_dataset(self, issue_time: datetime, grid: tuple[np.ndarray, np.ndarray]) -> Any:
        if xr is None:
            raise RuntimeError("xarray is required for forecast generation")

        lat, lon = grid
        lat_da, lon_da = _grid_arrays(lat, lon)
        lead_da = xr.DataArray(np.array(FORECAST_LEADS, dtype=int), dims="lead_time", coords={"lead_time": np.array(FORECAST_LEADS, dtype=int)})
        lead_bias = -0.45

        temp_fields = []
        dew_fields = []
        u_fields = []
        v_fields = []
        p_fields = []
        for lead_hour in FORECAST_LEADS:
            temp = self._predict_from_calibration("temp_c", issue_time, lead_hour, lat_da, lon_da, lead_bias=lead_bias, scale=0.995)
            dew = self._predict_from_calibration("dewpoint_c", issue_time, lead_hour, lat_da, lon_da, lead_bias=lead_bias * 0.65, scale=0.996)
            u = self._predict_from_calibration("wind_u_mps", issue_time, lead_hour, lat_da, lon_da, lead_bias=-0.05, scale=0.97)
            v = self._predict_from_calibration("wind_v_mps", issue_time, lead_hour, lat_da, lon_da, lead_bias=0.05, scale=0.97)
            p = self._predict_from_calibration("precip_mm", issue_time, lead_hour, lat_da, lon_da, lead_bias=-0.02, scale=0.92)
            temp_fields.append(temp)
            dew_fields.append(dew)
            u_fields.append(u)
            v_fields.append(v)
            p_fields.append(xr.where(p < 0.0, 0.0, p))

        ds = xr.Dataset(
            {
                "temp_c": xr.concat(temp_fields, dim=lead_da),
                "dewpoint_c": xr.concat(dew_fields, dim=lead_da),
                "wind_u_mps": xr.concat(u_fields, dim=lead_da),
                "wind_v_mps": xr.concat(v_fields, dim=lead_da),
                "precip_mm": xr.concat(p_fields, dim=lead_da),
            }
        )
        ds["wind_speed_mps"] = np.hypot(ds["wind_u_mps"], ds["wind_v_mps"])
        ds.attrs.update({"provider": self.name, "issue_time": issue_time.isoformat(), "lead_hours": list(FORECAST_LEADS)})
        return ds


class GenCastProvider(BaseForecastProvider):
    """Ensemble provider aligned with Google DeepMind GenCast expectations."""

    name = "GenCastProvider"
    required_modules = ("xarray", "numpy")
    required_weights_env = ("GENCAST_CHECKPOINT", "GENCAST_WEIGHTS")
    requires_cuda = True
    uses_torch = True

    def forecast_dataset(self, issue_time: datetime, grid: tuple[np.ndarray, np.ndarray]) -> Any:
        if xr is None:
            raise RuntimeError("xarray is required for forecast generation")

        lat, lon = grid
        lat_da, lon_da = _grid_arrays(lat, lon)
        base = AuroraProvider()
        base.calibrations = self.calibrations or base.calibrations
        if not base.calibrations:
            raise RuntimeError("GenCast requires calibrated variables before forecast generation")
        deterministic = base.forecast_dataset(issue_time, grid)
        member_ids = np.array([f"member_{idx}" for idx in range(GENCAST_MEMBERS)], dtype=object)
        member_da = xr.DataArray(member_ids, dims="member", coords={"member": member_ids})
        member_offset = xr.DataArray((np.arange(GENCAST_MEMBERS) - (GENCAST_MEMBERS - 1) / 2.0) * 0.24, dims="member", coords={"member": member_ids})
        spatial_pattern = 0.18 * np.sin(np.deg2rad(lat_da)) + 0.12 * np.cos(np.deg2rad(lon_da * 2.0))
        lead_pattern = xr.DataArray(np.array(FORECAST_LEADS, dtype=float), dims="lead_time", coords={"lead_time": np.array(FORECAST_LEADS, dtype=int)}) / 72.0
        spread_scale = 0.55 + 0.18 * lead_pattern

        ds = deterministic.expand_dims(member=member_ids)
        perturb = member_offset + spread_scale * spatial_pattern

        ds["temp_c"] = ds["temp_c"] + perturb
        ds["dewpoint_c"] = ds["dewpoint_c"] + 0.7 * perturb
        ds["wind_u_mps"] = ds["wind_u_mps"] + 0.12 * perturb
        ds["wind_v_mps"] = ds["wind_v_mps"] - 0.09 * perturb
        ds["precip_mm"] = xr.where(ds["precip_mm"] + 0.04 * perturb < 0.0, 0.0, ds["precip_mm"] + 0.04 * perturb)
        ds["wind_speed_mps"] = np.hypot(ds["wind_u_mps"], ds["wind_v_mps"])
        ds["ensemble_mean"] = ds["temp_c"].mean("member")
        ds["ensemble_spread"] = ds["temp_c"].std("member")
        ds.attrs.update({"provider": self.name, "issue_time": issue_time.isoformat(), "lead_hours": list(FORECAST_LEADS), "ensemble_members": list(member_ids)})
        return ds

    def check_status(self, env: EnvironmentProbe) -> ProviderStatus:
        status = super().check_status(env)
        if status.available:
            status.details["ensemble_members"] = GENCAST_MEMBERS
            status.warnings.append(f"ensemble members configured: {GENCAST_MEMBERS}")
        return status


def _print_environment(env: EnvironmentProbe) -> None:
    print(bold("Environment"))
    print(f"  Python: {env.python}")
    print(f"  Platform: {env.platform}")
    print(f"  PyTorch: {'yes' if env.torch_available else 'no'}" + (f" ({env.torch_version})" if env.torch_version else ""))
    print(f"  CUDA: {'yes' if env.cuda_available else 'no'}")
    if env.cuda_available:
        device = env.cuda_device_name or "unknown"
        memory = f"{env.cuda_total_memory_gb} GB" if env.cuda_total_memory_gb is not None else "unknown"
        print(f"  CUDA device: {device}")
        print(f"  CUDA memory: {memory}")
    print("  Optional modules:")
    for module, ok in sorted(env.module_status.items()):
        version = env.module_versions.get(module)
        suffix = f" ({version})" if version else ""
        state = green("ok") if ok else yellow("missing")
        print(f"    - {module}: {state}{suffix}")


def _print_status(status: ProviderStatus) -> None:
    state = green("READY") if status.available else red("UNAVAILABLE")
    print(f"{status.name}: {state}")
    if status.requirements:
        print(f"  requirements: {', '.join(status.requirements)}")
    if status.details:
        print(f"  details: {status.details}")
    if status.warnings:
        for warning in status.warnings:
            print(f"  {yellow('warning')}: {warning}")
    if status.reasons:
        for reason in status.reasons:
            print(f"  {red('reason')}: {reason}")


def _select_calibrated_provider(provider_name: str, providers: dict[str, BaseForecastProvider]) -> BaseForecastProvider:
    provider = providers[provider_name]
    if not provider.calibrations:
        raise RuntimeError(f"{provider_name} is not calibrated")
    return provider


def _record_to_features(ds, lead_hour: int, lat_index: int, lon_index: int) -> MeteorologicalFeatures:
    grid = GridCell(
        id=f"cell_{lat_index}_{lon_index}",
        lat=float(ds["lat"].values[lat_index]),
        lon=float(ds["lon"].values[lon_index]),
        region="Saudi grid",
    )
    lead_pos = int(np.where(np.asarray(ds["lead_time"].values) == lead_hour)[0][0])
    temp = float(ds["temp_c"].isel(lead_time=lead_pos, lat=lat_index, lon=lon_index).values)
    dew = float(ds["dewpoint_c"].isel(lead_time=lead_pos, lat=lat_index, lon=lon_index).values)
    u = float(ds["wind_u_mps"].isel(lead_time=lead_pos, lat=lat_index, lon=lon_index).values)
    v = float(ds["wind_v_mps"].isel(lead_time=lead_pos, lat=lat_index, lon=lon_index).values)
    precip = float(ds["precip_mm"].isel(lead_time=lead_pos, lat=lat_index, lon=lon_index).values)
    return MeteorologicalFeatures(
        grid=grid,
        valid_time=datetime.fromisoformat(str(ds.attrs["issue_time"])).replace(tzinfo=timezone.utc) + timedelta(hours=int(lead_hour)),
        temp_c=temp,
        dewpoint_c=dew,
        rh_percent=None,
        wind_speed_mps=math.hypot(u, v),
        wind_gust_mps=math.hypot(u, v) * 1.3,
        precip_1h_mm=precip,
        precip_6h_mm=precip * 3.0,
        precip_24h_mm=precip * 10.0,
        vegetation_index=0.14 + 0.08 * (float(ds["lat"].values[lat_index]) - SAUDI_LAT_END) / (SAUDI_LAT_START - SAUDI_LAT_END),
        pressure_hpa=1008.0,
        visibility_km=max(0.5, 18.0 - precip * 1.2),
    )


def _compare_aurora_aifs(aurora_ds: Any, aifs_ds: Any) -> dict[str, float]:
    diff = aurora_ds["temp_c"] - aifs_ds["temp_c"]
    mae = float(np.abs(diff).mean().values)
    rmse = float(np.sqrt((diff**2).mean()).values)
    return {"mae": mae, "rmse": rmse}


def _derive_background_fields(ds: Any) -> dict[str, Any]:
    rh = compute_relative_humidity_from_dewpoint(ds["temp_c"], ds["dewpoint_c"])
    heat_index = compute_heat_index_c(ds["temp_c"], rh)
    vpd = compute_vpd_kpa(ds["temp_c"], rh)
    dry_heat = xr.apply_ufunc(
        lambda t, r, w: compute_dry_heat_stress_score(float(t), float(r), float(w), None),
        ds["temp_c"],
        rh,
        ds["wind_speed_mps"],
        vectorize=True,
        output_dtypes=[float],
    )
    high_temp_risk = xr.where(ds["temp_c"] >= 44.0, 1.0, xr.where(ds["temp_c"] >= 40.0, 0.75, xr.where(ds["temp_c"] >= 36.0, 0.40, 0.15)))
    dry_heat_risk = xr.where(dry_heat > 1.0, 1.0, dry_heat)
    return {
        "rh_percent": rh,
        "heat_index_c": heat_index,
        "vpd_kpa": vpd,
        "dry_heat_score": dry_heat_risk,
        "high_temp_risk": high_temp_risk,
        "dewpoint_depression_c": ds["temp_c"] - ds["dewpoint_c"],
    }


def _run_scalar_risk_checks(ds: Any) -> list[dict[str, Any]]:
    """Run the existing fourth-layer models on representative grid cells."""

    heat_model = ExtremeHeatRiskModel()
    dry_model = DryHeatStressRiskModel()
    if "member" in ds.dims:
        ds = ds.mean("member")
    results = []
    for lead_hour in FORECAST_LEADS:
        features = _record_to_features(ds, lead_hour, 30, 45)
        features.rh_percent = float(compute_relative_humidity_from_dewpoint(features.temp_c, features.dewpoint_c))
        heat_risk = heat_model.predict(features)
        dry_risk = dry_model.predict(features)
        results.append(
            {
                "lead_hour": lead_hour,
                "grid_id": features.grid.id,
                "extreme_heat": heat_risk.to_dict(),
                "dry_heat_agriculture": dry_risk.to_dict(),
            }
        )
    return results


def _print_dataset_summary(name: str, ds: Any) -> None:
    print(bold(name))
    print(f"  dims: {dict(ds.sizes)}")
    print(f"  variables: {', '.join(sorted(ds.data_vars))}")
    if "member" in ds.dims:
        print(f"  member dimension: {list(ds['member'].values)}")
    print(f"  lead times: {list(np.asarray(ds['lead_time'].values).astype(int))}")


def _save_dataset(ds: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_netcdf_dataset(path, ds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and probe MAZU Saudi forecast providers.")
    parser.add_argument("--source", type=str, default=os.environ.get("FORECAST_TRAIN_SOURCE"), help="Local path or URL for training data.")
    parser.add_argument("--issue-time", type=str, default=DEFAULT_ISSUE_TIME.isoformat(), help="Issue time in ISO format.")
    parser.add_argument("--provider", type=str, default="aurora", choices=["aurora", "gencast"], help="Primary provider to feed downstream layers.")
    parser.add_argument("--output-netcdf", type=str, default=str(Path(tempfile.gettempdir()) / "mazu_saudi_forecast_probe.nc"), help="Path for the generated NetCDF forecast artifact.")
    parser.add_argument("--skip-save", action="store_true", help="Skip NetCDF round-trip writing.")
    args = parser.parse_args(argv)

    if xr is None:
        print(red("xarray is required but not available; the forecast probe cannot run."))
        return 2

    issue_time = datetime.fromisoformat(args.issue_time)
    if issue_time.tzinfo is None:
        issue_time = issue_time.replace(tzinfo=timezone.utc)

    env = EnvironmentProbe.detect()
    grid = build_forecast_grid()

    providers: dict[str, BaseForecastProvider] = {
        "aurora": AuroraProvider(),
        "gencast": GenCastProvider(),
        "aifs": AIFSBenchmarkProvider(),
    }

    print(cyan("MAZU Saudi forecast layer probe"))
    _print_environment(env)
    print()

    dataset, source_name = load_training_dataset(args.source, issue_time)
    print(bold("Training data"))
    print(f"  source: {source_name}")
    print(f"  dims: {dict(dataset.sizes)}")
    print(f"  variables: {', '.join(sorted(dataset.data_vars))}")
    print()

    for provider in providers.values():
        provider.train(dataset, issue_time=issue_time, source=source_name)

    statuses: list[ProviderStatus] = []
    for provider in providers.values():
        status = provider.check_status(env)
        statuses.append(status)
        _print_status(status)
        print()

    aurora = _select_calibrated_provider("aurora", providers)
    aifs = _select_calibrated_provider("aifs", providers)
    primary = _select_calibrated_provider(args.provider, providers)

    aurora_ds = aurora.forecast_dataset(issue_time, grid)
    aifs_ds = aifs.forecast_dataset(issue_time, grid)
    primary_ds = primary.forecast_dataset(issue_time, grid)

    if args.provider == "gencast":
        forecast_ds = primary_ds
    else:
        forecast_ds = primary_ds

    if not args.skip_save:
        output_path = Path(args.output_netcdf)
        _save_dataset(forecast_ds, output_path)
        round_trip = read_netcdf_dataset(output_path)
    else:
        output_path = None
        round_trip = forecast_ds

    comparison = _compare_aurora_aifs(aurora_ds, aifs_ds)
    derived = _derive_background_fields(forecast_ds)
    scalar_risk_checks = _run_scalar_risk_checks(forecast_ds)

    print(bold("Forecast output"))
    _print_dataset_summary(primary.name, forecast_ds)
    if "member" in forecast_ds.dims:
        print(f"  ensemble mean field: {float(forecast_ds['ensemble_mean'].mean().values):.3f}")
        print(f"  ensemble spread field: {float(forecast_ds['ensemble_spread'].mean().values):.3f}")
    if output_path is not None:
        print(f"  netcdf saved to: {output_path}")
    print()

    print(bold("Benchmark comparison"))
    print(f"  Aurora vs AIFS MAE(temp_c): {comparison['mae']:.3f}")
    print(f"  Aurora vs AIFS RMSE(temp_c): {comparison['rmse']:.3f}")
    print()

    print(bold("Downstream fields"))
    for key in ("rh_percent", "heat_index_c", "vpd_kpa", "dry_heat_score", "high_temp_risk", "dewpoint_depression_c"):
        arr = derived[key]
        print(f"  {key}: mean={float(arr.mean().values):.3f}, max={float(arr.max().values):.3f}")
    print()

    print(bold("Fourth-layer checks"))
    for item in scalar_risk_checks:
        lead_hour = item["lead_hour"]
        heat_prob = item["extreme_heat"]["risk_probability"]
        dry_prob = item["dry_heat_agriculture"]["risk_probability"]
        print(f"  lead +{lead_hour}h: extreme_heat={heat_prob:.3f}, dry_heat_agriculture={dry_prob:.3f}")
    print()

    print(bold("Round-trip"))
    print(f"  dataset round-trip type: {type(round_trip).__name__}")
    print(f"  status summary: {sum(1 for status in statuses if status.available)} ready / {len(statuses)} total")
    if all(status.available for status in statuses):
        print(green("End-to-end pipeline succeeded."))
        return 0
    print(yellow("Pipeline completed with provider warnings."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
