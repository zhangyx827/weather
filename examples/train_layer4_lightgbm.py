"""Train hazard-specific Layer-4 LightGBM boosters from daily inputs."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from mazu_saudi.data import read_netcdf_dataset
from mazu_saudi.indicators.physical import (
    compute_cape_placeholder,
    compute_dry_heat_stress_score,
    compute_flash_flood_screening_score,
    compute_ivt_placeholder,
    compute_pwat_placeholder,
)
from mazu_saudi.risk.layer4_features import (
    feature_matrix_from_dataset,
    feature_names_for_hazard,
    prepare_feature_frame,
    required_feature_names_for_hazard,
)
from mazu_saudi.risk.ml import LightGBMAdapter

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

DEFAULT_SOURCE = ROOT / "data" / "raw" / "era5_saudi_20250616.nc"
DEFAULT_MODEL_DIR = ROOT / "models" / "layer4"
SOURCE_FORMATS = ("auto", "era5", "indicator-netcdf", "indicator-parquet", "indicator-csv", "indicator-json")
HAZARD_TYPES = ("extreme_heat", "dry_heat_agriculture", "flash_flood")


def normalize_dataset(dataset):
    ds = dataset.copy()

    rename_dims = {}
    if "lat" in ds.dims and "latitude" not in ds.dims:
        rename_dims["lat"] = "latitude"
    if "lon" in ds.dims and "longitude" not in ds.dims:
        rename_dims["lon"] = "longitude"
    if rename_dims:
        ds = ds.rename(rename_dims)

    return ds


def to_celsius(temperature):
    return temperature - 273.15


def saturation_vapor_pressure_kpa(temp_c):
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def relative_humidity_from_dewpoint(temp_c, dewpoint_c):
    es = saturation_vapor_pressure_kpa(temp_c)
    ea = saturation_vapor_pressure_kpa(dewpoint_c)
    return np.clip(100.0 * ea / np.maximum(es, 1e-6), 0.0, 100.0)


def compute_vpd_kpa(temp_c, dewpoint_c):
    es = saturation_vapor_pressure_kpa(temp_c)
    ea = saturation_vapor_pressure_kpa(dewpoint_c)
    return np.maximum(es - ea, 0.0)


def compute_heat_index_c(temp_c, rh_percent):
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


def compute_wind_speed_mps(u10, v10):
    return np.sqrt(np.asarray(u10) ** 2 + np.asarray(v10) ** 2)


def build_target_from_features(features: np.ndarray, hazard_type: str) -> np.ndarray:
    names = feature_names_for_hazard(hazard_type)
    index = {name: position for position, name in enumerate(names)}
    if hazard_type == "extreme_heat":
        temp = features[:, index["temp_c"]]
        hi = features[:, index["heat_index_c"]]
        tmax = features[:, index["tmax_c"]]
        target = np.maximum((temp - 38.0) / 12.0, (hi - 41.0) / 12.0)
        target = np.maximum(target, (tmax - 42.0) / 10.0)
        return np.clip(target, 0.0, 1.0).astype(np.float32)
    if hazard_type == "dry_heat_agriculture":
        temp = features[:, index["temp_c"]]
        rh = features[:, index["relative_humidity_percent"]]
        wind = features[:, index["wind_speed_mps"]]
        return np.array(
            [compute_dry_heat_stress_score(t, rh_value, wind_value) for t, rh_value, wind_value in zip(temp, rh, wind)],
            dtype=np.float32,
        )
    if hazard_type == "flash_flood":
        precip = features[:, index["daily_precip_total"]]
        conv = features[:, index["daily_convective_precip"]]
        cape = features[:, index["cape"]]
        pwat = features[:, index["pwat"]]
        ivt = features[:, index["ivt"]]
        shear = features[:, index["wind_shear_850_200"]]
        screen = features[:, index["flash_flood_risk"]]
        target = np.maximum.reduce(
            [
                precip / 50.0,
                conv / 30.0,
                cape / 2500.0,
                pwat / 50.0,
                ivt / 400.0,
                shear / 60.0,
                screen / 4.0,
            ]
        )
        return np.clip(target, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"Unsupported hazard type: {hazard_type}")


def _require_pandas():
    if pd is None:
        raise RuntimeError("pandas is required for daily Layer-4 training inputs")


def _normalize_date_series(series):
    _require_pandas()
    def _parse_one(value):
        try:
            return pd.Timestamp(value)
        except Exception:
            return pd.NaT

    parsed = series.map(_parse_one)
    if parsed.isna().any():
        raise ValueError("daily training data contains invalid date values")
    if ((parsed.dt.hour != 0) | (parsed.dt.minute != 0) | (parsed.dt.second != 0)).any():
        raise ValueError("daily training data requires day-level timestamps without sub-daily time components")
    return parsed.dt.strftime("%Y-%m-%d")


def _ensure_daily_date_column(table):
    _require_pandas()
    working = table.copy()
    if "date" in working.columns:
        working["date"] = _normalize_date_series(working["date"])
        return working
    for candidate in ("time", "valid_time"):
        if candidate in working.columns:
            working["date"] = _normalize_date_series(working[candidate])
            return working
    raise KeyError("daily training tables require a 'date', 'time', or 'valid_time' column")


def _numeric_series(table, names, *, scale: float | None = None):
    _require_pandas()
    for name in names:
        if name in table.columns:
            values = pd.to_numeric(table[name], errors="coerce")
            if scale is not None:
                values = values * scale
            return values.astype(np.float32)
    raise KeyError(names[0])


def _optional_numeric_series(table, names, *, scale: float | None = None):
    try:
        return _numeric_series(table, names, scale=scale)
    except KeyError:
        return None


def _full_like(series, fill_value=np.nan):
    return pd.Series(np.full(len(series), fill_value, dtype=np.float32), index=series.index, dtype=np.float32)


def _resolve_temperature_c(table):
    try:
        return _numeric_series(table, ("t2m_c", "temp_c"))
    except KeyError:
        return _numeric_series(table, ("t2m",), scale=1.0).astype(np.float32) - np.float32(273.15)


def _resolve_tmax_c(table, fallback_temp):
    try:
        return _numeric_series(table, ("tmax_c", "mx2t_c"))
    except KeyError:
        mx2t = _optional_numeric_series(table, ("mx2t",))
        if mx2t is not None:
            return mx2t.astype(np.float32) - np.float32(273.15)
    return fallback_temp.astype(np.float32)


def _resolve_tmin_c(table, fallback_temp):
    try:
        return _numeric_series(table, ("tmin_c", "mn2t_c"))
    except KeyError:
        mn2t = _optional_numeric_series(table, ("mn2t",))
        if mn2t is not None:
            return mn2t.astype(np.float32) - np.float32(273.15)
    return fallback_temp.astype(np.float32)


def _resolve_relative_humidity(table, temp_c):
    direct = _optional_numeric_series(table, ("rh2m", "relative_humidity_percent", "rh_percent"))
    if direct is not None:
        return direct
    dewpoint_c = _optional_numeric_series(table, ("d2m_c", "dewpoint_c"))
    if dewpoint_c is None:
        dewpoint_k = _optional_numeric_series(table, ("d2m",))
        if dewpoint_k is not None:
            dewpoint_c = dewpoint_k.astype(np.float32) - np.float32(273.15)
    if dewpoint_c is None:
        raise KeyError("rh2m")
    values = relative_humidity_from_dewpoint(temp_c.to_numpy(dtype=np.float32), dewpoint_c.to_numpy(dtype=np.float32))
    return pd.Series(values.astype(np.float32), index=temp_c.index)


def _resolve_vpd(table, temp_c, rh):
    direct = _optional_numeric_series(table, ("vpd_kpa",))
    if direct is not None:
        return direct
    values = compute_vpd_kpa(temp_c.to_numpy(dtype=np.float32), rh.to_numpy(dtype=np.float32))
    return pd.Series(np.asarray(values, dtype=np.float32), index=temp_c.index)


def _resolve_heat_index(table, temp_c, rh):
    direct = _optional_numeric_series(table, ("heat_index_c",))
    if direct is not None:
        return direct
    values = compute_heat_index_c(temp_c.to_numpy(dtype=np.float32), rh.to_numpy(dtype=np.float32))
    return pd.Series(np.asarray(values, dtype=np.float32), index=temp_c.index)


def _resolve_wind10_speed(table):
    direct = _optional_numeric_series(table, ("wind10_speed", "wind_speed_mps"))
    if direct is not None:
        return direct
    u10 = _optional_numeric_series(table, ("u10", "u10m"))
    v10 = _optional_numeric_series(table, ("v10", "v10m"))
    if u10 is None or v10 is None:
        raise KeyError("wind10_speed")
    values = compute_wind_speed_mps(u10.to_numpy(dtype=np.float32), v10.to_numpy(dtype=np.float32))
    return pd.Series(np.asarray(values, dtype=np.float32), index=u10.index)


def _resolve_daily_precip_total(table):
    total = _optional_numeric_series(table, ("daily_precip_total", "gpm_daily_precip"))
    if total is not None:
        return total
    return _numeric_series(table, ("tp",), scale=1000.0)


def _resolve_daily_convective_precip(table, daily_total):
    convective = _optional_numeric_series(table, ("daily_convective_precip",))
    if convective is not None:
        return convective
    cp = _optional_numeric_series(table, ("cp",), scale=1000.0)
    if cp is not None:
        return cp
    large_scale = _optional_numeric_series(table, ("daily_large_scale_precip",))
    if large_scale is not None:
        return (daily_total - large_scale).clip(lower=0.0).astype(np.float32)
    raise KeyError("daily_convective_precip")


def _resolve_daily_large_scale_precip(table, daily_total, convective):
    large_scale = _optional_numeric_series(table, ("daily_large_scale_precip",))
    if large_scale is not None:
        return large_scale
    return (daily_total - convective).clip(lower=0.0).astype(np.float32)


def _resolve_wind850_speed(table):
    direct = _optional_numeric_series(table, ("wind850_speed",))
    if direct is not None:
        return direct
    u850 = _optional_numeric_series(table, ("u850",))
    v850 = _optional_numeric_series(table, ("v850",))
    if u850 is None or v850 is None:
        raise KeyError("wind850_speed")
    values = compute_wind_speed_mps(u850.to_numpy(dtype=np.float32), v850.to_numpy(dtype=np.float32))
    return pd.Series(np.asarray(values, dtype=np.float32), index=u850.index)


def _resolve_wind_shear_850_200(table, wind850_speed):
    direct = _optional_numeric_series(table, ("wind_shear_850_200",))
    if direct is not None:
        return direct
    u850 = _optional_numeric_series(table, ("u850",))
    v850 = _optional_numeric_series(table, ("v850",))
    u200 = _optional_numeric_series(table, ("u200",))
    v200 = _optional_numeric_series(table, ("v200",))
    if all(series is not None for series in (u850, v850, u200, v200)):
        values = compute_wind_speed_mps(
            u200.to_numpy(dtype=np.float32) - u850.to_numpy(dtype=np.float32),
            v200.to_numpy(dtype=np.float32) - v850.to_numpy(dtype=np.float32),
        )
        return pd.Series(np.asarray(values, dtype=np.float32), index=wind850_speed.index)
    wind200_speed = _optional_numeric_series(table, ("wind200_speed", "jet200_speed"))
    if wind200_speed is not None:
        return np.abs(wind200_speed - wind850_speed).astype(np.float32)
    raise KeyError("wind_shear_850_200")


def _resolve_cape(table, temp_c, rh):
    direct = _optional_numeric_series(table, ("cape",))
    if direct is not None:
        return direct
    values = compute_cape_placeholder(temp_c.to_numpy(dtype=np.float32), rh.to_numpy(dtype=np.float32))
    return pd.Series(np.asarray(values, dtype=np.float32), index=temp_c.index)


def _resolve_pwat(table, temp_c, rh):
    direct = _optional_numeric_series(table, ("pwat", "pwat_mm"))
    if direct is not None:
        return direct
    pressure_hpa = _optional_numeric_series(table, ("surface_pressure_hpa",))
    if pressure_hpa is None:
        sp = _optional_numeric_series(table, ("surface_pressure", "sp"))
        if sp is not None:
            pressure_hpa = (sp / 100.0).astype(np.float32)
    values = compute_pwat_placeholder(
        temp_c.to_numpy(dtype=np.float32),
        rh.to_numpy(dtype=np.float32),
        None if pressure_hpa is None else pressure_hpa.to_numpy(dtype=np.float32),
    )
    return pd.Series(np.asarray(values, dtype=np.float32), index=temp_c.index)


def _resolve_ivt(table, wind850_speed, pwat):
    direct = _optional_numeric_series(table, ("ivt", "ivt_kg_m_s"))
    if direct is not None:
        return direct
    values = compute_ivt_placeholder(wind850_speed.to_numpy(dtype=np.float32), pwat.to_numpy(dtype=np.float32))
    return pd.Series(np.asarray(values, dtype=np.float32), index=wind850_speed.index)


def _resolve_flash_flood_risk(table, daily_total):
    direct = _optional_numeric_series(table, ("flash_flood_risk",))
    if direct is not None:
        return direct
    slope = _optional_numeric_series(table, ("slope", "slope_deg"))
    soil = _optional_numeric_series(table, ("soil_moisture_frac",))
    impervious = _optional_numeric_series(table, ("impervious_frac",))
    values = [
        compute_flash_flood_screening_score(
            None,
            None,
            precip_24h_mm=precip,
            slope_deg=None if slope is None else slope.iloc[index],
            soil_moisture_frac=None if soil is None else soil.iloc[index],
            impervious_frac=None if impervious is None else impervious.iloc[index],
        )
        for index, precip in enumerate(daily_total.to_numpy(dtype=np.float32))
    ]
    return pd.Series(np.asarray(values, dtype=np.float32), index=daily_total.index)


def build_daily_feature_frame(table, hazard_type: str):
    working = _ensure_daily_date_column(table)
    data: dict[str, object] = {"date": working["date"]}
    for name in (
        "latitude",
        "longitude",
        "label",
        "label_status",
        "label_source_mode",
        "matched_event_ids",
        "label_provenance",
        "hazard_type",
        "sst_celsius",
        "t2m_anomaly_c",
        "tmax_anomaly_c",
        "heatwave_day_flag",
        "heatwave_duration_days",
        "daily_precip_anomaly",
    ):
        if name in working.columns:
            data[name] = working[name]

    if hazard_type in {"extreme_heat", "dry_heat_agriculture"}:
        temp_c = _resolve_temperature_c(working)
        tmax_c = _resolve_tmax_c(working, temp_c)
        tmin_c = _resolve_tmin_c(working, temp_c)
        rh = _resolve_relative_humidity(working, temp_c)
        vpd = _resolve_vpd(working, temp_c, rh)
        heat_index = _resolve_heat_index(working, temp_c, rh)
        wind10 = _resolve_wind10_speed(working)
        data.update(
            {
                "t2m_c": temp_c,
                "temp_c": temp_c,
                "tmax_c": tmax_c,
                "tmin_c": tmin_c,
                "rh2m": rh,
                "relative_humidity_percent": rh,
                "vpd_kpa": vpd,
                "heat_index_c": heat_index,
                "wind10_speed": wind10,
                "wind_speed_mps": wind10,
            }
        )
    elif hazard_type == "flash_flood":
        daily_total = _resolve_daily_precip_total(working)
        convective = _resolve_daily_convective_precip(working, daily_total)
        large_scale = _resolve_daily_large_scale_precip(working, daily_total, convective)
        wind850_speed = _resolve_wind850_speed(working)
        shear = _resolve_wind_shear_850_200(working, wind850_speed)
        temp_c = rh = None
        if "cape" not in working.columns or "pwat" not in working.columns:
            temp_c = _resolve_temperature_c(working)
            rh = _resolve_relative_humidity(working, temp_c)
        cape = _resolve_cape(working, temp_c, rh) if temp_c is not None and rh is not None else _numeric_series(working, ("cape",))
        pwat = _resolve_pwat(working, temp_c, rh) if temp_c is not None and rh is not None else _numeric_series(working, ("pwat", "pwat_mm"))
        ivt = _resolve_ivt(working, wind850_speed, pwat)
        flash_risk = _resolve_flash_flood_risk(working, daily_total)
        data.update(
            {
                "daily_precip_total": daily_total,
                "daily_convective_precip": convective,
                "daily_large_scale_precip": large_scale,
                "cape": cape,
                "pwat": pwat,
                "ivt": ivt,
                "wind850_speed": wind850_speed,
                "wind_shear_850_200": shear,
                "flash_flood_risk": flash_risk,
            }
        )

    return pd.DataFrame(data)


def build_training_table(dataset, hazard_type: str):
    ds = normalize_dataset(dataset)
    try:
        features, _ = feature_matrix_from_dataset(ds, hazard_type=hazard_type)
    except Exception:
        if pd is None:
            raise
        frame = ds.to_dataframe().reset_index()
        daily_frame = build_daily_feature_frame(frame, hazard_type)
        prepared = prepare_feature_frame(daily_frame, hazard_type=hazard_type)
        features = prepared.loc[:, list(feature_names_for_hazard(hazard_type))].to_numpy(dtype=np.float32)
    if features.size == 0:
        raise RuntimeError("No valid training features available after sanitization")
    target = build_target_from_features(features, hazard_type)
    return features, target


def summarize_frame_training_targets(table, hazard_type: str) -> dict[str, object]:
    summary: dict[str, object] = {
        "target_source": "pseudo_target",
        "input_rows": int(len(table)),
        "rows_after_label_filter": int(len(table)),
        "rows_with_explicit_label": 0,
    }
    if hazard_type != "flash_flood" or "label" not in table.columns:
        return summary

    working = table.reset_index(drop=True).copy()
    if "label_status" in working.columns:
        label_status = working["label_status"].astype(str).str.lower()
        status_counts = label_status.value_counts(dropna=False).to_dict()
        filtered = working[label_status.isin(("positive", "negative"))].copy()
        summary["label_status_counts"] = {str(key): int(value) for key, value in status_counts.items()}
    else:
        filtered = working
    summary["rows_after_label_filter"] = int(len(filtered))

    labels = filtered["label"].astype(np.float32)
    explicit_mask = labels.notna()
    summary["rows_with_explicit_label"] = int(explicit_mask.sum())
    if explicit_mask.any():
        summary["target_source"] = "explicit_label"
        summary["positive_labels"] = int((labels[explicit_mask] > 0.5).sum())
        summary["negative_labels"] = int((labels[explicit_mask] <= 0.5).sum())
        if "label_source_mode" in filtered.columns:
            source_counts = (
                filtered.loc[explicit_mask, "label_source_mode"]
                .astype(str)
                .value_counts(dropna=False)
                .to_dict()
            )
            summary["label_source_mode_counts"] = {str(key): int(value) for key, value in source_counts.items()}
    return summary


def build_training_table_from_frame(table, hazard_type: str):
    _require_pandas()
    working = build_daily_feature_frame(table.reset_index(drop=True).copy(), hazard_type)
    target = None
    if hazard_type == "flash_flood" and "label" in working.columns:
        labels = working["label"]
        if "label_status" in working.columns:
            label_status = working["label_status"].astype(str).str.lower()
            working = working[label_status.isin(("positive", "negative"))].copy()
            working = working.reset_index(drop=True)
            labels = working["label"]
        labels = labels.astype(np.float32)
        if labels.notna().any():
            labels = labels.reset_index(drop=True)
            target = labels

    frame = prepare_feature_frame(working, hazard_type=hazard_type)
    names = list(feature_names_for_hazard(hazard_type))
    features = frame.loc[:, names].to_numpy(dtype=np.float32)
    required_names = list(required_feature_names_for_hazard(hazard_type))
    required_indexes = [names.index(name) for name in required_names]
    valid_mask = np.all(np.isfinite(features[:, required_indexes]), axis=1)
    features = features[valid_mask]
    if target is not None:
        target = target.loc[frame.index].to_numpy(dtype=np.float32)[valid_mask]
    else:
        target = build_target_from_features(features, hazard_type)
    if features.size == 0:
        raise RuntimeError("No valid training features available after sanitization")
    return features, target


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Layer-4 LightGBM boosters from daily gridded inputs.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Daily NetCDF dataset or daily feature table.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory for saved booster files.")
    parser.add_argument("--source-format", choices=SOURCE_FORMATS, default="auto", help="Input format: daily ERA5-like NetCDF, daily indicator NetCDF, or daily indicator table.")
    parser.add_argument("--hazard-type", choices=HAZARD_TYPES, default="extreme_heat", help="Hazard-specific Layer-4 model to train.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/valid split and LightGBM.")
    return parser.parse_args()


def infer_source_format(path: Path) -> str:
    suffixes = path.suffixes
    if ".parquet" in suffixes:
        return "indicator-parquet"
    if ".csv" in suffixes:
        return "indicator-csv"
    if ".json" in suffixes:
        return "indicator-json"
    return "era5" if "era5" in path.name else "indicator-netcdf"


def load_training_source(path: Path, source_format: str):
    normalized = infer_source_format(path) if source_format == "auto" else source_format
    if normalized not in SOURCE_FORMATS[1:]:
        raise ValueError(f"Unsupported source format: {normalized}")
    if normalized in {"indicator-parquet", "indicator-csv", "indicator-json"}:
        import pandas as pd

        if normalized == "indicator-parquet":
            return pd.read_parquet(path), normalized
        if normalized == "indicator-csv":
            return pd.read_csv(path), normalized
        return pd.read_json(path), normalized
    return read_netcdf_dataset(path), normalized


def main() -> int:
    args = parse_args()
    source = args.source
    model_dir = args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset, resolved_source_format = load_training_source(source, args.source_format)
    target_summary = None
    if resolved_source_format in {"indicator-parquet", "indicator-csv", "indicator-json"}:
        target_summary = summarize_frame_training_targets(dataset, args.hazard_type)
        features, target = build_training_table_from_frame(dataset, args.hazard_type)
    else:
        features, target = build_training_table(dataset, args.hazard_type)

    adapter = LightGBMAdapter()
    training_summary = adapter.train(
        {
            "features": features,
            "labels": target,
            "feature_names": list(feature_names_for_hazard(args.hazard_type)),
        },
        validation_fraction=0.1,
        seed=args.seed,
        num_boost_round=250,
        early_stopping_rounds=20,
    )
    model_filenames = {
        "extreme_heat": "extreme_heat.txt",
        "dry_heat_agriculture": "dry_heat_stress.txt",
        "flash_flood": "flash_flood.txt",
    }
    model_path = model_dir / model_filenames[args.hazard_type]
    adapter.save_model(model_path)

    summary = {
        "source": display_path(source),
        "source_format": resolved_source_format,
        "hazard_type": args.hazard_type,
        "samples": int(features.shape[0]),
        "feature_names": list(feature_names_for_hazard(args.hazard_type)),
        "model": {
            "path": display_path(model_path),
            "backend": training_summary["backend"],
            "objective": training_summary["objective"],
            "metric": training_summary["metric"],
            "validation_metric": training_summary["validation_metric"],
            "best_iteration": training_summary["best_iteration"],
        },
    }
    if target_summary is not None:
        summary["training_target"] = target_summary
    (model_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
