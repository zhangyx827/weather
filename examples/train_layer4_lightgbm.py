"""Train hazard-specific Layer-4 LightGBM boosters from ERA5-derived inputs."""

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
from mazu_saudi.indicators.physical import compute_dry_heat_stress_score
from mazu_saudi.risk.layer4_features import feature_matrix_from_dataset, feature_names_for_hazard, prepare_feature_frame

DEFAULT_SOURCE = ROOT / "data" / "raw" / "era5_saudi_20250616.nc"
DEFAULT_MODEL_DIR = ROOT / "models" / "layer4"
SOURCE_FORMATS = ("auto", "era5", "indicator-netcdf", "indicator-parquet")
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
        anomaly = np.nan_to_num(features[:, index["tmax_anomaly_c"]], nan=0.0)
        heatwave = np.nan_to_num(features[:, index["heatwave_day_flag"]], nan=0.0)
        target = np.maximum((temp - 38.0) / 12.0, (hi - 41.0) / 12.0)
        target = np.maximum(target, (tmax - 42.0) / 10.0)
        target = target + np.where(anomaly >= 3.0, 0.08, 0.0) + np.where(heatwave >= 1.0, 0.08, 0.0)
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
        anomaly = np.nan_to_num(features[:, index["daily_precip_anomaly"]], nan=0.0)
        target = np.maximum.reduce(
            [
                precip / 50.0,
                conv / 30.0,
                cape / 2500.0,
                pwat / 50.0,
                ivt / 400.0,
                shear / 60.0,
                screen / 4.0,
                np.maximum(anomaly, 0.0) / 25.0,
            ]
        )
        return np.clip(target, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"Unsupported hazard type: {hazard_type}")


def build_training_table(dataset, hazard_type: str):
    ds = normalize_dataset(dataset)
    if "t2m" in ds.data_vars:
        if not any(dim not in {"latitude", "longitude"} for dim in ds["t2m"].dims):
            raise RuntimeError("Expected a temporal dimension in the ERA5 sample")
        t2m_c = to_celsius(ds["t2m"].values)
        d2m_c = to_celsius(ds["d2m"].values)
        rh = relative_humidity_from_dewpoint(t2m_c, d2m_c)
        vpd = compute_vpd_kpa(t2m_c, d2m_c)
        hi = compute_heat_index_c(t2m_c, rh)
        wind = compute_wind_speed_mps(ds["u10"].values, ds["v10"].values)
        era5_feature_frame = {
            "t2m_c": np.asarray(t2m_c, dtype=np.float32).reshape(-1),
            "tmax_c": np.asarray(t2m_c, dtype=np.float32).reshape(-1),
            "tmin_c": np.asarray(t2m_c, dtype=np.float32).reshape(-1),
            "vpd_kpa": np.asarray(vpd, dtype=np.float32).reshape(-1),
            "heat_index_c": np.asarray(hi, dtype=np.float32).reshape(-1),
            "wind10_speed": np.asarray(wind, dtype=np.float32).reshape(-1),
            "rh2m": np.asarray(rh, dtype=np.float32).reshape(-1),
            "sst_celsius": np.full(np.asarray(t2m_c).size, np.nan, dtype=np.float32),
            "t2m_anomaly_c": np.full(np.asarray(t2m_c).size, np.nan, dtype=np.float32),
            "tmax_anomaly_c": np.full(np.asarray(t2m_c).size, np.nan, dtype=np.float32),
            "heatwave_day_flag": np.full(np.asarray(t2m_c).size, np.nan, dtype=np.float32),
            "heatwave_duration_days": np.full(np.asarray(t2m_c).size, np.nan, dtype=np.float32),
        }
        if hazard_type == "flash_flood":
            precip = np.asarray(ds["tp"].values if "tp" in ds.data_vars else np.zeros_like(t2m_c), dtype=np.float32).reshape(-1) * 1000.0
            era5_feature_frame.update(
                {
                    "daily_precip_total": precip,
                    "daily_convective_precip": np.zeros_like(precip),
                    "daily_large_scale_precip": precip,
                    "cape": np.asarray(ds["cape"].values if "cape" in ds.data_vars else np.zeros_like(t2m_c), dtype=np.float32).reshape(-1),
                    "pwat": np.zeros_like(precip),
                    "ivt": np.zeros_like(precip),
                    "wind850_speed": np.zeros_like(precip),
                    "wind_shear_850_200": np.zeros_like(precip),
                    "flash_flood_risk": np.zeros_like(precip),
                    "daily_precip_anomaly": np.full_like(precip, np.nan),
                }
            )
        import pandas as pd
        features_frame = pd.DataFrame(era5_feature_frame)
        frame = prepare_feature_frame(features_frame, hazard_type=hazard_type)
        features = frame.loc[:, list(feature_names_for_hazard(hazard_type))].to_numpy(dtype=np.float32)
    else:
        features, _ = feature_matrix_from_dataset(ds, hazard_type=hazard_type)

    required_names = [name for name in feature_names_for_hazard(hazard_type) if name not in {"sst_celsius", "t2m_anomaly_c", "tmax_anomaly_c", "heatwave_day_flag", "heatwave_duration_days", "daily_precip_anomaly"}]
    required_indexes = [feature_names_for_hazard(hazard_type).index(name) for name in required_names]
    valid_mask = np.all(np.isfinite(features[:, required_indexes]), axis=1)
    features = features[valid_mask]
    if features.size == 0:
        raise RuntimeError("No valid training features available after sanitization")
    target = build_target_from_features(features, hazard_type)
    return features, target


def build_training_table_from_frame(table, hazard_type: str):
    frame = prepare_feature_frame(table, hazard_type=hazard_type)
    names = list(feature_names_for_hazard(hazard_type))
    features = frame.loc[:, names].to_numpy(dtype=np.float32)
    target = build_target_from_features(features, hazard_type)
    return features, target


def train_booster(features, target, hazard_type: str, seed=42):
    import lightgbm as lgb

    indices = np.arange(features.shape[0])
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    split = int(indices.size * 0.9)
    train_idx = indices[:split]
    valid_idx = indices[split:]

    feature_names = list(feature_names_for_hazard(hazard_type))
    train_set = lgb.Dataset(features[train_idx], label=target[train_idx], feature_name=feature_names, free_raw_data=False)
    valid_set = lgb.Dataset(features[valid_idx], label=target[valid_idx], feature_name=feature_names, free_raw_data=False)

    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
    }

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=250,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )

    valid_pred = booster.predict(features[valid_idx])
    rmse = float(np.sqrt(np.mean((valid_pred - target[valid_idx]) ** 2)))
    return booster, rmse


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Layer-4 LightGBM boosters from ERA5 data.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="NetCDF or CDS ZIP bundle to train from.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Directory for saved booster files.")
    parser.add_argument("--source-format", choices=SOURCE_FORMATS, default="auto", help="Input format: ERA5 sample, daily indicator NetCDF, or indicator Parquet.")
    parser.add_argument("--hazard-type", choices=HAZARD_TYPES, default="extreme_heat", help="Hazard-specific Layer-4 model to train.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/valid split and LightGBM.")
    return parser.parse_args()


def infer_source_format(path: Path) -> str:
    suffixes = path.suffixes
    if ".parquet" in suffixes:
        return "indicator-parquet"
    return "era5" if "era5" in path.name else "indicator-netcdf"


def load_training_source(path: Path, source_format: str):
    normalized = infer_source_format(path) if source_format == "auto" else source_format
    if normalized not in SOURCE_FORMATS[1:]:
        raise ValueError(f"Unsupported source format: {normalized}")
    if normalized == "indicator-parquet":
        import pandas as pd

        return pd.read_parquet(path), normalized
    return read_netcdf_dataset(path), normalized


def main() -> int:
    args = parse_args()
    source = args.source
    model_dir = args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset, resolved_source_format = load_training_source(source, args.source_format)
    if resolved_source_format == "indicator-parquet":
        features, target = build_training_table_from_frame(dataset, args.hazard_type)
    else:
        features, target = build_training_table(dataset, args.hazard_type)

    model, rmse = train_booster(features, target, args.hazard_type, seed=args.seed)
    model_filenames = {
        "extreme_heat": "extreme_heat.txt",
        "dry_heat_agriculture": "dry_heat_stress.txt",
        "flash_flood": "flash_flood.txt",
    }
    model_path = model_dir / model_filenames[args.hazard_type]
    model.save_model(str(model_path))

    summary = {
        "source": display_path(source),
        "source_format": resolved_source_format,
        "hazard_type": args.hazard_type,
        "samples": int(features.shape[0]),
        "feature_names": list(feature_names_for_hazard(args.hazard_type)),
        "model": {"path": display_path(model_path), "valid_rmse": rmse},
    }
    (model_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
