"""Train Layer-4 LightGBM boosters from the bundled ERA5 sample.

This script builds the two binary risk scorers expected by
`examples/real_data_probe.py`:

- `models/layer4/extreme_heat.txt`
- `models/layer4/dry_heat_stress.txt`

The training targets are deterministic pseudo-labels derived from the same
physical indicators used by the probe. That keeps the workflow self-contained
until a labeled hazard archive is available.
"""

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
from mazu_saudi.risk.layer4_features import LAYER4_FEATURE_NAMES, feature_matrix_from_dataset, prepare_feature_frame

DEFAULT_SOURCE = ROOT / "data" / "raw" / "era5_saudi_20250616.nc"
DEFAULT_MODEL_DIR = ROOT / "models" / "layer4"
SOURCE_FORMATS = ("auto", "era5", "indicator-netcdf", "indicator-parquet")


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


def build_targets_from_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    temp = features[:, 0]
    hi_flat = features[:, 2]
    rh_flat = features[:, 4]
    wind_flat = features[:, 3]

    extreme_heat = np.maximum((temp - 38.0) / 12.0, (hi_flat - 41.0) / 12.0)
    extreme_heat = np.where(temp >= 45.0, extreme_heat + 0.2, extreme_heat)
    extreme_heat = np.clip(extreme_heat, 0.0, 1.0).astype(np.float32)

    dry_heat = np.array(
        [compute_dry_heat_stress_score(t, rh_value, wind_value) for t, rh_value, wind_value in zip(temp, rh_flat, wind_flat)],
        dtype=np.float32,
    )
    return extreme_heat, dry_heat


def build_training_table(dataset):
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
        features = np.stack(
            [
                np.asarray(t2m_c, dtype=np.float32).reshape(-1),
                np.asarray(vpd, dtype=np.float32).reshape(-1),
                np.asarray(hi, dtype=np.float32).reshape(-1),
                np.asarray(wind, dtype=np.float32).reshape(-1),
                np.asarray(rh, dtype=np.float32).reshape(-1),
            ],
            axis=1,
        )
    else:
        features, _ = feature_matrix_from_dataset(ds)

    valid_mask = np.all(np.isfinite(features), axis=1)
    features = features[valid_mask]
    if features.size == 0:
        raise RuntimeError("No valid training features available after sanitization")
    extreme_heat, dry_heat = build_targets_from_features(features)
    return features, extreme_heat, dry_heat


def build_training_table_from_frame(table):
    frame = prepare_feature_frame(table)
    features = frame.loc[:, list(LAYER4_FEATURE_NAMES)].to_numpy(dtype=np.float32)
    extreme_heat, dry_heat = build_targets_from_features(features)
    return features, extreme_heat, dry_heat


def train_booster(features, target, seed=42):
    import lightgbm as lgb

    indices = np.arange(features.shape[0])
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    split = int(indices.size * 0.9)
    train_idx = indices[:split]
    valid_idx = indices[split:]

    train_set = lgb.Dataset(features[train_idx], label=target[train_idx], feature_name=list(LAYER4_FEATURE_NAMES), free_raw_data=False)
    valid_set = lgb.Dataset(features[valid_idx], label=target[valid_idx], feature_name=list(LAYER4_FEATURE_NAMES), free_raw_data=False)

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
        features, extreme_heat_target, dry_heat_target = build_training_table_from_frame(dataset)
    else:
        features, extreme_heat_target, dry_heat_target = build_training_table(dataset)

    heat_model, heat_rmse = train_booster(features, extreme_heat_target, seed=args.seed)
    dry_model, dry_rmse = train_booster(features, dry_heat_target, seed=args.seed + 1)

    heat_path = model_dir / "extreme_heat.txt"
    dry_path = model_dir / "dry_heat_stress.txt"
    heat_model.save_model(str(heat_path))
    dry_model.save_model(str(dry_path))

    summary = {
        "source": display_path(source),
        "source_format": resolved_source_format,
        "samples": int(features.shape[0]),
        "feature_names": list(LAYER4_FEATURE_NAMES),
        "models": {
            "extreme_heat": {"path": display_path(heat_path), "valid_rmse": heat_rmse},
            "dry_heat_stress": {"path": display_path(dry_path), "valid_rmse": dry_rmse},
        },
    }
    (model_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
