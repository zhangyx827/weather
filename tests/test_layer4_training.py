from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "examples" / "train_layer4_lightgbm.py"
BUILD_TABLE_SCRIPT_PATH = ROOT / "scripts" / "build_layer4_training_table.py"


def _load_training_module():
    spec = importlib.util.spec_from_file_location("train_layer4_lightgbm", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_build_table_module():
    spec = importlib.util.spec_from_file_location("build_layer4_training_table", BUILD_TABLE_SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False


def _indicator_frame(rows: int = 512) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "t2m_c": rng.uniform(24.0, 48.0, rows).astype(np.float32),
            "tmax_c": rng.uniform(28.0, 52.0, rows).astype(np.float32),
            "tmin_c": rng.uniform(18.0, 34.0, rows).astype(np.float32),
            "vpd_kpa": rng.uniform(0.2, 6.0, rows).astype(np.float32),
            "heat_index_c": rng.uniform(25.0, 52.0, rows).astype(np.float32),
            "wind10_speed": rng.uniform(0.5, 14.0, rows).astype(np.float32),
            "rh2m": rng.uniform(10.0, 95.0, rows).astype(np.float32),
            "sst_celsius": rng.uniform(20.0, 35.0, rows).astype(np.float32),
            "t2m_anomaly_c": rng.uniform(-5.0, 7.0, rows).astype(np.float32),
            "tmax_anomaly_c": rng.uniform(-5.0, 8.0, rows).astype(np.float32),
            "heatwave_day_flag": rng.integers(0, 2, rows).astype(np.int16),
            "heatwave_duration_days": rng.integers(0, 10, rows).astype(np.int16),
            "daily_precip_total": rng.uniform(0.0, 40.0, rows).astype(np.float32),
            "daily_convective_precip": rng.uniform(0.0, 25.0, rows).astype(np.float32),
            "daily_large_scale_precip": rng.uniform(0.0, 25.0, rows).astype(np.float32),
            "cape": rng.uniform(0.0, 4000.0, rows).astype(np.float32),
            "pwat": rng.uniform(5.0, 60.0, rows).astype(np.float32),
            "ivt": rng.uniform(20.0, 500.0, rows).astype(np.float32),
            "wind850_speed": rng.uniform(1.0, 25.0, rows).astype(np.float32),
            "wind_shear_850_200": rng.uniform(1.0, 70.0, rows).astype(np.float32),
            "flash_flood_risk": rng.integers(0, 4, rows).astype(np.int16),
            "daily_precip_anomaly": rng.uniform(-10.0, 30.0, rows).astype(np.float32),
        }
    )


def _indicator_dataset() -> xr.Dataset:
    lat = np.array([16.0, 16.1, 16.2], dtype=np.float32)
    lon = np.array([34.0, 34.1, 34.2], dtype=np.float32)
    time = np.array(["2025-01-01"], dtype="datetime64[ns]")
    base = np.arange(lat.size * lon.size, dtype=np.float32).reshape(lat.size, lon.size)
    return xr.Dataset(
        data_vars={
            "t2m_c": (("time", "latitude", "longitude"), (35.0 + base)[None, :, :]),
            "tmax_c": (("time", "latitude", "longitude"), (40.0 + base)[None, :, :]),
            "tmin_c": (("time", "latitude", "longitude"), (28.0 + base * 0.2)[None, :, :]),
            "vpd_kpa": (("time", "latitude", "longitude"), (1.5 + base * 0.05)[None, :, :]),
            "heat_index_c": (("time", "latitude", "longitude"), (37.0 + base * 0.2)[None, :, :]),
            "wind10_speed": (("time", "latitude", "longitude"), (3.0 + base * 0.1)[None, :, :]),
            "rh2m": (("time", "latitude", "longitude"), (45.0 + base)[None, :, :]),
            "sst_celsius": (("time", "lat", "lon"), (30.0 + base * 0.05)[None, :, :]),
            "t2m_anomaly_c": (("time", "latitude", "longitude"), (base * 0.1)[None, :, :]),
            "tmax_anomaly_c": (("time", "latitude", "longitude"), (base * 0.12)[None, :, :]),
            "heatwave_day_flag": (("time", "latitude", "longitude"), np.where(base > 2, 1, 0)[None, :, :]),
            "heatwave_duration_days": (("time", "latitude", "longitude"), (1 + base)[None, :, :]),
            "daily_precip_total": (("time", "latitude", "longitude"), (10.0 + base)[None, :, :]),
            "daily_convective_precip": (("time", "latitude", "longitude"), (4.0 + base * 0.5)[None, :, :]),
            "daily_large_scale_precip": (("time", "latitude", "longitude"), (3.0 + base * 0.4)[None, :, :]),
            "cape": (("time", "latitude", "longitude"), (500.0 + base * 50.0)[None, :, :]),
            "pwat": (("time", "latitude", "longitude"), (20.0 + base)[None, :, :]),
            "ivt": (("time", "latitude", "longitude"), (80.0 + base * 5.0)[None, :, :]),
            "wind850_speed": (("time", "latitude", "longitude"), (6.0 + base * 0.3)[None, :, :]),
            "wind_shear_850_200": (("time", "latitude", "longitude"), (18.0 + base)[None, :, :]),
            "flash_flood_risk": (("time", "latitude", "longitude"), np.where(base > 4, 2, 1)[None, :, :]),
            "daily_precip_anomaly": (("time", "latitude", "longitude"), (base - 2.0)[None, :, :]),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )


def test_indicator_parquet_training_smoke():
    if not _parquet_available():
        return
    module = _load_training_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "saudi_indicator_samples_2025.parquet"
        model_dir = tmp_path / "models"
        _indicator_frame().to_parquet(source, index=False)

        old_argv = sys.argv
        sys.argv = ["train_layer4_lightgbm.py", "--source", str(source), "--source-format", "indicator-parquet", "--model-dir", str(model_dir), "--hazard-type", "extreme_heat"]
        try:
            assert module.main() == 0
        finally:
            sys.argv = old_argv

        summary = json.loads((model_dir / "train_summary.json").read_text(encoding="utf-8"))
        assert summary["source_format"] == "indicator-parquet"
        assert summary["hazard_type"] == "extreme_heat"
        assert (model_dir / "extreme_heat.txt").exists()


def test_indicator_netcdf_training_table():
    module = _load_training_module()
    ds = _indicator_dataset()
    features, target = module.build_training_table(ds, "extreme_heat")
    assert features.shape[1] == len(module.feature_names_for_hazard("extreme_heat"))
    assert features.shape[0] == ds.latitude.size * ds.longitude.size
    assert target.shape == (features.shape[0],)


def test_indicator_netcdf_training_table_flash_flood():
    module = _load_training_module()
    ds = _indicator_dataset()
    features, target = module.build_training_table(ds, "flash_flood")
    assert features.shape[1] == len(module.feature_names_for_hazard("flash_flood"))
    assert features.shape[0] == ds.latitude.size * ds.longitude.size
    assert target.shape == (features.shape[0],)


def test_build_layer4_training_table_script_exports_parquet():
    if not _parquet_available():
        return
    module = _load_build_table_module()
    ds = _indicator_dataset()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_dir = tmp_path / "indicators"
        output_dir = tmp_path / "tables"
        input_dir.mkdir()
        source = input_dir / "saudi_indicators_20250101.nc"
        ds.to_netcdf(source)

        result = module.main(["--input", str(input_dir), "--output-dir", str(output_dir), "--hazard-type", "flash_flood"])
        assert result == 0

        table = pd.read_parquet(output_dir / "flash_flood_training.parquet")
        assert len(table) == ds.latitude.size * ds.longitude.size
        assert set(["date", "hazard_type", "latitude", "longitude", "source_status", "degradation_metadata"]).issubset(table.columns)
        assert set(module.HAZARD_TYPES) >= {"flash_flood"}
        assert table["hazard_type"].nunique() == 1
        assert table["hazard_type"].iloc[0] == "flash_flood"
