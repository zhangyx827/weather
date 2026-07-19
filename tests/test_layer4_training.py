from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "examples" / "train_layer4_lightgbm.py"
BUILD_TABLE_SCRIPT_PATH = ROOT / "scripts" / "build_layer4_training_table.py"
BUILD_FLASH_FLOOD_LABELS_SCRIPT_PATH = ROOT / "scripts" / "build_flash_flood_training_labels.py"
BUILD_SUPERVISED_TABLE_SCRIPT_PATH = ROOT / "scripts" / "build_flash_flood_supervised_training_table.py"
BUILD_DRY_HEAT_SUPERVISED_TABLE_SCRIPT_PATH = ROOT / "scripts" / "build_dry_heat_agriculture_supervised_training_table.py"
BUILD_DUST_STORM_SUPERVISED_TABLE_SCRIPT_PATH = ROOT / "scripts" / "build_dust_storm_supervised_training_table.py"
BUILD_EXTREME_HEAT_SUPERVISED_TABLE_SCRIPT_PATH = ROOT / "scripts" / "build_extreme_heat_supervised_training_table.py"
COMPARE_EXTREME_HEAT_SUPERVISION_VARIANTS_SCRIPT_PATH = ROOT / "scripts" / "compare_extreme_heat_supervision_variants.py"
DEMO_SUPERVISED_SCRIPT_PATH = ROOT / "examples" / "demo_flash_flood_supervised_training.py"


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


def _load_build_flash_flood_labels_module():
    spec = importlib.util.spec_from_file_location("build_flash_flood_training_labels", BUILD_FLASH_FLOOD_LABELS_SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_build_supervised_table_module():
    spec = importlib.util.spec_from_file_location("build_flash_flood_supervised_training_table", BUILD_SUPERVISED_TABLE_SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_build_dry_heat_supervised_table_module():
    spec = importlib.util.spec_from_file_location(
        "build_dry_heat_agriculture_supervised_training_table",
        BUILD_DRY_HEAT_SUPERVISED_TABLE_SCRIPT_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_build_dust_storm_supervised_table_module():
    spec = importlib.util.spec_from_file_location(
        "build_dust_storm_supervised_training_table",
        BUILD_DUST_STORM_SUPERVISED_TABLE_SCRIPT_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_build_extreme_heat_supervised_table_module():
    spec = importlib.util.spec_from_file_location(
        "build_extreme_heat_supervised_training_table",
        BUILD_EXTREME_HEAT_SUPERVISED_TABLE_SCRIPT_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_compare_extreme_heat_supervision_variants_module():
    spec = importlib.util.spec_from_file_location(
        "compare_extreme_heat_supervision_variants",
        COMPARE_EXTREME_HEAT_SUPERVISION_VARIANTS_SCRIPT_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_demo_supervised_module():
    spec = importlib.util.spec_from_file_location("demo_flash_flood_supervised_training", DEMO_SUPERVISED_SCRIPT_PATH)
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


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:
        return False


def _indicator_frame(rows: int = 512) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=rows, freq="D").strftime("%Y-%m-%d"),
            "latitude": rng.uniform(16.0, 33.0, rows).astype(np.float32),
            "longitude": rng.uniform(34.0, 57.0, rows).astype(np.float32),
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


def _dry_heat_daily_region_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2024-01-15", "region_id": "asir", "temp_c": 31.0, "tmax_c": 37.0, "heat_index_c": 33.0, "vpd_kpa": 1.6, "wind_speed_mps": 3.2, "relative_humidity_percent": 40.0, "t2m_anomaly_c": 0.5, "heatwave_day_flag": 0, "heatwave_duration_days": 0},
            {"date": "2024-04-10", "region_id": "asir", "temp_c": 35.0, "tmax_c": 41.0, "heat_index_c": 38.0, "vpd_kpa": 2.1, "wind_speed_mps": 4.0, "relative_humidity_percent": 31.0, "t2m_anomaly_c": 1.2, "heatwave_day_flag": 1, "heatwave_duration_days": 2},
            {"date": "2024-08-02", "region_id": "asir", "temp_c": 39.0, "tmax_c": 45.0, "heat_index_c": 42.0, "vpd_kpa": 3.2, "wind_speed_mps": 5.0, "relative_humidity_percent": 24.0, "t2m_anomaly_c": 2.5, "heatwave_day_flag": 1, "heatwave_duration_days": 4},
            {"date": "2024-02-05", "region_id": "qassim", "temp_c": 28.0, "tmax_c": 34.0, "heat_index_c": 30.0, "vpd_kpa": 1.2, "wind_speed_mps": 2.8, "relative_humidity_percent": 45.0, "t2m_anomaly_c": -0.3, "heatwave_day_flag": 0, "heatwave_duration_days": 0},
            {"date": "2024-07-12", "region_id": "qassim", "temp_c": 41.0, "tmax_c": 47.0, "heat_index_c": 44.0, "vpd_kpa": 3.6, "wind_speed_mps": 6.1, "relative_humidity_percent": 20.0, "t2m_anomaly_c": 3.1, "heatwave_day_flag": 1, "heatwave_duration_days": 5},
            {"date": "2025-03-03", "region_id": "asir", "temp_c": 32.0, "tmax_c": 38.0, "heat_index_c": 34.0, "vpd_kpa": 1.8, "wind_speed_mps": 3.5, "relative_humidity_percent": 36.0, "t2m_anomaly_c": 0.7, "heatwave_day_flag": 0, "heatwave_duration_days": 1},
        ]
    )


def _daily_era5_dataset() -> xr.Dataset:
    lat = np.array([16.0, 16.1], dtype=np.float32)
    lon = np.array([34.0, 34.1], dtype=np.float32)
    time = np.array(["2025-01-01"], dtype="datetime64[ns]")
    base = np.arange(lat.size * lon.size, dtype=np.float32).reshape(lat.size, lon.size)
    return xr.Dataset(
        data_vars={
            "t2m": (("time", "latitude", "longitude"), (305.0 + base)[None, :, :]),
            "mx2t": (("time", "latitude", "longitude"), (309.0 + base)[None, :, :]),
            "mn2t": (("time", "latitude", "longitude"), (300.0 + base * 0.5)[None, :, :]),
            "d2m": (("time", "latitude", "longitude"), (293.0 + base * 0.3)[None, :, :]),
            "u10": (("time", "latitude", "longitude"), (2.0 + base * 0.1)[None, :, :]),
            "v10": (("time", "latitude", "longitude"), (1.0 + base * 0.1)[None, :, :]),
            "tp": (("time", "latitude", "longitude"), (0.01 + base * 0.001)[None, :, :]),
            "cp": (("time", "latitude", "longitude"), (0.003 + base * 0.0005)[None, :, :]),
            "cape": (("time", "latitude", "longitude"), (800.0 + base * 50.0)[None, :, :]),
            "u850": (("time", "latitude", "longitude"), (8.0 + base * 0.3)[None, :, :]),
            "v850": (("time", "latitude", "longitude"), (4.0 + base * 0.2)[None, :, :]),
            "u200": (("time", "latitude", "longitude"), (18.0 + base * 0.3)[None, :, :]),
            "v200": (("time", "latitude", "longitude"), (12.0 + base * 0.2)[None, :, :]),
            "sp": (("time", "latitude", "longitude"), (100800.0 + base * 10.0)[None, :, :]),
            "slope_deg": (("time", "latitude", "longitude"), (3.0 + base * 0.5)[None, :, :]),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
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


def _extreme_heat_indicator_dataset(date: str, *, heat_index_offset: float) -> xr.Dataset:
    lat = np.array([21.5, 21.6], dtype=np.float32)
    lon = np.array([39.1, 39.2], dtype=np.float32)
    time = np.array([date], dtype="datetime64[ns]")
    base = np.arange(lat.size * lon.size, dtype=np.float32).reshape(lat.size, lon.size)
    return xr.Dataset(
        data_vars={
            "t2m_c": (("time", "latitude", "longitude"), (40.0 + base)[None, :, :]),
            "tmax_c": (("time", "latitude", "longitude"), (44.0 + base)[None, :, :]),
            "tmin_c": (("time", "latitude", "longitude"), (31.0 + base * 0.2)[None, :, :]),
            "vpd_kpa": (("time", "latitude", "longitude"), (1.7 + base * 0.1)[None, :, :]),
            "heat_index_c": (("time", "latitude", "longitude"), (43.0 + heat_index_offset + base)[None, :, :]),
            "wind10_speed": (("time", "latitude", "longitude"), (2.5 + base * 0.2)[None, :, :]),
            "rh2m": (("time", "latitude", "longitude"), (38.0 + base)[None, :, :]),
            "sst_celsius": (("time", "latitude", "longitude"), (29.0 + base * 0.05)[None, :, :]),
            "t2m_anomaly_c": (("time", "latitude", "longitude"), (base * 0.1)[None, :, :]),
            "tmax_anomaly_c": (("time", "latitude", "longitude"), (base * 0.15)[None, :, :]),
            "heatwave_day_flag": (("time", "latitude", "longitude"), np.where(base > 1, 1, 0)[None, :, :]),
            "heatwave_duration_days": (("time", "latitude", "longitude"), (1 + base)[None, :, :]),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )


def _extreme_heat_multi_region_indicator_dataset(date: str, *, heat_index_offset: float) -> xr.Dataset:
    lat = np.array([21.5, 24.7], dtype=np.float32)
    lon = np.array([39.2, 46.7], dtype=np.float32)
    time = np.array([date], dtype="datetime64[ns]")
    base = np.arange(lat.size * lon.size, dtype=np.float32).reshape(lat.size, lon.size)
    return xr.Dataset(
        data_vars={
            "t2m_c": (("time", "latitude", "longitude"), (39.0 + base)[None, :, :]),
            "tmax_c": (("time", "latitude", "longitude"), (44.0 + base)[None, :, :]),
            "tmin_c": (("time", "latitude", "longitude"), (30.0 + base * 0.2)[None, :, :]),
            "vpd_kpa": (("time", "latitude", "longitude"), (1.6 + base * 0.1)[None, :, :]),
            "heat_index_c": (("time", "latitude", "longitude"), (42.0 + heat_index_offset + base)[None, :, :]),
            "wind10_speed": (("time", "latitude", "longitude"), (2.0 + base * 0.2)[None, :, :]),
            "rh2m": (("time", "latitude", "longitude"), (36.0 + base)[None, :, :]),
            "sst_celsius": (("time", "latitude", "longitude"), (29.0 + base * 0.05)[None, :, :]),
            "t2m_anomaly_c": (("time", "latitude", "longitude"), (base * 0.08)[None, :, :]),
            "tmax_anomaly_c": (("time", "latitude", "longitude"), (base * 0.1)[None, :, :]),
            "heatwave_day_flag": (("time", "latitude", "longitude"), np.where(base >= 1, 1, 0)[None, :, :]),
            "heatwave_duration_days": (("time", "latitude", "longitude"), (1 + base)[None, :, :]),
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
        assert summary["model"]["backend"] == "lightgbm"
        assert summary["model"]["objective"] == "regression"
        assert summary["model"]["metric"] == "rmse"
        assert (model_dir / "extreme_heat.txt").exists()
        assert (model_dir / "extreme_heat.txt.metadata.json").exists()


def test_indicator_csv_training_smoke():
    module = _load_training_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "flash_flood_supervised.csv"
        model_dir = tmp_path / "models"
        frame = _indicator_frame(rows=16)
        frame["label"] = np.array([1.0, 0.0] * 8, dtype=np.float32)
        frame["label_status"] = ["positive", "negative"] * 8
        frame["label_source_mode"] = ["point_buffer", "no_event_day"] * 8
        frame["matched_event_ids"] = ["ff_event_a", "", "ff_event_a", "", "ff_event_b", "", "ff_event_b", ""] * 2
        frame.to_csv(source, index=False)

        old_argv = sys.argv
        sys.argv = [
            "train_layer4_lightgbm.py",
            "--source",
            str(source),
            "--source-format",
            "indicator-csv",
            "--model-dir",
            str(model_dir),
            "--hazard-type",
            "flash_flood",
        ]
        try:
            assert module.main() == 0
        finally:
            sys.argv = old_argv

        summary = json.loads((model_dir / "train_summary.json").read_text(encoding="utf-8"))
        assert summary["source_format"] == "indicator-csv"
        assert summary["hazard_type"] == "flash_flood"
        assert summary["model"]["objective"] == "binary"
        assert summary["model"]["split_strategy"] == "group_shuffle"
        assert summary["training_target"]["target_source"] == "explicit_label"
        assert (model_dir / "flash_flood.txt").exists()


def test_indicator_csv_training_summary_surfaces_weak_supervision_flags():
    module = _load_training_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "flash_flood_supervised.csv"
        model_dir = tmp_path / "models"
        frame = _indicator_frame(rows=8)
        frame["label"] = np.array([1.0, 0.0] * 4, dtype=np.float32)
        frame["label_status"] = ["positive", "negative"] * 4
        frame["label_source_mode"] = ["point_buffer", "no_event_day"] * 4
        frame["matched_event_ids"] = ["ff_event_a", "", "ff_event_a", "", "", "", "", ""]
        frame.to_csv(source, index=False)

        old_argv = sys.argv
        sys.argv = [
            "train_layer4_lightgbm.py",
            "--source",
            str(source),
            "--source-format",
            "indicator-csv",
            "--model-dir",
            str(model_dir),
            "--hazard-type",
            "flash_flood",
        ]
        try:
            assert module.main() == 0
        finally:
            sys.argv = old_argv

        summary = json.loads((model_dir / "train_summary.json").read_text(encoding="utf-8"))
        quality = summary["training_target"]["supervision_quality"]
        assert quality["status"] == "warning"
        assert "no_geometry_backed_positives" in quality["warnings"]
        assert "few_event_groups" in quality["warnings"]
        assert "fallback_date_groups_dominate" in quality["warnings"]
        assert summary["model"]["supervision_quality"] == quality


def test_indicator_json_training_smoke_dry_heat():
    module = _load_training_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "dry_heat_daily.json"
        model_dir = tmp_path / "models"
        _indicator_frame(rows=24).to_json(source, orient="records")

        old_argv = sys.argv
        sys.argv = [
            "train_layer4_lightgbm.py",
            "--source",
            str(source),
            "--source-format",
            "indicator-json",
            "--model-dir",
            str(model_dir),
            "--hazard-type",
            "dry_heat_agriculture",
        ]
        try:
            assert module.main() == 0
        finally:
            sys.argv = old_argv

        summary = json.loads((model_dir / "train_summary.json").read_text(encoding="utf-8"))
        assert summary["source_format"] == "indicator-json"
        assert summary["hazard_type"] == "dry_heat_agriculture"
        assert summary["model"]["objective"] == "regression"
        assert (model_dir / "dry_heat_stress.txt").exists()


def test_indicator_csv_training_smoke_dry_heat_with_explicit_region_season_labels():
    module = _load_training_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "dry_heat_region_season.csv"
        model_dir = tmp_path / "models"
        frame = _indicator_frame(rows=18).drop(columns=["date", "latitude", "longitude"])
        frame["region_id"] = ["asir", "qassim", "jazan"] * 6
        frame["season"] = ["spring", "summer", "winter"] * 6
        frame["year"] = [2021, 2022, 2023] * 6
        frame["crop_type"] = ["wheat", "dates", "sorghum"] * 6
        frame["yield_anomaly"] = np.linspace(-0.35, 0.4, len(frame), dtype=np.float32)
        frame["validation_status"] = ["verified"] * len(frame)
        frame.to_csv(source, index=False)

        old_argv = sys.argv
        sys.argv = [
            "train_layer4_lightgbm.py",
            "--source",
            str(source),
            "--source-format",
            "indicator-csv",
            "--model-dir",
            str(model_dir),
            "--hazard-type",
            "dry_heat_agriculture",
        ]
        try:
            assert module.main() == 0
        finally:
            sys.argv = old_argv

        summary = json.loads((model_dir / "train_summary.json").read_text(encoding="utf-8"))
        assert summary["training_target"]["target_source"] == "explicit_label"
        assert summary["training_target"]["target_column"] == "yield_anomaly"
        assert summary["training_target"]["sample_unit"] == "region-season"
        assert summary["model"]["objective"] == "regression"
        assert (model_dir / "dry_heat_stress.txt").exists()


def test_indicator_netcdf_training_table():
    module = _load_training_module()
    ds = _indicator_dataset()
    features, target = module.build_training_table(ds, "extreme_heat")
    assert features.shape[1] == len(module.feature_names_for_hazard("extreme_heat"))
    assert features.shape[0] == ds.latitude.size * ds.longitude.size
    assert target.shape == (features.shape[0],)


def test_daily_era5_training_table_flash_flood():
    module = _load_training_module()
    ds = _daily_era5_dataset()
    features, target = module.build_training_table(ds, "flash_flood")
    assert features.shape[1] == len(module.feature_names_for_hazard("flash_flood"))
    assert features.shape[0] == ds.latitude.size * ds.longitude.size
    assert target.shape == (features.shape[0],)


def test_indicator_netcdf_training_table_flash_flood():
    module = _load_training_module()
    ds = _indicator_dataset()
    features, target = module.build_training_table(ds, "flash_flood")
    assert features.shape[1] == len(module.feature_names_for_hazard("flash_flood"))
    assert features.shape[0] == ds.latitude.size * ds.longitude.size
    assert target.shape == (features.shape[0],)


def test_build_layer4_training_table_script_exports_csv_by_default():
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

        table = pd.read_csv(output_dir / "flash_flood_training.csv")
        assert len(table) == ds.latitude.size * ds.longitude.size
        assert set(["date", "hazard_type", "latitude", "longitude", "source_status", "degradation_metadata"]).issubset(table.columns)
        assert set(module.HAZARD_TYPES) >= {"flash_flood"}
        assert table["hazard_type"].nunique() == 1
        assert table["hazard_type"].iloc[0] == "flash_flood"


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

        result = module.main(
            ["--input", str(input_dir), "--output-dir", str(output_dir), "--hazard-type", "flash_flood", "--format", "parquet"]
        )
        assert result == 0

        table = pd.read_parquet(output_dir / "flash_flood_training.parquet")
        assert len(table) == ds.latitude.size * ds.longitude.size


def test_build_layer4_training_table_script_is_incremental():
    module = _load_build_table_module()
    ds = _indicator_dataset()
    ds_next = ds.assign_coords(time=np.array(["2025-01-02"], dtype="datetime64[ns]"))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_dir = tmp_path / "indicators"
        output_dir = tmp_path / "tables"
        input_dir.mkdir()

        (input_dir / "saudi_indicators_20250101.nc").write_bytes(ds.to_netcdf())

        result = module.main(["--input", str(input_dir), "--output-dir", str(output_dir), "--hazard-type", "flash_flood"])
        assert result == 0

        table = pd.read_csv(output_dir / "flash_flood_training.csv")
        assert len(table) == ds.latitude.size * ds.longitude.size
        assert table["source_file"].nunique() == 1

        (input_dir / "saudi_indicators_20250102.nc").write_bytes(ds_next.to_netcdf())

        result = module.main(["--input", str(input_dir), "--output-dir", str(output_dir), "--hazard-type", "flash_flood"])
        assert result == 0

        table = pd.read_csv(output_dir / "flash_flood_training.csv")
        assert len(table) == ds.latitude.size * ds.longitude.size * 2
        assert table["source_file"].nunique() == 2
        assert set(table["source_file"]) == {"saudi_indicators_20250101.nc", "saudi_indicators_20250102.nc"}

        result = module.main(["--input", str(input_dir), "--output-dir", str(output_dir), "--hazard-type", "flash_flood"])
        assert result == 0

        table = pd.read_csv(output_dir / "flash_flood_training.csv")
        assert len(table) == ds.latitude.size * ds.longitude.size * 2
        assert table["source_file"].nunique() == 2


def test_build_layer4_training_table_script_parquet_legacy_ns_precision_is_incremental():
    if not _parquet_available():
        return

    module = _load_build_table_module()
    ds = _indicator_dataset()
    ds_next = ds.assign_coords(time=np.array(["2025-01-02"], dtype="datetime64[ns]"))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_dir = tmp_path / "indicators"
        output_dir = tmp_path / "tables"
        input_dir.mkdir()

        (input_dir / "saudi_indicators_20250101.nc").write_bytes(ds.to_netcdf())
        (input_dir / "saudi_indicators_20250102.nc").write_bytes(ds_next.to_netcdf())

        result = module.main(
            ["--input", str(input_dir), "--output-dir", str(output_dir), "--hazard-type", "flash_flood", "--format", "parquet"]
        )
        assert result == 0

        output_path = output_dir / "flash_flood_training.parquet"
        table = pd.read_parquet(output_path)
        assert table["source_file"].nunique() == 2

        # Simulate a legacy parquet written without the microsecond column. Once the
        # column widens to float64, nanosecond precision is no longer reliable.
        legacy = table.drop(columns=["source_mtime_us"]).copy()
        legacy["source_mtime_ns"] = legacy["source_mtime_ns"].astype(np.float64)
        legacy.to_parquet(output_path, index=False)

        result = module.main(["--input", str(input_dir), "--output-dir", str(output_dir), "--hazard-type", "flash_flood", "--format", "parquet"])
        assert result == 0

        table = pd.read_parquet(output_path)
        assert len(table) == ds.latitude.size * ds.longitude.size * 2
        assert table["source_file"].nunique() == 2
        assert "source_mtime_us" not in legacy.columns


def test_layer4_feature_schema_separates_evidence_only_fields():
    from mazu_saudi.risk.layer4_features import (
        evidence_feature_names_for_hazard,
        feature_frame_from_dataset,
        feature_names_for_hazard,
        required_feature_names_for_hazard,
    )

    ds = _indicator_dataset()

    assert required_feature_names_for_hazard("flash_flood") == (
        "daily_precip_total",
        "daily_convective_precip",
        "daily_large_scale_precip",
        "cape",
        "pwat",
        "ivt",
        "wind850_speed",
        "wind_shear_850_200",
        "flash_flood_risk",
    )
    assert evidence_feature_names_for_hazard("flash_flood") == ("daily_precip_anomaly",)
    assert "daily_precip_anomaly" not in feature_names_for_hazard("flash_flood")

    frame = feature_frame_from_dataset(ds, hazard_type="flash_flood", include_evidence_only=True)
    assert "daily_precip_anomaly" in frame.columns


def test_build_training_table_from_frame_uses_explicit_flash_flood_labels():
    module = _load_training_module()
    frame = _indicator_frame(rows=8)
    frame["label"] = np.array([1.0, 0.0, 1.0, np.nan, 0.0, 1.0, np.nan, 0.0], dtype=np.float32)
    frame["label_status"] = ["positive", "negative", "positive", "uncertain", "negative", "positive", "uncertain", "negative"]

    features, target = module.build_training_table_from_frame(frame, "flash_flood")

    assert features.shape[0] == 6
    assert target.tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_build_training_payload_from_frame_adds_flash_flood_split_groups():
    module = _load_training_module()
    frame = _indicator_frame(rows=6)
    frame["date"] = ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"]
    frame["label"] = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    frame["label_status"] = ["positive", "negative", "positive", "negative", "negative", "positive"]
    frame["matched_event_ids"] = ["ff_a", "", "ff_b", "", "", "ff_c"]

    payload = module.build_training_payload_from_frame(frame, "flash_flood")

    assert payload["features"].shape[0] == 6
    assert payload["labels"].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    assert payload["split_groups"].tolist() == ["ff_a", "date:2025-01-01", "ff_b", "date:2025-01-02", "date:2025-01-03", "ff_c"]


def test_build_training_table_from_frame_filters_rows_with_missing_required_features():
    module = _load_training_module()
    frame = _indicator_frame(rows=4)
    frame.loc[1, "daily_precip_total"] = np.nan
    frame["label"] = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    frame["label_status"] = ["positive", "negative", "positive", "negative"]

    features, target = module.build_training_table_from_frame(frame, "flash_flood")

    assert features.shape[0] == 3
    assert target.tolist() == [1.0, 1.0, 0.0]


def test_build_training_table_from_frame_requires_daily_date_column():
    module = _load_training_module()
    frame = _indicator_frame(rows=4).drop(columns=["date"])

    try:
        module.build_training_table_from_frame(frame, "extreme_heat")
    except KeyError as exc:
        assert "date" in str(exc)
    else:
        raise AssertionError("Expected missing date column to raise")


def test_build_training_table_from_frame_supports_non_daily_supervision_tables():
    module = _load_training_module()
    frame = _indicator_frame(rows=6).drop(columns=["date", "latitude", "longitude"])
    frame["region_id"] = ["asir", "asir", "qassim", "qassim", "jazan", "jazan"]
    frame["season"] = ["spring", "summer", "spring", "summer", "spring", "summer"]
    frame["year"] = [2021, 2021, 2022, 2022, 2023, 2023]
    frame["crop_type"] = ["wheat"] * 6
    frame["yield_anomaly"] = np.array([-0.2, -0.1, 0.0, 0.1, 0.15, 0.3], dtype=np.float32)

    features, target = module.build_training_table_from_frame(frame, "dry_heat_agriculture")

    assert features.shape[0] == 6
    assert np.allclose(target, np.array([-0.2, -0.1, 0.0, 0.1, 0.15, 0.3], dtype=np.float32))


def test_build_training_table_from_frame_rejects_invalid_daily_dates():
    module = _load_training_module()
    frame = _indicator_frame(rows=4)
    frame["date"] = ["2025-01-01", "not-a-date", "2025-01-03", "2025-01-04"]

    try:
        module.build_training_table_from_frame(frame, "extreme_heat")
    except ValueError as exc:
        assert "invalid date" in str(exc)
    else:
        raise AssertionError("Expected invalid date to raise")


def test_build_training_table_from_frame_rejects_subdaily_timestamps():
    module = _load_training_module()
    frame = _indicator_frame(rows=4)
    frame["date"] = ["2025-01-01T00:00:00", "2025-01-02T06:00:00", "2025-01-03", "2025-01-04"]

    try:
        module.build_training_table_from_frame(frame, "extreme_heat")
    except ValueError as exc:
        assert "sub-daily" in str(exc)
    else:
        raise AssertionError("Expected sub-daily timestamp to raise")


def test_summarize_frame_training_targets_reports_explicit_label_usage():
    module = _load_training_module()
    frame = _indicator_frame(rows=6)
    frame["date"] = ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"]
    frame["label"] = np.array([1.0, 0.0, np.nan, 1.0, np.nan, 0.0], dtype=np.float32)
    frame["label_status"] = ["positive", "negative", "uncertain", "positive", "uncertain", "negative"]
    frame["label_source_mode"] = [
        "point_buffer",
        "no_event_day",
        "event_day_unresolved",
        "point_buffer",
        "event_day_unresolved",
        "no_event_day",
    ]
    frame["matched_event_ids"] = ["ff_a", "", "", "ff_b", "", ""]

    summary = module.summarize_frame_training_targets(frame, "flash_flood")

    assert summary["target_source"] == "explicit_label"
    assert summary["input_rows"] == 6
    assert summary["rows_after_label_filter"] == 4
    assert summary["rows_with_explicit_label"] == 4
    assert summary["positive_labels"] == 2
    assert summary["negative_labels"] == 2
    assert summary["label_status_counts"] == {"positive": 2, "negative": 2, "uncertain": 2}
    assert summary["label_source_mode_counts"] == {"point_buffer": 2, "no_event_day": 2}
    assert summary["split_group_count"] == 4
    assert summary["event_group_count"] == 2
    assert summary["fallback_date_group_count"] == 2
    assert summary["rows_with_matched_event_ids"] == 2
    assert summary["rows_using_fallback_date_groups"] == 2
    assert summary["rows_with_multi_event_groups"] == 0


def test_summarize_frame_training_targets_reports_boundary_grounding_dominance():
    module = _load_training_module()
    frame = _indicator_frame(rows=4)
    frame["date"] = ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"]
    frame["label"] = np.array([1.0, 1.0, 0.0, np.nan], dtype=np.float32)
    frame["label_status"] = ["positive", "positive", "negative", "uncertain"]
    frame["label_source_mode"] = [
        "province_day",
        "province_day",
        "no_event_day",
        "event_day_unresolved",
    ]
    frame["matched_event_ids"] = ["ff_a", "ff_b", "", ""]
    frame["label_provenance"] = [
        json.dumps(
            {
                "date": "2025-01-01",
                "matched_event_ids": ["ff_a"],
                "matched_geometry_sources": ["province_boundary"],
                "matched_geometry_wkts": [],
            }
        ),
        json.dumps(
            {
                "date": "2025-01-01",
                "matched_event_ids": ["ff_b"],
                "matched_geometry_sources": ["province_boundary"],
                "matched_geometry_wkts": [],
            }
        ),
        "{}",
        "{}",
    ]

    summary = module.summarize_frame_training_targets(frame, "flash_flood")

    quality = summary["supervision_quality"]
    assert quality["boundary_grounded_positive_rows"] == 2
    assert quality["explicit_geometry_positive_rows"] == 0
    assert "boundary_grounding_dominates" in quality["warnings"]


def test_summarize_frame_training_targets_reports_dry_heat_explicit_outcomes():
    module = _load_training_module()
    frame = _indicator_frame(rows=5).drop(columns=["date", "latitude", "longitude"])
    frame["region_id"] = ["asir", "asir", "qassim", "jazan", "jazan"]
    frame["season"] = ["spring", "summer", "summer", "winter", "winter"]
    frame["year"] = [2021, 2021, 2022, 2023, 2023]
    frame["crop_type"] = ["wheat", "wheat", "dates", "sorghum", "sorghum"]
    frame["yield_anomaly"] = np.array([-0.25, np.nan, 0.05, 0.2, 0.35], dtype=np.float32)
    frame["validation_status"] = ["verified", "verified", "verified", "rejected", "verified"]

    summary = module.summarize_frame_training_targets(frame, "dry_heat_agriculture")

    assert summary["target_source"] == "explicit_label"
    assert summary["sample_unit"] == "region-season"
    assert summary["target_column"] == "yield_anomaly"
    assert summary["rows_after_label_filter"] == 3
    assert summary["rows_with_explicit_label"] == 3
    assert summary["validation_status_counts"] == {"verified": 4, "rejected": 1}


def test_summarize_frame_training_targets_infers_region_year_sample_unit():
    module = _load_training_module()
    frame = _indicator_frame(rows=4).drop(columns=["date", "latitude", "longitude"])
    frame["region_id"] = ["asir", "qassim", "jazan", "hail"]
    frame["year"] = [2021, 2021, 2022, 2023]
    frame["crop_type"] = ["wheat", "wheat", "dates", "sorghum"]
    frame["yield_anomaly"] = np.array([-0.2, 0.1, 0.0, 0.25], dtype=np.float32)
    frame["validation_status"] = ["verified"] * len(frame)

    summary = module.summarize_frame_training_targets(frame, "dry_heat_agriculture")

    assert summary["target_source"] == "explicit_label"
    assert summary["sample_unit"] == "region-year"
    assert summary["target_column"] == "yield_anomaly"


def test_build_dry_heat_agriculture_supervised_training_dataset_aggregates_region_year():
    from mazu_saudi.data import build_dry_heat_agriculture_supervised_training_dataset

    features = _dry_heat_daily_region_frame()
    labels = pd.DataFrame(
        [
            {"region_id": "asir", "year": 2024, "crop_type": "wheat", "yield_anomaly": -0.15, "yield_value": 2.4, "harvest_area": 10.0, "source_name": "FAOSTAT", "source_url": "https://example.test/faostat", "validation_status": "verified"},
            {"region_id": "qassim", "year": 2024, "crop_type": "wheat", "yield_anomaly": 0.2, "yield_value": 3.1, "harvest_area": 8.0, "source_name": "MOA", "source_url": "https://example.test/moa", "validation_status": "verified"},
            {"region_id": "asir", "year": 2025, "crop_type": "wheat", "yield_anomaly": 0.05, "yield_value": 2.8, "harvest_area": 11.0, "source_name": "FAOSTAT", "source_url": "https://example.test/faostat-2025", "validation_status": "verified"},
        ]
    )

    merged = build_dry_heat_agriculture_supervised_training_dataset(features, labels, sample_unit="region-year")

    assert len(merged) == 3
    assert set(["sample_unit", "aggregation_start_date", "aggregation_end_date", "feature_row_count", "temp_c_mean", "tmax_c_days_ge_45", "vpd_kpa_days_ge_3", "yield_anomaly", "crop_type"]).issubset(merged.columns)
    asir_2024 = merged[(merged["region_id"] == "asir") & (merged["year"] == 2024)].iloc[0]
    assert asir_2024["sample_unit"] == "region-year"
    assert asir_2024["aggregation_start_date"] == "2024-01-15"
    assert asir_2024["aggregation_end_date"] == "2024-08-02"
    assert asir_2024["feature_row_count"] == 3.0
    assert np.isclose(asir_2024["temp_c_mean"], (31.0 + 35.0 + 39.0) / 3.0)
    assert asir_2024["temp_c_days_ge_35"] == 2.0
    assert asir_2024["tmax_c_days_ge_45"] == 1.0
    assert asir_2024["vpd_kpa_days_ge_3"] == 1.0
    assert bool(asir_2024["is_labeled"])


def test_build_dry_heat_agriculture_supervised_training_dataset_infers_season_keys():
    from mazu_saudi.data import build_dry_heat_agriculture_supervised_training_dataset

    features = _dry_heat_daily_region_frame()
    labels = pd.DataFrame(
        [
            {"region_id": "asir", "year": 2024, "season": "winter", "crop_type": "wheat", "yield_anomaly": -0.05, "validation_status": "verified"},
            {"region_id": "asir", "year": 2024, "season": "spring", "crop_type": "wheat", "yield_anomaly": -0.12, "validation_status": "verified"},
            {"region_id": "asir", "year": 2024, "season": "summer", "crop_type": "wheat", "yield_anomaly": -0.25, "validation_status": "verified"},
        ]
    )

    merged = build_dry_heat_agriculture_supervised_training_dataset(features, labels, sample_unit="region-season")

    assert len(merged) == 3
    assert set(merged["season"]) == {"winter", "spring", "summer"}
    assert set(merged["sample_unit"]) == {"region-season"}


def test_build_dry_heat_agriculture_supervised_training_table_script_exports_csv(tmp_path: Path):
    module = _load_build_dry_heat_supervised_table_module()
    features = _dry_heat_daily_region_frame()
    labels = pd.DataFrame(
        [
            {"region_id": "asir", "year": 2024, "crop_type": "wheat", "yield_anomaly": -0.15, "validation_status": "verified"},
            {"region_id": "qassim", "year": 2024, "crop_type": "wheat", "yield_anomaly": 0.2, "validation_status": "verified"},
            {"region_id": "asir", "year": 2025, "crop_type": "wheat", "yield_anomaly": 0.05, "validation_status": "verified"},
        ]
    )

    feature_path = tmp_path / "dry_heat_features.csv"
    label_path = tmp_path / "dry_heat_labels.csv"
    output_path = tmp_path / "dry_heat_supervised.csv"
    features.to_csv(feature_path, index=False)
    labels.to_csv(label_path, index=False)

    result = module.main(
        [
            "--features",
            str(feature_path),
            "--labels",
            str(label_path),
            "--output",
            str(output_path),
            "--sample-unit",
            "region-year",
        ]
    )
    assert result == 0

    merged = pd.read_csv(output_path)
    assert len(merged) == 3
    assert set(["region_id", "year", "yield_anomaly", "temp_c_mean", "training_join_key"]).issubset(merged.columns)


def test_build_extreme_heat_supervised_training_table_script_exports_csv(tmp_path: Path):
    module = _load_build_extreme_heat_supervised_table_module()
    input_dir = tmp_path / "indicators"
    input_dir.mkdir()

    positive_date = "2024-06-14"
    negative_date = "2024-06-15"
    positive_ds = _extreme_heat_indicator_dataset(positive_date, heat_index_offset=6.0)
    negative_ds = _extreme_heat_indicator_dataset(negative_date, heat_index_offset=-2.0)
    positive_path = input_dir / "saudi_indicators_20240614.nc"
    negative_path = input_dir / "saudi_indicators_20240615.nc"
    positive_ds.to_netcdf(positive_path)
    negative_ds.to_netcdf(negative_path)

    labels = pd.DataFrame(
        [
            {
                "record_id": "Mecca_20240614",
                "event_id": "SA-HEAT-2024-001",
                "hazard_type": "Extreme Heat",
                "start_date": positive_date,
                "end_date": positive_date,
                "location_name": "Mecca",
                "country_code": "SA",
                "validation_status": "verified",
                "label_status": "Labeled",
                "impact_level": "High temperature",
                "impact_count": "51.8°C",
                "label": "Maximum temperature in Mecca reached 51.8°C.",
            }
        ]
    )
    label_path = tmp_path / "verified_extreme_heat.csv"
    labels.to_csv(label_path, index=False)

    output_path = tmp_path / "extreme_heat_supervised.csv"
    assert (
        module.main(
            [
                "--input",
                str(input_dir),
                "--labels",
                str(label_path),
                "--output",
                str(output_path),
                "--format",
                "csv",
                "--sample-unit",
                "single_point_day",
                "--negative-sample-size",
                "1",
                "--seed",
                "7",
            ]
        )
        == 0
    )

    merged = pd.read_csv(output_path)
    summary = json.loads(output_path.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert merged["label"].tolist() == [1.0, 0.0]
    assert merged["label_status"].tolist() == ["positive", "negative"]
    assert merged["hazard_type"].tolist() == ["extreme_heat", "extreme_heat"]
    assert {"temp_c", "tmax_c", "tmin_c", "heat_index_c", "vpd_kpa", "wind_speed_mps", "relative_humidity_percent"}.issubset(merged.columns)
    assert summary["positive_rows"] == 1
    assert summary["negative_rows"] == 1


def test_build_extreme_heat_region_day_supervised_training_table_exports_region_rows(tmp_path: Path):
    module = _load_build_extreme_heat_supervised_table_module()
    input_dir = tmp_path / "indicators"
    input_dir.mkdir()

    positive_date = "2024-06-14"
    negative_date = "2024-06-15"
    _extreme_heat_multi_region_indicator_dataset(positive_date, heat_index_offset=4.0).to_netcdf(
        input_dir / "saudi_indicators_20240614.nc"
    )
    _extreme_heat_multi_region_indicator_dataset(negative_date, heat_index_offset=-2.0).to_netcdf(
        input_dir / "saudi_indicators_20240615.nc"
    )

    labels = pd.DataFrame(
        [
            {
                "record_id": "Mecca_20240614",
                "event_id": "SA-HEAT-2024-001",
                "hazard_type": "Extreme Heat",
                "start_date": positive_date,
                "end_date": positive_date,
                "location_name": "Mecca",
                "country_code": "SA",
                "validation_status": "verified",
                "label_status": "Labeled",
                "impact_level": "High temperature",
                "impact_count": "51.8°C",
                "label": "Maximum temperature in Mecca reached 51.8°C.",
            }
        ]
    )
    label_path = tmp_path / "verified_extreme_heat.csv"
    labels.to_csv(label_path, index=False)

    output_path = tmp_path / "extreme_heat_region_day.csv"
    assert (
        module.main(
            [
                "--input",
                str(input_dir),
                "--labels",
                str(label_path),
                "--output",
                str(output_path),
                "--format",
                "csv",
                "--sample-unit",
                "region_day",
                "--top-k",
                "2",
                "--negative-sample-size",
                "2",
                "--seed",
                "11",
            ]
        )
        == 0
    )

    merged = pd.read_csv(output_path)
    summary = json.loads(output_path.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert "region_id" in merged.columns
    assert "training_join_key" in merged.columns
    assert summary["sample_unit"] == "region_day"
    assert "makkah" in set(merged["region_id"])
    assert merged.loc[merged["region_id"] == "makkah", "label_status"].iloc[0] == "positive"
    assert "outside_event_region" in set(merged["label_source_mode"])
    assert {"grid_cell_count", "pooled_grid_cell_count", "heat_index_c_max", "heat_index_c_p90"}.issubset(merged.columns)


def test_extreme_heat_location_resolution_is_conservative():
    from mazu_saudi.config import ExtremeHeatLabelMappingConfig
    from mazu_saudi.data.extreme_heat_training_dataset import _resolved_region_ids

    config = ExtremeHeatLabelMappingConfig.from_env()
    assert config.location_to_region_ids["mecca and hajj sites"] == ("makkah",)
    assert config.location_to_region_ids["eastern province and riyadh"] == ("eastern_province", "riyadh")
    assert _resolved_region_ids("Al-Qaisumah") == ["eastern_province"]
    assert _resolved_region_ids("Mecca and Hajj sites") == ["makkah"]
    assert _resolved_region_ids("Eastern Province and Riyadh") == ["eastern_province", "riyadh"]
    assert _resolved_region_ids("Al-Ahsa, Al-Kharj") == ["eastern_province", "riyadh"]
    assert _resolved_region_ids("Multiple cities") == ["saudi_arabia"]


def test_summarize_frame_training_targets_reports_extreme_heat_region_day_sample_unit():
    module = _load_training_module()
    frame = pd.DataFrame(
        [
            {
                "date": "2024-06-14",
                "region_id": "makkah",
                "sample_unit": "region_day",
                "label": 1.0,
                "label_status": "positive",
                "label_source_mode": "region_day_overlap",
                "matched_event_ids": "SA-HEAT-2024-001",
                "temp_c": 41.0,
                "tmax_c": 45.0,
                "heat_index_c": 49.0,
                "vpd_kpa": 2.2,
                "wind_speed_mps": 3.5,
                "relative_humidity_percent": 34.0,
            },
            {
                "date": "2024-06-15",
                "region_id": "makkah",
                "sample_unit": "region_day",
                "label": 0.0,
                "label_status": "negative",
                "label_source_mode": "outside_event_region",
                "matched_event_ids": "",
                "temp_c": 38.0,
                "tmax_c": 42.0,
                "heat_index_c": 44.0,
                "vpd_kpa": 1.8,
                "wind_speed_mps": 3.0,
                "relative_humidity_percent": 36.0,
            },
        ]
    )

    summary = module.summarize_frame_training_targets(frame, "extreme_heat")

    assert summary["target_source"] == "explicit_label"
    assert summary["sample_unit"] == "region_day"
    assert summary["target_column"] == "label"


def test_compare_extreme_heat_supervision_variants_script_exports_json(tmp_path: Path):
    module = _load_compare_extreme_heat_supervision_variants_module()
    input_dir = tmp_path / "indicators"
    input_dir.mkdir()

    dates = [
        "2024-06-01",
        "2024-06-02",
        "2024-06-03",
        "2024-06-04",
        "2024-06-05",
        "2024-06-06",
        "2024-06-07",
        "2024-06-08",
        "2024-06-09",
        "2024-06-10",
        "2024-06-11",
        "2024-06-12",
        "2024-06-13",
        "2024-06-14",
    ]
    positive_dates = {"2024-06-01", "2024-06-03", "2024-06-05", "2024-06-07", "2024-06-09", "2024-06-11"}
    for index, date in enumerate(dates):
        ds = _extreme_heat_indicator_dataset(date, heat_index_offset=6.0 if date in positive_dates else -2.0)
        ds.to_netcdf(input_dir / f"saudi_indicators_{date.replace('-', '')}.nc")

    label_rows = [
        {
            "record_id": f"Mecca_{date.replace('-', '')}",
            "event_id": f"SA-HEAT-{index:03d}",
            "hazard_type": "Extreme Heat",
            "start_date": date,
            "end_date": date,
            "location_name": "Mecca",
            "country_code": "SA",
            "validation_status": "verified",
            "label_status": "Labeled",
            "impact_level": "High temperature",
            "impact_count": "51.8°C",
            "label": f"Maximum temperature in Mecca reached 51.8°C on {date}.",
        }
        for index, date in enumerate(sorted(positive_dates), start=1)
    ]
    label_path = tmp_path / "verified_extreme_heat.csv"
    pd.DataFrame(label_rows).to_csv(label_path, index=False)

    output_path = tmp_path / "comparison.json"
    assert (
        module.main(
            [
                "--input",
                str(input_dir),
                "--labels",
                str(label_path),
                "--output",
                str(output_path),
                "--num-boost-round",
                "15",
                "--early-stopping-rounds",
                "5",
            ]
        )
        == 0
    )

    comparison = json.loads(output_path.read_text(encoding="utf-8"))
    assert comparison["selection"]["selected_positive_dates"] == len(positive_dates)
    assert comparison["selection"]["selected_negative_dates"] == len(dates) - len(positive_dates)
    assert comparison["selection"]["sample_unit"] == "region_day"
    assert comparison["selection"]["top_k_values"] == [1, 3]
    assert len(comparison["variants"]) == 12
    assert comparison["best_variant"] in {variant["name"] for variant in comparison["variants"]}
    assert all("validation_metric" in variant for variant in comparison["variants"])
    assert all(variant["top_k"] in {1, 3} for variant in comparison["variants"])
    assert all(variant["name"].startswith("region_day_") for variant in comparison["variants"])


def test_build_flash_flood_supervised_training_table_script_exports_csv(tmp_path: Path):
    module = _load_build_supervised_table_module()
    features = pd.DataFrame(
        [
            {"date": "2022-11-24", "hazard_type": "flash_flood", "latitude": 21.49004, "longitude": 39.19996, "daily_precip_total": 30.0},
            {"date": "2022-11-24", "hazard_type": "flash_flood", "latitude": 24.71004, "longitude": 46.67004, "daily_precip_total": 1.0},
            {"date": "2022-11-25", "hazard_type": "flash_flood", "latitude": 24.71004, "longitude": 46.67004, "daily_precip_total": 0.0},
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "date": "2022-11-24",
                "hazard_type": "flash_flood",
                "latitude": 21.49,
                "longitude": 39.20,
                "label": 1.0,
                "label_status": "positive",
                "label_source_mode": "point_buffer",
                "matched_event_ids": "ff_jeddah_20221124",
                "label_provenance": "{}",
            },
            {
                "date": "2022-11-24",
                "hazard_type": "flash_flood",
                "latitude": 24.71,
                "longitude": 46.67,
                "label": np.nan,
                "label_status": "uncertain",
                "label_source_mode": "event_day_unresolved",
                "matched_event_ids": "",
                "label_provenance": "{}",
            },
            {
                "date": "2022-11-25",
                "hazard_type": "flash_flood",
                "latitude": 24.71,
                "longitude": 46.67,
                "label": 0.0,
                "label_status": "negative",
                "label_source_mode": "no_event_day",
                "matched_event_ids": "",
                "label_provenance": "{}",
            },
        ]
    )

    feature_path = tmp_path / "features.csv"
    label_path = tmp_path / "labels.csv"
    output_path = tmp_path / "merged.csv"
    features.to_csv(feature_path, index=False)
    labels.to_csv(label_path, index=False)

    assert module.main(["--features", str(feature_path), "--labels", str(label_path), "--output", str(output_path)]) == 0

    merged = pd.read_csv(output_path)
    summary = json.loads(output_path.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert merged["label_status"].tolist() == ["positive", "negative"]
    assert merged["training_join_mode"].nunique() == 1
    assert merged["training_join_mode"].iloc[0] == "grid_day"
    assert summary["label_input_audit"]["input_rows"] == 3
    assert summary["label_input_audit"]["input_label_status_counts"] == {"positive": 1, "uncertain": 1, "negative": 1}
    assert summary["label_input_audit"]["input_event_day_unresolved_rows"] == 1


def test_build_flash_flood_supervised_training_table_script_reports_supervision_quality(tmp_path: Path):
    module = _load_build_supervised_table_module()
    features = pd.DataFrame(
        [
            {"date": "2022-11-24", "hazard_type": "flash_flood", "latitude": 21.50, "longitude": 39.20, "flash_flood_risk": 3},
            {"date": "2022-11-24", "hazard_type": "flash_flood", "latitude": 24.71, "longitude": 46.67, "flash_flood_risk": 1},
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "date": "2022-11-24",
                "hazard_type": "flash_flood",
                "latitude": 21.50,
                "longitude": 39.20,
                "label": 1.0,
                "label_status": "positive",
                "label_source_mode": "point_buffer",
                "matched_event_ids": "ff_event_a",
                "label_provenance": "{}",
            },
            {
                "date": "2022-11-24",
                "hazard_type": "flash_flood",
                "latitude": 24.71,
                "longitude": 46.67,
                "label": 0.0,
                "label_status": "negative",
                "label_source_mode": "no_event_day",
                "matched_event_ids": "",
                "label_provenance": "{}",
            },
        ]
    )

    feature_path = tmp_path / "features.csv"
    label_path = tmp_path / "labels.csv"
    output_path = tmp_path / "merged.csv"
    features.to_csv(feature_path, index=False)
    labels.to_csv(label_path, index=False)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        assert module.main(["--features", str(feature_path), "--labels", str(label_path), "--output", str(output_path)]) == 0

    summary = json.loads(stdout.getvalue())
    assert summary["supervision_quality"]["status"] == "warning"
    assert "no_geometry_backed_positives" in summary["supervision_quality"]["warnings"]
    assert summary["supervision_quality"]["matched_event_fraction"] == 0.5
    assert summary["geometry_positive_source_counts"] == {}


def test_indicator_csv_training_summary_loads_source_label_audit_from_sidecar():
    module = _load_training_module()
    build_module = _load_build_supervised_table_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        features = _indicator_frame(rows=32)
        labels = pd.DataFrame(
            [
                {
                    "date": features.loc[0, "date"],
                    "hazard_type": "flash_flood",
                    "latitude": features.loc[0, "latitude"],
                    "longitude": features.loc[0, "longitude"],
                    "label": 1.0,
                    "label_status": "positive",
                    "label_source_mode": "point_buffer",
                    "matched_event_ids": "ff_event_a",
                    "label_provenance": "{}",
                },
                {
                    "date": features.loc[1, "date"],
                    "hazard_type": "flash_flood",
                    "latitude": features.loc[1, "latitude"],
                    "longitude": features.loc[1, "longitude"],
                    "label": np.nan,
                    "label_status": "uncertain",
                    "label_source_mode": "event_day_unresolved",
                    "matched_event_ids": "",
                    "label_provenance": "{}",
                },
                {
                    "date": features.loc[2, "date"],
                    "hazard_type": "flash_flood",
                    "latitude": features.loc[2, "latitude"],
                    "longitude": features.loc[2, "longitude"],
                    "label": 0.0,
                    "label_status": "negative",
                    "label_source_mode": "no_event_day",
                    "matched_event_ids": "",
                    "label_provenance": "{}",
                },
            ]
        )

        labels = pd.concat(
            [
                labels,
                pd.DataFrame(
                    [
                        {
                            "date": features.loc[index, "date"],
                            "hazard_type": "flash_flood",
                            "latitude": features.loc[index, "latitude"],
                            "longitude": features.loc[index, "longitude"],
                            "label": 1.0 if index % 2 == 0 else 0.0,
                            "label_status": "positive" if index % 2 == 0 else "negative",
                            "label_source_mode": "point_buffer" if index % 2 == 0 else "no_event_day",
                            "matched_event_ids": f"ff_event_{index}",
                            "label_provenance": "{}",
                        }
                        for index in range(3, len(features))
                    ]
                ),
            ],
            ignore_index=True,
        )

        feature_path = tmp_path / "features.csv"
        label_path = tmp_path / "labels.csv"
        supervised_path = tmp_path / "supervised.csv"
        model_dir = tmp_path / "models"
        features.to_csv(feature_path, index=False)
        labels.to_csv(label_path, index=False)

        assert (
            build_module.main(
                [
                    "--features",
                    str(feature_path),
                    "--labels",
                    str(label_path),
                    "--output",
                    str(supervised_path),
                ]
            )
            == 0
        )

        old_argv = sys.argv
        sys.argv = [
            "train_layer4_lightgbm.py",
            "--source",
            str(supervised_path),
            "--source-format",
            "indicator-csv",
            "--model-dir",
            str(model_dir),
            "--hazard-type",
            "flash_flood",
        ]
        try:
            assert module.main() == 0
        finally:
            sys.argv = old_argv

        summary = json.loads((model_dir / "train_summary.json").read_text(encoding="utf-8"))
        assert summary["training_target"]["source_label_audit"]["input_rows"] == 32
        assert summary["training_target"]["source_label_audit"]["input_event_day_unresolved_rows"] == 1


def test_indicator_csv_training_summary_derives_source_audit_without_sidecar():
    module = _load_training_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = _indicator_frame(rows=8)
        source["hazard_type"] = "flash_flood"
        source["label"] = np.where(np.arange(len(source)) == 0, 1.0, 0.0)
        source["label_status"] = np.where(np.arange(len(source)) == 0, "positive", "negative")
        source["label_source_mode"] = np.where(np.arange(len(source)) == 0, "point_buffer", "no_event_day")
        source["matched_event_ids"] = np.where(np.arange(len(source)) == 0, "ff_event_a", "")
        source["label_provenance"] = "{}"

        source_path = tmp_path / "source.csv"
        model_dir = tmp_path / "models"
        source.to_csv(source_path, index=False)

        old_argv = sys.argv
        sys.argv = [
            "train_layer4_lightgbm.py",
            "--source",
            str(source_path),
            "--source-format",
            "indicator-csv",
            "--model-dir",
            str(model_dir),
            "--hazard-type",
            "flash_flood",
        ]
        try:
            assert module.main() == 0
        finally:
            sys.argv = old_argv

        summary = json.loads((model_dir / "train_summary.json").read_text(encoding="utf-8"))
        assert summary["training_target"]["source_label_audit"]["input_rows"] == 8
        assert summary["training_target"]["source_label_audit"]["input_event_day_unresolved_rows"] == 0
        assert summary["training_target"]["source_supervision_quality"]["status"] in {"warning", "ok"}


def test_build_flash_flood_supervised_training_table_uses_geometry_provenance_for_quality_audit(tmp_path: Path):
    module = _load_build_supervised_table_module()
    features = pd.DataFrame(
        [
            {"date": "2022-11-24", "hazard_type": "flash_flood", "latitude": 21.50, "longitude": 39.20, "flash_flood_risk": 3},
            {"date": "2022-11-24", "hazard_type": "flash_flood", "latitude": 24.71, "longitude": 46.67, "flash_flood_risk": 1},
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "date": "2022-11-24",
                "hazard_type": "flash_flood",
                "latitude": 21.50,
                "longitude": 39.20,
                "label": 1.0,
                "label_status": "positive",
                "label_source_mode": "point_buffer",
                "matched_event_ids": "ff_event_a",
                "label_provenance": json.dumps(
                    {
                        "date": "2022-11-24",
                        "matched_event_ids": ["ff_event_a"],
                        "matched_geometry_sources": ["derived_point_buffer"],
                        "matched_geometry_wkts": [],
                    }
                ),
            },
            {
                "date": "2022-11-24",
                "hazard_type": "flash_flood",
                "latitude": 24.71,
                "longitude": 46.67,
                "label": 0.0,
                "label_status": "negative",
                "label_source_mode": "no_event_day",
                "matched_event_ids": "",
                "label_provenance": "{}",
            },
        ]
    )

    feature_path = tmp_path / "features.csv"
    label_path = tmp_path / "labels.csv"
    output_path = tmp_path / "merged.csv"
    features.to_csv(feature_path, index=False)
    labels.to_csv(label_path, index=False)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        assert module.main(["--features", str(feature_path), "--labels", str(label_path), "--output", str(output_path)]) == 0

    summary = json.loads(stdout.getvalue())
    assert summary["supervision_quality"]["status"] == "ok"
    assert "no_geometry_backed_positives" not in summary["supervision_quality"]["warnings"]
    assert summary["supervision_quality"]["geometry_positive_fraction_of_positives"] == 1.0
    assert summary["geometry_positive_source_counts"] == {"derived_point_buffer": 1}


def test_build_flash_flood_training_labels_script_writes_summary_sidecar(tmp_path: Path):
    module = _load_build_flash_flood_labels_module()
    samples = _indicator_frame(rows=3)
    sample_path = tmp_path / "samples.csv"
    output_path = tmp_path / "labels.csv"
    samples.to_csv(sample_path, index=False)

    result = module.main(["--samples", str(sample_path), "--output", str(output_path)])
    assert result == 0

    summary_path = output_path.with_suffix(".summary.json")
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["rows"] == 3
    assert summary["supervision_quality"]["status"] == "insufficient"


def test_build_flash_flood_supervised_training_table_rejects_missing_positive_labels(tmp_path: Path):
    module = _load_build_supervised_table_module()
    features = pd.DataFrame(
        [
            {"date": "2022-11-24", "hazard_type": "flash_flood", "latitude": 21.50, "longitude": 39.20, "flash_flood_risk": 3},
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "date": "2022-11-24",
                "hazard_type": "flash_flood",
                "latitude": 21.50,
                "longitude": 39.20,
                "label": 1.0,
                "label_status": "positive",
                "label_source_mode": "point_buffer",
                "matched_event_ids": "ff_event_a",
                "label_provenance": "{}",
            },
            {
                "date": "2022-11-24",
                "hazard_type": "flash_flood",
                "latitude": 24.71,
                "longitude": 46.67,
                "label": 1.0,
                "label_status": "positive",
                "label_source_mode": "point_buffer",
                "matched_event_ids": "ff_event_b",
                "label_provenance": "{}",
            },
        ]
    )

    feature_path = tmp_path / "features.csv"
    label_path = tmp_path / "labels.csv"
    output_path = tmp_path / "merged.csv"
    features.to_csv(feature_path, index=False)
    labels.to_csv(label_path, index=False)

    with pytest.raises(RuntimeError, match=r"missing 1 positive label rows"):
        module.main(["--features", str(feature_path), "--labels", str(label_path), "--output", str(output_path)])


def test_build_flash_flood_supervised_training_table_script_streams_compatible_label_parquet(tmp_path: Path):
    module = _load_build_supervised_table_module()
    labels = pd.DataFrame(
        [
            {
                "date": "2022-11-24",
                "hazard_type": "flash_flood",
                "latitude": 21.49004,
                "longitude": 39.19996,
                "daily_precip_total": 30.0,
                "label": 1.0,
                "label_status": "positive",
                "label_source_mode": "point_buffer",
                "matched_event_ids": "ff_jeddah_20221124",
                "label_provenance": "{}",
            },
            {
                "date": "2022-11-24",
                "hazard_type": "flash_flood",
                "latitude": 24.71004,
                "longitude": 46.67004,
                "daily_precip_total": 1.0,
                "label": np.nan,
                "label_status": "uncertain",
                "label_source_mode": "event_day_unresolved",
                "matched_event_ids": "",
                "label_provenance": "{}",
            },
            {
                "date": "2022-11-25",
                "hazard_type": "flash_flood",
                "latitude": 24.71004,
                "longitude": 46.67004,
                "daily_precip_total": 0.0,
                "label": 0.0,
                "label_status": "negative",
                "label_source_mode": "no_event_day",
                "matched_event_ids": "",
                "label_provenance": "{}",
            },
        ]
    )
    features = labels.drop(columns=["label", "label_status", "label_source_mode", "matched_event_ids", "label_provenance"]).copy()

    feature_path = tmp_path / "features.parquet"
    label_path = tmp_path / "labels.parquet"
    output_path = tmp_path / "merged.parquet"
    features.to_parquet(feature_path, index=False)
    labels.to_parquet(label_path, index=False)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        assert (
            module.main(
                [
                    "--features",
                    str(feature_path),
                    "--labels",
                    str(label_path),
                    "--output",
                    str(output_path),
                    "--batch-rows",
                    "1",
                ]
            )
            == 0
        )

    summary = json.loads(stdout.getvalue())
    merged = pd.read_parquet(output_path)

    assert merged["label_status"].tolist() == ["positive", "negative"]
    assert merged["training_join_mode"].nunique() == 1
    assert merged["training_join_mode"].iloc[0] == "grid_day"
    assert summary["rows"] == 2
    assert summary["labeled_rows"] == 2


def test_build_dust_storm_supervised_training_table_script_exports_csv(tmp_path: Path):
    module = _load_build_dust_storm_supervised_table_module()
    features = pd.DataFrame(
        [
            {"date": "2025-05-04", "hazard_type": "dust_storm", "province_name": "Qassim", "dust_aod": 0.45},
            {"date": "2025-05-04", "hazard_type": "dust_storm", "province_name": "Madinah", "dust_aod": 0.12},
            {"date": "2025-05-06", "hazard_type": "dust_storm", "province_name": "Madinah", "dust_aod": 0.08},
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "date": "2025-05-04",
                "hazard_type": "dust_storm",
                "province_name": "Qassim",
                "label": 1.0,
                "label_status": "positive",
                "label_source_mode": "region_day_text",
                "matched_event_ids": "dust_20250504_qassim_riyadh",
                "label_provenance": "{}",
            },
            {
                "date": "2025-05-04",
                "hazard_type": "dust_storm",
                "province_name": "Madinah",
                "label": 0.0,
                "label_status": "negative",
                "label_source_mode": "outside_event_regions",
                "matched_event_ids": "",
                "label_provenance": "{}",
            },
            {
                "date": "2025-05-06",
                "hazard_type": "dust_storm",
                "province_name": "Madinah",
                "label": 0.0,
                "label_status": "negative",
                "label_source_mode": "no_event_day",
                "matched_event_ids": "",
                "label_provenance": "{}",
            },
        ]
    )

    feature_path = tmp_path / "features.csv"
    label_path = tmp_path / "labels.csv"
    output_path = tmp_path / "merged.csv"
    features.to_csv(feature_path, index=False)
    labels.to_csv(label_path, index=False)

    assert module.main(["--features", str(feature_path), "--labels", str(label_path), "--output", str(output_path)]) == 0

    merged = pd.read_csv(output_path)
    assert merged["label_status"].tolist() == ["positive", "negative", "negative"]
    assert merged["training_join_mode"].nunique() == 1
    assert merged["training_join_mode"].iloc[0] == "region_day:province_name"


def test_build_dust_storm_supervised_training_table_script_builds_features_from_processed_indicators(tmp_path: Path):
    import scripts.build_dust_storm_supervised_training_table as dust_script

    input_dir = tmp_path / "lightgbm_indicators_nc"
    input_dir.mkdir()
    valid_path = input_dir / "saudi_indicators_20250504.nc"
    output_path = tmp_path / "merged.csv"

    ds = xr.Dataset(
        {
            "dust_aod": (("time", "latitude", "longitude"), np.array([[[0.1]]], dtype=np.float32)),
            "dust_column_mass": (("time", "latitude", "longitude"), np.array([[[0.2]]], dtype=np.float32)),
            "dust_surface_mass": (("time", "latitude", "longitude"), np.array([[[0.3]]], dtype=np.float32)),
        },
        coords={
            "time": np.array(["2025-05-04"], dtype="datetime64[ns]"),
            "latitude": np.array([24.71]),
            "longitude": np.array([46.67]),
        },
    )
    ds.to_netcdf(valid_path)

    labels = pd.DataFrame(
        [
            {
                "date": "2025-05-04",
                "hazard_type": "dust_storm",
                "province_name": "riyadh",
                "label": 1.0,
                "label_status": "positive",
                "label_source_mode": "region_day_text",
                "matched_event_ids": "dust_20250504_riyadh",
                "label_provenance": "{}",
            }
        ]
    )
    label_path = tmp_path / "labels.csv"
    labels.to_csv(label_path, index=False)

    captured: dict[str, object] = {}

    original_build_features = dust_script.build_dust_storm_province_day_feature_table

    def _fake_build(feature_paths, **kwargs):
        captured["feature_paths"] = [Path(path).name for path in feature_paths]
        captured["kwargs"] = kwargs
        return pd.DataFrame(
            [
                {
                    "date": "2025-05-04",
                    "hazard_type": "dust_storm",
                    "province_name": "riyadh",
                    "region_id": "riyadh",
                    "dust_aod": 0.1,
                }
            ]
        )

    dust_script.build_dust_storm_province_day_feature_table = _fake_build
    try:
        assert dust_script.main(["--input", str(input_dir), "--labels", str(label_path), "--output", str(output_path)]) == 0
    finally:
        dust_script.build_dust_storm_province_day_feature_table = original_build_features

    assert captured["feature_paths"] == ["saudi_indicators_20250504.nc"]
    assert captured["kwargs"]["boundary_path"] == dust_script.parse_args(["--labels", str(label_path)]).boundary_path
    assert output_path.exists()


def test_build_dust_storm_province_day_feature_table_script_skips_invalid_inputs(tmp_path: Path):
    import scripts.build_dust_storm_province_day_feature_table as dust_script

    input_dir = tmp_path / "dust"
    input_dir.mkdir()
    valid_path = input_dir / "MERRA2_20250504.nc4"
    invalid_path = input_dir / "MERRA2_20250505.nc4"
    output_path = tmp_path / "province_day.csv"

    ds = xr.Dataset(
        {
            "DUEXTTAU": (("lat", "lon"), np.array([[0.1]])),
            "DUCMASS": (("lat", "lon"), np.array([[0.2]])),
            "DUSMASS": (("lat", "lon"), np.array([[0.3]])),
        },
        coords={"lat": np.array([24.71]), "lon": np.array([46.67])},
    )
    ds.to_netcdf(valid_path)
    invalid_path.write_bytes(b"%PDF-1.7\nfake pdf payload\n")

    captured: dict[str, object] = {}

    def _fake_build(feature_paths, **kwargs):
        captured["feature_paths"] = [Path(path).name for path in feature_paths]
        return pd.DataFrame([{"date": "2025-05-04", "province_name": "riyadh", "region_id": "riyadh"}])

    original = dust_script.build_dust_storm_province_day_feature_table
    dust_script.build_dust_storm_province_day_feature_table = _fake_build
    try:
        assert (
            dust_script.main(
                [
                    "--input",
                    str(input_dir),
                    "--output",
                    str(output_path),
                    "--glob",
                    "MERRA2_*.nc4",
                ]
            )
            == 0
        )
    finally:
        dust_script.build_dust_storm_province_day_feature_table = original


def test_build_dust_storm_province_day_feature_table_script_defaults_to_processed_indicators(tmp_path: Path):
    import scripts.build_dust_storm_province_day_feature_table as dust_script

    input_dir = tmp_path / "lightgbm_indicators_nc"
    input_dir.mkdir()
    valid_path = input_dir / "saudi_indicators_20250504.nc"
    output_path = tmp_path / "province_day.csv"

    ds = xr.Dataset(
        {
            "dust_aod": (("time", "latitude", "longitude"), np.array([[[0.1]]], dtype=np.float32)),
            "dust_column_mass": (("time", "latitude", "longitude"), np.array([[[0.2]]], dtype=np.float32)),
            "dust_surface_mass": (("time", "latitude", "longitude"), np.array([[[0.3]]], dtype=np.float32)),
        },
        coords={
            "time": np.array(["2025-05-04"], dtype="datetime64[ns]"),
            "latitude": np.array([24.71]),
            "longitude": np.array([46.67]),
        },
    )
    ds.to_netcdf(valid_path)

    captured: dict[str, object] = {}

    def _fake_build(feature_paths, **kwargs):
        captured["feature_paths"] = [Path(path).name for path in feature_paths]
        return pd.DataFrame([{"date": "2025-05-04", "province_name": "riyadh", "region_id": "riyadh"}])

    original = dust_script.build_dust_storm_province_day_feature_table
    dust_script.build_dust_storm_province_day_feature_table = _fake_build
    try:
        assert dust_script.main(["--output", str(output_path), "--input", str(input_dir)]) == 0
    finally:
        dust_script.build_dust_storm_province_day_feature_table = original

    assert captured["feature_paths"] == ["saudi_indicators_20250504.nc"]
    assert output_path.exists()


def test_dust_storm_training_payload_keeps_dust_feature_columns():
    module = _load_training_module()
    frame = pd.DataFrame(
        [
            {
                "date": "2025-05-04",
                "hazard_type": "dust_storm",
                "province_name": "Qassim",
                "DUEXTTAU": 0.45,
                "DUCMASS": 0.12,
                "DUSMASS": 0.003,
                "label": 1.0,
                "label_status": "positive",
                "label_source_mode": "region_day_text",
                "matched_event_ids": "dust_20250504_qassim_riyadh",
                "label_provenance": "{}",
            },
            {
                "date": "2025-05-06",
                "hazard_type": "dust_storm",
                "province_name": "Madinah",
                "DUEXTTAU": 0.12,
                "DUCMASS": 0.05,
                "DUSMASS": 0.001,
                "label": 0.0,
                "label_status": "negative",
                "label_source_mode": "no_event_day",
                "matched_event_ids": "",
                "label_provenance": "{}",
            },
        ]
    )

    payload = module.build_training_payload_from_frame(frame, "dust_storm")

    assert payload["feature_names"] == ["dust_aod", "dust_column_mass", "dust_surface_mass"]
    assert payload["features"].shape == (2, 3)
    assert payload["labels"].tolist() == [1.0, 0.0]


def test_demo_flash_flood_supervised_training_builds_balanced_dataset(tmp_path: Path):
    module = _load_demo_supervised_module()

    summary = module.run_demo(tmp_path, rows_per_bucket=3, train_model=False)

    assert summary["feature_rows"] == 36
    assert summary["label_rows"] == 36
    assert summary["supervised_rows"] == 36
    assert summary["positive_rows"] == 18
    assert summary["negative_rows"] == 18
    assert summary["training_skipped_reason"] == "train_model_disabled"
    assert (tmp_path / "flash_flood_demo_features.csv").exists()
    assert (tmp_path / "flash_flood_demo_labels.csv").exists()
    if _parquet_available():
        assert (tmp_path / "flash_flood_demo_supervised.parquet").exists()


def test_demo_flash_flood_supervised_training_trains_with_adapter(tmp_path: Path):
    if not _lightgbm_available():
        return

    module = _load_demo_supervised_module()
    summary = module.run_demo(tmp_path, rows_per_bucket=3, train_model=True)

    assert summary["model_path"].endswith("flash_flood.txt")
    assert Path(summary["model_path"]).exists()
    assert Path(f"{summary['model_path']}.metadata.json").exists()
    train_summary = json.loads(Path(summary["train_summary_path"]).read_text(encoding="utf-8"))
    assert train_summary["hazard_type"] == "flash_flood"
    assert train_summary["model"]["backend"] == "lightgbm"
    assert train_summary["model"]["objective"] == "binary"
    assert train_summary["model"]["metric"] == "binary_logloss"
    assert train_summary["training_target"]["target_source"] == "explicit_label"
    assert train_summary["model"]["split_group_count"] >= 1
    assert train_summary["model"]["event_group_count"] >= 1
    assert train_summary["model"]["rows_with_matched_event_ids"] >= 1
    assert "training_split_group_audit" in train_summary["training_target"]
