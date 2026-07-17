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
BUILD_SUPERVISED_TABLE_SCRIPT_PATH = ROOT / "scripts" / "build_flash_flood_supervised_training_table.py"
BUILD_DRY_HEAT_SUPERVISED_TABLE_SCRIPT_PATH = ROOT / "scripts" / "build_dry_heat_agriculture_supervised_training_table.py"
BUILD_DUST_STORM_SUPERVISED_TABLE_SCRIPT_PATH = ROOT / "scripts" / "build_dust_storm_supervised_training_table.py"
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
        assert summary["training_target"]["target_source"] == "explicit_label"
        assert (model_dir / "flash_flood.txt").exists()


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

    summary = module.summarize_frame_training_targets(frame, "flash_flood")

    assert summary["target_source"] == "explicit_label"
    assert summary["input_rows"] == 6
    assert summary["rows_after_label_filter"] == 4
    assert summary["rows_with_explicit_label"] == 4
    assert summary["positive_labels"] == 2
    assert summary["negative_labels"] == 2
    assert summary["label_status_counts"] == {"positive": 2, "negative": 2, "uncertain": 2}
    assert summary["label_source_mode_counts"] == {"point_buffer": 2, "no_event_day": 2}


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
    assert merged["label_status"].tolist() == ["positive", "negative"]
    assert merged["training_join_mode"].nunique() == 1
    assert merged["training_join_mode"].iloc[0] == "grid_day"


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
