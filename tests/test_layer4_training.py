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


def _load_training_module():
    spec = importlib.util.spec_from_file_location("train_layer4_lightgbm", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _indicator_frame(rows: int = 512) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "t2m_c": rng.uniform(24.0, 48.0, rows).astype(np.float32),
            "vpd_kpa": rng.uniform(0.2, 6.0, rows).astype(np.float32),
            "heat_index_c": rng.uniform(25.0, 52.0, rows).astype(np.float32),
            "wind10_speed": rng.uniform(0.5, 14.0, rows).astype(np.float32),
            "rh2m": rng.uniform(10.0, 95.0, rows).astype(np.float32),
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
            "vpd_kpa": (("time", "latitude", "longitude"), (1.5 + base * 0.05)[None, :, :]),
            "heat_index_c": (("time", "latitude", "longitude"), (37.0 + base * 0.2)[None, :, :]),
            "wind10_speed": (("time", "latitude", "longitude"), (3.0 + base * 0.1)[None, :, :]),
            "rh2m": (("time", "latitude", "longitude"), (45.0 + base)[None, :, :]),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )


def test_indicator_parquet_training_smoke():
    module = _load_training_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "saudi_indicator_samples_2025.parquet"
        model_dir = tmp_path / "models"
        _indicator_frame().to_parquet(source, index=False)

        old_argv = sys.argv
        sys.argv = ["train_layer4_lightgbm.py", "--source", str(source), "--source-format", "indicator-parquet", "--model-dir", str(model_dir)]
        try:
            assert module.main() == 0
        finally:
            sys.argv = old_argv

        summary = json.loads((model_dir / "train_summary.json").read_text(encoding="utf-8"))
        assert summary["source_format"] == "indicator-parquet"
        assert (model_dir / "extreme_heat.txt").exists()
        assert (model_dir / "dry_heat_stress.txt").exists()


def test_indicator_netcdf_training_table():
    module = _load_training_module()
    ds = _indicator_dataset()
    features, extreme_heat, dry_heat = module.build_training_table(ds)
    assert features.shape[1] == 5
    assert features.shape[0] == ds.latitude.size * ds.longitude.size
    assert extreme_heat.shape == (features.shape[0],)
    assert dry_heat.shape == (features.shape[0],)
