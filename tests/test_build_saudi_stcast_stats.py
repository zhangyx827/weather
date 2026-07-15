"""Tests for Saudi STCast statistics generation."""

from __future__ import annotations

import importlib.util
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_saudi_stcast_stats.py"
    spec = importlib.util.spec_from_file_location("build_saudi_stcast_stats", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load build_saudi_stcast_stats.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_sample_tree(root: Path, year: int, value: float) -> None:
    module = _load_script_module()
    stamp = datetime(year, 1, 1, 0, 0)
    pressure_day = root / f"{year}" / f"{year}-01-01"
    surface_day = root / "single" / f"{year}" / f"{year}-01-01"
    pressure_day.mkdir(parents=True, exist_ok=True)
    surface_day.mkdir(parents=True, exist_ok=True)

    for var in module.PRESSURE_VARIABLES:
        for level in module.PRESSURE_LEVELS:
            np.save(pressure_day / f"{stamp:%H}:00:00-{var}-{level}.npy", np.full((1, 1), value, dtype=np.float32))
    for name in module.SURFACE_VARIABLES:
        np.save(surface_day / f"{stamp:%H}:00:00-{name}.npy", np.full((1, 1), value, dtype=np.float32))


def test_build_stats_merges_multiple_sources():
    module = _load_script_module()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root_2022 = tmp_path / "stcast_2022"
        root_2024 = tmp_path / "stcast_2024"
        stats_dir = tmp_path / "joint_stats"

        _write_sample_tree(root_2022, 2022, 1.0)
        _write_sample_tree(root_2024, 2024, 3.0)

        module.build_stats(
            [
                {"root_dir": root_2022, "train_start": datetime(2022, 1, 1, 0, 0), "train_end": datetime(2022, 1, 1, 0, 0)},
                {"root_dir": root_2024, "train_start": datetime(2024, 1, 1, 0, 0), "train_end": datetime(2024, 1, 1, 0, 0)},
            ],
            stats_dir,
            6,
        )

        with (stats_dir / "mean_std.json").open("r", encoding="utf-8") as fh:
            level_stats = module.json.load(fh)
        with (stats_dir / "mean_std_single.json").open("r", encoding="utf-8") as fh:
            surface_stats = module.json.load(fh)

    assert level_stats["count"] == 2
    assert surface_stats["count"] == 2
    assert level_stats["mean"]["z"][0] == 2.0
    assert level_stats["mean"]["z_overall"] == 2.0
    assert surface_stats["mean"]["t2m"] == 2.0
    assert surface_stats["std"]["t2m"] == 1.0


def test_parse_source_spec():
    module = _load_script_module()

    parsed = module._parse_source_spec("data/processed/stcast_saudi_2022_6h,2022-01-01T00:00,2022-12-31T18:00")

    assert parsed["root_dir"] == Path("data/processed/stcast_saudi_2022_6h")
    assert parsed["train_start"].isoformat(timespec="minutes") == "2022-01-01T00:00"
    assert parsed["train_end"].isoformat(timespec="minutes") == "2022-12-31T18:00"
