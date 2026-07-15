from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "audit_saudi_stcast_dataset.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("audit_saudi_stcast_dataset", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_npy(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.zeros((2, 2), dtype=np.float32))


def _build_stcast_tree(root: Path, *, start: datetime, end: datetime, step_hours: int) -> None:
    module = _load_script_module()
    current = start
    while current <= end:
        pressure_day = root / "2024" / f"{current:%Y-%m-%d}"
        surface_day = root / "single" / "2024" / f"{current:%Y-%m-%d}"
        for var_name in module.PRESSURE_VARIABLES:
            for level in module.PRESSURE_LEVELS:
                _write_npy(pressure_day / f"{current:%H}:00:00-{var_name}-{level}.npy")
        for var_name in module.SURFACE_VARIABLES:
            _write_npy(surface_day / f"{current:%H}:00:00-{var_name}.npy")
        current += timedelta(hours=step_hours)


def test_audit_stcast_dataset_passes_complete_dataset(tmp_path: Path):
    module = _load_script_module()
    root = tmp_path / "stcast"
    _build_stcast_tree(
        root,
        start=datetime(2024, 1, 1, 0),
        end=datetime(2024, 12, 31, 18),
        step_hours=6,
    )
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    expected_count = 366 * 4
    (stats_dir / "mean_std.json").write_text(
        json.dumps({"count": expected_count, "current_date": "2024/2024-12-31T18:00:00.nc"}),
        encoding="utf-8",
    )
    (stats_dir / "mean_std_single.json").write_text(json.dumps({"count": expected_count}), encoding="utf-8")

    result = module.audit_stcast_dataset(
        root,
        year=2024,
        cadence_hours=6,
        stats_dir=stats_dir,
        stats_start=datetime(2024, 1, 1, 0),
        stats_end=datetime(2024, 12, 31, 18),
        stats_step_hours=6,
    )

    assert result.is_complete is True
    assert result.expected_hours == ["00:00:00", "06:00:00", "12:00:00", "18:00:00"]
    assert result.pressure_hour_count_distribution == {"4": 366}
    assert result.surface_hour_count_distribution == {"4": 366}
    assert result.stats_validation["valid"] is True


def test_audit_stcast_dataset_flags_missing_files_and_bad_stats(tmp_path: Path):
    module = _load_script_module()
    root = tmp_path / "stcast"
    _build_stcast_tree(
        root,
        start=datetime(2024, 1, 1, 0),
        end=datetime(2024, 12, 31, 18),
        step_hours=6,
    )
    missing_pressure = root / "2024" / "2024-01-01" / "06:00:00-q-925.0.npy"
    missing_surface = root / "single" / "2024" / "2024-01-02" / "12:00:00-msl.npy"
    missing_pressure.unlink()
    missing_surface.unlink()

    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    (stats_dir / "mean_std.json").write_text(json.dumps({"count": 10}), encoding="utf-8")
    (stats_dir / "mean_std_single.json").write_text(json.dumps({"count": 11}), encoding="utf-8")

    result = module.audit_stcast_dataset(
        root,
        year=2024,
        cadence_hours=6,
        stats_dir=stats_dir,
        stats_start=datetime(2024, 1, 1, 0),
        stats_end=datetime(2024, 12, 31, 18),
        stats_step_hours=6,
    )

    assert result.is_complete is False
    assert "2024-01-01T06:00:00" in result.missing_pressure_files
    assert result.missing_pressure_files["2024-01-01T06:00:00"] == ["q-925.0"]
    assert "2024-01-02T12:00:00" in result.missing_surface_files
    assert result.missing_surface_files["2024-01-02T12:00:00"] == ["msl"]
    assert result.stats_validation["valid"] is False
    assert result.stats_validation["reason"] == "count_mismatch"


def test_audit_script_compact_output_and_json_artifact(tmp_path: Path):
    module = _load_script_module()
    root = tmp_path / "stcast"
    _build_stcast_tree(
        root,
        start=datetime(2024, 1, 1, 0),
        end=datetime(2024, 1, 2, 18),
        step_hours=6,
    )
    for hour in ("01", "02", "03", "04", "05", "07", "08", "09", "10", "11", "13", "14", "15", "16", "17", "19", "20", "21", "22", "23"):
        for var_name in module.PRESSURE_VARIABLES:
            for level in module.PRESSURE_LEVELS:
                _write_npy(root / "2024" / "2024-01-01" / f"{hour}:00:00-{var_name}-{level}.npy")
        for var_name in module.SURFACE_VARIABLES:
            _write_npy(root / "single" / "2024" / "2024-01-01" / f"{hour}:00:00-{var_name}.npy")

    output_json = tmp_path / "audit.json"
    buffer = StringIO()
    with redirect_stdout(buffer):
        exit_code = module.main(
            [
                "--root-dir",
                str(root),
                "--year",
                "2024",
                "--cadence-hours",
                "6",
                "--compact",
                "--output-json",
                str(output_json),
            ]
        )

    assert exit_code == 1
    payload = json.loads(buffer.getvalue())
    full_payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["unexpected_pressure_hours"]["day_count"] == 1
    assert payload["unexpected_pressure_hours"]["sample_days"][0]["day"] == "2024-01-01"
    assert payload["unexpected_pressure_hours"]["sample_days"][0]["hour_count"] == 20
    assert "2024-01-01" in full_payload["unexpected_pressure_hours"]
    assert len(full_payload["unexpected_pressure_hours"]["2024-01-01"]) == 20
