from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "convert_era5_to_stcast_npy.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("convert_era5_to_stcast_npy", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_years_and_months():
    module = _load_script_module()
    args = argparse.Namespace(years=[2022, 2021], year=[2024], months=[12, 1, 12])

    assert module._parse_years(args) == [2021, 2022, 2024]
    assert module._parse_months(args.months) == [1, 12]


def test_build_stats_sources_cover_each_month():
    module = _load_script_module()
    sources = module._build_stats_sources(Path("data/processed/stcast_global"), [2021, 2022], [1, 12])

    assert len(sources) == 4
    first = sources[0]
    last = sources[-1]
    assert first["root_dir"] == Path("data/processed/stcast_global")
    assert first["train_start"] == datetime(2021, 1, 1, 0, 0)
    assert first["train_end"] == datetime(2021, 1, 31, 18, 0)
    assert last["train_start"] == datetime(2022, 12, 1, 0, 0)
    assert last["train_end"] == datetime(2022, 12, 31, 18, 0)


def test_convert_years_iterates_selected_years_and_months(monkeypatch):
    module = _load_script_module()
    calls: list[tuple[Path, int, int]] = []

    def fake_convert_month(**kwargs):
        calls.append((kwargs["surface_file"], kwargs["year"], kwargs["month"]))
        return 1

    monkeypatch.setattr(module, "_convert_month", fake_convert_month)

    total = module.convert_years(
        surface_dir=Path("/input/surface"),
        pressure_dir=Path("/input/pressure"),
        missing_pressure_dir=Path("/input/missing"),
        output_dir=Path("/output"),
        years=[2022, 2021],
        months=[1, 3],
        surface_template="surface_{year}_{month02}.nc",
        pressure_template="pressure_{year}_{month02}_{long_name}.nc",
        missing_pressure_template="missing_{year}_{month02}_{long_name}.nc",
    )

    assert total == 4
    assert calls == [
        (Path("/input/surface/surface_2022_01.nc"), 2022, 1),
        (Path("/input/surface/surface_2022_03.nc"), 2022, 3),
        (Path("/input/surface/surface_2021_01.nc"), 2021, 1),
        (Path("/input/surface/surface_2021_03.nc"), 2021, 3),
    ]


def test_build_joint_stats_uses_generated_month_windows(monkeypatch):
    module = _load_script_module()
    captured: dict[str, object] = {}
    fake_stats = types.ModuleType("build_saudi_stcast_stats")

    def fake_build_stats(sources, stats_dir, step_hours):
        captured["sources"] = sources
        captured["stats_dir"] = stats_dir
        captured["step_hours"] = step_hours

    fake_stats.build_stats = fake_build_stats
    monkeypatch.setitem(sys.modules, "build_saudi_stcast_stats", fake_stats)

    module.build_joint_stats(
        output_dir=Path("data/processed/stcast_global"),
        stats_dir=Path("data/processed/stcast_global_stats"),
        years=[2021],
        months=[1, 2],
        step_hours=6,
    )

    assert captured["stats_dir"] == Path("data/processed/stcast_global_stats")
    assert captured["step_hours"] == 6
    assert len(captured["sources"]) == 2
    assert captured["sources"][0]["train_start"] == datetime(2021, 1, 1, 0, 0)
    assert captured["sources"][0]["train_end"] == datetime(2021, 1, 31, 18, 0)
    assert captured["sources"][1]["train_start"] == datetime(2021, 2, 1, 0, 0)
    assert captured["sources"][1]["train_end"] == datetime(2021, 2, 28, 18, 0)


def test_convert_month_selects_single_time_slice_with_duplicate_valid_time(monkeypatch, tmp_path):
    module = _load_script_module()
    monkeypatch.setattr(module, "PRESSURE_LEVELS", [1000.0])
    monkeypatch.setattr(module, "PRESSURE_VARIABLES", {"t": "temperature"})
    monkeypatch.setattr(module, "SURFACE_VARIABLES", ("t2m",))

    valid_time = np.array(["2024-01-01T00:00:00", "2024-01-01T00:00:00"], dtype="datetime64[ns]")
    pressure_data = xr.DataArray(
        np.array(
            [
                [
                    [[[1.0, 1.0], [1.0, 1.0]]],
                    [[[9.0, 9.0], [9.0, 9.0]]],
                ],
            ],
            dtype=np.float32,
        ).reshape(1, 2, 1, 2, 2),
        dims=("valid_time", "time", "pressure_level", "latitude", "longitude"),
        coords={
            "valid_time": valid_time[:1],
            "time": np.array([0, 1], dtype=np.int64),
            "pressure_level": np.array([1000.0], dtype=np.float32),
            "latitude": np.array([24.0, 25.0], dtype=np.float32),
            "longitude": np.array([46.0, 47.0], dtype=np.float32),
        },
    )
    surface_ds = xr.Dataset(
        {
            "t2m": (("valid_time", "latitude", "longitude"), np.array([[[3.0, 4.0], [5.0, 6.0]]], dtype=np.float32)),
        },
        coords={
            "valid_time": np.array(["2024-01-01T00:00:00"], dtype="datetime64[ns]"),
            "latitude": np.array([24.0, 25.0], dtype=np.float32),
            "longitude": np.array([46.0, 47.0], dtype=np.float32),
        },
    )

    written: list[tuple[str, tuple[int, ...], np.ndarray]] = []

    def fake_open_surface_dataset(_surface_file):
        return nullcontext(surface_ds)

    def fake_open_pressure_data(*_args, **_kwargs):
        return pressure_data

    def fake_write_array(dst_path, array):
        written.append((str(dst_path), array.shape, array.copy()))

    monkeypatch.setattr(module, "_open_surface_dataset", fake_open_surface_dataset)
    monkeypatch.setattr(module, "_open_pressure_data", fake_open_pressure_data)
    monkeypatch.setattr(module, "_write_array", fake_write_array)

    count = module._convert_month(
        surface_file=Path("surface.nc"),
        pressure_dir=Path("pressure"),
        missing_pressure_dir=Path("missing"),
        year=2024,
        month=1,
        surface_template="surface_{year}_{month02}.nc",
        pressure_template="pressure_{year}_{month02}_{long_name}.nc",
        missing_pressure_template="missing_{year}_{month02}_{long_name}.nc",
        output_dir=tmp_path,
    )

    assert count == 1
    assert len(written) == 2
    assert all(shape == (2, 2) for _, shape, _ in written)
    pressure_write = next(item for item in written if item[0].endswith("t-1000.0.npy"))
    surface_write = next(item for item in written if item[0].endswith("t2m.npy"))
    np.testing.assert_array_equal(pressure_write[2], np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32))
    np.testing.assert_array_equal(surface_write[2], np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32))
