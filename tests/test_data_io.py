"""Tests for standard data I/O helpers."""

from __future__ import annotations

import sys
import tempfile
import zipfile
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
    import xarray as xr
except Exception:
    np = None
    xr = None

from mazu_saudi.data import (
    check_missing_values,
    compute_daily_precipitation_statistics,
    crop_to_bbox,
    crop_to_saudi,
    derive_xarray_physical_indicators,
    generate_standard_grid,
    read_json_features,
    write_json_features,
)
from mazu_saudi.schemas import GridCell, MeteorologicalFeatures


class DataIOTests(unittest.TestCase):
    def test_json_feature_round_trip(self):
        features = MeteorologicalFeatures(
            grid=GridCell(id="riyadh", lat=24.7, lon=46.7),
            temp_c=42.0,
            rh_percent=20.0,
            wind_speed_mps=8.0,
            precip_1h_mm=0.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.json"
            write_json_features(path, features)
            loaded = read_json_features(path)
        self.assertEqual(loaded.grid.id, "riyadh")
        self.assertEqual(loaded.temp_c, 42.0)

    def test_saudi_bbox_crop(self):
        cells = [
            GridCell(id="inside", lat=24.0, lon=46.0),
            GridCell(id="outside", lat=10.0, lon=46.0),
        ]
        cropped = crop_to_saudi(cells)
        self.assertEqual([cell.id for cell in cropped], ["inside"])

    def test_standard_grid_generation(self):
        grid = generate_standard_grid(bbox=(16.0, 34.0, 16.1, 34.2), resolution_deg=0.1)
        self.assertEqual(len(grid), 6)
        self.assertEqual(grid[0].lat, 16.0)
        self.assertEqual(grid[-1].lon, 34.2)

    def test_missing_value_check(self):
        features = MeteorologicalFeatures(grid=GridCell(id="x", lat=24.0, lon=46.0), temp_c=None, rh_percent=10.0)
        report = check_missing_values(features, required_fields=["temp_c", "rh_percent"])
        self.assertFalse(report["ok"])
        self.assertEqual(report["missing"][0]["field"], "temp_c")

    def test_netcdf_reader_prefers_netcdf4_engine(self):
        calls: list[str | None] = []

        def fake_open_dataset(path, engine=None):
            calls.append(engine)
            if engine == "netcdf4":
                raise OSError("netcdf4 unavailable")
            if engine == "h5netcdf":
                return {"path": str(path), "engine": engine}
            raise OSError(f"engine unavailable: {engine}")

        fake_xarray = types.SimpleNamespace(open_dataset=fake_open_dataset)

        with mock.patch.dict(sys.modules, {"xarray": fake_xarray}):
            from mazu_saudi.data import read_netcdf_dataset as patched_read_netcdf_dataset

            dataset = patched_read_netcdf_dataset(Path("sample.nc"))

        self.assertEqual(dataset["engine"], "h5netcdf")
        self.assertEqual(calls, ["netcdf4", "h5netcdf"])

    def test_netcdf_reader_opens_zipped_cds_downloads(self):
        calls: list[Path] = []

        class FakeDataset:
            def __init__(self, path: Path) -> None:
                self.path = path

            def load(self):
                return {"path": str(self.path)}

            def close(self):
                return None

        def fake_open_dataset(path, engine=None):
            calls.append(Path(path))
            return FakeDataset(Path(path))

        def fake_merge(datasets, compat=None):
            return {"merged": [dataset["path"] for dataset in datasets], "compat": compat}

        fake_xarray = types.SimpleNamespace(open_dataset=fake_open_dataset, merge=fake_merge)

        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "download.nc"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("data_stream-oper_stepType-instant.nc", b"instant")
                archive.writestr("data_stream-oper_stepType-accum.nc", b"accum")

            with mock.patch.dict(sys.modules, {"xarray": fake_xarray}):
                from mazu_saudi.data import read_netcdf_dataset as patched_read_netcdf_dataset

                dataset = patched_read_netcdf_dataset(archive_path)

        self.assertEqual(
            [path.name for path in calls],
            ["data_stream-oper_stepType-instant.nc", "data_stream-oper_stepType-accum.nc"],
        )
        self.assertEqual(
            [Path(item).name for item in dataset["merged"]],
            ["data_stream-oper_stepType-instant.nc", "data_stream-oper_stepType-accum.nc"],
        )
        self.assertEqual(dataset["compat"], "override")

    def test_xarray_bbox_crop_handles_descending_latitudes(self):
        if xr is None or np is None:
            self.skipTest("xarray/numpy optional dependencies are not installed")
        ds = xr.Dataset(
            data_vars={
                "precipitation": (("time", "lat", "lon"), np.arange(12, dtype=np.float32).reshape(1, 3, 4)),
            },
            coords={
                "time": np.array(["2025-01-01"], dtype="datetime64[ns]"),
                "lat": np.array([32.0, 31.9, 31.8], dtype=np.float32),
                "lon": np.array([34.0, 34.1, 34.2, 34.3], dtype=np.float32),
            },
        )

        cropped = crop_to_bbox(ds, (31.8, 34.05, 32.0, 34.25))
        self.assertEqual(cropped.sizes["lat"], 3)
        self.assertEqual(cropped.sizes["lon"], 2)

    def test_daily_precipitation_statistics(self):
        if xr is None or np is None:
            self.skipTest("xarray/numpy optional dependencies are not installed")
        ds = xr.Dataset(
            data_vars={
                "precipitation": (
                    ("time", "lat", "lon"),
                    np.array([[[10.0]], [[20.0]], [[30.0]], [[25.0]]], dtype=np.float32),
                ),
            },
            coords={
                "time": np.array(
                    ["2025-01-01T00:00", "2025-01-01T12:00", "2025-01-02T00:00", "2025-01-02T12:00"],
                    dtype="datetime64[ns]",
                ),
                "lat": np.array([24.0], dtype=np.float32),
                "lon": np.array([46.0], dtype=np.float32),
            },
        )

        stats = compute_daily_precipitation_statistics(ds)

        self.assertEqual(stats.sizes["time"], 2)
        self.assertEqual(stats["daily_precip_total_mm"].values[:, 0, 0].tolist(), [30.0, 55.0])
        self.assertEqual(stats["daily_precip_extreme_flag"].values[:, 0, 0].tolist(), [1.0, 2.0])

    def test_derive_xarray_physical_indicators(self):
        if xr is None or np is None:
            self.skipTest("xarray/numpy optional dependencies are not installed")
        ds = xr.Dataset(
            data_vars={
                "temp_c": (("time",), np.array([35.0], dtype=np.float32)),
                "dewpoint_c": (("time",), np.array([20.0], dtype=np.float32)),
                "u_wind_mps": (("time",), np.array([3.0], dtype=np.float32)),
                "v_wind_mps": (("time",), np.array([4.0], dtype=np.float32)),
            },
            coords={"time": np.array(["2025-01-01T00:00"], dtype="datetime64[ns]")},
        )

        indicators = derive_xarray_physical_indicators(ds)

        self.assertIn("rh_percent", indicators)
        self.assertIn("vpd_kpa", indicators)
        self.assertIn("heat_index_c", indicators)
        self.assertIn("wind_speed_mps", indicators)
        self.assertAlmostEqual(float(indicators["wind_speed_mps"].values[0]), 5.0)


if __name__ == "__main__":
    unittest.main()
