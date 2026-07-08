"""Tests for standard data I/O helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mazu_saudi.data import check_missing_values, crop_to_saudi, generate_standard_grid, read_json_features, write_json_features
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


if __name__ == "__main__":
    unittest.main()
