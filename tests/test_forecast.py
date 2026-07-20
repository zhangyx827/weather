"""Tests for forecast provider contracts."""

import unittest
from datetime import datetime, timezone
from pathlib import Path
import json
import tempfile

import numpy as np
import xarray as xr

from mazu_saudi.forecast import ERA5MSWEPForecastProvider, GenCastForecastProvider, MockForecastProvider


def _find_real_era5_mswep_tree() -> tuple[int, Path, Path, Path] | None:
    precip = Path("data/raw/precip")
    if not precip.exists():
        return None
    candidates = (
        (2024, Path("era5_single_levels_2024"), Path("era5_pressure_levels_2024")),
        (2022, Path("era5_single_levels_2022_6h"), Path("era5_pressure_levels_2022_6h")),
        (2021, Path("era5_single_levels_2021_6h"), Path("era5_pressure_levels_2021_6h")),
        (2019, Path("era5_single_levels_2019_6h"), Path("era5_pressure_levels_2019_6h")),
        (2015, Path("era5_single_levels_2015_6h"), Path("era5_pressure_levels_2015_6h")),
        (2009, Path("era5_single_levels_2009_6h"), Path("era5_pressure_levels_2009_6h")),
    )
    for year, single, pressure in candidates:
        if not single.exists() or not pressure.exists():
            continue
        if not any(single.glob(f"era5_single_levels_{year}_01*.nc")):
            continue
        if not all((pressure / f"era5_pl_{year}_01_{name}.nc").exists() for name in ("specific_humidity", "u_component_of_wind", "v_component_of_wind")):
            continue
        if not (
            (precip / f"{year}001.nc").exists()
            or (precip / f"GPM_3IMERGDF_{year}0101.nc4").exists()
        ):
            continue
        return year, single, pressure, precip
    return None


REAL_ERA5_MSWEP_TREE = _find_real_era5_mswep_tree()


def _write_fake_heat_climatology_tree(root: Path) -> Path:
    climatology_dir = root / "heat_climatology"
    climatology_dir.mkdir(parents=True, exist_ok=True)
    times = np.array(["2025-07-11T00:00:00"], dtype="datetime64[ns]")
    lats = np.array([24.7, 25.7], dtype=np.float32)
    lons = np.array([46.7, 47.7], dtype=np.float32)
    for index, (temp_c, tmax_c) in enumerate(((28.0, 33.0), (29.0, 34.0)), start=1):
        dataset = xr.Dataset(
            {
                "t2m_c": (("time", "latitude", "longitude"), np.full((1, len(lats), len(lons)), temp_c, dtype=np.float32)),
                "tmax_c": (("time", "latitude", "longitude"), np.full((1, len(lats), len(lons)), tmax_c, dtype=np.float32)),
            },
            coords={"time": times, "latitude": lats, "longitude": lons},
        )
        dataset.to_netcdf(climatology_dir / f"saudi_indicators_2024010{index}.nc")
    return climatology_dir


class ForecastTests(unittest.TestCase):
    def test_era5_mswep_precip_handles_swapped_spatial_coords(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            precip_path = Path(temp_dir) / "GPM_3IMERGDF_20240101.nc4"
            dataset = xr.Dataset(
                data_vars={
                    "precipitation": (
                        ("time", "lon", "lat"),
                        np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
                    ),
                },
                coords={
                    "time": [np.datetime64("2024-01-01T00:00:00")],
                    "lon": np.array([24.0, 25.0], dtype=np.float32),
                    "lat": np.array([45.0, 46.0], dtype=np.float32),
                },
            )
            dataset.to_netcdf(precip_path)
            dataset.close()

            provider = ERA5MSWEPForecastProvider(precip_dir=temp_dir)
            try:
                valid = datetime(2024, 1, 1, tzinfo=timezone.utc)
                precip = provider.fetch("precip_24h_mm", valid_time=valid)
            finally:
                provider.close()

        self.assertEqual(len(precip.values), 4)
        self.assertEqual(sorted({cell.lat for cell in precip.grid}), [24.0, 25.0])
        self.assertEqual(sorted({cell.lon for cell in precip.grid}), [45.0, 46.0])
        self.assertEqual(sorted(precip.values), [1.0, 2.0, 3.0, 4.0])

    def test_get_forecast_uses_lead_hours(self):
        provider = MockForecastProvider()
        issue = datetime(2026, 7, 8, tzinfo=timezone.utc)
        fields = provider.get_forecast(issue, [0, 6], variables=["temp_c"])
        self.assertIn("temp_c:+0h", fields)
        self.assertIn("temp_c:+6h", fields)

    def test_gencast_ensemble_statistics(self):
        provider = GenCastForecastProvider()
        field = provider.fetch("temp_c")
        self.assertEqual(field.provider_role, "secondary_ensemble")
        self.assertEqual(field.provider_status, "degraded_mock")
        self.assertEqual(provider.member_count(field), 4)
        self.assertEqual(len(provider.ensemble_mean(field)), len(field.values))
        self.assertEqual(len(provider.ensemble_spread(field)), len(field.values))
        probs = provider.exceedance_probability(field, threshold=40.0, variable="temp_c")
        self.assertEqual(len(probs), len(field.values))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in probs))
        self.assertIn("ensemble_stats", field.metadata)

    def test_mock_provider_marks_degraded_runtime(self):
        provider = MockForecastProvider()
        field = provider.fetch("temp_c")
        self.assertEqual(field.provider_status, "degraded_mock")
        self.assertEqual(field.source_status, "degraded")
        self.assertEqual(field.degradation_metadata["reason"], "mock_provider_used")

    @unittest.skipUnless(REAL_ERA5_MSWEP_TREE is not None, "real forecast data not available")
    def test_era5_mswep_provider_reads_real_fields(self):
        assert REAL_ERA5_MSWEP_TREE is not None
        year, single_dir, pressure_dir, precip_dir = REAL_ERA5_MSWEP_TREE
        with tempfile.TemporaryDirectory() as tmp:
            heat_climatology_dir = _write_fake_heat_climatology_tree(Path(tmp))
            provider = ERA5MSWEPForecastProvider(
                era5_dir=single_dir,
                precip_dir=precip_dir,
                pressure_dir=pressure_dir,
                heat_climatology_dir=heat_climatology_dir,
            )
            try:
                valid = datetime(year, 1, 1, tzinfo=timezone.utc)

                temp = provider.fetch("temp_c", valid_time=valid)
                tmax = provider.fetch("tmax_c", valid_time=valid)
                tmin = provider.fetch("tmin_c", valid_time=valid)
                precip = provider.fetch("precip_24h_mm", valid_time=valid)
                pwat = provider.fetch("pwat", valid_time=valid)
                fields = provider.get_forecast(valid, 0)
                dataset = provider.forecast_dataset(valid, 0)

                self.assertEqual(temp.provider, "era5_mswep")
                self.assertEqual(temp.units, "degC")
                self.assertTrue(temp.values)
                self.assertEqual(tmax.units, "degC")
                self.assertEqual(tmin.units, "degC")
                self.assertTrue(tmax.values)
                self.assertTrue(tmin.values)
                self.assertEqual(precip.metadata["source"], "MSWEP")
                self.assertEqual(precip.units, "mm")
                self.assertEqual(pwat.metadata["source"], "ERA5 pressure levels")
                self.assertTrue(pwat.values)
                self.assertIn("temp_c:+0h", fields)
                self.assertIn("tmax_c:+0h", fields)
                self.assertIn("tmin_c:+0h", fields)
                self.assertIn("rh_percent:+0h", fields)
                self.assertIn("wind_speed_mps:+0h", fields)
                self.assertIn("precip_1h_mm:+0h", fields)
                self.assertIn("tmax_c", dataset.data_vars)
                self.assertIn("tmin_c", dataset.data_vars)
                self.assertIn("daily_precip_total", dataset.data_vars)
                self.assertIn("daily_convective_precip", dataset.data_vars)
                self.assertIn("daily_large_scale_precip", dataset.data_vars)
                source_metadata = json.loads(dataset.attrs["source_metadata_json"])
                flood_metadata = source_metadata["grounding_gap"]["flash_flood_features"]
                self.assertEqual(dataset.attrs["source_status"], "normal")
                self.assertEqual(json.loads(dataset.attrs["degradation_metadata_json"]), {})
                self.assertEqual(flood_metadata["feature_status"]["pwat"]["status"], "direct")
                self.assertEqual(flood_metadata["feature_status"]["ivt"]["status"], "direct")
                self.assertEqual(flood_metadata["feature_status"]["wind850_speed"]["status"], "direct")
                self.assertEqual(flood_metadata["feature_status"]["wind_shear_850_200"]["status"], "direct")
                self.assertEqual(flood_metadata["feature_status"]["daily_convective_precip"]["status"], "direct")
                self.assertEqual(flood_metadata["feature_status"]["daily_large_scale_precip"]["status"], "derived")
                self.assertEqual(flood_metadata["precipitation_partition"]["status"], "same_source_residual")
                self.assertEqual(flood_metadata["precipitation_partition"]["source_pair"], ["era5_mswep", "era5_mswep"])
                self.assertEqual(source_metadata["grounding_gap"]["heat_features"]["feature_status"]["t2m_anomaly_c"]["method"], "historical_indicator_archive_mean_fallback")
                self.assertEqual(source_metadata["grounding_gap"]["heat_features"]["feature_status"]["tmax_anomaly_c"]["method"], "historical_indicator_archive_mean_fallback")
            finally:
                provider.close()


if __name__ == "__main__":
    unittest.main()
