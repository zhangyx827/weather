"""Tests for forecast provider contracts."""

import unittest
from datetime import datetime, timezone
from pathlib import Path

from mazu_saudi.forecast import ERA5MSWEPForecastProvider, GenCastForecastProvider, MockForecastProvider


class ForecastTests(unittest.TestCase):
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

    @unittest.skipUnless(Path("era5_single_levels_2025").exists() and Path("precip").exists(), "real forecast data not available")
    def test_era5_mswep_provider_reads_real_fields(self):
        provider = ERA5MSWEPForecastProvider()
        try:
            valid = datetime(2025, 1, 1, tzinfo=timezone.utc)
            bbox = (24.5, 46.5, 24.9, 46.9)

            temp = provider.fetch("temp_c", valid_time=valid, bbox=bbox)
            precip = provider.fetch("precip_24h_mm", valid_time=valid, bbox=bbox)
            fields = provider.get_forecast(valid, 0, bbox=bbox)

            self.assertEqual(temp.provider, "era5_mswep")
            self.assertEqual(temp.units, "degC")
            self.assertTrue(temp.values)
            self.assertEqual(precip.metadata["source"], "MSWEP")
            self.assertEqual(precip.units, "mm")
            self.assertTrue(precip.values)
            self.assertIn("temp_c:+0h", fields)
            self.assertIn("rh_percent:+0h", fields)
            self.assertIn("wind_speed_mps:+0h", fields)
            self.assertIn("precip_1h_mm:+0h", fields)
        finally:
            provider.close()


if __name__ == "__main__":
    unittest.main()
