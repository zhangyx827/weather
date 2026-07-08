"""Tests for forecast provider contracts."""

import unittest
from datetime import datetime, timezone

from mazu_saudi.forecast import GenCastForecastProvider, MockForecastProvider


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
        self.assertEqual(provider.member_count(field), 4)
        self.assertEqual(len(provider.ensemble_mean(field)), len(field.values))
        self.assertEqual(len(provider.ensemble_spread(field)), len(field.values))
        probs = provider.exceedance_probability(field, threshold=40.0, variable="temp_c")
        self.assertEqual(len(probs), len(field.values))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in probs))


if __name__ == "__main__":
    unittest.main()
