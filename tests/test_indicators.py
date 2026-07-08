"""Tests for physical indicators."""

import math
import unittest

try:
    import numpy as np
except Exception:
    np = None

from mazu_saudi.indicators import (
    compute_dewpoint_depression,
    compute_extreme_precip_flags,
    compute_heat_index_c,
    compute_precip_anomaly,
    compute_relative_humidity_from_dewpoint,
    compute_vpd_kpa,
    compute_wind_shear,
)


class IndicatorTests(unittest.TestCase):
    def test_vpd_calculation(self):
        vpd = compute_vpd_kpa(30.0, 50.0)
        self.assertTrue(math.isclose(vpd, 2.12, rel_tol=0.03))

    def test_vpd_missing_returns_nan(self):
        self.assertTrue(math.isnan(compute_vpd_kpa(None, 50.0)))

    def test_heat_index_calculation(self):
        hi = compute_heat_index_c(32.0, 70.0)
        self.assertGreater(hi, 38.0)
        self.assertLess(hi, 43.0)

    def test_heat_index_cool_returns_temperature(self):
        self.assertEqual(compute_heat_index_c(20.0, 80.0), 20.0)

    def test_vpd_boundaries(self):
        self.assertGreater(compute_vpd_kpa(35.0, 0.0), 5.0)
        self.assertEqual(compute_vpd_kpa(35.0, 100.0), 0.0)

    def test_high_temperature_heat_index(self):
        self.assertGreater(compute_heat_index_c(50.0, 40.0), 60.0)

    def test_numpy_array_inputs(self):
        if np is None:
            self.skipTest("numpy optional dependency is not installed")
        values = compute_dewpoint_depression(np.array([35.0, 40.0]), np.array([25.0, 30.0]))
        self.assertEqual(values.tolist(), [10.0, 10.0])

    def test_nan_propagation(self):
        self.assertTrue(math.isnan(compute_relative_humidity_from_dewpoint(float("nan"), 20.0)))

    def test_new_precip_and_shear_indicators(self):
        self.assertEqual(compute_precip_anomaly(30.0, 10.0), 20.0)
        self.assertEqual(compute_extreme_precip_flags(60.0), 2.0)
        self.assertEqual(compute_wind_shear(20.0, 8.0), 12.0)


if __name__ == "__main__":
    unittest.main()
