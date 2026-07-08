"""Tests for physical indicators."""

import math
import unittest

from mazu_saudi.indicators import compute_heat_index_c, compute_vpd_kpa


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


if __name__ == "__main__":
    unittest.main()
