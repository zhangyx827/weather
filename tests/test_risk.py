"""Tests for risk models and levels."""

import unittest

from mazu_saudi.risk import all_default_models, probability_to_level
from mazu_saudi.schemas import GridCell, MeteorologicalFeatures, RiskLevel


def sample_features():
    return MeteorologicalFeatures(
        grid=GridCell(id="test", lat=24.7, lon=46.7, region="Riyadh"),
        temp_c=44.0,
        rh_percent=35.0,
        precip_1h_mm=20.0,
        precip_6h_mm=50.0,
        precip_24h_mm=70.0,
        wind_speed_mps=14.0,
        wind_gust_mps=20.0,
        soil_moisture_frac=0.1,
        slope_deg=10.0,
        impervious_frac=0.3,
        vegetation_index=0.1,
        visibility_km=5.0,
        coastal_distance_km=30.0,
    )


class RiskModelTests(unittest.TestCase):
    def test_probability_to_level(self):
        self.assertEqual(probability_to_level(0.1), RiskLevel.LOW)
        self.assertEqual(probability_to_level(0.4), RiskLevel.MEDIUM)
        self.assertEqual(probability_to_level(0.7), RiskLevel.HIGH)
        self.assertEqual(probability_to_level(0.9), RiskLevel.EXTREME)

    def test_all_models_output_range(self):
        features = sample_features()
        risks = [model.predict(features) for model in all_default_models()]
        self.assertEqual(len(risks), 5)
        for risk in risks:
            self.assertGreaterEqual(risk.risk_probability, 0.0)
            self.assertLessEqual(risk.risk_probability, 1.0)
            self.assertIn(risk.risk_level, list(RiskLevel))
            self.assertTrue(risk.contributing_factors)


if __name__ == "__main__":
    unittest.main()
