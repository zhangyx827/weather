"""Tests for risk models and levels."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mazu_saudi.risk import MLBackedRiskModel, all_default_models, probability_to_level
from mazu_saudi.risk.ml import LightGBMAdapter
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
        self.assertEqual(probability_to_level(0.25), RiskLevel.MEDIUM)
        self.assertEqual(probability_to_level(0.5), RiskLevel.HIGH)
        self.assertEqual(probability_to_level(0.75), RiskLevel.EXTREME)

    def test_all_models_output_range(self):
        features = sample_features()
        risks = [model.predict(features) for model in all_default_models()]
        self.assertEqual(len(risks), 5)
        for risk in risks:
            self.assertGreaterEqual(risk.risk_probability, 0.0)
            self.assertLessEqual(risk.risk_probability, 1.0)
            self.assertIn(risk.risk_level, list(RiskLevel))
            self.assertTrue(risk.contributing_factors)
            self.assertTrue(risk.model_version)
            self.assertTrue(risk.model_family)
            self.assertTrue(risk.inference_mode)

    def test_batch_prediction_and_explain(self):
        model = all_default_models()[0]
        features = sample_features()
        self.assertEqual(len(model.predict_batch([features, features])), 2)
        explanation = model.explain(features)
        self.assertIn("contributing_factors", explanation)
        self.assertTrue(explanation["model_version"])

    def test_ml_fallback_interface(self):
        model = MLBackedRiskModel()
        self.assertEqual(model.train([sample_features()])["status"], "trained_stub")
        self.assertEqual(model.predict_proba(sample_features()), 0.0)
        self.assertFalse(model.shap_explain(sample_features())["available"])

    def test_lightgbm_wrappers_degrade_to_rule_when_model_unavailable(self):
        features = sample_features()
        risks = [model.predict(features) for model in all_default_models() if model.hazard_type in {"flash_flood", "extreme_heat", "dry_heat_agriculture"}]
        self.assertEqual(len(risks), 3)
        for risk in risks:
            self.assertEqual(risk.model_family, "lightgbm")
            self.assertIn(risk.inference_mode, {"degraded_rule_fallback", "rule"})
            self.assertIn("degradation_metadata", risk.evidence)

    def test_lightgbm_adapter_train_save_load_roundtrip(self):
        adapter = LightGBMAdapter()
        rng = np.random.default_rng(42)
        features = rng.normal(size=(48, 3)).astype(np.float32)
        labels = (features[:, 0] + 0.5 * features[:, 1] > 0.0).astype(np.float32)

        summary = adapter.train(
            {
                "features": features,
                "labels": labels,
                "feature_names": ["temp_c", "rh2m", "wind10_speed"],
            }
        )

        self.assertEqual(summary["status"], "trained")
        self.assertEqual(summary["backend"], "lightgbm")
        self.assertEqual(summary["feature_names"], ["temp_c", "rh2m", "wind10_speed"])
        self.assertEqual(summary["objective"], "binary")
        self.assertEqual(summary["metric"], "binary_logloss")
        self.assertGreater(summary["train_rows"], 0)
        self.assertGreater(summary["validation_rows"], 0)
        self.assertIsInstance(summary["validation_metric"], float)
        prediction = adapter.predict_proba(features[0])
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)
        shap = adapter.shap_explain(features[0])
        self.assertTrue(shap["available"])
        self.assertEqual(shap["backend"], "lightgbm")
        self.assertEqual(set(shap["values"].keys()), {"temp_c", "rh2m", "wind10_speed"})

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "flash_flood.txt"
            adapter.save_model(model_path)
            self.assertTrue(model_path.exists())
            self.assertTrue(Path(f"{model_path}.metadata.json").exists())

            reloaded = LightGBMAdapter().load_model(model_path)
            self.assertTrue(reloaded.trained)
            self.assertEqual(reloaded.metadata["feature_names"], ["temp_c", "rh2m", "wind10_speed"])
            self.assertEqual(reloaded.metadata["objective"], "binary")
            self.assertEqual(reloaded.metadata["metric"], "binary_logloss")
            reloaded_prediction = reloaded.predict_proba(features[0])
            self.assertGreaterEqual(reloaded_prediction, 0.0)
            self.assertLessEqual(reloaded_prediction, 1.0)


if __name__ == "__main__":
    unittest.main()
