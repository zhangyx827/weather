"""Tests for risk models and levels."""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np

from mazu_saudi.risk import FlashFloodRiskModel, MLBackedRiskModel, all_default_models, probability_to_level
from mazu_saudi.risk.ml import LightGBMAdapter
from mazu_saudi.risk import model_paths
from mazu_saudi.schemas import GridCell, IndicatorFieldSet, MeteorologicalFeatures, RiskLevel


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
            self.assertIn(risk.inference_mode, {"degraded_rule_fallback", "rule", "lightgbm"})
            self.assertIn("degradation_metadata", risk.evidence)

    def test_flash_flood_runtime_model_path_falls_back_to_verified_chain_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            verified_model = repo_root / "data" / "processed" / "models" / "flash_flood_province_day_verified_chain_baseline_quick" / "flash_flood.txt"
            verified_model.parent.mkdir(parents=True, exist_ok=True)
            verified_model.write_text("stub", encoding="utf-8")
            with mock.patch.object(model_paths, "REPO_ROOT", repo_root), mock.patch.object(
                model_paths,
                "DEFAULT_LAYER4_MODEL_DIR",
                repo_root / "models" / "layer4",
            ):
                resolved = model_paths.resolve_layer4_model_path(
                    "flash_flood",
                    default_name="flash_flood.txt",
                    allow_missing=True,
                )
        self.assertEqual(resolved, verified_model)

    def test_lightgbm_hybrid_success_path_reports_lightgbm_inference_mode(self):
        class FakeAdapter:
            metadata = {"feature_names": ["daily_precip_total", "daily_convective_precip", "daily_large_scale_precip", "cape", "pwat", "ivt", "wind850_speed", "wind_shear_850_200", "flash_flood_risk"]}

            def predict_proba(self, features):
                return 0.8

            def shap_explain(self, features):
                return {"values": {"daily_precip_total": 0.4, "cape": 0.2}}

        model = next(item for item in all_default_models() if item.hazard_type == "flash_flood")
        model.adapter = FakeAdapter()
        features = IndicatorFieldSet(
            grid=GridCell(id="ff", lat=24.7, lon=46.7, region="Riyadh"),
            valid_time=datetime(2026, 7, 19, tzinfo=timezone.utc),
            values={
                "daily_precip_total": 35.0,
                "daily_convective_precip": 12.0,
                "daily_large_scale_precip": 8.0,
                "cape": 900.0,
                "pwat": 32.0,
                "ivt": 280.0,
                "wind850_speed": 14.0,
                "wind_shear_850_200": 20.0,
                "flash_flood_risk": 2.5,
                "ds10_max_1h": 18.0,
                "ds10_max_6h": 42.0,
                "slope_deg": 5.0,
                "soil_moisture_frac": 0.2,
                "impervious_frac": 0.15,
            },
        )

        risk = model.predict(features)

        self.assertEqual(risk.inference_mode, "lightgbm")
        self.assertEqual(risk.model_family, "lightgbm")
        self.assertEqual(risk.degradation_metadata, {})
        self.assertEqual(risk.evidence["inference_mode"], "lightgbm")

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

    def test_lightgbm_adapter_supports_grouped_validation_split(self):
        adapter = LightGBMAdapter()
        rng = np.random.default_rng(7)
        features = rng.normal(size=(24, 3)).astype(np.float32)
        labels = np.array(([1.0] * 6) + ([0.0] * 6) + ([1.0] * 6) + ([0.0] * 6), dtype=np.float32)
        split_groups = np.array(
            (["event_a"] * 6) + (["event_b"] * 6) + (["date:2025-01-03"] * 6) + (["date:2025-01-04"] * 6),
            dtype=object,
        )

        summary = adapter.train(
            {
                "features": features,
                "labels": labels,
                "feature_names": ["temp_c", "rh2m", "wind10_speed"],
                "split_groups": split_groups,
            },
            validation_fraction=0.25,
            seed=11,
        )

        self.assertEqual(summary["split_strategy"], "group_shuffle")
        self.assertEqual(summary["split_group_count"], 4)
        self.assertGreaterEqual(summary["validation_group_count"], 1)
        self.assertGreater(summary["train_rows"], 0)
        self.assertGreater(summary["validation_rows"], 0)


if __name__ == "__main__":
    unittest.main()
