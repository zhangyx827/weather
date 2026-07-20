"""Tests for the demo workflow."""

import unittest
from datetime import datetime, timezone

from mazu_saudi.agent import run_demo_pipeline
from mazu_saudi.agent.workflow import (
    DataCheckNode,
    ForecastNode,
    HazardScanNode,
    IndicatorDeriveNode,
    KGReasoningNode,
    ModelInferenceNode,
    SaudiWarningPipeline,
)
from mazu_saudi.forecast.providers import BaseForecastProvider
from mazu_saudi.schemas import ForecastField, GridCell, MeteorologicalFeatures


class ForecastOnlyProvider(BaseForecastProvider):
    """Forecast fixture carrying provenance without an IndicatorFieldSet."""

    name = "forecast_fixture"

    def fetch(self, variable, valid_time=None, bbox=None):
        del bbox
        valid = valid_time or datetime(2026, 7, 20, tzinfo=timezone.utc)
        grid = [GridCell(id="riyadh", lat=24.7, lon=46.7, region="Riyadh")]
        metadata = {
            "source_metadata": {
                "resolved_sources": {
                    "temperature": {"dataset_id": "forecast-primary", "role": "primary"}
                },
                "source_status": "degraded",
            },
            "grounding_gap": {
                "forecast_temperature": {
                    "source_pair": ["forecast-primary", "forecast-fallback"],
                    "units": "degC",
                    "absolute_difference": 2.5,
                    "status": "fallback",
                }
            },
        }
        return ForecastField(
            provider=self.name,
            variable=variable,
            units="unknown",
            valid_time=valid,
            values=[1.0],
            grid=grid,
            metadata=metadata,
            provider_role="forecast_model",
            provider_status="fallback",
            source_status="degraded",
            degradation_metadata={"reason": "forecast_fixture_fallback", "promoted_source": "forecast-fallback"},
        )


class PipelineTests(unittest.TestCase):
    def test_demo_pipeline_runs(self):
        result = run_demo_pipeline()
        self.assertEqual(len(result["risks"]), 5)
        self.assertEqual(len(result["warning_product"]["briefings"]), 6)
        self.assertIn("kg_reasoning", result["trace"])
        self.assertIn("pipeline_trace", result)
        self.assertTrue(all("duration_ms" in item for item in result["pipeline_trace"]))
        self.assertGreater(result["kg_explanation"]["triple_count"], 0)
        self.assertIn("semantic_evidence", result["kg_explanation"])
        self.assertIn("flash_flood", result["kg_explanation"]["semantic_evidence"])
        self.assertEqual(
            result["kg_explanation"]["semantic_evidence"]["flash_flood"]["disaster_type_uri"],
            "https://sakuna.ph/FlashFlood",
        )

    def test_forecast_only_provenance_reaches_kg_grounding_records(self):
        features = MeteorologicalFeatures(
            grid=GridCell(id="riyadh", lat=24.7, lon=46.7, region="Riyadh"),
            valid_time=datetime(2026, 7, 20, tzinfo=timezone.utc),
            temp_c=45.0,
            rh_percent=20.0,
            precip_1h_mm=8.0,
            precip_24h_mm=30.0,
            wind_speed_mps=10.0,
            pressure_hpa=1008.0,
        )
        context = SaudiWarningPipeline(
            nodes=[
                DataCheckNode(),
                HazardScanNode(),
                ForecastNode(ForecastOnlyProvider()),
                IndicatorDeriveNode(),
                ModelInferenceNode(),
                KGReasoningNode(),
            ]
        ).run(features)

        self.assertFalse(context.get("errors"))
        confidence = context["forecast_confidence"]
        self.assertEqual(confidence["source_metadata"]["resolved_sources"]["temperature"]["dataset_id"], "forecast-primary")
        self.assertEqual(confidence["grounding_gap"]["forecast_temperature"]["source_pair"][1], "forecast-fallback")
        self.assertTrue(all(context["kg_explanation"]["grounding_gap_uris"][risk.hazard_type] for risk in context["risks"]))
        ttl = context["kg"].export_instances()
        self.assertIn("forecast-primary", ttl)
        self.assertIn("forecast-fallback", ttl)
        self.assertIn("forecast_fixture_fallback", ttl)
        provenance = context["kg_explanation"]["runtime_provenance_uri"]
        self.assertIsNotNone(provenance)
        self.assertTrue(
            any(
                triple.subject == provenance and "forecast-primary" in triple.object
                for triple in context["kg"].triples
                if triple.predicate.endswith("provenancePayload")
            )
        )


if __name__ == "__main__":
    unittest.main()
