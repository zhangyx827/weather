"""Tests for the knowledge graph loop."""

import unittest
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from mazu_saudi.kg import HazardKnowledgeGraph, validate_instance_ttl
from mazu_saudi.risk import ExtremeHeatRiskModel
from mazu_saudi.schemas import GridCell, HazardRisk, MeteorologicalFeatures, RiskLevel


class KnowledgeGraphTests(unittest.TestCase):
    def test_triple_generation_and_query(self):
        graph = HazardKnowledgeGraph()
        initial_count = len(graph.triples)
        risk = ExtremeHeatRiskModel().predict(
            MeteorologicalFeatures(
                grid=GridCell(id="jeddah", lat=21.4, lon=39.2),
                temp_c=45.0,
                rh_percent=60.0,
            )
        )
        graph.add_risk_evidence(risk)
        self.assertGreater(len(graph.triples), initial_count)
        impacts = graph.query_hazard_impacts("extreme_heat")
        self.assertIn("outdoor_workers", impacts["exposures"])
        self.assertIn("open_cooling_centers", impacts["actions"])

    def test_serialization_and_configured_queries(self):
        graph = HazardKnowledgeGraph()
        ttl = graph.to_ttl()
        self.assertIn("HazardScenario", ttl)
        actions = graph.query_response_actions_by_hazard("flash_flood")
        self.assertIn("activate_flood_watch", actions["actions"])
        products = graph.query_warning_products_by_department("meteorology")
        self.assertIn("flash_flood", products["hazards"])
        hazards = graph.query_hazards_by_exposure("ports")
        self.assertIn("coastal_humid_heat", hazards["hazards"])

    def test_sakunagraph_backend_is_used_for_disaster_type_mapping(self):
        graph = HazardKnowledgeGraph(backend="sakuna")
        self.assertEqual(graph.backend_metadata["backend"], "sakuna")
        self.assertGreater(graph.backend_metadata["ontology_triple_count"], 0)
        impacts = graph.query_hazard_impacts("flash_flood")
        self.assertEqual(impacts["disaster_type_uri"], "https://sakuna.ph/FlashFlood")

    def test_sakuna_instance_export_contains_risk_event_and_indicators(self):
        graph = HazardKnowledgeGraph(backend="sakuna")
        risk = HazardRisk(
            hazard_type="flash_flood",
            risk_probability=0.82,
            risk_level=RiskLevel.HIGH,
            contributing_factors=["pwat", "ivt"],
            grid=GridCell(id="riyadh", lat=24.7, lon=46.7, region="Riyadh"),
            valid_time=datetime(2026, 7, 19, tzinfo=timezone.utc),
            model_name="lightgbm_hybrid_v1",
            model_version="verified_chain_quick",
            indicator_evidence={"pwat": 48.2, "ivt": 310.0},
            model_family="lightgbm",
            inference_mode="lightgbm",
            shap_summary={"top_features": [{"feature": "pwat", "value": 0.31}]},
        )
        graph.add_risk_evidence(risk)
        ttl = graph.export_instances()
        self.assertIn("sakuna:DisasterEvent", ttl)
        self.assertIn("sakuna:hasDisasterType sakuna:FlashFlood", ttl)
        self.assertIn("baw:hasClimateParameterMeasurement", ttl)
        evidence = graph.semantic_evidence_for_risk(risk)
        self.assertEqual(evidence["model"]["family"], "lightgbm")
        self.assertEqual(evidence["disaster_type_uri"], "https://sakuna.ph/FlashFlood")

    def test_graphdb_push_skips_without_endpoint(self):
        graph = HazardKnowledgeGraph(backend="sakuna")
        result = graph.push_to_graphdb()
        self.assertFalse(result["persisted"])
        self.assertEqual(result["reason"], "graphdb_endpoint_not_configured")

    def test_graphdb_push_posts_instance_ttl_to_endpoint(self):
        received = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                received["body"] = self.rfile.read(length).decode("utf-8")
                received["content_type"] = self.headers.get("Content-Type")
                self.send_response(204)
                self.end_headers()

            def log_message(self, format, *args):
                return

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            graph = HazardKnowledgeGraph(backend="sakuna")
            risk = HazardRisk(
                hazard_type="flash_flood",
                risk_probability=0.7,
                risk_level=RiskLevel.HIGH,
                contributing_factors=["pwat"],
                grid=GridCell(id="jeddah", lat=21.4, lon=39.2),
                indicator_evidence={"pwat": 44.0},
            )
            graph.add_risk_evidence(risk)
            url = f"http://127.0.0.1:{server.server_port}/repositories/mazu/statements"
            result = graph.push_to_graphdb(url)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertTrue(result["persisted"])
        self.assertEqual(result["http_status"], 204)
        self.assertEqual(received["content_type"], "text/turtle")
        self.assertIn("sakuna:hasDisasterType sakuna:FlashFlood", received["body"])

    def test_runtime_provenance_records_stcast_provider_metadata(self):
        graph = HazardKnowledgeGraph(backend="sakuna")
        uri = graph.add_runtime_provenance(
            {
                "provider": "STCast",
                "provider_role": "regional_forecast_model",
                "provider_status": "ready",
                "model_version": "saudi_local",
            }
        )
        self.assertIsNotNone(uri)
        ttl = graph.export_instances()
        self.assertIn("STCast", ttl)
        self.assertIn("regional_forecast_model", ttl)

    def test_grounding_gap_is_linked_and_queryable(self):
        graph = HazardKnowledgeGraph(backend="sakuna")
        risk = HazardRisk(
            hazard_type="extreme_heat",
            risk_probability=0.9,
            risk_level=RiskLevel.HIGH,
            contributing_factors=["heat_index_c"],
            grid=GridCell(id="jeddah", lat=21.4, lon=39.2),
            valid_time=datetime(2026, 7, 19, tzinfo=timezone.utc),
        )
        records = graph.add_grounding_gap_evidence(
            risk,
            source_metadata={
                "resolved_sources": {"temperature": {"dataset_id": "era5_mswep", "role": "primary"}},
                "source_status": "degraded",
            },
            grounding_gap={
                "heat_features": {
                    "source_pair": ["era5_mswep", "historical_archive"],
                    "comparison_time": "2026-07-19T00:00:00Z",
                    "units": "degC",
                    "absolute_difference": 6.25,
                    "summary": {"mean": 4.1},
                }
            },
            degradation_metadata={"reason": "archive_fallback", "promoted_source": "historical_archive"},
        )
        self.assertEqual(len(records), 1)
        ttl = graph.export_instances()
        self.assertIn("hasGroundingGap", ttl)
        self.assertIn("sourcePair", ttl)
        self.assertIn("era5_mswep", ttl)
        self.assertIn("absoluteDelta", ttl)
        self.assertIn("archive_fallback", ttl)

    def test_missing_grounding_gap_is_harmless(self):
        graph = HazardKnowledgeGraph()
        risk = HazardRisk(hazard_type="flash_flood", risk_probability=0.2, risk_level=RiskLevel.LOW, contributing_factors=[])
        before = len(graph.triples)
        self.assertEqual(graph.add_grounding_gap_evidence(risk, source_metadata={}), [])
        self.assertEqual(len(graph.triples), before)

    def test_runtime_ttl_validator_requires_grounding_contract(self):
        graph = HazardKnowledgeGraph(backend="sakuna")
        risk = HazardRisk(
            hazard_type="flash_flood",
            risk_probability=0.7,
            risk_level=RiskLevel.HIGH,
            contributing_factors=["pwat"],
            grid=GridCell(id="jeddah", lat=21.4, lon=39.2),
            indicator_evidence={"pwat": 44.0},
        )
        graph.add_risk_evidence(risk)
        graph.add_grounding_gap_evidence(
            risk,
            grounding_gap={"precip": {"source_pair": ["era5", "gpm"], "absolute_difference": 1.0}},
        )
        result = validate_instance_ttl(graph.export_instances())
        self.assertTrue(result["valid"])
        self.assertEqual(result["missing_markers"], [])


if __name__ == "__main__":
    unittest.main()
