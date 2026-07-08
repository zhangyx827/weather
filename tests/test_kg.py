"""Tests for the knowledge graph loop."""

import unittest

from mazu_saudi.kg import HazardKnowledgeGraph
from mazu_saudi.risk import ExtremeHeatRiskModel
from mazu_saudi.schemas import GridCell, MeteorologicalFeatures


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


if __name__ == "__main__":
    unittest.main()
