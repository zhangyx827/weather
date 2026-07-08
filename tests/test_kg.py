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


if __name__ == "__main__":
    unittest.main()
