"""Tests for the demo workflow."""

import unittest

from mazu_saudi.agent import run_demo_pipeline


class PipelineTests(unittest.TestCase):
    def test_demo_pipeline_runs(self):
        result = run_demo_pipeline()
        self.assertEqual(len(result["risks"]), 5)
        self.assertEqual(len(result["warning_product"]["briefings"]), 6)
        self.assertIn("kg_reasoning", result["trace"])
        self.assertIn("pipeline_trace", result)
        self.assertTrue(all("duration_ms" in item for item in result["pipeline_trace"]))
        self.assertGreater(result["kg_explanation"]["triple_count"], 0)


if __name__ == "__main__":
    unittest.main()
