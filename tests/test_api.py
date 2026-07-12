"""FastAPI tests for optional API layer."""

import asyncio
import os
import unittest
from unittest.mock import patch

import httpx

from mazu_saudi.api.app import app
from mazu_saudi.agent.strands import StrandsError
from mazu_saudi.agent.workflow import load_sample_features


@unittest.skipIf(app is None, "FastAPI/pydantic optional dependencies are not installed")
class APITests(unittest.TestCase):
    def _request(self, method: str, path: str, **kwargs):
        async def _call():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(_call())

    def test_batch_risk_scan(self):
        response = self._request("POST", "/risk/scan", json={"features": [load_sample_features()]})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["batch"])
        self.assertEqual(len(payload["risks"]), 5)

    def test_kg_query_type(self):
        response = self._request("GET", "/kg/query", params={"query_type": "response_actions_by_hazard", "value": "flash_flood"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("activate_flood_watch", response.json()["actions"])

    def test_demo_trace(self):
        response = self._request("GET", "/demo/run")
        self.assertEqual(response.status_code, 200)
        self.assertIn("pipeline_trace", response.json())

    def test_indicator_netcdf_warning_endpoint(self):
        response = self._request(
            "POST",
            "/warning/generate-from-netcdf",
            json={
                "path": "data/processed/lightgbm_indicators_nc/saudi_indicators_20250101.nc",
                "latitude": 24.7,
                "longitude": 46.7,
                "region": "Riyadh",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["input_contract"], "indicator_field_set")
        self.assertEqual(len(payload["risks"]), 5)

    def test_warning_generate_exposes_llm_and_strands_metadata(self):
        features = load_sample_features()

        class FakeStrandsAgent:
            def __init__(self, settings, pipeline_factory=None):
                self.settings = settings
                self.pipeline_factory = pipeline_factory

            def execute(self, payload):
                del payload
                return type(
                    "Result",
                    (),
                    {
                        "output": {
                            "generation_metadata": {"provider": "llm", "status": "ok"},
                            "llm_raw": {"response": {"id": "abc"}},
                            "briefings": [{"industry": "meteorology", "zh": "气象简报"}],
                            "briefing_text": [{"industry": "meteorology", "text": "气象简报"}],
                            "requested_language": "zh",
                            "strands_run": {"run_id": "strands-123", "status": "completed"},
                        }
                    },
                )()

        with patch.dict(
            os.environ,
            {"MAZU_STRANDS_ENABLED": "true", "MAZU_STRANDS_MODEL_ID": "amazon.nova-lite-v1:0"},
            clear=False,
        ):
            with patch("mazu_saudi.api.app.StrandsWarningAgent", FakeStrandsAgent):
                response = self._request("POST", "/warning/generate", json={"features": features})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["generation_metadata"]["provider"], "llm")
        self.assertEqual(payload["llm_raw"]["response"]["id"], "abc")
        self.assertEqual(payload["strands_run"]["run_id"], "strands-123")

    def test_warning_generate_returns_empty_strands_run_when_disabled(self):
        response = self._request("POST", "/warning/generate", json={"features": load_sample_features(), "industries": ["meteorology"]})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["strands_run"], {})
        self.assertEqual(len(payload["briefing_text"]), 1)

    def test_warning_generate_maps_strands_errors_to_integration_failure(self):
        class FakeStrandsAgent:
            def __init__(self, settings, pipeline_factory=None):
                del settings, pipeline_factory

            def execute(self, payload):
                del payload
                raise StrandsError("bedrock timeout")

        with patch.dict(os.environ, {"MAZU_STRANDS_ENABLED": "true"}, clear=False):
            with patch("mazu_saudi.api.app.StrandsWarningAgent", FakeStrandsAgent):
                response = self._request("POST", "/warning/generate", json={"features": load_sample_features()})
        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]
        self.assertEqual(detail["code"], "strands_integration_failed")
        self.assertIn("timeout", detail["message"])


if __name__ == "__main__":
    unittest.main()
