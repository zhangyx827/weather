"""FastAPI tests for optional API layer."""

import asyncio
import unittest

import httpx

from mazu_saudi.api.app import app
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


if __name__ == "__main__":
    unittest.main()
