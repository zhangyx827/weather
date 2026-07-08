"""Call the FastAPI app in-process and print API responses."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "examples" / "sample_features.json"


def main() -> None:
    from mazu_saudi.api.app import app

    features = json.loads(FEATURES.read_text(encoding="utf-8"))

    async def run() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            risk_response = await client.post("/risk/scan", json={"features": [features]})
            warning_response = await client.post(
                "/warning/generate",
                json={"features": features, "industries": ["meteorology"], "language": "zh"},
            )
            print(
                json.dumps(
                    {
                        "risk_count": len(risk_response.json()["risks"]),
                        "warning": warning_response.json(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

    asyncio.run(run())


if __name__ == "__main__":
    main()
