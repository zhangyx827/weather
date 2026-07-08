"""FastAPI API for the MAZU Saudi MVP."""

from __future__ import annotations

from typing import Any

from mazu_saudi.agent.briefing import build_warning_product
from mazu_saudi.agent.workflow import run_demo_pipeline
from mazu_saudi.kg import HazardKnowledgeGraph
from mazu_saudi.risk import all_default_models
from mazu_saudi.schemas import MeteorologicalFeatures

try:
    from fastapi import FastAPI, Query
except Exception as exc:  # pragma: no cover
    FastAPI = None
    _FASTAPI_IMPORT_ERROR = exc
else:
    _FASTAPI_IMPORT_ERROR = None


def create_app():
    """Create the FastAPI app, raising a clear error if FastAPI is missing."""

    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed. Install dependencies with: python3 -m pip install -r requirements.txt") from _FASTAPI_IMPORT_ERROR

    app = FastAPI(title="MAZU Saudi Early Warning MVP", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": "mazu-saudi", "version": "0.1.0"}

    @app.post("/risk/scan")
    def risk_scan(payload: dict[str, Any]) -> dict[str, Any]:
        features = MeteorologicalFeatures.from_dict(payload)
        risks = [model.predict(features).to_dict() for model in all_default_models()]
        return {"risks": risks}

    @app.post("/warning/generate")
    def warning_generate(payload: dict[str, Any]) -> dict[str, Any]:
        features = MeteorologicalFeatures.from_dict(payload)
        risks = [model.predict(features) for model in all_default_models()]
        graph = HazardKnowledgeGraph()
        impacts = {}
        for risk in risks:
            graph.add_risk_evidence(risk)
            impacts[risk.hazard_type] = graph.query_hazard_impacts(risk.hazard_type)
        product = build_warning_product(features.grid.region or features.grid.id, risks, {"impacts": impacts, "triple_count": len(graph.triples)})
        return product.to_dict()

    @app.get("/kg/query")
    def kg_query(hazard_type: str = Query("flash_flood")) -> dict[str, Any]:
        graph = HazardKnowledgeGraph()
        return graph.query_hazard_impacts(hazard_type)

    @app.get("/demo/run")
    def demo_run() -> dict[str, Any]:
        return run_demo_pipeline()

    return app


app = create_app() if FastAPI is not None else None
