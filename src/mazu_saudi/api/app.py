"""FastAPI API for the MAZU Saudi MVP."""

from __future__ import annotations

from typing import Any

from mazu_saudi.agent.briefing import build_warning_product
from mazu_saudi.agent.workflow import run_demo_pipeline, run_indicator_netcdf_pipeline
from mazu_saudi.data import indicator_point_from_netcdf
from mazu_saudi.kg import HazardKnowledgeGraph
from mazu_saudi.risk import all_default_models
from mazu_saudi.schemas import IndicatorFieldSet, MeteorologicalFeatures

try:
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover
    FastAPI = None
    HTTPException = None
    BaseModel = object
    Field = None
    _FASTAPI_IMPORT_ERROR = exc
else:
    _FASTAPI_IMPORT_ERROR = None


if FastAPI is not None:
    class RiskScanRequest(BaseModel):
        """Single or batch risk scan request."""

        features: dict[str, Any] | list[dict[str, Any]] | None = Field(default=None)

    class RiskScanResponse(BaseModel):
        risks: list[dict[str, Any]]
        batch: bool = False
        count: int = 1

    class WarningGenerateRequest(BaseModel):
        features: dict[str, Any]
        industries: list[str] | None = None
        language: str = "zh"

    class NetCDFWarningGenerateRequest(BaseModel):
        path: str
        latitude: float
        longitude: float
        region: str | None = None

    class ErrorResponse(BaseModel):
        error: dict[str, Any]


def create_app():
    """Create the FastAPI app, raising a clear error if FastAPI is missing."""

    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed. Install dependencies with: python3 -m pip install -r requirements.txt") from _FASTAPI_IMPORT_ERROR

    app = FastAPI(
        title="MAZU Saudi Early Warning MVP",
        version="0.2.0",
        openapi_tags=[
            {"name": "health", "description": "Service health checks"},
            {"name": "risk", "description": "Multi-hazard risk scans"},
            {"name": "warning", "description": "Warning product generation"},
            {"name": "kg", "description": "Knowledge graph queries"},
            {"name": "demo", "description": "Runnable demo workflow"},
        ],
    )

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, Any]:
        return {"status": "ok", "service": "mazu-saudi", "version": "0.2.0"}

    @app.post("/risk/scan", tags=["risk"], response_model=RiskScanResponse, responses={400: {"model": ErrorResponse}})
    async def risk_scan(payload: dict[str, Any] | RiskScanRequest) -> dict[str, Any]:
        try:
            raw_payload = payload.model_dump() if hasattr(payload, "model_dump") else payload
            raw_features = raw_payload.get("features", raw_payload)
            feature_items = raw_features if isinstance(raw_features, list) else [raw_features]
            models = all_default_models()
            risks = []
            for item in feature_items:
                features = IndicatorFieldSet.from_dict(item) if isinstance(item, dict) and "values" in item else MeteorologicalFeatures.from_dict(item)
                risks.extend(model.predict_one(features).to_dict() for model in models)
            return {"risks": risks, "batch": isinstance(raw_features, list), "count": len(feature_items)}
        except Exception as exc:
            raise _http_error("risk_scan_failed", str(exc))

    @app.post("/warning/generate", tags=["warning"], responses={400: {"model": ErrorResponse}})
    async def warning_generate(payload: dict[str, Any] | WarningGenerateRequest) -> dict[str, Any]:
        try:
            raw_payload = payload.model_dump() if hasattr(payload, "model_dump") else payload
            raw_features = raw_payload.get("features", raw_payload)
            requested_industries = raw_payload.get("industries")
            language = raw_payload.get("language", "zh")
            features = IndicatorFieldSet.from_dict(raw_features) if isinstance(raw_features, dict) and "values" in raw_features else MeteorologicalFeatures.from_dict(raw_features)
            risks = [model.predict_one(features) for model in all_default_models()]
            graph = HazardKnowledgeGraph()
            impacts = {}
            for risk in risks:
                graph.add_risk_evidence(risk)
                impacts[risk.hazard_type] = graph.query_hazard_impacts(risk.hazard_type)
            product = build_warning_product(features.grid.region or features.grid.id, risks, {"impacts": impacts, "triple_count": len(graph.triples)})
            if requested_industries:
                product.briefings = [briefing for briefing in product.briefings if briefing.industry in requested_industries]
            output = product.to_dict()
            output["requested_language"] = language
            output["briefing_text"] = [
                {"industry": item["industry"], "text": item.get(language, item["zh"])}
                for item in output["briefings"]
            ]
            return output
        except Exception as exc:
            raise _http_error("warning_generation_failed", str(exc))

    @app.post("/warning/generate-from-netcdf", tags=["warning"], responses={400: {"model": ErrorResponse}})
    async def warning_generate_from_netcdf(payload: dict[str, Any] | NetCDFWarningGenerateRequest) -> dict[str, Any]:
        try:
            raw_payload = payload.model_dump() if hasattr(payload, "model_dump") else payload
            return run_indicator_netcdf_pipeline(
                raw_payload["path"],
                float(raw_payload["latitude"]),
                float(raw_payload["longitude"]),
                raw_payload.get("region"),
            )
        except Exception as exc:
            raise _http_error("warning_generation_from_netcdf_failed", str(exc))

    @app.post("/risk/scan-indicators", tags=["risk"], response_model=RiskScanResponse, responses={400: {"model": ErrorResponse}})
    async def risk_scan_indicators(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            if "path" in payload:
                feature_items = [
                    indicator_point_from_netcdf(
                        payload["path"],
                        float(payload["latitude"]),
                        float(payload["longitude"]),
                        region=payload.get("region"),
                    )
                ]
            else:
                raw_features = payload.get("features", payload)
                raw_items = raw_features if isinstance(raw_features, list) else [raw_features]
                feature_items = [IndicatorFieldSet.from_dict(item) if isinstance(item, dict) else item for item in raw_items]
            models = all_default_models()
            risks = []
            for features in feature_items:
                risks.extend(model.predict_one(features).to_dict() for model in models)
            return {"risks": risks, "batch": len(feature_items) > 1, "count": len(feature_items)}
        except Exception as exc:
            raise _http_error("indicator_risk_scan_failed", str(exc))

    @app.get("/kg/query", tags=["kg"], responses={400: {"model": ErrorResponse}})
    async def kg_query(
        query_type: str = Query("hazard_impacts"),
        value: str | None = Query(None),
        hazard_type: str | None = Query(None),
    ) -> dict[str, Any]:
        try:
            graph = HazardKnowledgeGraph()
            query_value = value or hazard_type or "flash_flood"
            return graph.query(query_type, query_value)
        except Exception as exc:
            raise _http_error("kg_query_failed", str(exc))

    @app.get("/demo/run", tags=["demo"])
    async def demo_run() -> dict[str, Any]:
        return run_demo_pipeline()

    return app


def _http_error(code: str, message: str):
    if HTTPException is None:  # pragma: no cover
        raise RuntimeError(f"{code}: {message}")
    raise HTTPException(status_code=400, detail={"code": code, "message": message})


app = create_app() if FastAPI is not None else None
