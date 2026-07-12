"""Lightweight MAZU Saudi agent workflow."""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mazu_saudi.data import check_missing_values, indicator_point_from_netcdf, read_json_features
from mazu_saudi.agent.briefing import build_warning_product, create_default_briefing_generator
from mazu_saudi.forecast import AIFSBenchmarkProvider, AuroraForecastProvider, GenCastForecastProvider, MockForecastProvider
from mazu_saudi.indicators import (
    compute_cape_placeholder,
    compute_heat_index_c,
    compute_ivt_placeholder,
    compute_pwat_placeholder,
    compute_vpd_kpa,
)
from mazu_saudi.kg import HazardKnowledgeGraph
from mazu_saudi.risk import all_default_models
from mazu_saudi.schemas import IndicatorFieldSet, MeteorologicalFeatures
from mazu_saudi.utils.math import is_missing

try:
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pragma: no cover - pydantic optional in minimal env
    BaseModel = None


if BaseModel is not None:
    class WorkflowContext(BaseModel):
        """Pydantic workflow context for API/runtime validation."""

        model_config = ConfigDict(arbitrary_types_allowed=True)
        features: IndicatorFieldSet | MeteorologicalFeatures | dict[str, Any]
        trace: list[str] = Field(default_factory=list)
        node_trace: list[dict[str, Any]] = Field(default_factory=list)
        errors: list[dict[str, Any]] = Field(default_factory=list)
        data: dict[str, Any] = Field(default_factory=dict)
else:
    @dataclass
    class WorkflowContext:
        """Fallback context with the same public fields when pydantic is absent."""

        features: IndicatorFieldSet | MeteorologicalFeatures | dict[str, Any]
        trace: list[str] = field(default_factory=list)
        node_trace: list[dict[str, Any]] = field(default_factory=list)
        errors: list[dict[str, Any]] = field(default_factory=list)
        data: dict[str, Any] = field(default_factory=dict)

        def model_dump(self) -> dict[str, Any]:
            return {
                "features": self.features,
                "trace": self.trace,
                "node_trace": self.node_trace,
                "errors": self.errors,
                "data": self.data,
            }


Context = dict[str, Any]


def create_default_forecast_provider():
    """Create the runtime forecast provider from environment or defaults."""

    provider_name = os.getenv("MAZU_FORECAST_PROVIDER", "aurora").strip().lower()
    if provider_name == "gencast":
        return GenCastForecastProvider()
    if provider_name == "aifs":
        return AIFSBenchmarkProvider()
    if provider_name == "mock":
        return MockForecastProvider()
    return AuroraForecastProvider()


class WorkflowNode(ABC):
    """Base workflow node."""

    name = "node"

    @abstractmethod
    def run(self, context: Context) -> Context:
        """Run node and return context."""

    def input_summary(self, context: Context) -> dict[str, Any]:
        return {"keys": sorted(context.keys())}

    def output_summary(self, context: Context) -> dict[str, Any]:
        summary = {"keys": sorted(context.keys())}
        if "risks" in context:
            summary["risk_count"] = len(context["risks"])
        if "indicators" in context:
            summary["indicator_count"] = len(context["indicators"])
        return summary


class DataCheckNode(WorkflowNode):
    """Validate and normalize input feature payload."""

    name = "data_check"

    def run(self, context: Context) -> Context:
        features = context.get("features")
        if isinstance(features, dict):
            features = IndicatorFieldSet.from_dict(features) if "values" in features else MeteorologicalFeatures.from_dict(features)
        if not isinstance(features, (IndicatorFieldSet, MeteorologicalFeatures)):
            raise ValueError("context['features'] must be IndicatorFieldSet, MeteorologicalFeatures, or dict")
        context["features"] = features
        context["input_contract"] = "indicator_field_set" if isinstance(features, IndicatorFieldSet) else "meteorological_features"
        if isinstance(features, IndicatorFieldSet):
            missing = [
                name
                for name in ("t2m_c", "rh2m", "vpd_kpa", "heat_index_c", "wind10_speed")
                if features.values.get(name) is None
            ]
            context["data_quality"] = {"ok": not missing, "missing_count": len(missing), "missing": missing}
            context["indicators"] = dict(features.values)
        else:
            context["data_quality"] = check_missing_values(features)
        context.setdefault("trace", []).append(self.name)
        return context


class HazardScanNode(WorkflowNode):
    """Declare candidate hazards for this MVP scan."""

    name = "hazard_scan"

    def run(self, context: Context) -> Context:
        context["candidate_hazards"] = [model.hazard_type for model in all_default_models()]
        context.setdefault("trace", []).append(self.name)
        return context


class ForecastNode(WorkflowNode):
    """Attach mock forecast background fields."""

    name = "forecast"

    def __init__(self, provider=None):
        self.provider = provider or create_default_forecast_provider()

    def run(self, context: Context) -> Context:
        if isinstance(context["features"], IndicatorFieldSet):
            context["forecast_fields"] = {}
            context["forecast_confidence"] = {
                "status": "not_required",
                "provider": "processed_indicator_dataset",
                "provider_role": "preprocessed_indicator_dataset",
            }
            context.setdefault("trace", []).append(self.name)
            return context
        features: MeteorologicalFeatures = context["features"]
        forecast = self.provider.get_forecast(features.valid_time, lead_hours=0)
        context["forecast_fields"] = {key: field.to_dict() for key, field in forecast.items()}
        first_field = next(iter(forecast.values())) if forecast else None
        context["forecast_confidence"] = {
            "status": "ready" if first_field is not None else "unavailable",
            "provider": self.provider.name,
            "provider_role": getattr(first_field, "provider_role", getattr(self.provider, "provider_role", "deterministic")),
            "provider_status": getattr(first_field, "provider_status", getattr(self.provider, "provider_status", "unknown")),
            "source_status": getattr(first_field, "source_status", getattr(self.provider, "source_status", "unknown")),
            "degradation_metadata": getattr(first_field, "degradation_metadata", {}),
        }
        context.setdefault("trace", []).append(self.name)
        return context


class IndicatorDeriveNode(WorkflowNode):
    """Derive physical indicators and fill placeholders when needed."""

    name = "indicator_derive"

    def run(self, context: Context) -> Context:
        if isinstance(context["features"], IndicatorFieldSet):
            indicators = context["features"].values
            required = ("t2m_c", "rh2m", "vpd_kpa", "heat_index_c", "wind10_speed")
            context["indicator_status"] = {
                name: (name in indicators and not is_missing(indicators.get(name)))
                for name in required
            }
            context.setdefault("trace", []).append(self.name)
            return context
        features: MeteorologicalFeatures = context["features"]
        pwat = features.pwat_mm if not is_missing(features.pwat_mm) else compute_pwat_placeholder(features.temp_c, features.rh_percent, features.pressure_hpa)
        ivt = features.ivt_kg_m_s if not is_missing(features.ivt_kg_m_s) else compute_ivt_placeholder(features.wind_speed_mps or 0.0, pwat)
        cape = features.cape_j_kg if not is_missing(features.cape_j_kg) else compute_cape_placeholder(features.temp_c, features.rh_percent)
        context["indicators"] = {
            "vpd_kpa": compute_vpd_kpa(features.temp_c, features.rh_percent),
            "heat_index_c": compute_heat_index_c(features.temp_c, features.rh_percent),
            "pwat_mm": pwat,
            "ivt_kg_m_s": ivt,
            "cape_j_kg": cape,
        }
        features.pwat_mm = None if is_missing(pwat) else float(pwat)
        features.ivt_kg_m_s = None if is_missing(ivt) else float(ivt)
        features.cape_j_kg = None if is_missing(cape) else float(cape)
        context.setdefault("trace", []).append(self.name)
        return context


class ModelInferenceNode(WorkflowNode):
    """Run all hazard risk models."""

    name = "model_inference"

    def __init__(self, models=None):
        self.models = models or all_default_models()

    def run(self, context: Context) -> Context:
        features = context["features"]
        context["risks"] = [model.predict(features) for model in self.models]
        context["risk_runtime"] = {
            "model_families": {risk.hazard_type: risk.model_family for risk in context["risks"]},
            "inference_modes": {risk.hazard_type: risk.inference_mode for risk in context["risks"]},
        }
        context.setdefault("trace", []).append(self.name)
        return context


class KGReasoningNode(WorkflowNode):
    """Build risk evidence triples and query hazard impacts."""

    name = "kg_reasoning"

    def run(self, context: Context) -> Context:
        graph = HazardKnowledgeGraph()
        impacts = {}
        for risk in context["risks"]:
            graph.add_risk_evidence(risk)
            impacts[risk.hazard_type] = graph.query_hazard_impacts(risk.hazard_type)
        context["kg"] = graph
        context["kg_explanation"] = {"impacts": impacts, "triple_count": len(graph.triples)}
        context.setdefault("trace", []).append(self.name)
        return context


class EvidenceCheckNode(WorkflowNode):
    """Check that risk outputs include evidence and factors."""

    name = "evidence_check"

    def run(self, context: Context) -> Context:
        required_indicators = ["vpd_kpa", "heat_index_c", "pwat_mm", "ivt_kg_m_s", "cape_j_kg"]
        if isinstance(context["features"], IndicatorFieldSet):
            required_indicators = ["vpd_kpa", "heat_index_c", "pwat", "ivt", "cape"]
        indicators = context.get("indicators", {})
        indicator_status = {
            name: (name in indicators and not is_missing(indicators.get(name)))
            for name in required_indicators
        }
        risk_status = {}
        for risk in context["risks"]:
            high_risk = risk.risk_level.value in {"high", "extreme"}
            risk_status[risk.hazard_type] = {
                "has_model_version": bool(risk.model_version or risk.evidence.get("model_version")),
                "has_contributing_factors": bool(risk.contributing_factors),
                "has_indicator_evidence": bool(risk.indicator_evidence),
                "high_risk_requires_factors_ok": (not high_risk) or bool(risk.contributing_factors),
                "model_family": risk.model_family,
                "inference_mode": risk.inference_mode,
                "source_status": risk.source_status,
                "has_shap_summary": bool(risk.shap_summary.get("available") or risk.shap_summary.get("top_features")),
            }
        source_metadata = getattr(context["features"], "source_metadata", {}) if isinstance(context["features"], IndicatorFieldSet) else {}
        context["evidence_status"] = {
            "physical_indicators_complete": all(indicator_status.values()),
            "indicator_status": indicator_status,
            "risk_status": risk_status,
            "gencast_confidence": context.get("forecast_confidence", {"status": "unavailable"}),
            "grounding_metadata_present": bool(source_metadata.get("resolved_sources") or source_metadata.get("grounding_gap")),
            "source_status": getattr(context["features"], "source_status", "normal") if isinstance(context["features"], IndicatorFieldSet) else "normal",
        }
        context.setdefault("trace", []).append(self.name)
        return context


class BriefingNode(WorkflowNode):
    """Generate warning product and industry briefings."""

    name = "briefing"

    def __init__(self, generator=None):
        self.generator = generator or create_default_briefing_generator()

    def run(self, context: Context) -> Context:
        features = context["features"]
        area = features.grid.region or features.grid.id
        context["warning_product"] = build_warning_product(
            area,
            context["risks"],
            context["kg_explanation"],
            generator=self.generator,
            context=context,
        )
        context["generation_metadata"] = context["warning_product"].generation_metadata
        context["llm_raw"] = context["warning_product"].llm_raw
        context.setdefault("trace", []).append(self.name)
        return context


class HumanReviewNode(WorkflowNode):
    """Mark high/extreme risk drafts as requiring human review."""

    name = "human_review"

    def run(self, context: Context) -> Context:
        required = any(risk.risk_level.value in {"high", "extreme"} for risk in context.get("risks", []))
        context["human_review"] = {"required": required, "human_review_required": required, "status": "pending" if required else "not_required"}
        context.setdefault("trace", []).append(self.name)
        return context


class PublishNode(WorkflowNode):
    """Prepare structured publishable output without external side effects."""

    name = "publish"

    def run(self, context: Context) -> Context:
        product = context["warning_product"]
        context["publish_payload"] = product.to_dict()
        context.setdefault("trace", []).append(self.name)
        return context


class SaudiWarningPipeline:
    """End-to-end MAZU Saudi MVP workflow."""

    def __init__(self, nodes: list[WorkflowNode] | None = None):
        self.nodes = nodes or [
            DataCheckNode(),
            HazardScanNode(),
            ForecastNode(),
            IndicatorDeriveNode(),
            ModelInferenceNode(),
            KGReasoningNode(),
            EvidenceCheckNode(),
            BriefingNode(),
            HumanReviewNode(),
            PublishNode(),
        ]

    def run(self, features: IndicatorFieldSet | MeteorologicalFeatures | dict[str, Any]) -> Context:
        """Run all workflow nodes."""

        workflow_context = WorkflowContext(features=features)
        context: Context = workflow_context.model_dump()
        for node in self.nodes:
            start = time.perf_counter()
            record = {
                "node": node.name,
                "status": "running",
                "input_summary": node.input_summary(context),
                "output_summary": {},
                "duration_ms": 0.0,
                "error": None,
            }
            try:
                context = node.run(context)
                record["status"] = "ok"
                record["output_summary"] = node.output_summary(context)
            except Exception as exc:
                record["status"] = "error"
                record["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
                context.setdefault("errors", []).append({"node": node.name, **record["error"]})
                context.setdefault("trace", []).append(node.name)
                context["failed"] = True
                record["output_summary"] = node.output_summary(context)
                break
            finally:
                record["duration_ms"] = round((time.perf_counter() - start) * 1000.0, 3)
                context.setdefault("node_trace", []).append(record)
        return context


def load_sample_features() -> dict[str, Any]:
    """Load bundled demo feature payload."""

    path = Path(__file__).resolve().parents[3] / "examples" / "sample_features.json"
    features = read_json_features(path)
    return features.to_dict() if hasattr(features, "to_dict") else json.loads(path.read_text(encoding="utf-8"))


def run_demo_pipeline() -> dict[str, Any]:
    """Run the demo pipeline and return JSON-serializable output."""

    context = SaudiWarningPipeline().run(load_sample_features())
    return {
        "trace": context["trace"],
        "pipeline_trace": context["node_trace"],
        "errors": context.get("errors", []),
        "data_quality": context.get("data_quality", {}),
        "features": context["features"].to_dict(),
        "indicators": context["indicators"],
        "risks": [risk.to_dict() for risk in context["risks"]],
        "kg_explanation": context["kg_explanation"],
        "evidence_status": context["evidence_status"],
        "human_review": context["human_review"],
        "warning_product": context["publish_payload"],
    }


def run_indicator_netcdf_pipeline(path: str | Path, latitude: float, longitude: float, region: str | None = None) -> dict[str, Any]:
    """Run the agent against one point selected from a processed indicator NetCDF."""

    features = indicator_point_from_netcdf(path, latitude, longitude, region=region)
    context = SaudiWarningPipeline().run(features)
    return {
        "trace": context["trace"],
        "pipeline_trace": context["node_trace"],
        "errors": context.get("errors", []),
        "input_contract": context.get("input_contract"),
        "data_quality": context.get("data_quality", {}),
        "features": context["features"].to_dict(),
        "indicators": context.get("indicators", {}),
        "risks": [risk.to_dict() for risk in context.get("risks", [])],
        "kg_explanation": context.get("kg_explanation", {}),
        "evidence_status": context.get("evidence_status", {}),
        "human_review": context.get("human_review", {}),
        "warning_product": context.get("publish_payload", {}),
    }
