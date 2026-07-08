"""Lightweight MAZU Saudi agent workflow."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from mazu_saudi.agent.briefing import build_warning_product
from mazu_saudi.forecast import MockForecastProvider
from mazu_saudi.indicators import (
    compute_cape_placeholder,
    compute_heat_index_c,
    compute_ivt_placeholder,
    compute_pwat_placeholder,
    compute_vpd_kpa,
)
from mazu_saudi.kg import HazardKnowledgeGraph
from mazu_saudi.risk import all_default_models
from mazu_saudi.schemas import MeteorologicalFeatures
from mazu_saudi.utils.math import is_missing

Context = dict[str, Any]


class WorkflowNode(ABC):
    """Base workflow node."""

    name = "node"

    @abstractmethod
    def run(self, context: Context) -> Context:
        """Run node and return context."""


class DataCheckNode(WorkflowNode):
    """Validate and normalize input feature payload."""

    name = "data_check"

    def run(self, context: Context) -> Context:
        features = context.get("features")
        if isinstance(features, dict):
            features = MeteorologicalFeatures.from_dict(features)
        if not isinstance(features, MeteorologicalFeatures):
            raise ValueError("context['features'] must be MeteorologicalFeatures or dict")
        context["features"] = features
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
        self.provider = provider or MockForecastProvider()

    def run(self, context: Context) -> Context:
        features: MeteorologicalFeatures = context["features"]
        context["forecast_fields"] = {
            variable: self.provider.fetch(variable, valid_time=features.valid_time).to_dict()
            for variable in ["temp_c", "rh_percent", "wind_speed_mps", "precip_1h_mm"]
        }
        context.setdefault("trace", []).append(self.name)
        return context


class IndicatorDeriveNode(WorkflowNode):
    """Derive physical indicators and fill placeholders when needed."""

    name = "indicator_derive"

    def run(self, context: Context) -> Context:
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
        features: MeteorologicalFeatures = context["features"]
        context["risks"] = [model.predict(features) for model in self.models]
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
        context["evidence_status"] = {
            risk.hazard_type: bool(risk.contributing_factors and risk.evidence) for risk in context["risks"]
        }
        context.setdefault("trace", []).append(self.name)
        return context


class BriefingNode(WorkflowNode):
    """Generate warning product and industry briefings."""

    name = "briefing"

    def run(self, context: Context) -> Context:
        features: MeteorologicalFeatures = context["features"]
        area = features.grid.region or features.grid.id
        context["warning_product"] = build_warning_product(area, context["risks"], context["kg_explanation"])
        context.setdefault("trace", []).append(self.name)
        return context


class HumanReviewNode(WorkflowNode):
    """Mark the draft as requiring human review."""

    name = "human_review"

    def run(self, context: Context) -> Context:
        context["human_review"] = {"required": True, "status": "pending"}
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

    def run(self, features: MeteorologicalFeatures | dict[str, Any]) -> Context:
        """Run all workflow nodes."""

        context: Context = {"features": features}
        for node in self.nodes:
            context = node.run(context)
        return context


def load_sample_features() -> dict[str, Any]:
    """Load bundled demo feature payload."""

    path = Path(__file__).resolve().parents[3] / "examples" / "sample_features.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run_demo_pipeline() -> dict[str, Any]:
    """Run the demo pipeline and return JSON-serializable output."""

    context = SaudiWarningPipeline().run(load_sample_features())
    return {
        "trace": context["trace"],
        "features": context["features"].to_dict(),
        "indicators": context["indicators"],
        "risks": [risk.to_dict() for risk in context["risks"]],
        "kg_explanation": context["kg_explanation"],
        "evidence_status": context["evidence_status"],
        "human_review": context["human_review"],
        "warning_product": context["publish_payload"],
    }
