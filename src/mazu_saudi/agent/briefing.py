"""Warning briefing generation via template or configurable LLM providers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import httpx

from mazu_saudi.config import BriefingProviderSettings, LLMSettings
from mazu_saudi.schemas import HazardRisk, IndustryBriefing, WarningProduct

INDUSTRIES = ["meteorology", "emergency", "agriculture", "transport", "port", "public_health"]

INDUSTRY_NAMES_ZH = {
    "meteorology": "气象部门",
    "emergency": "应急管理",
    "agriculture": "农业部门",
    "transport": "交通部门",
    "port": "港口部门",
    "public_health": "公共卫生",
}

FOCUS = {
    "meteorology": "加强临近监测、订正预报场并滚动发布风险等级。",
    "emergency": "关注高等级风险区域，准备人员转移、避险提示和联动响应。",
    "agriculture": "关注干热胁迫、灌溉调度和设施农业降温防风措施。",
    "transport": "关注道路积水、低能见度和横风影响，准备交通管制预案。",
    "port": "关注沿海湿热、水汽输送和大风对港口作业的影响。",
    "public_health": "关注户外作业者、老人和慢病人群热健康风险。",
}


class BriefingGenerationError(RuntimeError):
    """Raised when structured briefing generation fails."""


@dataclass
class BriefingGenerationResult:
    """Structured briefing generation output."""

    briefings: list[IndustryBriefing]
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)


class BriefingGenerator(Protocol):
    """Generator contract for warning briefings."""

    def generate(
        self,
        area: str,
        risks: list[HazardRisk],
        kg_explanation: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> BriefingGenerationResult:
        """Generate structured briefings."""


def generate_industry_briefings(risks: list[HazardRisk]) -> list[IndustryBriefing]:
    """Generate deterministic fallback briefings."""

    significant = [r for r in risks if r.risk_level.value in {"medium", "high", "extreme"}]
    source = significant or risks
    hazard_summary = "；".join(f"{r.hazard_type}:{r.risk_level.value}({r.risk_probability:.2f})" for r in source)
    max_factors = "；".join({factor for risk in source for factor in risk.contributing_factors[:2]})
    uncertainty_note = ""
    if any(r.source_status == "degraded" or r.inference_mode == "degraded_rule_fallback" for r in source):
        uncertainty_note = " 说明：当前存在数据源或模型降级，建议结合人工复核与后续滚动订正。"
    briefings = []
    for industry in INDUSTRIES:
        zh = (
            f"{INDUSTRY_NAMES_ZH[industry]}预警简报：当前识别风险为 {hazard_summary}。"
            f"主要依据：{max_factors or '暂无显著触发因子'}。"
            f"建议：{FOCUS[industry]}{uncertainty_note}"
        )
        en = f"Placeholder English briefing for {industry}: hazards={hazard_summary}."
        ar = f"Arabic placeholder briefing for {industry}: hazards={hazard_summary}."
        briefings.append(IndustryBriefing(industry=industry, zh=zh, en=en, ar=ar, hazards=[r.hazard_type for r in source]))
    return briefings


class TemplateBriefingGenerator:
    """Generate deterministic template briefings."""

    provider_name = "template"

    def generate(
        self,
        area: str,
        risks: list[HazardRisk],
        kg_explanation: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> BriefingGenerationResult:
        del area, kg_explanation, context
        return BriefingGenerationResult(
            briefings=generate_industry_briefings(risks),
            metadata={"provider": self.provider_name, "status": "ok"},
            raw_response={},
        )


def _extract_json_payload(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end >= start:
        candidate = candidate[start : end + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise BriefingGenerationError("LLM response JSON must be an object")
    return parsed


def _coerce_briefings(payload: dict[str, Any]) -> list[IndustryBriefing]:
    raw_items = payload.get("briefings", payload)
    if isinstance(raw_items, dict):
        if "items" in raw_items and isinstance(raw_items["items"], list):
            raw_items = raw_items["items"]
        else:
            raise BriefingGenerationError("LLM payload does not contain a briefing list")
    if not isinstance(raw_items, list):
        raise BriefingGenerationError("LLM payload briefings must be a list")
    items = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise BriefingGenerationError("Each briefing entry must be an object")
        industry = item.get("industry")
        if industry not in INDUSTRIES:
            raise BriefingGenerationError(f"Unexpected industry in LLM output: {industry}")
        items.append(
            IndustryBriefing(
                industry=industry,
                zh=str(item.get("zh", "")).strip(),
                en=str(item.get("en", "")).strip() or f"Placeholder English briefing for {industry}.",
                ar=str(item.get("ar", "")).strip() or f"Arabic placeholder briefing for {industry}.",
                hazards=[str(value) for value in item.get("hazards", []) if str(value).strip()],
            )
        )
    observed = {item.industry for item in items}
    missing = [industry for industry in INDUSTRIES if industry not in observed]
    if missing:
        raise BriefingGenerationError(f"LLM payload missing industries: {', '.join(missing)}")
    return sorted(items, key=lambda item: INDUSTRIES.index(item.industry))


def _build_generation_payload(
    area: str,
    risks: list[HazardRisk],
    kg_explanation: dict[str, Any],
    context: dict[str, Any] | None,
) -> dict[str, Any]:
    evidence_status = (context or {}).get("evidence_status", {})
    human_review = (context or {}).get("human_review", {})
    risk_items = []
    for risk in risks:
        risk_items.append(
            {
                "hazard_type": risk.hazard_type,
                "risk_level": risk.risk_level.value,
                "risk_probability": round(risk.risk_probability, 3),
                "contributing_factors": risk.contributing_factors,
                "model_family": risk.model_family,
                "inference_mode": risk.inference_mode,
                "source_status": risk.source_status,
                "degradation_metadata": risk.degradation_metadata,
                "indicator_evidence": risk.indicator_evidence,
            }
        )
    return {
        "area": area,
        "industries": INDUSTRIES,
        "industry_names_zh": INDUSTRY_NAMES_ZH,
        "industry_focus": FOCUS,
        "risks": risk_items,
        "kg_explanation": kg_explanation,
        "evidence_status": evidence_status,
        "human_review": human_review,
        "instructions": {
            "language": "zh",
            "must_include_uncertainty_when_degraded": True,
            "must_return_json": True,
        },
    }


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part.strip())
    raise BriefingGenerationError("Unsupported LLM content format")


class OpenAICompatibleBriefingGenerator:
    """Generate structured briefings via a generic OpenAI-compatible endpoint."""

    provider_name = "llm"

    def __init__(self, settings: LLMSettings | None = None, http_client: httpx.Client | None = None):
        self.settings = settings or LLMSettings.from_env()
        self.http_client = http_client

    def _require_config(self) -> None:
        missing = [
            name
            for name, value in (
                ("MAZU_LLM_BASE_URL", self.settings.base_url),
                ("MAZU_LLM_API_KEY", self.settings.api_key),
                ("MAZU_LLM_MODEL", self.settings.model),
            )
            if not value
        ]
        if missing:
            raise BriefingGenerationError(f"Missing LLM configuration: {', '.join(missing)}")

    def _create_http_client(self) -> httpx.Client:
        return httpx.Client(timeout=self.settings.timeout_seconds)

    def _request_payload(
        self,
        area: str,
        risks: list[HazardRisk],
        kg_explanation: dict[str, Any],
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        user_payload = _build_generation_payload(area, risks, kg_explanation, context)
        return {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": self.settings.system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Generate warning briefings for the given weather-risk context. "
                        "Return JSON only in the shape "
                        '{"briefings":[{"industry":"meteorology","zh":"...","en":"...","ar":"...","hazards":["..."]}]}.'
                        "\nContext:\n"
                        + json.dumps(user_payload, ensure_ascii=False)
                    ),
                },
            ],
            "temperature": self.settings.temperature,
            "stream": False,
            "response_format": {"type": "json_object"},
            **self.settings.extra_body,
        }

    def _parse_response(self, response_payload: dict[str, Any]) -> tuple[dict[str, Any], list[IndustryBriefing]]:
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise BriefingGenerationError("LLM response missing choices")
        message = choices[0].get("message", {})
        content = _content_to_text(message.get("content"))
        parsed = _extract_json_payload(content)
        return parsed, _coerce_briefings(parsed)

    def generate(
        self,
        area: str,
        risks: list[HazardRisk],
        kg_explanation: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> BriefingGenerationResult:
        self._require_config()
        payload = self._request_payload(area, risks, kg_explanation, context)
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        started = time.perf_counter()
        client = self.http_client or self._create_http_client()
        close_client = self.http_client is None
        try:
            response = client.post(
                f"{self.settings.base_url.rstrip('/')}{self.settings.chat_path}",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            response_payload = response.json()
            structured_payload, briefings = self._parse_response(response_payload)
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            return BriefingGenerationResult(
                briefings=briefings,
                metadata={
                    "provider": self.provider_name,
                    "model": self.settings.model,
                    "status": "ok",
                    "latency_ms": latency_ms,
                    "prompt_version": "v1",
                },
                raw_response={"request": payload, "response": response_payload, "structured": structured_payload},
            )
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise BriefingGenerationError(f"LLM generation failed: {exc}") from exc
        finally:
            if close_client:
                client.close()


def create_default_briefing_generator() -> BriefingGenerator:
    """Create the default runtime briefing generator."""

    settings = BriefingProviderSettings.from_env()
    if settings.provider == "llm":
        return OpenAICompatibleBriefingGenerator()
    return TemplateBriefingGenerator()


class LlamaIndexRAGBriefingGenerator(TemplateBriefingGenerator):
    """Backward-compatible placeholder alias."""


def build_warning_product(
    area: str,
    risks: list[HazardRisk],
    kg_explanation: dict[str, Any],
    *,
    generator: BriefingGenerator | None = None,
    context: dict[str, Any] | None = None,
) -> WarningProduct:
    """Build a draft warning product from risk and KG outputs."""

    result = (generator or create_default_briefing_generator()).generate(
        area,
        risks,
        kg_explanation,
        context=context,
    )
    return WarningProduct(
        id="mazu-saudi-demo-warning",
        issued_at=datetime.now(timezone.utc),
        area=area,
        risks=risks,
        briefings=result.briefings,
        kg_explanation=kg_explanation,
        generation_metadata=result.metadata,
        llm_raw=result.raw_response,
    )
