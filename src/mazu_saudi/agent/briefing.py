"""Template warning briefing generation."""

from __future__ import annotations

from datetime import datetime, timezone

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


def generate_industry_briefings(risks: list[HazardRisk]) -> list[IndustryBriefing]:
    """Generate Chinese briefings plus English/Arabic placeholders."""

    significant = [r for r in risks if r.risk_level.value in {"medium", "high", "extreme"}]
    source = significant or risks
    hazard_summary = "；".join(f"{r.hazard_type}:{r.risk_level.value}({r.risk_probability:.2f})" for r in source)
    max_factors = "；".join({factor for risk in source for factor in risk.contributing_factors[:2]})
    briefings = []
    for industry in INDUSTRIES:
        zh = (
            f"{INDUSTRY_NAMES_ZH[industry]}预警简报：当前识别风险为 {hazard_summary}。"
            f"主要依据：{max_factors or '暂无显著触发因子'}。"
            f"建议：{FOCUS[industry]}"
        )
        en = f"Placeholder English briefing for {industry}: hazards={hazard_summary}."
        ar = f"Arabic placeholder briefing for {industry}: hazards={hazard_summary}."
        briefings.append(IndustryBriefing(industry=industry, zh=zh, en=en, ar=ar, hazards=[r.hazard_type for r in source]))
    return briefings


class LlamaIndexRAGBriefingGenerator:
    """Placeholder for future LlamaIndex RAG briefing generation."""

    def generate(self, risks: list[HazardRisk]) -> list[IndustryBriefing]:
        """Use template output until RAG dependencies and corpus are connected."""

        return generate_industry_briefings(risks)


def build_warning_product(area: str, risks: list[HazardRisk], kg_explanation: dict) -> WarningProduct:
    """Build a draft warning product from risk and KG outputs."""

    return WarningProduct(
        id="mazu-saudi-demo-warning",
        issued_at=datetime.now(timezone.utc),
        area=area,
        risks=risks,
        briefings=generate_industry_briefings(risks),
        kg_explanation=kg_explanation,
    )
