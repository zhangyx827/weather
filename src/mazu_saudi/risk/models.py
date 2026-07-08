"""Rule-based hazard risk models.

Each model implements ``predict(features) -> HazardRisk`` and keeps the public
interface narrow so a later LightGBM/XGBoost/SHAP model can replace the rule
engine without changing callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mazu_saudi.indicators import (
    compute_dry_heat_stress_score,
    compute_dust_potential_score,
    compute_flash_flood_screening_score,
    compute_heat_index_c,
    compute_ivt_placeholder,
    compute_pwat_placeholder,
    compute_vpd_kpa,
)
from mazu_saudi.schemas import HazardRisk, MeteorologicalFeatures
from mazu_saudi.utils.math import clamp, is_missing


class BaseRiskModel(ABC):
    """Base interface for rule or ML risk models."""

    hazard_type: str
    model_name = "rule_screening_v1"

    @abstractmethod
    def predict(self, features: MeteorologicalFeatures) -> HazardRisk:
        """Predict one hazard risk."""

    def _risk(self, features: MeteorologicalFeatures, probability: float, factors: list[str], evidence: dict[str, Any]) -> HazardRisk:
        p = clamp(probability)
        from mazu_saudi.risk.levels import probability_to_level

        return HazardRisk(
            hazard_type=self.hazard_type,
            risk_probability=p,
            risk_level=probability_to_level(p),
            contributing_factors=factors or ["未触发显著阈值"],
            grid=features.grid,
            valid_time=features.valid_time,
            model_name=self.model_name,
            evidence=evidence,
        )


class FlashFloodRiskModel(BaseRiskModel):
    """Flash-flood and heavy-rain screening for wadis and urban drainage."""

    hazard_type = "flash_flood"

    def predict(self, features: MeteorologicalFeatures) -> HazardRisk:
        score = compute_flash_flood_screening_score(
            features.precip_1h_mm,
            features.precip_6h_mm,
            features.precip_24h_mm,
            features.slope_deg,
            features.soil_moisture_frac,
            features.impervious_frac,
        )
        factors = []
        if (features.precip_1h_mm or 0) >= 25:
            factors.append("1小时强降水超过山洪初筛阈值")
        if (features.precip_6h_mm or 0) >= 60:
            factors.append("6小时累计降水偏高")
        if (features.slope_deg or 0) >= 12:
            factors.append("地形坡度提高汇流速度")
        if (features.impervious_frac or 0) >= 0.35:
            factors.append("不透水面比例较高")
        return self._risk(features, score, factors, {"screening_score": score})


class ExtremeHeatRiskModel(BaseRiskModel):
    """Extreme heat and heat-health screening."""

    hazard_type = "extreme_heat"

    def predict(self, features: MeteorologicalFeatures) -> HazardRisk:
        hi = compute_heat_index_c(features.temp_c, features.rh_percent)
        temp = 0.0 if is_missing(features.temp_c) else float(features.temp_c)
        hi_value = temp if is_missing(hi) else float(hi)
        probability = max((temp - 38.0) / 12.0, (hi_value - 41.0) / 12.0)
        if temp >= 45:
            probability += 0.2
        factors = []
        if temp >= 42:
            factors.append("气温达到极端高温关注阈值")
        if hi_value >= 45:
            factors.append("热指数显示人体热负荷显著升高")
        if (features.rh_percent or 0) >= 55:
            factors.append("湿度提高体感热风险")
        return self._risk(features, probability, factors, {"heat_index_c": hi_value, "temp_c": temp})


class DryHeatStressRiskModel(BaseRiskModel):
    """Agricultural dry-heat stress screening."""

    hazard_type = "dry_heat_agriculture"

    def predict(self, features: MeteorologicalFeatures) -> HazardRisk:
        score = compute_dry_heat_stress_score(
            features.temp_c,
            features.rh_percent,
            features.wind_speed_mps,
            features.vegetation_index,
        )
        vpd = compute_vpd_kpa(features.temp_c, features.rh_percent)
        factors = []
        if not is_missing(vpd) and float(vpd) >= 4:
            factors.append("VPD偏高，作物蒸散胁迫增强")
        if (features.temp_c or 0) >= 40:
            factors.append("高温增加干热胁迫")
        if (features.wind_speed_mps or 0) >= 8:
            factors.append("较大风速增强蒸发")
        return self._risk(features, score, factors, {"vpd_kpa": None if is_missing(vpd) else float(vpd), "screening_score": score})


class DustPotentialRiskModel(BaseRiskModel):
    """Strong-wind dust emission potential screening."""

    hazard_type = "dust_potential"

    def predict(self, features: MeteorologicalFeatures) -> HazardRisk:
        score = compute_dust_potential_score(
            features.wind_speed_mps,
            features.wind_gust_mps,
            features.soil_moisture_frac,
            features.vegetation_index,
            features.visibility_km,
        )
        factors = []
        if (features.wind_speed_mps or 0) >= 12:
            factors.append("近地面风速达到起沙关注阈值")
        if (features.wind_gust_mps or 0) >= 18:
            factors.append("阵风增强扬沙潜势")
        if (features.soil_moisture_frac or 1) <= 0.12:
            factors.append("土壤干燥利于起沙")
        if (features.visibility_km or 99) <= 5:
            factors.append("能见度下降提示沙尘影响")
        return self._risk(features, score, factors, {"screening_score": score})


class CoastalHumidHeatRiskModel(BaseRiskModel):
    """Coastal humid heat and vapor-transport screening."""

    hazard_type = "coastal_humid_heat"

    def predict(self, features: MeteorologicalFeatures) -> HazardRisk:
        pwat = features.pwat_mm
        if is_missing(pwat):
            pwat = compute_pwat_placeholder(features.temp_c, features.rh_percent, features.pressure_hpa)
        ivt = features.ivt_kg_m_s
        if is_missing(ivt):
            ivt = compute_ivt_placeholder(features.wind_speed_mps or 0.0, pwat)
        hi = compute_heat_index_c(features.temp_c, features.rh_percent)
        coastal_factor = 1.0 - clamp(((features.coastal_distance_km if features.coastal_distance_km is not None else 120.0) - 20.0) / 180.0)
        probability = 0.35 * clamp(((features.rh_percent or 0) - 55.0) / 35.0)
        probability += 0.30 * clamp((float(hi) - 38.0) / 12.0) if not is_missing(hi) else 0.0
        probability += 0.20 * clamp((float(pwat) - 35.0) / 25.0) if not is_missing(pwat) else 0.0
        probability += 0.15 * coastal_factor
        factors = []
        if (features.coastal_distance_km or 999) <= 80:
            factors.append("靠近红海或海湾沿岸，湿热暴露较高")
        if (features.rh_percent or 0) >= 65:
            factors.append("相对湿度较高")
        if not is_missing(ivt) and float(ivt) >= 250:
            factors.append("水汽输送占位指标偏强")
        return self._risk(features, probability, factors, {"pwat_mm": None if is_missing(pwat) else float(pwat), "ivt_kg_m_s": None if is_missing(ivt) else float(ivt)})


def all_default_models() -> list[BaseRiskModel]:
    """Return the five MVP hazard models."""

    return [
        FlashFloodRiskModel(),
        ExtremeHeatRiskModel(),
        DryHeatStressRiskModel(),
        DustPotentialRiskModel(),
        CoastalHumidHeatRiskModel(),
    ]
