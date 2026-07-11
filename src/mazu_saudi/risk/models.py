"""Rule-based hazard risk models.

Each model implements ``predict(features) -> HazardRisk`` and keeps the public
interface narrow so a later LightGBM/XGBoost/SHAP model can replace the rule
engine without changing callers.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
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
from mazu_saudi.schemas import HazardRisk, IndicatorFieldSet, MeteorologicalFeatures
from mazu_saudi.utils.math import clamp, is_missing
from .levels import RiskThresholdConfig, probability_to_level
from .ml import OptionalMLAdapter, create_ml_adapter


RiskInput = IndicatorFieldSet | MeteorologicalFeatures


def _value(record: RiskInput, *indicator_names: str, legacy: str | None = None, default: Any = None) -> Any:
    if isinstance(record, IndicatorFieldSet):
        for name in indicator_names:
            if name in record.values:
                return record.values.get(name)
        return default
    if legacy is not None:
        return getattr(record, legacy, default)
    for name in indicator_names:
        if hasattr(record, name):
            return getattr(record, name)
    return default


def _indicator_evidence(record: RiskInput, names: list[str]) -> dict[str, Any]:
    if not isinstance(record, IndicatorFieldSet):
        return {}
    return {name: record.values.get(name) for name in names if name in record.values}


def _input_grid(record: RiskInput):
    return record.grid


def _input_valid_time(record: RiskInput):
    return record.valid_time


class BaseRiskModel(ABC):
    """Base interface for rule or ML risk models."""

    hazard_type: str
    model_name = "rule_screening_v1"
    model_version = "v1"

    def __init__(
        self,
        threshold_config: RiskThresholdConfig | dict[str, Any] | None = None,
        model_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.threshold_config = (
            threshold_config
            if isinstance(threshold_config, RiskThresholdConfig)
            else RiskThresholdConfig.from_mapping(threshold_config)
        )
        if model_version is not None:
            self.model_version = model_version
        self.metadata = metadata or {}

    @abstractmethod
    def predict(self, features: RiskInput) -> HazardRisk:
        """Predict one hazard risk."""

    def predict_one(self, features: RiskInput | dict[str, Any]) -> HazardRisk:
        """Predict one hazard risk with dict/dataclass input compatibility."""

        if isinstance(features, dict):
            normalized = IndicatorFieldSet.from_dict(features) if "values" in features else MeteorologicalFeatures.from_dict(features)
        else:
            normalized = features
        return self.predict(normalized)

    def predict_batch(self, features_list: list[RiskInput | dict[str, Any]]) -> list[HazardRisk]:
        """Predict a list of feature records."""

        return [self.predict_one(features) for features in features_list]

    def explain(self, features: RiskInput | dict[str, Any]) -> dict[str, Any]:
        """Return contributing factors and model metadata for one prediction."""

        risk = self.predict_one(features)
        return {
            "hazard_type": self.hazard_type,
            "risk_probability": risk.risk_probability,
            "risk_level": risk.risk_level.value,
            "contributing_factors": risk.contributing_factors,
            "model_version": self.model_version,
            "metadata": dict(self.metadata),
        }

    def load_threshold_config(self, path: str | Path) -> None:
        """Load threshold config from JSON or simple YAML."""

        payload = _load_config_mapping(path)
        self.threshold_config = RiskThresholdConfig.from_mapping(payload.get("risk_thresholds", payload))

    def _risk(self, features: RiskInput, probability: float, factors: list[str], evidence: dict[str, Any]) -> HazardRisk:
        p = clamp(probability)
        indicator_evidence = dict(evidence.pop("indicator_evidence", {}))

        return HazardRisk(
            hazard_type=self.hazard_type,
            risk_probability=p,
            risk_level=probability_to_level(p, self.threshold_config),
            contributing_factors=factors or ["未触发显著阈值"],
            grid=_input_grid(features),
            valid_time=_input_valid_time(features),
            model_name=self.model_name,
            model_version=self.model_version,
            metadata={"threshold_config": self.threshold_config.to_dict(), **self.metadata},
            evidence={"model_version": self.model_version, **evidence},
            indicator_evidence=indicator_evidence,
        )


class RuleBasedRiskModel(BaseRiskModel):
    """Base class for deterministic threshold/rule models."""


class MLBackedRiskModel(BaseRiskModel):
    """Placeholder ML-backed risk model using an optional adapter."""

    hazard_type = "ml_backed_hazard"
    model_name = "ml_backed_placeholder"

    def __init__(self, adapter: OptionalMLAdapter | None = None, backend: str = "fallback", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.adapter = adapter or create_ml_adapter(backend)
        self.metadata.update({"backend": self.adapter.backend})

    def train(self, dataset: Any) -> dict[str, Any]:
        return self.adapter.train(dataset)

    def save_model(self, path: str | Path) -> None:
        self.adapter.save_model(path)

    def load_model(self, path: str | Path) -> "MLBackedRiskModel":
        self.adapter.load_model(path)
        return self

    def predict_proba(self, features: RiskInput | dict[str, Any]) -> float:
        return clamp(self.adapter.predict_proba(features))

    def shap_explain(self, features: RiskInput | dict[str, Any]) -> dict[str, Any]:
        return self.adapter.shap_explain(features)

    def predict(self, features: RiskInput) -> HazardRisk:
        probability = self.predict_proba(features)
        explanation = self.shap_explain(features)
        factors = ["ML模型占位接口已执行"] if not explanation.get("values") else list(explanation["values"].keys())
        return self._risk(features, probability, factors, {"shap": explanation})


class FlashFloodRiskModel(RuleBasedRiskModel):
    """Flash-flood and heavy-rain screening for wadis and urban drainage."""

    hazard_type = "flash_flood"

    def predict(self, features: RiskInput) -> HazardRisk:
        p1 = _value(features, "ds10_max_1h", legacy="precip_1h_mm")
        p6 = _value(features, "ds10_max_6h", "precip_6h", legacy="precip_6h_mm")
        p24 = _value(features, "daily_precip_total", "gpm_daily_precip", legacy="precip_24h_mm")
        slope = _value(features, "slope_deg", legacy="slope_deg")
        soil = _value(features, "soil_moisture_frac", legacy="soil_moisture_frac")
        impervious = _value(features, "impervious_frac", legacy="impervious_frac")
        score = compute_flash_flood_screening_score(
            p1,
            p6,
            p24,
            slope,
            soil,
            impervious,
        )
        factors = []
        if (p1 or 0) >= 25:
            factors.append("1小时强降水超过山洪初筛阈值")
        if (p6 or 0) >= 60:
            factors.append("6小时累计降水偏高")
        if (slope or 0) >= 12:
            factors.append("地形坡度提高汇流速度")
        if (impervious or 0) >= 0.35:
            factors.append("不透水面比例较高")
        evidence_names = ["ds10_max_1h", "ds10_max_6h", "daily_precip_total", "gpm_daily_precip", "slope_deg"]
        return self._risk(features, score, factors, {"screening_score": score, "indicator_evidence": _indicator_evidence(features, evidence_names)})


class ExtremeHeatRiskModel(RuleBasedRiskModel):
    """Extreme heat and heat-health screening."""

    hazard_type = "extreme_heat"

    def predict(self, features: RiskInput) -> HazardRisk:
        temp_raw = _value(features, "t2m_c", legacy="temp_c")
        rh_raw = _value(features, "rh2m", legacy="rh_percent")
        hi = _value(features, "heat_index_c")
        if is_missing(hi):
            hi = compute_heat_index_c(temp_raw, rh_raw)
        temp = 0.0 if is_missing(temp_raw) else float(temp_raw)
        hi_value = temp if is_missing(hi) else float(hi)
        probability = max((temp - 38.0) / 12.0, (hi_value - 41.0) / 12.0)
        if temp >= 45:
            probability += 0.2
        factors = []
        if temp >= 42:
            factors.append("气温达到极端高温关注阈值")
        if hi_value >= 45:
            factors.append("热指数显示人体热负荷显著升高")
        if (rh_raw or 0) >= 55:
            factors.append("湿度提高体感热风险")
        evidence_names = ["t2m_c", "rh2m", "heat_index_c", "tmax_c", "hot_day_flag", "hot_night_flag"]
        return self._risk(features, probability, factors, {"heat_index_c": hi_value, "t2m_c": temp, "indicator_evidence": _indicator_evidence(features, evidence_names)})


class DryHeatStressRiskModel(RuleBasedRiskModel):
    """Agricultural dry-heat stress screening."""

    hazard_type = "dry_heat_agriculture"

    def predict(self, features: RiskInput) -> HazardRisk:
        temp = _value(features, "t2m_c", legacy="temp_c")
        rh = _value(features, "rh2m", legacy="rh_percent")
        wind = _value(features, "wind10_speed", legacy="wind_speed_mps")
        vegetation = _value(features, "vegetation_index", legacy="vegetation_index")
        score = compute_dry_heat_stress_score(
            temp,
            rh,
            wind,
            vegetation,
        )
        vpd = _value(features, "vpd_kpa")
        if is_missing(vpd):
            vpd = compute_vpd_kpa(temp, rh)
        factors = []
        if not is_missing(vpd) and float(vpd) >= 4:
            factors.append("VPD偏高，作物蒸散胁迫增强")
        if (temp or 0) >= 40:
            factors.append("高温增加干热胁迫")
        if (wind or 0) >= 8:
            factors.append("较大风速增强蒸发")
        evidence_names = ["t2m_c", "rh2m", "vpd_kpa", "wind10_speed", "bowen_ratio"]
        return self._risk(features, score, factors, {"vpd_kpa": None if is_missing(vpd) else float(vpd), "screening_score": score, "indicator_evidence": _indicator_evidence(features, evidence_names)})


class DustPotentialRiskModel(RuleBasedRiskModel):
    """Strong-wind dust emission potential screening."""

    hazard_type = "dust_potential"

    def predict(self, features: RiskInput) -> HazardRisk:
        wind = _value(features, "wind10_speed", legacy="wind_speed_mps")
        gust = _value(features, "wind_gust_mps", legacy="wind_gust_mps")
        soil = _value(features, "soil_moisture_frac", legacy="soil_moisture_frac")
        vegetation = _value(features, "vegetation_index", legacy="vegetation_index")
        visibility = _value(features, "visibility_km", legacy="visibility_km")
        score = compute_dust_potential_score(
            wind,
            gust,
            soil,
            vegetation,
            visibility,
        )
        dust_risk = _value(features, "dust_risk_proxy")
        if not is_missing(dust_risk):
            score = max(score, float(dust_risk))
        factors = []
        if (wind or 0) >= 12:
            factors.append("近地面风速达到起沙关注阈值")
        if (gust or 0) >= 18:
            factors.append("阵风增强扬沙潜势")
        if (soil or 1) <= 0.12:
            factors.append("土壤干燥利于起沙")
        if (visibility or 99) <= 5:
            factors.append("能见度下降提示沙尘影响")
        evidence_names = ["wind10_speed", "strong_wind_flag", "dust_aod", "dust_surface_mass", "dust_risk_proxy"]
        return self._risk(features, score, factors, {"screening_score": score, "indicator_evidence": _indicator_evidence(features, evidence_names)})


class CoastalHumidHeatRiskModel(RuleBasedRiskModel):
    """Coastal humid heat and vapor-transport screening."""

    hazard_type = "coastal_humid_heat"

    def predict(self, features: RiskInput) -> HazardRisk:
        temp = _value(features, "t2m_c", legacy="temp_c")
        rh = _value(features, "rh2m", legacy="rh_percent")
        wind = _value(features, "wind10_speed", legacy="wind_speed_mps", default=0.0)
        pressure_hpa = _value(features, "surface_pressure", legacy="pressure_hpa")
        if not is_missing(pressure_hpa) and pressure_hpa and pressure_hpa > 2000:
            pressure_hpa = float(pressure_hpa) / 100.0
        pwat = _value(features, "pwat", legacy="pwat_mm")
        if is_missing(pwat):
            pwat = compute_pwat_placeholder(temp, rh, pressure_hpa)
        ivt = _value(features, "ivt", legacy="ivt_kg_m_s")
        if is_missing(ivt):
            ivt = compute_ivt_placeholder(wind or 0.0, pwat)
        hi = _value(features, "heat_index_c")
        if is_missing(hi):
            hi = compute_heat_index_c(temp, rh)
        coastal_distance = _value(features, "coastal_distance_km", legacy="coastal_distance_km", default=120.0)
        coastal_factor = 1.0 - clamp(((coastal_distance if coastal_distance is not None else 120.0) - 20.0) / 180.0)
        probability = 0.35 * clamp(((rh or 0) - 55.0) / 35.0)
        probability += 0.30 * clamp((float(hi) - 38.0) / 12.0) if not is_missing(hi) else 0.0
        probability += 0.20 * clamp((float(pwat) - 35.0) / 25.0) if not is_missing(pwat) else 0.0
        probability += 0.15 * coastal_factor
        factors = []
        if (coastal_distance or 999) <= 80:
            factors.append("靠近红海或海湾沿岸，湿热暴露较高")
        if (rh or 0) >= 65:
            factors.append("相对湿度较高")
        if not is_missing(ivt) and float(ivt) >= 250:
            factors.append("水汽输送占位指标偏强")
        evidence_names = ["rh2m", "heat_index_c", "pwat", "ivt", "sst_celsius"]
        return self._risk(features, probability, factors, {"pwat": None if is_missing(pwat) else float(pwat), "ivt": None if is_missing(ivt) else float(ivt), "indicator_evidence": _indicator_evidence(features, evidence_names)})


def all_default_models() -> list[BaseRiskModel]:
    """Return the five MVP hazard models."""

    return [
        FlashFloodRiskModel(),
        ExtremeHeatRiskModel(),
        DryHeatStressRiskModel(),
        DustPotentialRiskModel(),
        CoastalHumidHeatRiskModel(),
    ]


def _load_config_mapping(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except Exception:
        return _parse_simple_yaml(text)
    payload = yaml.safe_load(text)
    return payload or {}


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the simple threshold YAML shape used by examples/tests."""

    result: dict[str, Any] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current = line[:-1].strip()
            result[current] = {}
            continue
        if current and ":" in line:
            key, value = line.strip().split(":", 1)
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                result[current][key] = [float(item.strip()) for item in value.strip("[]").split(",")]
            else:
                result[current][key] = float(value) if value.replace(".", "", 1).isdigit() else value
    return result
