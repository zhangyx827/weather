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

import numpy as np

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
from .layer4_features import feature_names_for_hazard, optional_feature_names_for_hazard
from .levels import RiskThresholdConfig, probability_to_level
from .ml import OptionalMLAdapter, create_ml_adapter


RiskInput = IndicatorFieldSet | MeteorologicalFeatures
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LAYER4_MODEL_DIR = REPO_ROOT / "models" / "layer4"
FEATURE_LABELS = {
    "temp_c": "气温",
    "tmax_c": "最高气温",
    "tmin_c": "最低气温",
    "vpd_kpa": "VPD",
    "heat_index_c": "热指数",
    "wind_speed_mps": "近地风速",
    "relative_humidity_percent": "相对湿度",
    "sst_celsius": "海温",
    "daily_precip_total": "日累计降水",
    "daily_convective_precip": "对流降水",
    "daily_large_scale_precip": "大尺度降水",
    "cape": "CAPE",
    "pwat": "可降水量",
    "ivt": "整层水汽输送",
    "wind850_speed": "850风速",
    "wind_shear_850_200": "850-200切变",
    "flash_flood_risk": "山洪筛查分数",
    "daily_precip_anomaly": "降水距平",
    "t2m_anomaly_c": "气温距平",
    "tmax_anomaly_c": "最高温距平",
    "heatwave_day_flag": "热浪日标志",
    "heatwave_duration_days": "热浪持续日数",
}


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


def _default_runtime_source_status(features: RiskInput) -> tuple[str, dict[str, Any]]:
    if isinstance(features, IndicatorFieldSet):
        return features.source_status or "normal", dict(features.degradation_metadata)
    return "normal", {}


def _standard_indicator_values(features: RiskInput) -> dict[str, Any]:
    temp = _value(features, "t2m_c", legacy="temp_c")
    rh = _value(features, "rh2m", legacy="rh_percent")
    wind = _value(features, "wind10_speed", legacy="wind_speed_mps")
    vpd = _value(features, "vpd_kpa")
    if is_missing(vpd):
        vpd = compute_vpd_kpa(temp, rh)
    heat_index = _value(features, "heat_index_c")
    if is_missing(heat_index):
        heat_index = compute_heat_index_c(temp, rh)
    return {
        "temp_c": None if is_missing(temp) else float(temp),
        "tmax_c": None if is_missing(_value(features, "tmax_c")) else float(_value(features, "tmax_c")),
        "tmin_c": None if is_missing(_value(features, "tmin_c")) else float(_value(features, "tmin_c")),
        "vpd_kpa": None if is_missing(vpd) else float(vpd),
        "heat_index_c": None if is_missing(heat_index) else float(heat_index),
        "wind_speed_mps": None if is_missing(wind) else float(wind),
        "relative_humidity_percent": None if is_missing(rh) else float(rh),
        "sst_celsius": None if is_missing(_value(features, "sst_celsius")) else float(_value(features, "sst_celsius")),
        "daily_precip_total": None if is_missing(_value(features, "daily_precip_total")) else float(_value(features, "daily_precip_total")),
        "daily_convective_precip": None if is_missing(_value(features, "daily_convective_precip")) else float(_value(features, "daily_convective_precip")),
        "daily_large_scale_precip": None if is_missing(_value(features, "daily_large_scale_precip")) else float(_value(features, "daily_large_scale_precip")),
        "cape": None if is_missing(_value(features, "cape")) else float(_value(features, "cape")),
        "pwat": None if is_missing(_value(features, "pwat", legacy="pwat_mm")) else float(_value(features, "pwat", legacy="pwat_mm")),
        "ivt": None if is_missing(_value(features, "ivt", legacy="ivt_kg_m_s")) else float(_value(features, "ivt", legacy="ivt_kg_m_s")),
        "wind850_speed": None if is_missing(_value(features, "wind850_speed")) else float(_value(features, "wind850_speed")),
        "wind_shear_850_200": None if is_missing(_value(features, "wind_shear_850_200")) else float(_value(features, "wind_shear_850_200")),
        "flash_flood_risk": None if is_missing(_value(features, "flash_flood_risk")) else float(_value(features, "flash_flood_risk")),
        "daily_precip_anomaly": None if is_missing(_value(features, "daily_precip_anomaly")) else float(_value(features, "daily_precip_anomaly")),
        "t2m_anomaly_c": None if is_missing(_value(features, "t2m_anomaly_c")) else float(_value(features, "t2m_anomaly_c")),
        "tmax_anomaly_c": None if is_missing(_value(features, "tmax_anomaly_c")) else float(_value(features, "tmax_anomaly_c")),
        "heatwave_day_flag": None if is_missing(_value(features, "heatwave_day_flag")) else float(_value(features, "heatwave_day_flag")),
        "heatwave_duration_days": None if is_missing(_value(features, "heatwave_duration_days")) else float(_value(features, "heatwave_duration_days")),
    }


def _layer4_feature_vector(features: RiskInput, hazard_type: str) -> np.ndarray | None:
    values = _standard_indicator_values(features)
    ordered = [values.get(name) for name in feature_names_for_hazard(hazard_type)]
    optional = set(optional_feature_names_for_hazard(hazard_type))
    ordered = [np.nan if value is None and name in optional else value for name, value in zip(feature_names_for_hazard(hazard_type), ordered)]
    if any(value is None for value in ordered):
        return None
    return np.asarray(ordered, dtype=np.float32)


def _layer4_feature_vector_for_names(features: RiskInput, feature_names: list[str] | tuple[str, ...]) -> np.ndarray | None:
    values = _standard_indicator_values(features)
    optional_names = set().union(*(optional_feature_names_for_hazard(name) for name in ("extreme_heat", "dry_heat_agriculture", "flash_flood")))
    ordered: list[float] = []
    for name in feature_names:
        value = values.get(name)
        if value is None and name in optional_names:
            ordered.append(np.nan)
            continue
        if value is None:
            return None
        ordered.append(float(value))
    return np.asarray(ordered, dtype=np.float32)


def _top_shap_factors(shap_summary: dict[str, Any]) -> list[str]:
    factors = []
    for item in shap_summary.get("top_features", []):
        name = str(item.get("feature", ""))
        label = FEATURE_LABELS.get(name, name)
        contribution = float(item.get("contribution", 0.0))
        direction = "抬升" if contribution >= 0 else "压低"
        factors.append(f"{label}贡献{direction}风险({contribution:.3f})")
    return factors


class BaseRiskModel(ABC):
    """Base interface for rule or ML risk models."""

    hazard_type: str
    model_name = "rule_screening_v1"
    model_version = "v1"
    model_family = "rule"
    inference_mode = "rule"

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
        source_status, degradation_metadata = _default_runtime_source_status(features)

        return HazardRisk(
            hazard_type=self.hazard_type,
            risk_probability=p,
            risk_level=probability_to_level(p, self.threshold_config),
            contributing_factors=factors or ["未触发显著阈值"],
            grid=_input_grid(features),
            valid_time=_input_valid_time(features),
            model_name=self.model_name,
            model_version=self.model_version,
            metadata={"threshold_config": self.threshold_config.to_dict(), "model_family": self.model_family, **self.metadata},
            evidence={"model_version": self.model_version, "model_family": self.model_family, "inference_mode": self.inference_mode, **evidence},
            indicator_evidence=indicator_evidence,
            model_family=self.model_family,
            inference_mode=self.inference_mode,
            source_status=source_status,
            degradation_metadata=degradation_metadata,
            shap_summary=dict(evidence.get("shap_summary", evidence.get("shap", {}))),
        )


class RuleBasedRiskModel(BaseRiskModel):
    """Base class for deterministic threshold/rule models."""


class MLBackedRiskModel(BaseRiskModel):
    """Placeholder ML-backed risk model using an optional adapter."""

    hazard_type = "ml_backed_hazard"
    model_name = "ml_backed_placeholder"
    model_family = "ml"

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


class LightGBMHybridRiskModel(BaseRiskModel):
    """Use a LightGBM booster when available, otherwise degrade to a rule model."""

    model_family = "lightgbm"
    model_name = "lightgbm_hybrid_v1"

    def __init__(
        self,
        hazard_type: str,
        fallback_model: BaseRiskModel,
        model_path: str | Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hazard_type = hazard_type
        self.fallback_model = fallback_model
        self.model_path = Path(model_path)
        self.adapter: OptionalMLAdapter | None = None
        self.metadata.update({"fallback_model": fallback_model.model_name, "model_path": str(self.model_path)})

    def _load_adapter(self) -> OptionalMLAdapter | None:
        if self.adapter is not None:
            return self.adapter
        if not self.model_path.exists():
            return None
        try:
            adapter = create_ml_adapter("lightgbm")
            adapter.load_model(self.model_path)
        except Exception:
            return None
        self.adapter = adapter
        return adapter

    def predict(self, features: RiskInput) -> HazardRisk:
        fallback = self.fallback_model.predict(features)
        adapter = self._load_adapter()
        trained_feature_names = list(adapter.metadata.get("feature_names", [])) if adapter is not None else []
        feature_vector = (
            _layer4_feature_vector_for_names(features, trained_feature_names)
            if trained_feature_names
            else _layer4_feature_vector(features, self.hazard_type)
        )
        fallback_reason = None
        if adapter is None:
            fallback_reason = "missing_model_file"
        elif feature_vector is None:
            fallback_reason = "incomplete_ml_features" if not trained_feature_names else "incompatible_model_features"
        if fallback_reason is not None:
            fallback.model_family = self.model_family
            fallback.inference_mode = "degraded_rule_fallback"
            fallback.source_status = "degraded" if adapter is None else fallback.source_status
            fallback.degradation_metadata = {
                **fallback.degradation_metadata,
                "fallback_reason": fallback_reason,
                "fallback_model": self.fallback_model.model_name,
            }
            if trained_feature_names:
                fallback.degradation_metadata["trained_feature_names"] = trained_feature_names
            fallback.evidence.update(
                {
                    "model_family": self.model_family,
                    "inference_mode": "degraded_rule_fallback",
                    "degradation_metadata": dict(fallback.degradation_metadata),
                }
            )
            return fallback

        probability = adapter.predict_proba(feature_vector)
        shap_summary = adapter.shap_explain(feature_vector)
        factors = _top_shap_factors(shap_summary) or list(fallback.contributing_factors)
        indicator_values = _standard_indicator_values(features)
        risk = self._risk(
            features,
            probability,
            factors,
            {
                "shap_summary": shap_summary,
                "indicator_evidence": {
                    **dict(fallback.indicator_evidence),
                    **{key: value for key, value in indicator_values.items() if value is not None},
                },
                "fallback_model": self.fallback_model.model_name,
                "trained_feature_names": trained_feature_names or list(feature_names_for_hazard(self.hazard_type)),
                "feature_source_contract_version": "layer4_v2",
            },
        )
        risk.evidence["degradation_metadata"] = {}
        risk.degradation_metadata = {}
        risk.source_status = fallback.source_status
        risk.valid_time = fallback.valid_time
        risk.grid = fallback.grid
        return risk


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

    flash_flood_rule = FlashFloodRiskModel()
    extreme_heat_rule = ExtremeHeatRiskModel()
    dry_heat_rule = DryHeatStressRiskModel()
    return [
        LightGBMHybridRiskModel("flash_flood", flash_flood_rule, DEFAULT_LAYER4_MODEL_DIR / "flash_flood.txt"),
        LightGBMHybridRiskModel("extreme_heat", extreme_heat_rule, DEFAULT_LAYER4_MODEL_DIR / "extreme_heat.txt"),
        LightGBMHybridRiskModel("dry_heat_agriculture", dry_heat_rule, DEFAULT_LAYER4_MODEL_DIR / "dry_heat_stress.txt"),
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
