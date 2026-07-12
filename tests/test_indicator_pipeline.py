from __future__ import annotations

from pathlib import Path

from mazu_saudi.agent.workflow import SaudiWarningPipeline, run_indicator_netcdf_pipeline
from mazu_saudi.data import indicator_point_from_netcdf
from mazu_saudi.risk import all_default_models
from mazu_saudi.schemas import IndicatorFieldSet


ROOT = Path(__file__).resolve().parents[1]
INDICATOR_NC = ROOT / "data" / "processed" / "lightgbm_indicators_nc" / "saudi_indicators_20250101.nc"


def test_indicator_netcdf_point_contract():
    features = indicator_point_from_netcdf(INDICATOR_NC, 24.7, 46.7, region="Riyadh")

    assert isinstance(features, IndicatorFieldSet)
    assert features.grid.id == "saudi_24.7_46.7"
    assert features.valid_time.year == 2025
    assert features.source_status in {"normal", "degraded"}
    assert isinstance(features.source_metadata, dict)
    for name in ("t2m_c", "rh2m", "vpd_kpa", "heat_index_c", "wind10_speed"):
        assert name in features.values
        assert features.values[name] is not None


def test_indicator_models_emit_indicator_evidence():
    features = indicator_point_from_netcdf(INDICATOR_NC, 24.7, 46.7, region="Riyadh")
    risks = [model.predict(features) for model in all_default_models()]

    assert len(risks) == 5
    assert all(risk.indicator_evidence for risk in risks)
    assert any("t2m_c" in risk.indicator_evidence for risk in risks)


def test_indicator_pipeline_reaches_publish_and_kg():
    features = indicator_point_from_netcdf(INDICATOR_NC, 24.7, 46.7, region="Riyadh")
    context = SaudiWarningPipeline().run(features)

    assert context["errors"] == []
    assert context["input_contract"] == "indicator_field_set"
    assert context["trace"][-1] == "publish"
    assert len(context["risks"]) == 5
    assert context["kg_explanation"]["triple_count"] > 0
    assert len(context["publish_payload"]["briefings"]) == 6
    assert all(status["has_indicator_evidence"] for status in context["evidence_status"]["risk_status"].values())
    assert "grounding_metadata_present" in context["evidence_status"]
    assert "model_family" in next(iter(context["evidence_status"]["risk_status"].values()))


def test_run_indicator_netcdf_pipeline_payload():
    payload = run_indicator_netcdf_pipeline(INDICATOR_NC, 24.7, 46.7, region="Riyadh")

    assert payload["errors"] == []
    assert payload["input_contract"] == "indicator_field_set"
    assert len(payload["risks"]) == 5
    assert payload["warning_product"]["status"] == "draft"
