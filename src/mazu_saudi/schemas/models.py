"""Core schemas.

The MVP uses dataclasses to remain runnable in minimal Python environments. The
field layout mirrors a future Pydantic model layer and every schema exposes
``to_dict`` for API and demo serialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class RiskLevel(str, Enum):
    """Four-level warning risk scale."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

    @classmethod
    def from_probability(cls, probability: float) -> "RiskLevel":
        """Map a 0-1 probability-like score to a risk level."""

        p = max(0.0, min(1.0, float(probability)))
        if p >= 0.75:
            return cls.EXTREME
        if p >= 0.5:
            return cls.HIGH
        if p >= 0.25:
            return cls.MEDIUM
        return cls.LOW


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {k: _serialize(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    return value


@dataclass
class GridCell:
    """A Saudi-domain grid cell or point location."""

    id: str
    lat: float
    lon: float
    elevation_m: float | None = None
    region: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass
class MeteorologicalFeatures:
    """Meteorological feature vector.

    Units: temperature in degC, RH in percent, precipitation in mm, wind in m/s,
    soil moisture as 0-1 fraction, pressure in hPa, visibility in km.
    """

    grid: GridCell
    valid_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    temp_c: float | None = None
    rh_percent: float | None = None
    dewpoint_c: float | None = None
    precip_1h_mm: float | None = None
    precip_6h_mm: float | None = None
    precip_24h_mm: float | None = None
    wind_speed_mps: float | None = None
    wind_gust_mps: float | None = None
    soil_moisture_frac: float | None = None
    slope_deg: float | None = None
    impervious_frac: float | None = None
    vegetation_index: float | None = None
    pressure_hpa: float | None = None
    visibility_km: float | None = None
    coastal_distance_km: float | None = None
    pwat_mm: float | None = None
    ivt_kg_m_s: float | None = None
    cape_j_kg: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MeteorologicalFeatures":
        grid_payload = payload.get("grid", {})
        grid = grid_payload if isinstance(grid_payload, GridCell) else GridCell(**grid_payload)
        values = dict(payload)
        values["grid"] = grid
        if isinstance(values.get("valid_time"), str):
            values["valid_time"] = datetime.fromisoformat(values["valid_time"].replace("Z", "+00:00"))
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass
class IndicatorFieldSet:
    """Canonical indicator vector for one Saudi grid cell.

    The ``values`` keys intentionally match indicator NetCDF variable names.
    This is the primary risk/agent contract for processed indicator products.
    """

    grid: GridCell
    valid_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    values: dict[str, float | int | None] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    source: str | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
    source_status: str = "normal"
    primary_source_id: str | None = None
    secondary_source_ids: list[str] = field(default_factory=list)
    grounding_gap: dict[str, Any] = field(default_factory=dict)
    degradation_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IndicatorFieldSet":
        grid_payload = payload.get("grid", {})
        grid = grid_payload if isinstance(grid_payload, GridCell) else GridCell(**grid_payload)
        valid_time = payload.get("valid_time")
        if isinstance(valid_time, str):
            valid_time = datetime.fromisoformat(valid_time.replace("Z", "+00:00"))
        elif valid_time is None:
            valid_time = datetime.now(timezone.utc)
        return cls(
            grid=grid,
            valid_time=valid_time,
            values=dict(payload.get("values", {})),
            units=dict(payload.get("units", {})),
            source=payload.get("source"),
            source_metadata=dict(payload.get("source_metadata", {})),
            source_status=payload.get("source_status", "normal"),
            primary_source_id=payload.get("primary_source_id"),
            secondary_source_ids=list(payload.get("secondary_source_ids", [])),
            grounding_gap=dict(payload.get("grounding_gap", {})),
            degradation_metadata=dict(payload.get("degradation_metadata", {})),
        )

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass
class ForecastField:
    """Standard forecast field container."""

    provider: str
    variable: str
    units: str
    valid_time: datetime
    values: list[float]
    grid: list[GridCell]
    metadata: dict[str, Any] = field(default_factory=dict)
    provider_role: str = "deterministic"
    provider_status: str = "ready"
    source_status: str = "normal"
    degradation_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass
class HazardRisk:
    """Risk model output for one hazard."""

    hazard_type: str
    risk_probability: float
    risk_level: RiskLevel
    contributing_factors: list[str]
    grid: GridCell | None = None
    valid_time: datetime | None = None
    model_name: str = "rule_screening"
    model_version: str = "v1"
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    indicator_evidence: dict[str, Any] = field(default_factory=dict)
    model_family: str = "rule"
    inference_mode: str = "rule"
    source_status: str = "normal"
    degradation_metadata: dict[str, Any] = field(default_factory=dict)
    shap_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass
class ExposureObject:
    """An exposed asset, population group, ecosystem, or infrastructure object."""

    id: str
    name: str
    category: str
    location: GridCell | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass
class IndustryBriefing:
    """Warning text for one service industry."""

    industry: str
    zh: str
    en: str
    ar: str
    hazards: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass
class WarningProduct:
    """Compiled warning product with risks, KG explanation, and briefings."""

    id: str
    issued_at: datetime
    area: str
    risks: list[HazardRisk]
    briefings: list[IndustryBriefing]
    kg_explanation: dict[str, Any] = field(default_factory=dict)
    status: str = "draft"
    generation_metadata: dict[str, Any] = field(default_factory=dict)
    llm_raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)
