"""Configuration for hazard label generation workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_csv_set(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return tuple(values) or default


@dataclass(frozen=True)
class FlashFloodLabelMappingConfig:
    point_buffer_km: float = 25.0
    province_fallback_enabled: bool = True
    emit_event_day_negatives: bool = False
    min_spatial_confidence: str = "medium"
    positive_validation_statuses: tuple[str, ...] = ("seed", "verified")
    location_to_province: dict[str, str] = field(
        default_factory=lambda: {
            "jeddah": "makkah",
            "jedda": "makkah",
            "mecca": "makkah",
            "makkah": "makkah",
            "makkah province": "makkah",
        }
    )

    @classmethod
    def from_env(cls) -> "FlashFloodLabelMappingConfig":
        return cls(
            point_buffer_km=_env_float("MAZU_FLASH_FLOOD_POINT_BUFFER_KM", 25.0),
            province_fallback_enabled=_env_flag("MAZU_FLASH_FLOOD_PROVINCE_FALLBACK_ENABLED", True),
            emit_event_day_negatives=_env_flag("MAZU_FLASH_FLOOD_EMIT_EVENT_DAY_NEGATIVES", False),
            min_spatial_confidence=os.getenv("MAZU_FLASH_FLOOD_MIN_SPATIAL_CONFIDENCE", "medium").strip().lower() or "medium",
            positive_validation_statuses=_env_csv_set("MAZU_FLASH_FLOOD_POSITIVE_VALIDATION_STATUSES", ("seed", "verified")),
        )
