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
            "makkah region": "makkah",
            "makkah high altitude regions": "makkah",
            "makkah high-altitude regions": "makkah",
            "taif": "makkah",
            "maysan": "makkah",
            "rabigh": "makkah",
            "riyadh": "riyadh",
            "riyadh region": "riyadh",
            "eastern": "eastern province",
            "eastern province": "eastern province",
            "eastern region": "eastern province",
            "dammam": "eastern province",
            "medina": "madinah",
            "madinah": "madinah",
            "madinah province": "madinah",
            "al madinah region": "madinah",
            "yanbu": "madinah",
            "al hanakiyah": "madinah",
            "al ula": "madinah",
            "al-ula": "madinah",
            "al ula region": "madinah",
            "al-ula region": "madinah",
            "fadhlan valley": "madinah",
            "khaybar": "madinah",
            "hail": "hail",
            "hail region": "hail",
            "ha'il": "hail",
            "ha'il region": "hail",
            "hayel region": "hail",
            "qassim": "qassim",
            "qaseem": "qassim",
            "qassim region": "qassim",
            "al-qassim region": "qassim",
            "jazan": "jazan",
            "jizan": "jazan",
            "jazan region": "jazan",
            "tabuk": "tabuk",
            "tabuk region": "tabuk",
            "duba": "tabuk",
            "wajh": "tabuk",
            "umluj": "tabuk",
            "najran": "najran",
            "najran region": "najran",
            "sharurah": "najran",
            "asir": "asir",
            "aseer": "asir",
            "'asir region": "asir",
            "abha": "asir",
            "khamis mushait": "asir",
            "al bahah": "al bahah",
            "al baha": "al bahah",
            "al-baha": "al bahah",
            "baha": "al bahah",
            "al bahah region": "al bahah",
            "al jawf": "al jawf",
            "jawf": "al jawf",
            "al jawf region": "al jawf",
            "northern borders": "northern borders",
            "northern border province": "northern borders",
            "northern borders region": "northern borders",
            "arar": "northern borders",
            "rumah": "riyadh",
            "huraymila": "riyadh",
            "turaif": "northern borders",
            "alula": "madinah",
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


@dataclass(frozen=True)
class DustStormLabelMappingConfig:
    emit_event_day_negatives: bool = True
    positive_validation_statuses: tuple[str, ...] = ("verified",)
    location_aliases: dict[str, str] = field(
        default_factory=lambda: {
            "asir": "asir",
            "aseer": "asir",
            "dammam": "eastern_province",
            "eastern": "eastern_province",
            "eastern province": "eastern_province",
            "eastern_region": "eastern_province",
            "easternprovince": "eastern_province",
            "hafar al-batin": "eastern_province",
            "hail": "hail",
            "hafar al batin": "eastern_province",
            "hejaz": "hijaz",
            "hijaz": "hijaz",
            "jizan": "jizan",
            "madinah": "madinah",
            "medina": "madinah",
            "najran": "najran",
            "northern borders": "northern_borders",
            "northern_borders": "northern_borders",
            "qassim": "qassim",
            "qaseem": "qassim",
            "rafha": "northern_borders",
            "riyadh": "riyadh",
        }
    )

    @classmethod
    def from_env(cls) -> "DustStormLabelMappingConfig":
        return cls(
            emit_event_day_negatives=_env_flag("MAZU_DUST_STORM_EMIT_EVENT_DAY_NEGATIVES", True),
            positive_validation_statuses=_env_csv_set("MAZU_DUST_STORM_POSITIVE_VALIDATION_STATUSES", ("verified",)),
        )
