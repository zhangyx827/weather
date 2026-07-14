"""Seed flash-flood label utilities for Layer-4 real supervision."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


@dataclass(frozen=True)
class FlashFloodEvent:
    event_id: str
    hazard_type: str
    start_date: date
    end_date: date
    location_name: str
    country_code: str = "SAU"
    latitude: float | None = None
    longitude: float | None = None
    geometry_wkt: str | None = None
    spatial_confidence: str = "medium"
    temporal_confidence: str = "high"
    source_name: str = "handoff_seed"
    source_url: str = ""
    source_record_id: str = ""
    validation_status: str = "seed"
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["start_date"] = self.start_date.isoformat()
        payload["end_date"] = self.end_date.isoformat()
        return payload


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def seed_flash_flood_events() -> list[FlashFloodEvent]:
    return [
        FlashFloodEvent(
            event_id="ff_jeddah_20091125",
            hazard_type="flash_flood",
            start_date=_parse_date("2009-11-25"),
            end_date=_parse_date("2009-11-25"),
            location_name="Jeddah",
            latitude=21.4858,
            longitude=39.1925,
            source_record_id="handoff-jeddah-20091125",
            notes="Handoff-approved seed event.",
        ),
        FlashFloodEvent(
            event_id="ff_jeddah_20110126",
            hazard_type="flash_flood",
            start_date=_parse_date("2011-01-26"),
            end_date=_parse_date("2011-01-26"),
            location_name="Jeddah",
            latitude=21.4858,
            longitude=39.1925,
            source_record_id="handoff-jeddah-20110126",
            notes="Handoff-approved seed event.",
        ),
        FlashFloodEvent(
            event_id="ff_jeddah_20151117",
            hazard_type="flash_flood",
            start_date=_parse_date("2015-11-17"),
            end_date=_parse_date("2015-11-17"),
            location_name="Jeddah",
            latitude=21.4858,
            longitude=39.1925,
            source_record_id="handoff-jeddah-20151117",
            notes="Handoff-approved seed event.",
        ),
        FlashFloodEvent(
            event_id="ff_jeddah_20171121",
            hazard_type="flash_flood",
            start_date=_parse_date("2017-11-21"),
            end_date=_parse_date("2017-11-21"),
            location_name="Jeddah",
            latitude=21.4858,
            longitude=39.1925,
            source_record_id="handoff-jeddah-20171121",
            notes="Handoff-approved seed event.",
        ),
        FlashFloodEvent(
            event_id="ff_jeddah_20221124",
            hazard_type="flash_flood",
            start_date=_parse_date("2022-11-24"),
            end_date=_parse_date("2022-11-24"),
            location_name="Jeddah",
            latitude=21.4858,
            longitude=39.1925,
            source_record_id="handoff-jeddah-20221124",
            notes="Handoff-approved seed event.",
        ),
        FlashFloodEvent(
            event_id="ff_mecca_20221223",
            hazard_type="flash_flood",
            start_date=_parse_date("2022-12-23"),
            end_date=_parse_date("2022-12-23"),
            location_name="Mecca",
            latitude=21.3891,
            longitude=39.8579,
            source_record_id="handoff-mecca-20221223",
            notes="Handoff-approved seed event.",
        ),
    ]


def flash_flood_event_records(events: list[FlashFloodEvent] | None = None) -> list[dict[str, Any]]:
    return [event.to_record() for event in (events or seed_flash_flood_events())]


def flash_flood_event_table(events: list[FlashFloodEvent] | None = None):
    if pd is None:
        raise RuntimeError("pandas is required for flash-flood event-table creation")
    return pd.DataFrame(flash_flood_event_records(events))


def expand_flash_flood_events_to_daily_records(events: list[FlashFloodEvent] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for event in events or seed_flash_flood_events():
        current = event.start_date
        while current <= event.end_date:
            records.append(
                {
                    "event_id": event.event_id,
                    "hazard_type": event.hazard_type,
                    "date": current.isoformat(),
                    "location_name": event.location_name,
                    "country_code": event.country_code,
                    "latitude": event.latitude,
                    "longitude": event.longitude,
                    "geometry_wkt": event.geometry_wkt,
                    "spatial_confidence": event.spatial_confidence,
                    "temporal_confidence": event.temporal_confidence,
                    "source_name": event.source_name,
                    "source_url": event.source_url,
                    "source_record_id": event.source_record_id,
                    "validation_status": event.validation_status,
                    "label_status": "positive",
                    "notes": event.notes,
                }
            )
            current += timedelta(days=1)
    return records


def expand_flash_flood_events_to_daily_table(events: list[FlashFloodEvent] | None = None):
    if pd is None:
        raise RuntimeError("pandas is required for flash-flood daily table creation")
    return pd.DataFrame(expand_flash_flood_events_to_daily_records(events))
