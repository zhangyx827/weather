"""Dust-storm event-source normalization and daily expansion helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip()


def _normalize_slug(value: Any) -> str:
    return _normalize_text(value).lower().replace(" ", "_").replace("/", "_")


def _parse_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.fromisoformat(str(value).strip()).date()


def _coalesce(record: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return default


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    numeric = float(value)
    if numeric != numeric:
        return None
    return numeric


def _is_embedded_header_row(record: dict[str, Any]) -> bool:
    return (
        _normalize_text(record.get("record_id")) == "record_id"
        and _normalize_text(record.get("event_id")) == "event_id"
        and _normalize_text(record.get("hazard_type")) == "hazard_type"
        and _normalize_text(record.get("start_date")) == "start_date"
        and _normalize_text(record.get("end_date")) == "end_date"
    )


@dataclass(frozen=True)
class DustStormEvent:
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
    source_name: str = "user_session_handoff"
    source_url: str = ""
    source_record_id: str = ""
    validation_status: str = "verified"
    severity: str = ""
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["start_date"] = self.start_date.isoformat()
        payload["end_date"] = self.end_date.isoformat()
        return payload


def standardize_dust_storm_event_records(
    records: Any,
    *,
    source_name: str,
    validation_status: str = "verified",
    hazard_type: str = "dust_storm",
) -> list[DustStormEvent]:
    """Normalize external dust-event rows into a shared contract."""

    if pd is not None and isinstance(records, pd.DataFrame):
        source_rows = records.to_dict(orient="records")
    elif isinstance(records, list):
        source_rows = [dict(record) for record in records]
    else:
        raise TypeError("records must be a pandas.DataFrame or list of mapping rows")

    events: list[DustStormEvent] = []
    for index, record in enumerate(source_rows, start=1):
        if _is_embedded_header_row(record):
            continue
        location_name = _coalesce(record, "location_name", "location", "city", "place_name")
        if not _normalize_text(location_name):
            raise ValueError(f"dust event record {index} is missing a location field")

        start_date = _parse_date(_coalesce(record, "start_date", "date", "event_date"))
        end_date = _parse_date(_coalesce(record, "end_date", "date", "event_end_date", default=start_date))
        event_id = _normalize_text(record.get("event_id")) or (
            f"{_normalize_slug(source_name)}_{_normalize_slug(location_name)}_{start_date.isoformat().replace('-', '')}"
        )

        events.append(
            DustStormEvent(
                event_id=event_id,
                hazard_type=_normalize_text(_coalesce(record, "hazard_type", default=hazard_type)) or hazard_type,
                start_date=start_date,
                end_date=end_date,
                location_name=_normalize_text(location_name),
                country_code=_normalize_text(_coalesce(record, "country_code", default="SAU")) or "SAU",
                latitude=_float_or_none(_coalesce(record, "latitude", "lat")),
                longitude=_float_or_none(_coalesce(record, "longitude", "lon", "lng")),
                geometry_wkt=_normalize_text(record.get("geometry_wkt")) or None,
                spatial_confidence=_normalize_text(_coalesce(record, "spatial_confidence", default="medium")) or "medium",
                temporal_confidence=_normalize_text(_coalesce(record, "temporal_confidence", default="high")) or "high",
                source_name=_normalize_text(_coalesce(record, "source_name", default=source_name)) or source_name,
                source_url=_normalize_text(record.get("source_url")),
                source_record_id=_normalize_text(_coalesce(record, "source_record_id", "record_id", "id")),
                validation_status=_normalize_text(_coalesce(record, "validation_status", default=validation_status)) or validation_status,
                severity=_normalize_text(record.get("severity")),
                notes=_normalize_text(record.get("notes")),
            )
        )
    return events


def dust_storm_event_records(events: list[DustStormEvent]) -> list[dict[str, Any]]:
    return [event.to_record() for event in events]


def dust_storm_event_table(events: list[DustStormEvent]):
    if pd is None:
        raise RuntimeError("pandas is required for dust-storm event-table creation")
    return pd.DataFrame(dust_storm_event_records(events))


def expand_dust_storm_events_to_daily_records(events: list[DustStormEvent]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for event in events:
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
                    "severity": event.severity,
                    "label_status": "positive",
                    "notes": event.notes,
                }
            )
            current += timedelta(days=1)
    return records


def expand_dust_storm_events_to_daily_table(events: list[DustStormEvent]):
    if pd is None:
        raise RuntimeError("pandas is required for dust-storm daily table creation")
    return pd.DataFrame(expand_dust_storm_events_to_daily_records(events))
