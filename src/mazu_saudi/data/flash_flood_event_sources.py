"""Verified flash-flood event-source normalization and merge utilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any

from .flash_flood_labels import FlashFloodEvent, flash_flood_event_table, seed_flash_flood_events

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_slug(value: Any) -> str:
    return _normalize_text(value).lower().replace(" ", "_").replace("/", "_")


def _parse_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    parsed = datetime.fromisoformat(str(value).strip())
    return parsed.date()


def _coalesce(record: Mapping[str, Any], *names: str, default: Any = None) -> Any:
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


def _records_from_input(records: Any) -> list[dict[str, Any]]:
    if pd is not None and isinstance(records, pd.DataFrame):
        return records.to_dict(orient="records")
    if isinstance(records, Iterable) and not isinstance(records, (str, bytes, dict)):
        return [dict(record) for record in records]
    raise TypeError("records must be a pandas.DataFrame or iterable of mapping rows")


def _canonical_event_key(event: FlashFloodEvent) -> tuple[Any, ...]:
    normalized_location = _normalize_slug(event.location_name)
    normalized_geometry = _normalize_text(event.geometry_wkt)
    rounded_latitude = None if event.latitude is None else round(float(event.latitude), 2)
    rounded_longitude = None if event.longitude is None else round(float(event.longitude), 2)
    if normalized_geometry:
        spatial_key: tuple[Any, ...] = ("geometry", normalized_geometry)
    elif normalized_location:
        spatial_key = ("location", normalized_location)
    else:
        spatial_key = ("coordinates", rounded_latitude, rounded_longitude)
    return (
        _normalize_slug(event.hazard_type),
        event.start_date.isoformat(),
        event.end_date.isoformat(),
        _normalize_slug(event.country_code),
        *spatial_key,
    )


def standardize_flash_flood_event_records(
    records: Any,
    *,
    source_name: str,
    validation_status: str = "verified",
    hazard_type: str = "flash_flood",
) -> list[FlashFloodEvent]:
    """Normalize external event rows into the shared FlashFloodEvent contract."""

    events: list[FlashFloodEvent] = []
    for index, record in enumerate(_records_from_input(records), start=1):
        location_name = _coalesce(record, "location_name", "location", "city", "place_name")
        if not _normalize_text(location_name):
            raise ValueError(f"event record {index} is missing a location field")

        start_date = _parse_date(_coalesce(record, "start_date", "date", "event_date"))
        end_date = _parse_date(_coalesce(record, "end_date", "date", "event_end_date", default=start_date))
        source_record_id = _normalize_text(_coalesce(record, "source_record_id", "record_id", "event_id", "id"))
        event_id = _normalize_text(record.get("event_id")) or (
            f"{_normalize_slug(source_name)}_{_normalize_slug(location_name)}_{start_date.isoformat().replace('-', '')}"
        )

        events.append(
            FlashFloodEvent(
                event_id=event_id,
                hazard_type=hazard_type,
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
                source_record_id=source_record_id,
                validation_status=_normalize_text(_coalesce(record, "validation_status", default=validation_status)) or validation_status,
                notes=_normalize_text(record.get("notes")),
            )
        )
    return events


def merge_flash_flood_event_sources(
    *,
    seed_events: list[FlashFloodEvent] | None = None,
    verified_events: list[FlashFloodEvent] | None = None,
) -> list[FlashFloodEvent]:
    """Merge seed and verified events, preferring verified provenance on duplicates."""

    merged: dict[tuple[Any, ...], FlashFloodEvent] = {}
    for event in seed_events or seed_flash_flood_events():
        merged[_canonical_event_key(event)] = event

    for event in verified_events or []:
        key = _canonical_event_key(event)
        current = merged.get(key)
        if current is None:
            merged[key] = event
            continue
        current_status = _normalize_text(current.validation_status).lower()
        incoming_status = _normalize_text(event.validation_status).lower()
        if current_status != "verified" and incoming_status == "verified":
            merged[key] = event
            continue
        if event.source_record_id and not current.source_record_id:
            merged[key] = event

    return sorted(merged.values(), key=lambda item: (item.start_date.isoformat(), item.location_name, item.event_id))


def flash_flood_event_table_from_sources(
    verified_records: Any | None = None,
    *,
    source_name: str = "verified_source",
    seed_events: list[FlashFloodEvent] | None = None,
):
    """Return a combined event table from seed events and optional verified records."""

    verified_events = (
        standardize_flash_flood_event_records(verified_records, source_name=source_name)
        if verified_records is not None
        else []
    )
    return flash_flood_event_table(merge_flash_flood_event_sources(seed_events=seed_events, verified_events=verified_events))
