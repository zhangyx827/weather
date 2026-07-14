"""Map flash-flood event rows to trainable grid-day or province-day labels."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

import numpy as np

from mazu_saudi.config import FlashFloodLabelMappingConfig
from mazu_saudi.data.flash_flood_labels import expand_flash_flood_events_to_daily_table

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


_CONFIDENCE_LEVELS = {"low": 1, "medium": 2, "high": 3}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalize_date_column(series: Any):
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def _confidence_at_least(value: str, minimum: str) -> bool:
    return _CONFIDENCE_LEVELS.get(_normalize_text(value), 0) >= _CONFIDENCE_LEVELS.get(_normalize_text(minimum), 0)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def _resolved_province_name(row: Any, config: FlashFloodLabelMappingConfig) -> str:
    for key in ("province_name", "admin1_name", "region_name", "location_name"):
        value = _normalize_text(row.get(key))
        if not value:
            continue
        return config.location_to_province.get(value, value)
    return ""


def _event_mapping_mode(event: Any, config: FlashFloodLabelMappingConfig) -> str:
    validation_status = _normalize_text(event.get("validation_status"))
    spatial_confidence = _normalize_text(event.get("spatial_confidence"))
    if validation_status not in config.positive_validation_statuses:
        return "unsupported"
    if not _confidence_at_least(spatial_confidence, config.min_spatial_confidence):
        return "uncertain"
    latitude = event.get("latitude")
    longitude = event.get("longitude")
    if pd.notna(latitude) and pd.notna(longitude):
        return "point_buffer"
    if config.province_fallback_enabled and _resolved_province_name(event, config):
        return "province_day"
    return "uncertain"


def _sample_matches_event(sample: Any, event: Any, config: FlashFloodLabelMappingConfig) -> tuple[bool, str | None]:
    mode = _event_mapping_mode(event, config)
    if mode == "point_buffer":
        sample_lat = sample.get("latitude")
        sample_lon = sample.get("longitude")
        if pd.isna(sample_lat) or pd.isna(sample_lon):
            sample_province = _resolved_province_name(sample, config)
            event_province = _resolved_province_name(event, config)
            return (sample_province != "" and sample_province == event_province, "province_day")
        distance_km = _haversine_km(float(sample_lat), float(sample_lon), float(event["latitude"]), float(event["longitude"]))
        return (distance_km <= config.point_buffer_km, "point_buffer")
    if mode == "province_day":
        sample_province = _resolved_province_name(sample, config)
        event_province = _resolved_province_name(event, config)
        return (sample_province != "" and sample_province == event_province, "province_day")
    return (False, None)


def _supports_event_day_negative(sample_columns: Any, day_events: list[dict[str, Any]], config: FlashFloodLabelMappingConfig) -> bool:
    if not config.emit_event_day_negatives:
        return False
    if not day_events:
        return True
    if "latitude" in sample_columns and "longitude" in sample_columns:
        return all(_event_mapping_mode(event, config) == "point_buffer" for event in day_events)
    if any(column in sample_columns for column in ("province_name", "admin1_name", "region_name", "location_name")):
        return all(_event_mapping_mode(event, config) in {"point_buffer", "province_day"} for event in day_events)
    return False


def build_flash_flood_training_labels(
    samples: Any,
    event_daily_table: Any | None = None,
    config: FlashFloodLabelMappingConfig | None = None,
):
    """Attach conservative flash-flood labels to a sample table.

    Required sample columns:
    - ``date``
    Optional spatial columns:
    - ``latitude`` and ``longitude`` for grid-day mapping
    - ``province_name`` / ``admin1_name`` / ``region_name`` for province-day fallback
    """

    if pd is None:
        raise RuntimeError("pandas is required for flash-flood training-label mapping")
    if not isinstance(samples, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(samples)!r}")
    if "date" not in samples.columns:
        raise KeyError("flash-flood label mapping requires a 'date' column in the sample table")

    active_config = config or FlashFloodLabelMappingConfig()
    events = event_daily_table if event_daily_table is not None else expand_flash_flood_events_to_daily_table()
    if not isinstance(events, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for event_daily_table, got {type(events)!r}")
    if "date" not in events.columns:
        raise KeyError("flash-flood event table requires a 'date' column")

    normalized_samples = samples.copy()
    normalized_samples["date"] = _normalize_date_column(normalized_samples["date"])
    if normalized_samples["date"].isna().any():
        raise ValueError("sample table contains invalid 'date' values")

    normalized_events = events.copy()
    normalized_events["date"] = _normalize_date_column(normalized_events["date"])
    if normalized_events["date"].isna().any():
        raise ValueError("event table contains invalid 'date' values")

    events_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in normalized_events.to_dict(orient="records"):
        events_by_date[record["date"]].append(record)

    sample_columns = normalized_samples.columns
    labels: list[float] = []
    label_statuses: list[str] = []
    label_modes: list[str] = []
    matched_event_ids: list[str] = []
    provenance_json: list[str] = []

    for row in normalized_samples.to_dict(orient="records"):
        day_events = events_by_date.get(row["date"], [])
        matches: list[dict[str, Any]] = []
        modes: list[str] = []
        mappable_modes = [_event_mapping_mode(event, active_config) for event in day_events]

        for event in day_events:
            matched, mode = _sample_matches_event(row, event, active_config)
            if matched:
                matches.append(event)
                if mode is not None:
                    modes.append(mode)

        if matches:
            label = 1.0
            label_status = "positive"
            label_mode = ",".join(sorted(set(modes))) or "point_buffer"
        elif not day_events:
            label = 0.0
            label_status = "negative"
            label_mode = "no_event_day"
        elif _supports_event_day_negative(sample_columns, day_events, active_config) and all(mode in {"point_buffer", "province_day"} for mode in mappable_modes):
            label = 0.0
            label_status = "negative"
            label_mode = "outside_event_footprint"
        else:
            label = np.nan
            label_status = "uncertain"
            label_mode = "event_day_unresolved"

        labels.append(label)
        label_statuses.append(label_status)
        label_modes.append(label_mode)
        matched_event_ids.append(",".join(event["event_id"] for event in matches))
        provenance_json.append(
            json.dumps(
                {
                    "date": row["date"],
                    "event_count_for_day": len(day_events),
                    "matched_event_ids": [event["event_id"] for event in matches],
                    "matched_location_names": [event.get("location_name", "") for event in matches],
                    "mapping_modes": sorted(set(modes)),
                    "day_event_mapping_modes": mappable_modes,
                    "point_buffer_km": active_config.point_buffer_km,
                    "province_fallback_enabled": active_config.province_fallback_enabled,
                    "emit_event_day_negatives": active_config.emit_event_day_negatives,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    normalized_samples["hazard_type"] = "flash_flood"
    normalized_samples["label"] = labels
    normalized_samples["label_status"] = label_statuses
    normalized_samples["label_source_mode"] = label_modes
    normalized_samples["matched_event_ids"] = matched_event_ids
    normalized_samples["label_provenance"] = provenance_json
    return normalized_samples
