"""Map dust-storm event rows to trainable province-day or region-day labels."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

import numpy as np

from mazu_saudi.config import DustStormLabelMappingConfig

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


_LOCATION_COLUMNS = ("region_id", "province_name", "admin1_name", "region_name", "location_name")
_TOKEN_SPLIT_PATTERN = re.compile(r"\s*(?:,|/|;|\band\b|\&)\s*", re.IGNORECASE)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return ""
    return text


def _normalize_location_token(value: Any) -> str:
    token = _normalize_text(value)
    if not token:
        return ""
    token = token.strip("\"'`")
    token = token.replace("-", " ").replace("_", " ")
    token = " ".join(token.split())
    return token


def _normalize_date_column(series: Any):
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def _canonical_location_token(value: Any, config: DustStormLabelMappingConfig) -> str:
    token = _normalize_location_token(value)
    if not token:
        return ""
    mapped = config.location_aliases.get(token)
    if mapped:
        return mapped
    region_ids = config.location_to_region_ids.get(token)
    if region_ids:
        return region_ids[0]
    return token.replace(" ", "_")


def _resolved_sample_location(row: dict[str, Any], config: DustStormLabelMappingConfig) -> tuple[str, str]:
    for column in _LOCATION_COLUMNS:
        value = row.get(column)
        normalized = _normalize_text(value)
        if not normalized:
            continue
        return _canonical_location_token(value, config), column
    return "", ""


def _resolved_event_regions(event: dict[str, Any], config: DustStormLabelMappingConfig) -> list[str]:
    raw_value = event.get("location_name")
    normalized = _normalize_text(raw_value)
    if not normalized:
        return []
    tokens = [token for token in _TOKEN_SPLIT_PATTERN.split(normalized) if token]
    resolved: list[str] = []
    for token in tokens or [normalized]:
        canonical = _canonical_location_token(token, config)
        if canonical and canonical not in resolved:
            resolved.append(canonical)
    return resolved


def build_dust_storm_training_labels(
    samples: Any,
    event_daily_table: Any,
    *,
    config: DustStormLabelMappingConfig | None = None,
):
    """Attach conservative dust-storm labels to a region-day or province-day sample table."""

    if pd is None:
        raise RuntimeError("pandas is required for dust-storm training-label mapping")
    if not isinstance(samples, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(samples)!r}")
    if not isinstance(event_daily_table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for event_daily_table, got {type(event_daily_table)!r}")
    if "date" not in samples.columns:
        raise KeyError("dust-storm label mapping requires a 'date' column in the sample table")
    if "date" not in event_daily_table.columns:
        raise KeyError("dust-storm event table requires a 'date' column")
    if not any(column in samples.columns for column in _LOCATION_COLUMNS):
        raise KeyError(f"dust-storm label mapping requires one location column from {_LOCATION_COLUMNS}")

    active_config = config or DustStormLabelMappingConfig()
    normalized_samples = samples.copy()
    normalized_samples["date"] = _normalize_date_column(normalized_samples["date"])
    if normalized_samples["date"].isna().any():
        raise ValueError("sample table contains invalid 'date' values")

    normalized_events = event_daily_table.copy()
    normalized_events["date"] = _normalize_date_column(normalized_events["date"])
    if normalized_events["date"].isna().any():
        raise ValueError("event table contains invalid 'date' values")

    valid_statuses = {status.lower() for status in active_config.positive_validation_statuses}
    normalized_events = normalized_events[
        normalized_events["validation_status"].astype(str).str.lower().isin(valid_statuses)
    ].copy()

    events_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in normalized_events.to_dict(orient="records"):
        record["_resolved_regions"] = _resolved_event_regions(record, active_config)
        events_by_date[record["date"]].append(record)

    labels: list[float] = []
    label_statuses: list[str] = []
    label_modes: list[str] = []
    matched_event_ids: list[str] = []
    provenance_json: list[str] = []

    for row in normalized_samples.to_dict(orient="records"):
        sample_region, sample_column = _resolved_sample_location(row, active_config)
        day_events = events_by_date.get(row["date"], [])
        matched = [event for event in day_events if sample_region and sample_region in event["_resolved_regions"]]
        any_unresolved = any(not event["_resolved_regions"] for event in day_events)

        if matched:
            label = 1.0
            label_status = "positive"
            label_mode = "region_day_text"
        elif not day_events:
            label = 0.0
            label_status = "negative"
            label_mode = "no_event_day"
        elif sample_region and not any_unresolved and active_config.emit_event_day_negatives:
            label = 0.0
            label_status = "negative"
            label_mode = "outside_event_regions"
        else:
            label = np.nan
            label_status = "uncertain"
            label_mode = "event_day_unresolved"

        labels.append(label)
        label_statuses.append(label_status)
        label_modes.append(label_mode)
        matched_event_ids.append(",".join(str(event["event_id"]) for event in matched))
        provenance_json.append(
            json.dumps(
                {
                    "date": row["date"],
                    "sample_location_column": sample_column,
                    "sample_region_id": sample_region,
                    "event_count_for_day": len(day_events),
                    "matched_event_ids": [str(event["event_id"]) for event in matched],
                    "matched_location_names": [str(event.get("location_name", "")) for event in matched],
                    "resolved_event_regions": {
                        str(event["event_id"]): event["_resolved_regions"] for event in day_events
                    },
                    "emit_event_day_negatives": active_config.emit_event_day_negatives,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    normalized_samples["hazard_type"] = "dust_storm"
    normalized_samples["label"] = labels
    normalized_samples["label_status"] = label_statuses
    normalized_samples["label_source_mode"] = label_modes
    normalized_samples["matched_event_ids"] = matched_event_ids
    normalized_samples["label_provenance"] = provenance_json
    return normalized_samples
