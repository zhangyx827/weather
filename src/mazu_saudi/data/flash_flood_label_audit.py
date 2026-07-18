"""Audit helpers for province-day flash-flood label outputs."""

from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any

from mazu_saudi.config import FlashFloodLabelMappingConfig
from mazu_saudi.data.flash_flood_audit import count_flash_flood_geometry_backed_positive_rows
from mazu_saudi.data.flash_flood_mapping import (
    _event_mapping_mode,
    _normalize_date_column,
    _resolved_province_names,
)

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for flash-flood label audits")


def _normalize_label_table(label_table: Any):
    _require_pandas()
    if not isinstance(label_table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for label_table, got {type(label_table)!r}")
    missing = [column for column in ("date", "label_status", "label_source_mode", "matched_event_ids", "label_provenance") if column not in label_table.columns]
    if missing:
        raise KeyError(f"label_table is missing required columns: {missing}")
    labels = label_table.copy()
    labels["date"] = _normalize_date_column(labels["date"])
    if labels["date"].isna().any():
        raise ValueError("label_table contains invalid 'date' values")
    return labels


def _normalize_event_table(event_daily_table: Any):
    _require_pandas()
    if not isinstance(event_daily_table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for event_daily_table, got {type(event_daily_table)!r}")
    if "date" not in event_daily_table.columns:
        raise KeyError("event_daily_table requires a 'date' column")
    events = event_daily_table.copy()
    events["date"] = _normalize_date_column(events["date"])
    if events["date"].isna().any():
        raise ValueError("event_daily_table contains invalid 'date' values")
    return events


def _safe_json_loads(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _split_csv_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    tokens = [token.strip() for token in str(value).split(",")]
    return [token for token in tokens if token]


def _canonicalize_for_audit(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _is_supported_mapping_mode(value: Any) -> bool:
    return str(value).strip() in {"geometry_wkt", "point_buffer", "province_day"}


def _manual_annotation_residual_tokens(
    location_name: Any,
    resolved_province_names: tuple[str, ...],
    config: FlashFloodLabelMappingConfig,
) -> list[str]:
    canonical_location = _canonicalize_for_audit(location_name)
    if not canonical_location:
        return []
    residual = f" {canonical_location} "
    alias_candidates: set[str] = set()
    resolved_set = {str(province).strip().lower() for province in resolved_province_names if str(province).strip()}
    for province in resolved_province_names:
        canonical_province = _canonicalize_for_audit(province)
        if canonical_province:
            alias_candidates.add(canonical_province)
    for alias, province in config.location_to_province.items():
        if str(province).strip().lower() in resolved_set:
            canonical_alias = _canonicalize_for_audit(alias)
            if canonical_alias:
                alias_candidates.add(canonical_alias)
    for alias_candidate in sorted(alias_candidates, key=len, reverse=True):
        residual = residual.replace(f" {alias_candidate} ", " ")
    residual = re.sub(r"\b(and|or|the|region|regions|province|provinces|governorate|governorates)\b", " ", residual)
    tokens = [token for token in residual.split() if token]
    return tokens


def _classify_unresolved_candidate_event(
    *,
    mapping_mode: str,
    resolved_province_names: tuple[str, ...],
    location_name: Any,
    day_supported_event_count: int,
    day_unsupported_event_count: int,
    config: FlashFloodLabelMappingConfig,
) -> tuple[str, list[str]]:
    if day_supported_event_count > 0 and day_unsupported_event_count > 0:
        return ("mixed_supported_and_unsupported", [])
    if mapping_mode in {"point_buffer", "geometry_wkt"}:
        return ("policy_conservative", [])
    if mapping_mode == "uncertain" and not resolved_province_names:
        return ("source_too_vague", [])
    if mapping_mode == "province_day" and resolved_province_names:
        residual_tokens = _manual_annotation_residual_tokens(location_name, resolved_province_names, config)
        if residual_tokens:
            return ("manual_annotation_candidate", residual_tokens)
        return ("policy_conservative", [])
    return ("source_too_vague", [])


def _has_geometry_evidence(row: Any) -> bool:
    geometry_source = row.get("geometry_source")
    if geometry_source is not None:
        try:
            if pd is not None and pd.notna(geometry_source) and str(geometry_source).strip():
                return True
        except Exception:
            pass
    value = row.get("geometry_wkt")
    if value is None:
        return False
    try:
        if pd is not None and pd.isna(value):
            return False
    except Exception:
        pass
    text = str(value).strip().lower()
    return text not in {"", "nan", "none", "null"}


def audit_flash_flood_province_day_labels(
    label_table: Any,
    *,
    event_daily_table: Any | None = None,
    config: FlashFloodLabelMappingConfig | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """Summarize unresolved province-day rows and the positive rows that did resolve."""

    if top_n <= 0:
        raise ValueError("top_n must be positive")

    labels = _normalize_label_table(label_table)
    active_config = config or FlashFloodLabelMappingConfig()

    unresolved = labels.loc[labels["label_source_mode"].astype(str) == "event_day_unresolved"].copy()
    unresolved["label_provenance_payload"] = unresolved["label_provenance"].map(_safe_json_loads)
    unresolved["event_count_for_day"] = unresolved["label_provenance_payload"].map(
        lambda payload: int(payload.get("event_count_for_day", 0) or 0)
    )
    unresolved["day_event_mapping_modes"] = unresolved["label_provenance_payload"].map(
        lambda payload: [str(mode) for mode in payload.get("day_event_mapping_modes", []) if str(mode).strip()]
    )

    unresolved_mode_counts: Counter[str] = Counter()
    for modes in unresolved["day_event_mapping_modes"].tolist():
        unresolved_mode_counts.update(modes)

    positive = labels.loc[labels["label_status"].astype(str) == "positive"].copy()
    positive_event_counter: Counter[str] = Counter()
    for token_list in positive["matched_event_ids"].map(_split_csv_tokens).tolist():
        positive_event_counter.update(token_list)

    summary: dict[str, Any] = {
        "rows": int(len(labels)),
        "positive_rows": int(len(positive)),
        "geometry_backed_positive_rows": int(count_flash_flood_geometry_backed_positive_rows(labels)),
        "uncertain_rows": int((labels["label_status"].astype(str) == "uncertain").sum()),
        "unresolved_rows": int(len(unresolved)),
        "unresolved_fraction": float(len(unresolved) / len(labels)) if len(labels) else None,
        "unresolved_date_count": int(unresolved["date"].nunique()),
        "unresolved_day_event_mapping_mode_counts": dict(sorted(unresolved_mode_counts.items())),
        "top_unresolved_dates": [
            {"date": str(index), "rows": int(value)}
            for index, value in unresolved["date"].value_counts().head(top_n).items()
        ],
        "top_positive_event_ids": [
            {"event_id": str(event_id), "rows": int(count)}
            for event_id, count in positive_event_counter.most_common(top_n)
        ],
    }

    province_column = next(
        (column for column in ("province_name", "admin1_name", "region_name", "location_name") if column in labels.columns),
        None,
    )
    if province_column is not None and len(unresolved) > 0:
        unresolved_provinces = unresolved[province_column].fillna("").astype(str).str.strip()
        summary["top_unresolved_sample_provinces"] = [
            {"province_name": str(index), "rows": int(value)}
            for index, value in unresolved_provinces[unresolved_provinces.ne("")].value_counts().head(top_n).items()
        ]
    if province_column is not None and len(positive) > 0:
        positive_provinces = positive[province_column].fillna("").astype(str).str.strip()
        summary["top_positive_sample_provinces"] = [
            {"province_name": str(index), "rows": int(value)}
            for index, value in positive_provinces[positive_provinces.ne("")].value_counts().head(top_n).items()
        ]

    if event_daily_table is not None:
        events = _normalize_event_table(event_daily_table)
        candidate_events = events.loc[events["date"].isin(unresolved["date"])].copy()
        if not candidate_events.empty:
            candidate_events["mapping_mode"] = candidate_events.apply(
                lambda row: _event_mapping_mode(row.to_dict(), active_config),
                axis=1,
            )
            candidate_events["day_supported_event_count"] = candidate_events.groupby("date")["mapping_mode"].transform(
                lambda series: int(series.map(_is_supported_mapping_mode).sum())
            )
            candidate_events["day_unsupported_event_count"] = candidate_events.groupby("date")["mapping_mode"].transform(
                lambda series: int((~series.map(_is_supported_mapping_mode)).sum())
            )
            candidate_events["day_event_category"] = candidate_events.apply(
                lambda row: (
                    "mixed_supported_and_unsupported"
                    if int(row["day_supported_event_count"]) > 0 and int(row["day_unsupported_event_count"]) > 0
                    else "supported_only"
                    if int(row["day_supported_event_count"]) > 0
                    else "unsupported_only"
                ),
                axis=1,
            )
            candidate_events["resolved_province_names"] = candidate_events.apply(
                lambda row: _resolved_province_names(row.to_dict(), active_config),
                axis=1,
            )
            candidate_events["resolved_province_name"] = candidate_events["resolved_province_names"].map(lambda names: ",".join(names))
            unresolved_counts_by_date = unresolved["date"].value_counts().to_dict()
            candidate_events["unresolved_rows_for_date"] = candidate_events["date"].map(
                lambda value: int(unresolved_counts_by_date.get(value, 0))
            )
            candidate_events["unresolved_bucket_payload"] = candidate_events.apply(
                lambda row: _classify_unresolved_candidate_event(
                    mapping_mode=str(row["mapping_mode"]),
                    resolved_province_names=tuple(row["resolved_province_names"]),
                    location_name=row.get("location_name"),
                    day_supported_event_count=int(row["day_supported_event_count"]),
                    day_unsupported_event_count=int(row["day_unsupported_event_count"]),
                    config=active_config,
                ),
                axis=1,
            )
            candidate_events["unresolved_bucket"] = candidate_events["unresolved_bucket_payload"].map(lambda payload: payload[0])
            candidate_events["manual_annotation_residual_tokens"] = candidate_events["unresolved_bucket_payload"].map(
                lambda payload: payload[1]
            )
            candidate_events["has_geometry_source"] = candidate_events.apply(_has_geometry_evidence, axis=1)
            candidate_events["has_point_source"] = candidate_events.apply(
                lambda row: pd.notna(row.get("latitude")) and pd.notna(row.get("longitude")),
                axis=1,
            )
            grouped = (
                candidate_events.groupby(
                    [
                        "event_id",
                        "location_name",
                        "mapping_mode",
                        "resolved_province_name",
                        "unresolved_bucket",
                    ],
                    dropna=False,
                )
                .agg(
                    unresolved_rows=("unresolved_rows_for_date", "sum"),
                    unresolved_date_count=("date", "nunique"),
                    first_date=("date", "min"),
                    last_date=("date", "max"),
                    manual_annotation_residual_tokens=("manual_annotation_residual_tokens", "first"),
                    day_geometry_event_count=("has_geometry_source", "sum"),
                    day_point_event_count=("has_point_source", "sum"),
                    day_supported_event_count=("day_supported_event_count", "max"),
                    day_unsupported_event_count=("day_unsupported_event_count", "max"),
                    day_event_category=("day_event_category", "first"),
                )
                .reset_index()
                .sort_values(["unresolved_rows", "unresolved_date_count", "event_id"], ascending=[False, False, True])
                .head(top_n)
            )
            summary["top_unresolved_candidate_events"] = grouped.to_dict(orient="records")
            summary["unresolved_candidate_mapping_mode_counts"] = {
                str(key): int(value) for key, value in candidate_events["mapping_mode"].value_counts(dropna=False).to_dict().items()
            }
            summary["unresolved_day_category_counts"] = {
                str(key): int(value) for key, value in candidate_events["day_event_category"].value_counts(dropna=False).to_dict().items()
            }
            summary["unresolved_candidate_bucket_counts"] = {
                str(key): int(value)
                for key, value in candidate_events["unresolved_bucket"].value_counts(dropna=False).to_dict().items()
            }
            summary["top_unresolved_candidate_events_by_bucket"] = {}
            for bucket_name, bucket_frame in candidate_events.groupby("unresolved_bucket", dropna=False):
                bucket_grouped = (
                    bucket_frame.groupby(
                        [
                            "event_id",
                            "location_name",
                            "mapping_mode",
                            "resolved_province_name",
                            "unresolved_bucket",
                        ],
                        dropna=False,
                    )
                    .agg(
                        unresolved_rows=("unresolved_rows_for_date", "sum"),
                        unresolved_date_count=("date", "nunique"),
                        first_date=("date", "min"),
                        last_date=("date", "max"),
                        manual_annotation_residual_tokens=("manual_annotation_residual_tokens", "first"),
                        day_geometry_event_count=("has_geometry_source", "sum"),
                        day_point_event_count=("has_point_source", "sum"),
                        day_supported_event_count=("day_supported_event_count", "max"),
                        day_unsupported_event_count=("day_unsupported_event_count", "max"),
                        day_event_category=("day_event_category", "first"),
                    )
                    .reset_index()
                    .sort_values(["unresolved_rows", "unresolved_date_count", "event_id"], ascending=[False, False, True])
                    .head(top_n)
                )
                summary["top_unresolved_candidate_events_by_bucket"][str(bucket_name)] = bucket_grouped.to_dict(orient="records")
        else:
            summary["top_unresolved_candidate_events"] = []
            summary["unresolved_candidate_mapping_mode_counts"] = {}
            summary["unresolved_day_category_counts"] = {}
            summary["unresolved_candidate_bucket_counts"] = {}
            summary["top_unresolved_candidate_events_by_bucket"] = {}

    return summary
