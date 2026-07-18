"""Reusable audit helpers for flash-flood supervision summaries."""

from __future__ import annotations

import json
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


_EXPLICIT_GEOMETRY_SOURCES = {"source_geometry", "derived_point_buffer", "geometry_wkt"}
_BOUNDARY_GEOMETRY_SOURCES = {"province_boundary"}


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


def _positive_geometry_source_flags(label_table: Any) -> list[tuple[bool, bool]]:
    if pd is None or not isinstance(label_table, pd.DataFrame):
        return []
    if "label_status" not in label_table.columns:
        return []

    positive = label_table.loc[label_table["label_status"].astype(str) == "positive"].copy()
    if positive.empty:
        return []

    provenance_values = positive["label_provenance"].map(_safe_json_loads) if "label_provenance" in positive.columns else []
    source_modes = positive["label_source_mode"].astype(str) if "label_source_mode" in positive.columns else None

    flags: list[tuple[bool, bool]] = []
    for index, payload in enumerate(provenance_values):
        matched_geometry_sources = {
            str(value).strip()
            for value in payload.get("matched_geometry_sources", [])
            if str(value).strip()
        }
        matched_geometry_wkts = [str(value).strip() for value in payload.get("matched_geometry_wkts", []) if str(value).strip()]
        if matched_geometry_sources or matched_geometry_wkts:
            flags.append(
                (
                    bool(matched_geometry_sources & _BOUNDARY_GEOMETRY_SOURCES),
                    bool(matched_geometry_sources & _EXPLICIT_GEOMETRY_SOURCES),
                )
            )
            continue

        mode_tokens: set[str] = set()
        if source_modes is not None:
            mode_tokens = {token.strip() for token in str(source_modes.iloc[index]).split(",") if token.strip()}
        flags.append(("province_day" in mode_tokens, bool(mode_tokens & {"geometry_wkt", "point_buffer"})))

    return flags


def count_flash_flood_geometry_backed_positive_rows(label_table: Any) -> int:
    """Count positive label rows whose provenance retains geometry-backed evidence."""

    if pd is None or not isinstance(label_table, pd.DataFrame):
        return 0
    if "label_status" not in label_table.columns:
        return 0

    positive = label_table.loc[label_table["label_status"].astype(str) == "positive"].copy()
    if positive.empty:
        return 0

    geometry_mode_rows = 0
    if "label_source_mode" in positive.columns:
        geometry_mode_rows = int(positive["label_source_mode"].astype(str).eq("geometry_wkt").sum())

    if "label_provenance" not in positive.columns:
        return geometry_mode_rows

    geometry_backed_rows = 0
    for payload in positive["label_provenance"].map(_safe_json_loads).tolist():
        matched_geometry_sources = [str(value).strip() for value in payload.get("matched_geometry_sources", []) if str(value).strip()]
        matched_geometry_wkts = [str(value).strip() for value in payload.get("matched_geometry_wkts", []) if str(value).strip()]
        if matched_geometry_sources or matched_geometry_wkts:
            geometry_backed_rows += 1

    return max(geometry_backed_rows, geometry_mode_rows)


def count_flash_flood_boundary_grounded_positive_rows(label_table: Any) -> int:
    """Count positive label rows grounded by province-boundary fallback provenance."""

    return int(sum(boundary for boundary, _ in _positive_geometry_source_flags(label_table)))


def count_flash_flood_explicit_geometry_positive_rows(label_table: Any) -> int:
    """Count positive label rows grounded by explicit geometry or point evidence."""

    return int(sum(explicit for _, explicit in _positive_geometry_source_flags(label_table)))


def summarize_flash_flood_geometry_backed_positive_rows(label_table: Any) -> dict[str, int]:
    """Count positive label rows by the geometry evidence source recorded in provenance."""

    if pd is None or not isinstance(label_table, pd.DataFrame):
        return {}
    if "label_status" not in label_table.columns:
        return {}

    positive = label_table.loc[label_table["label_status"].astype(str) == "positive"].copy()
    if positive.empty:
        return {}

    if "label_provenance" not in positive.columns:
        source_counts: dict[str, int] = {}
        if "label_source_mode" in positive.columns:
            geometry_mode_rows = int(positive["label_source_mode"].astype(str).eq("geometry_wkt").sum())
            if geometry_mode_rows:
                source_counts["geometry_wkt"] = geometry_mode_rows
        return dict(sorted(source_counts.items()))

    source_counts: dict[str, int] = {}

    for payload in positive["label_provenance"].map(_safe_json_loads).tolist():
        matched_geometry_sources = [str(value).strip() for value in payload.get("matched_geometry_sources", []) if str(value).strip()]
        if not matched_geometry_sources:
            continue
        for source_name in sorted(set(matched_geometry_sources)):
            source_counts[source_name] = source_counts.get(source_name, 0) + 1

    return dict(sorted(source_counts.items()))


def summarize_flash_flood_supervision_quality(
    *,
    total_rows: int,
    positive_rows: int,
    negative_rows: int,
    uncertain_rows: int = 0,
    rows_with_matched_event_ids: int = 0,
    geometry_positive_rows: int = 0,
    geometry_positive_source_counts: dict[str, int] | None = None,
    boundary_grounded_positive_rows: int = 0,
    explicit_geometry_positive_rows: int = 0,
    outside_event_footprint_negative_rows: int = 0,
    event_day_negative_rows: int = 0,
    event_day_unresolved_rows: int = 0,
    event_group_count: int | None = None,
    fallback_date_group_count: int | None = None,
    rows_using_fallback_date_groups: int | None = None,
) -> dict[str, object]:
    """Summarize whether a flash-flood supervision table looks strong enough to trust."""

    safe_total = max(total_rows, 0)
    safe_positive = max(positive_rows, 0)
    safe_negative = max(negative_rows, 0)
    safe_uncertain = max(uncertain_rows, 0)
    safe_matched = max(rows_with_matched_event_ids, 0)
    safe_geometry = max(geometry_positive_rows, 0)
    safe_boundary_grounded = max(boundary_grounded_positive_rows, 0)
    safe_explicit_geometry = max(explicit_geometry_positive_rows, 0)
    safe_outside_negative = max(outside_event_footprint_negative_rows, 0)
    safe_event_day_negative = max(event_day_negative_rows, 0)
    safe_event_day_unresolved = max(event_day_unresolved_rows, 0)

    warnings: list[str] = []
    if safe_total <= 0:
        warnings.append("no_rows")
    if safe_positive <= 0:
        warnings.append("no_positive_labels")
    if safe_matched <= 0:
        warnings.append("no_matched_event_rows")
    if safe_positive > 0 and safe_geometry <= 0:
        warnings.append("no_geometry_backed_positives")
    if safe_positive > 0 and safe_boundary_grounded > 0 and (safe_boundary_grounded / safe_positive) > 0.9:
        warnings.append("boundary_grounding_dominates")
    if safe_total > 0 and (safe_uncertain / safe_total) > 0.25:
        warnings.append("high_uncertain_fraction")
    if event_group_count is not None and 0 < event_group_count < 5:
        warnings.append("few_event_groups")
    if rows_using_fallback_date_groups is not None and safe_total > 0:
        if (max(rows_using_fallback_date_groups, 0) / safe_total) > 0.5:
            warnings.append("fallback_date_groups_dominate")

    status = "ok"
    if any(code in warnings for code in ("no_rows", "no_positive_labels", "no_matched_event_rows")):
        status = "insufficient"
    elif warnings:
        status = "warning"

    summary: dict[str, object] = {
        "status": status,
        "warnings": warnings,
        "positive_fraction": (safe_positive / safe_total) if safe_total else None,
        "negative_fraction": (safe_negative / safe_total) if safe_total else None,
        "uncertain_fraction": (safe_uncertain / safe_total) if safe_total else None,
        "matched_event_fraction": (safe_matched / safe_total) if safe_total else None,
        "geometry_positive_fraction_of_positives": (safe_geometry / safe_positive) if safe_positive else None,
        "boundary_grounded_positive_rows": safe_boundary_grounded,
        "boundary_grounded_positive_fraction_of_positives": (safe_boundary_grounded / safe_positive) if safe_positive else None,
        "explicit_geometry_positive_rows": safe_explicit_geometry,
        "explicit_geometry_positive_fraction_of_positives": (safe_explicit_geometry / safe_positive) if safe_positive else None,
        "geometry_positive_source_counts": dict(sorted((geometry_positive_source_counts or {}).items())),
        "outside_event_footprint_negative_fraction_of_negatives": (safe_outside_negative / safe_negative) if safe_negative else None,
        "event_day_negative_fraction": (safe_event_day_negative / safe_total) if safe_total else None,
        "event_day_unresolved_fraction": (safe_event_day_unresolved / safe_total) if safe_total else None,
    }
    if event_group_count is not None:
        summary["event_group_count"] = int(event_group_count)
    if fallback_date_group_count is not None:
        summary["fallback_date_group_count"] = int(fallback_date_group_count)
    if rows_using_fallback_date_groups is not None:
        summary["fallback_date_row_fraction"] = (max(rows_using_fallback_date_groups, 0) / safe_total) if safe_total else None
    return summary
