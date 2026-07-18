"""Reusable audit helpers for flash-flood supervision summaries."""

from __future__ import annotations


def summarize_flash_flood_supervision_quality(
    *,
    total_rows: int,
    positive_rows: int,
    negative_rows: int,
    uncertain_rows: int = 0,
    rows_with_matched_event_ids: int = 0,
    geometry_positive_rows: int = 0,
    outside_event_footprint_negative_rows: int = 0,
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
    safe_outside_negative = max(outside_event_footprint_negative_rows, 0)

    warnings: list[str] = []
    if safe_total <= 0:
        warnings.append("no_rows")
    if safe_positive <= 0:
        warnings.append("no_positive_labels")
    if safe_matched <= 0:
        warnings.append("no_matched_event_rows")
    if safe_positive > 0 and safe_geometry <= 0:
        warnings.append("no_geometry_backed_positives")
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
        "outside_event_footprint_negative_fraction_of_negatives": (safe_outside_negative / safe_negative) if safe_negative else None,
    }
    if event_group_count is not None:
        summary["event_group_count"] = int(event_group_count)
    if fallback_date_group_count is not None:
        summary["fallback_date_group_count"] = int(fallback_date_group_count)
    if rows_using_fallback_date_groups is not None:
        summary["fallback_date_row_fraction"] = (max(rows_using_fallback_date_groups, 0) / safe_total) if safe_total else None
    return summary
