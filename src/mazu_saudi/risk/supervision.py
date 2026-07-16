"""Hazard-specific supervision contracts for Layer-4 training."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


@dataclass(frozen=True)
class HazardSupervisionSpec:
    hazard_type: str
    default_sample_unit: str
    explicit_target_columns: tuple[str, ...]
    explicit_metadata_columns: tuple[str, ...] = ()
    explicit_filter_columns: tuple[str, ...] = ()
    pseudo_target_supported: bool = True


LAYER4_SUPERVISION_SPECS: dict[str, HazardSupervisionSpec] = {
    "extreme_heat": HazardSupervisionSpec(
        hazard_type="extreme_heat",
        default_sample_unit="impact-region-day",
        explicit_target_columns=("impact_level", "impact_count", "label"),
        explicit_metadata_columns=("region_id", "site_cluster_id", "date", "source_name", "source_url", "validation_status"),
        explicit_filter_columns=("label_status", "validation_status"),
    ),
    "dry_heat_agriculture": HazardSupervisionSpec(
        hazard_type="dry_heat_agriculture",
        default_sample_unit="region-season",
        explicit_target_columns=("yield_anomaly", "yield_value", "label"),
        explicit_metadata_columns=(
            "region_id",
            "season",
            "year",
            "crop_type",
            "harvest_area",
            "source_name",
            "source_url",
            "validation_status",
        ),
        explicit_filter_columns=("label_status", "validation_status"),
    ),
    "flash_flood": HazardSupervisionSpec(
        hazard_type="flash_flood",
        default_sample_unit="grid-day",
        explicit_target_columns=("label",),
        explicit_metadata_columns=("date", "latitude", "longitude", "label_source_mode", "matched_event_ids", "label_provenance"),
        explicit_filter_columns=("label_status",),
    ),
}


def supervision_spec_for_hazard(hazard_type: str) -> HazardSupervisionSpec:
    normalized = hazard_type.strip().lower()
    if normalized not in LAYER4_SUPERVISION_SPECS:
        raise ValueError(f"Unsupported Layer-4 hazard type: {hazard_type}")
    return LAYER4_SUPERVISION_SPECS[normalized]


def _require_pandas():
    if pd is None:
        raise RuntimeError("pandas is required for Layer-4 supervision tables")


def find_explicit_target_column(table, hazard_type: str) -> str | None:
    spec = supervision_spec_for_hazard(hazard_type)
    for name in spec.explicit_target_columns:
        if name in table.columns:
            return name
    return None


def has_explicit_targets(table, hazard_type: str) -> bool:
    return find_explicit_target_column(table, hazard_type) is not None


def explicit_target_payload(table, hazard_type: str):
    _require_pandas()
    spec = supervision_spec_for_hazard(hazard_type)
    target_column = find_explicit_target_column(table, hazard_type)
    if target_column is None:
        return table.copy(), None, None

    working = table.reset_index(drop=True).copy()
    filter_details: dict[str, dict[str, int]] = {}
    for column in spec.explicit_filter_columns:
        if column not in working.columns:
            continue
        series = working[column].astype(str).str.lower()
        counts = series.value_counts(dropna=False).to_dict()
        filter_details[column] = {str(key): int(value) for key, value in counts.items()}
        if column == "label_status":
            working = working[series.isin(("positive", "negative"))].copy()
        elif column == "validation_status":
            working = working[~series.isin(("rejected", "invalid"))].copy()

    labels = pd.to_numeric(working[target_column], errors="coerce")
    explicit_mask = labels.notna()
    working = working.loc[explicit_mask].copy()
    labels = labels.loc[explicit_mask].astype("float32")
    metadata = {
        "target_column": target_column,
        "sample_unit": spec.default_sample_unit,
        "filter_details": filter_details,
    }
    return working.reset_index(drop=True), labels.reset_index(drop=True), metadata
