"""Join flash-flood Layer-4 feature tables with conservative event-derived labels."""

from __future__ import annotations

from typing import Any

from mazu_saudi.config import FlashFloodLabelMappingConfig

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


_PROVINCE_COLUMNS = ("province_name", "admin1_name", "region_name", "location_name")


def _normalize_date(series: Any):
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalize_province_series(series: Any, config: FlashFloodLabelMappingConfig):
    return series.map(lambda value: config.location_to_province.get(_normalize_text(value), _normalize_text(value)))


def _prepare_grid_join_keys(table: Any, *, coordinate_precision: int) -> Any:
    normalized = table.copy()
    normalized["date"] = _normalize_date(normalized["date"])
    normalized["latitude"] = pd.to_numeric(normalized["latitude"], errors="coerce").round(coordinate_precision)
    normalized["longitude"] = pd.to_numeric(normalized["longitude"], errors="coerce").round(coordinate_precision)
    return normalized


def _prepare_province_join_keys(table: Any, column: str, config: FlashFloodLabelMappingConfig) -> Any:
    normalized = table.copy()
    normalized["date"] = _normalize_date(normalized["date"])
    normalized["_province_join_key"] = _normalize_province_series(normalized[column], config)
    return normalized


def build_flash_flood_supervised_training_dataset(
    feature_table: Any,
    label_table: Any,
    *,
    config: FlashFloodLabelMappingConfig | None = None,
    drop_uncertain: bool = True,
    coordinate_precision: int = 4,
):
    """Join flash-flood features with labels using grid-day or province-day keys."""

    if pd is None:
        raise RuntimeError("pandas is required for flash-flood supervised training dataset assembly")
    if not isinstance(feature_table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for feature_table, got {type(feature_table)!r}")
    if not isinstance(label_table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for label_table, got {type(label_table)!r}")
    for name, table in (("feature_table", feature_table), ("label_table", label_table)):
        if "date" not in table.columns:
            raise KeyError(f"{name} requires a 'date' column")

    active_config = config or FlashFloodLabelMappingConfig()
    feature_frame = feature_table.reset_index(drop=True).copy()
    label_frame = label_table.reset_index(drop=True).copy()

    if "hazard_type" in feature_frame.columns:
        feature_frame = feature_frame[feature_frame["hazard_type"].astype(str).str.lower() == "flash_flood"].copy()
    if "hazard_type" in label_frame.columns:
        label_frame = label_frame[label_frame["hazard_type"].astype(str).str.lower() == "flash_flood"].copy()
    if feature_frame.empty:
        raise ValueError("feature_table has no flash_flood rows to join")
    if label_frame.empty:
        raise ValueError("label_table has no flash_flood rows to join")

    feature_has_grid = {"latitude", "longitude"}.issubset(feature_frame.columns)
    label_has_grid = {"latitude", "longitude"}.issubset(label_frame.columns)

    if feature_has_grid and label_has_grid:
        left = _prepare_grid_join_keys(feature_frame, coordinate_precision=coordinate_precision)
        right = _prepare_grid_join_keys(label_frame, coordinate_precision=coordinate_precision)
        join_columns = ["date", "latitude", "longitude"]
        join_mode = "grid_day"
    else:
        shared_province_columns = [column for column in _PROVINCE_COLUMNS if column in feature_frame.columns and column in label_frame.columns]
        if not shared_province_columns:
            raise KeyError(
                "flash-flood supervised join requires either shared latitude/longitude columns "
                f"or one shared province column from {_PROVINCE_COLUMNS}"
            )
        province_column = shared_province_columns[0]
        left = _prepare_province_join_keys(feature_frame, province_column, active_config)
        right = _prepare_province_join_keys(label_frame, province_column, active_config)
        join_columns = ["date", "_province_join_key"]
        join_mode = f"province_day:{province_column}"

    label_columns = ["label", "label_status", "label_source_mode", "matched_event_ids", "label_provenance"]
    missing_label_columns = [name for name in label_columns if name not in right.columns]
    if missing_label_columns:
        raise KeyError(f"label_table is missing required label columns: {missing_label_columns}")

    label_payload = right.loc[:, join_columns + label_columns].drop_duplicates(subset=join_columns, keep="last")
    merged = left.merge(label_payload, on=join_columns, how="left", validate="m:1")
    merged["hazard_type"] = "flash_flood"
    merged["training_join_mode"] = join_mode
    merged["training_join_key"] = merged[join_columns].astype(str).agg("|".join, axis=1)
    merged["is_labeled"] = merged["label"].notna()

    if drop_uncertain:
        merged = merged[merged["label_status"].isin(("positive", "negative"))].copy()
        merged["is_labeled"] = True

    return merged.reset_index(drop=True)
