"""Join flash-flood Layer-4 feature tables with conservative event-derived labels."""

from __future__ import annotations

from collections import Counter
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
    if isinstance(value, float) and value != value:
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


def _flash_flood_join_context(
    feature_frame: Any,
    label_frame: Any,
    *,
    active_config: FlashFloodLabelMappingConfig,
    coordinate_precision: int,
) -> tuple[Any, Any, list[str], str, str | None]:
    feature_has_grid = {"latitude", "longitude"}.issubset(feature_frame.columns)
    label_has_grid = {"latitude", "longitude"}.issubset(label_frame.columns)

    if feature_has_grid and label_has_grid:
        left = _prepare_grid_join_keys(feature_frame, coordinate_precision=coordinate_precision)
        right = _prepare_grid_join_keys(label_frame, coordinate_precision=coordinate_precision)
        return left, right, ["date", "latitude", "longitude"], "grid_day", None

    shared_province_columns = [column for column in _PROVINCE_COLUMNS if column in feature_frame.columns and column in label_frame.columns]
    if not shared_province_columns:
        raise KeyError(
            "flash-flood supervised join requires either shared latitude/longitude columns "
            f"or one shared province column from {_PROVINCE_COLUMNS}"
        )

    province_column = shared_province_columns[0]
    left = _prepare_province_join_keys(feature_frame, province_column, active_config)
    right = _prepare_province_join_keys(label_frame, province_column, active_config)
    return left, right, ["date", "_province_join_key"], f"province_day:{province_column}", province_column


def _join_key_counts(table: Any, join_columns: list[str], *, positive_only: bool) -> Counter[tuple[Any, ...]]:
    if positive_only:
        table = table.loc[table["label_status"].astype(str) == "positive"].copy()
    if table.empty:
        return Counter()
    return Counter(tuple(values) for values in table.loc[:, join_columns].itertuples(index=False, name=None))


def _flash_flood_positive_alignment_summary(
    feature_table: Any,
    label_table: Any,
    merged: Any,
    *,
    config: FlashFloodLabelMappingConfig | None = None,
    coordinate_precision: int = 4,
) -> dict[str, object]:
    active_config = config or FlashFloodLabelMappingConfig()
    _, normalized_labels, join_columns, join_mode, province_column = _flash_flood_join_context(
        feature_table,
        label_table,
        active_config=active_config,
        coordinate_precision=coordinate_precision,
    )

    label_positive_counts = _join_key_counts(normalized_labels, join_columns, positive_only=True)
    merged_positive_counts = _join_key_counts(merged, join_columns, positive_only=True)
    missing_positive_rows = 0
    extra_positive_rows = 0
    missing_positive_keys: list[dict[str, object]] = []
    extra_positive_keys: list[dict[str, object]] = []

    for key, label_count in label_positive_counts.items():
        merged_count = merged_positive_counts.get(key, 0)
        if label_count > merged_count:
            missing = label_count - merged_count
            missing_positive_rows += missing
            missing_positive_keys.append({"join_key": "|".join(str(part) for part in key), "missing_rows": int(missing)})

    for key, merged_count in merged_positive_counts.items():
        label_count = label_positive_counts.get(key, 0)
        if merged_count > label_count:
            extra = merged_count - label_count
            extra_positive_rows += extra
            extra_positive_keys.append({"join_key": "|".join(str(part) for part in key), "extra_rows": int(extra)})

    missing_positive_keys.sort(key=lambda item: (-int(item["missing_rows"]), str(item["join_key"])))
    extra_positive_keys.sort(key=lambda item: (-int(item["extra_rows"]), str(item["join_key"])))

    summary: dict[str, object] = {
        "status": "ok" if missing_positive_rows == 0 and extra_positive_rows == 0 else "mismatch",
        "ok": missing_positive_rows == 0 and extra_positive_rows == 0,
        "join_mode": join_mode,
        "join_key_columns": join_columns,
        "province_column": province_column,
        "label_positive_rows": int(sum(label_positive_counts.values())),
        "supervised_positive_rows": int(sum(merged_positive_counts.values())),
        "missing_positive_rows": int(missing_positive_rows),
        "extra_positive_rows": int(extra_positive_rows),
        "missing_positive_key_count": int(len(missing_positive_keys)),
        "extra_positive_key_count": int(len(extra_positive_keys)),
        "sample_missing_positive_keys": missing_positive_keys[:10],
        "sample_extra_positive_keys": extra_positive_keys[:10],
    }
    return summary


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
    left, right, join_columns, join_mode, _ = _flash_flood_join_context(
        feature_frame,
        label_frame,
        active_config=active_config,
        coordinate_precision=coordinate_precision,
    )

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

    alignment_summary = _flash_flood_positive_alignment_summary(
        feature_frame,
        label_frame,
        merged,
        config=active_config,
        coordinate_precision=coordinate_precision,
    )
    if not alignment_summary["ok"]:
        missing_sample = alignment_summary["sample_missing_positive_keys"]
        extra_sample = alignment_summary["sample_extra_positive_keys"]
        details: list[str] = []
        if alignment_summary["missing_positive_rows"]:
            details.append(
                f"missing {alignment_summary['missing_positive_rows']} positive label rows across "
                f"{alignment_summary['missing_positive_key_count']} join keys"
            )
            if missing_sample:
                details.append("sample missing keys: " + ", ".join(item["join_key"] for item in missing_sample[:3]))
        if alignment_summary["extra_positive_rows"]:
            details.append(
                f"extra {alignment_summary['extra_positive_rows']} positive supervised rows across "
                f"{alignment_summary['extra_positive_key_count']} join keys"
            )
            if extra_sample:
                details.append("sample extra keys: " + ", ".join(item["join_key"] for item in extra_sample[:3]))
        raise RuntimeError("flash-flood supervised training alignment check failed: " + "; ".join(details))

    return merged.reset_index(drop=True)
