"""Join dust-storm Layer-4 feature tables with event-derived province-day labels."""

from __future__ import annotations

from typing import Any

from mazu_saudi.config import DustStormLabelMappingConfig

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


_LOCATION_COLUMNS = ("region_id", "province_name", "admin1_name", "region_name", "location_name")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalize_date(series: Any):
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def _canonical_location(value: Any, config: DustStormLabelMappingConfig) -> str:
    token = _normalize_text(value).replace("-", " ").replace("_", " ")
    token = " ".join(token.split())
    return config.location_aliases.get(token, token.replace(" ", "_"))


def _prepare_location_join_keys(table: Any, column: str, config: DustStormLabelMappingConfig) -> Any:
    normalized = table.copy()
    normalized["date"] = _normalize_date(normalized["date"])
    normalized["_dust_location_join_key"] = normalized[column].map(lambda value: _canonical_location(value, config))
    return normalized


def build_dust_storm_supervised_training_dataset(
    feature_table: Any,
    label_table: Any,
    *,
    config: DustStormLabelMappingConfig | None = None,
    drop_uncertain: bool = True,
):
    """Join dust-storm features with labels using a shared region-day or province-day key."""

    if pd is None:
        raise RuntimeError("pandas is required for dust-storm supervised training dataset assembly")
    if not isinstance(feature_table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for feature_table, got {type(feature_table)!r}")
    if not isinstance(label_table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for label_table, got {type(label_table)!r}")
    for name, table in (("feature_table", feature_table), ("label_table", label_table)):
        if "date" not in table.columns:
            raise KeyError(f"{name} requires a 'date' column")

    active_config = config or DustStormLabelMappingConfig()
    feature_frame = feature_table.reset_index(drop=True).copy()
    label_frame = label_table.reset_index(drop=True).copy()

    if "hazard_type" in feature_frame.columns:
        feature_frame = feature_frame[feature_frame["hazard_type"].astype(str).str.lower() == "dust_storm"].copy()
    if "hazard_type" in label_frame.columns:
        label_frame = label_frame[label_frame["hazard_type"].astype(str).str.lower() == "dust_storm"].copy()
    if feature_frame.empty:
        raise ValueError("feature_table has no dust_storm rows to join")
    if label_frame.empty:
        raise ValueError("label_table has no dust_storm rows to join")

    shared_location_columns = [column for column in _LOCATION_COLUMNS if column in feature_frame.columns and column in label_frame.columns]
    if not shared_location_columns:
        raise KeyError(f"dust-storm supervised join requires one shared location column from {_LOCATION_COLUMNS}")

    location_column = shared_location_columns[0]
    left = _prepare_location_join_keys(feature_frame, location_column, active_config)
    right = _prepare_location_join_keys(label_frame, location_column, active_config)
    join_columns = ["date", "_dust_location_join_key"]
    join_mode = f"region_day:{location_column}"

    label_columns = ["label", "label_status", "label_source_mode", "matched_event_ids", "label_provenance"]
    missing_label_columns = [name for name in label_columns if name not in right.columns]
    if missing_label_columns:
        raise KeyError(f"label_table is missing required label columns: {missing_label_columns}")

    label_payload = right.loc[:, join_columns + label_columns].drop_duplicates(subset=join_columns, keep="last")
    merged = left.merge(label_payload, on=join_columns, how="left", validate="m:1")
    merged["hazard_type"] = "dust_storm"
    merged["training_join_mode"] = join_mode
    merged["training_join_key"] = merged[join_columns].astype(str).agg("|".join, axis=1)
    merged["is_labeled"] = merged["label"].notna()

    if drop_uncertain:
        merged = merged[merged["label_status"].isin(("positive", "negative"))].copy()
        merged["is_labeled"] = True

    return merged.reset_index(drop=True)
