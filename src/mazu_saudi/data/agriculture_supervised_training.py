"""Build dry-heat agriculture supervised training tables from daily regional features."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


DRY_HEAT_AGGREGATION_COLUMNS: tuple[str, ...] = (
    "temp_c",
    "tmax_c",
    "heat_index_c",
    "vpd_kpa",
    "wind_speed_mps",
    "relative_humidity_percent",
    "t2m_anomaly_c",
    "heatwave_day_flag",
    "heatwave_duration_days",
)

_MONTH_TO_SEASON = {
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
    12: "winter",
}


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for dry-heat agriculture supervised training dataset assembly")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_label_status(series):
    return series.astype(str).str.strip().str.lower()


def _coerce_numeric(frame, columns: Iterable[str]) -> Any:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _infer_season_from_date(series):
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        raise ValueError("feature_table contains invalid date values for season inference")
    return parsed.dt.month.map(_MONTH_TO_SEASON)


def _prepare_feature_keys(feature_table, *, sample_unit: str, region_column: str):
    working = feature_table.reset_index(drop=True).copy()
    if region_column not in working.columns:
        raise KeyError(f"feature_table requires a '{region_column}' column")
    if "date" not in working.columns:
        raise KeyError("feature_table requires a 'date' column")

    parsed = pd.to_datetime(working["date"], errors="coerce")
    if parsed.isna().any():
        raise ValueError("feature_table contains invalid date values")
    working["date"] = parsed.dt.strftime("%Y-%m-%d")
    working["year"] = parsed.dt.year.astype("int32")
    if sample_unit == "region-season":
        if "season" in working.columns:
            working["season"] = working["season"].astype(str).str.strip().str.lower()
        else:
            working["season"] = _infer_season_from_date(working["date"])
    working[region_column] = working[region_column].map(_normalize_text)
    return working


def _prepare_label_keys(label_table, *, sample_unit: str, region_column: str):
    working = label_table.reset_index(drop=True).copy()
    required = [region_column, "year"]
    if sample_unit == "region-season":
        required.append("season")
    missing = [column for column in required if column not in working.columns]
    if missing:
        raise KeyError(f"label_table is missing required supervision columns: {missing}")

    working[region_column] = working[region_column].map(_normalize_text)
    working["year"] = pd.to_numeric(working["year"], errors="coerce")
    if working["year"].isna().any():
        raise ValueError("label_table contains invalid year values")
    working["year"] = working["year"].astype("int32")
    if "season" in working.columns:
        working["season"] = working["season"].astype(str).str.strip().str.lower()
    if "validation_status" in working.columns:
        status = _normalize_label_status(working["validation_status"])
        working = working[~status.isin(("rejected", "invalid"))].copy()
    return working.reset_index(drop=True)


def _build_group_columns(sample_unit: str, region_column: str) -> list[str]:
    group_columns = [region_column, "year"]
    if sample_unit == "region-season":
        group_columns.append("season")
    return group_columns


def _aggregate_numeric_feature(series, feature_name: str):
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {}

    mean_value = float(clean.mean())
    result: dict[str, float] = {
        f"{feature_name}_mean": mean_value,
        f"{feature_name}_max": float(clean.max()),
    }
    if feature_name in {"temp_c", "tmax_c", "heat_index_c", "vpd_kpa", "wind_speed_mps", "relative_humidity_percent", "t2m_anomaly_c"}:
        result[feature_name] = mean_value
    elif feature_name == "heatwave_day_flag":
        result[feature_name] = mean_value
    elif feature_name == "heatwave_duration_days":
        result[feature_name] = float(clean.max())
    if feature_name in {"temp_c", "tmax_c", "heat_index_c", "vpd_kpa"}:
        result[f"{feature_name}_p90"] = float(clean.quantile(0.9))
    if feature_name == "relative_humidity_percent":
        result[f"{feature_name}_min"] = float(clean.min())
        result[f"{feature_name}_p10"] = float(clean.quantile(0.1))
    if feature_name == "t2m_anomaly_c":
        result[f"{feature_name}_min"] = float(clean.min())
    if feature_name == "heatwave_day_flag":
        result[f"{feature_name}_sum"] = float(clean.sum())
    if feature_name == "heatwave_duration_days":
        result[f"{feature_name}_sum"] = float(clean.sum())
    return result


def _aggregate_threshold_counts(group) -> dict[str, float]:
    result: dict[str, float] = {"feature_row_count": float(len(group))}
    if "temp_c" in group.columns:
        temp = pd.to_numeric(group["temp_c"], errors="coerce")
        result["temp_c_days_ge_35"] = float((temp >= 35.0).sum())
        result["temp_c_days_ge_40"] = float((temp >= 40.0).sum())
    if "tmax_c" in group.columns:
        tmax = pd.to_numeric(group["tmax_c"], errors="coerce")
        result["tmax_c_days_ge_40"] = float((tmax >= 40.0).sum())
        result["tmax_c_days_ge_45"] = float((tmax >= 45.0).sum())
    if "heat_index_c" in group.columns:
        heat_index = pd.to_numeric(group["heat_index_c"], errors="coerce")
        result["heat_index_c_days_ge_40"] = float((heat_index >= 40.0).sum())
    if "vpd_kpa" in group.columns:
        vpd = pd.to_numeric(group["vpd_kpa"], errors="coerce")
        result["vpd_kpa_days_ge_2"] = float((vpd >= 2.0).sum())
        result["vpd_kpa_days_ge_3"] = float((vpd >= 3.0).sum())
    return result


def aggregate_dry_heat_agriculture_features(
    feature_table: Any,
    *,
    sample_unit: str = "region-year",
    region_column: str = "region_id",
):
    """Aggregate daily regional features into region-year or region-season rows."""

    _require_pandas()
    if sample_unit not in {"region-year", "region-season"}:
        raise ValueError(f"Unsupported sample_unit: {sample_unit}")
    if not isinstance(feature_table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for feature_table, got {type(feature_table)!r}")

    working = _prepare_feature_keys(feature_table, sample_unit=sample_unit, region_column=region_column)
    _coerce_numeric(working, DRY_HEAT_AGGREGATION_COLUMNS)
    group_columns = _build_group_columns(sample_unit, region_column)

    rows: list[dict[str, Any]] = []
    for keys, group in working.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["sample_unit"] = sample_unit
        row["aggregation_start_date"] = str(group["date"].min())
        row["aggregation_end_date"] = str(group["date"].max())
        row.update(_aggregate_threshold_counts(group))
        for feature_name in DRY_HEAT_AGGREGATION_COLUMNS:
            row.update(_aggregate_numeric_feature(group[feature_name], feature_name))
        rows.append(row)

    if not rows:
        raise ValueError("feature_table has no aggregatable rows")
    return pd.DataFrame(rows)


def build_dry_heat_agriculture_supervised_training_dataset(
    feature_table: Any,
    label_table: Any,
    *,
    sample_unit: str = "region-year",
    region_column: str = "region_id",
    drop_unmatched: bool = True,
):
    """Join aggregated dry-heat features with explicit agricultural outcome labels."""

    _require_pandas()
    if not isinstance(label_table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for label_table, got {type(label_table)!r}")

    aggregated = aggregate_dry_heat_agriculture_features(
        feature_table,
        sample_unit=sample_unit,
        region_column=region_column,
    )
    labels = _prepare_label_keys(label_table, sample_unit=sample_unit, region_column=region_column)
    if labels.empty:
        raise ValueError("label_table has no valid rows after validation filtering")

    join_columns = _build_group_columns(sample_unit, region_column)
    if "crop_type" in labels.columns:
        labels["crop_type"] = labels["crop_type"].astype(str).str.strip().str.lower()
    duplicate_mask = labels.duplicated(subset=join_columns + (["crop_type"] if "crop_type" in labels.columns else []), keep=False)
    if duplicate_mask.any():
        duplicate_rows = labels.loc[duplicate_mask, join_columns].drop_duplicates().to_dict(orient="records")
        raise ValueError(f"label_table contains duplicate supervision rows for keys: {duplicate_rows}")

    merged = aggregated.merge(labels, on=join_columns, how="left", validate="1:m")
    merged["hazard_type"] = "dry_heat_agriculture"
    merged["training_join_key"] = merged[join_columns].astype(str).agg("|".join, axis=1)
    merged["is_labeled"] = False
    for target_name in ("yield_anomaly", "yield_value", "label"):
        if target_name in merged.columns:
            merged["is_labeled"] = merged["is_labeled"] | merged[target_name].notna()

    if drop_unmatched:
        merged = merged[merged["is_labeled"]].copy()
    if merged.empty:
        raise ValueError("No supervised dry-heat agriculture rows remained after joining labels")
    return merged.reset_index(drop=True)
