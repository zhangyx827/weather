"""Build province-day flash-flood feature tables from Layer-4 grid-day rows."""

from __future__ import annotations

from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for flash-flood province feature assembly")


def _normalize_date(series: Any):
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


_NON_FEATURE_NUMERIC_COLUMNS = {
    "latitude",
    "longitude",
    "source_mtime_ns",
    "source_mtime_us",
    "source_size_bytes",
}


def province_day_numeric_feature_columns(feature_table: Any) -> list[str]:
    """Return numeric model-feature columns that should survive province-day aggregation."""

    _require_pandas()
    if not isinstance(feature_table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for feature_table, got {type(feature_table)!r}")
    return [
        column
        for column in feature_table.select_dtypes(include="number").columns
        if column not in _NON_FEATURE_NUMERIC_COLUMNS
    ]


def enrich_flash_flood_features_with_province(
    feature_table: Any,
    province_lookup_table: Any,
    *,
    coordinate_precision: int = 4,
    province_column: str = "province_name",
    lookup_province_column: str = "province_name",
    overwrite_existing: bool = False,
):
    """Attach a province column to a flash-flood feature table using rounded lat/lon keys."""

    _require_pandas()
    if not isinstance(feature_table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for feature_table, got {type(feature_table)!r}")
    if not isinstance(province_lookup_table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for province_lookup_table, got {type(province_lookup_table)!r}")

    for name, table in (("feature_table", feature_table), ("province_lookup_table", province_lookup_table)):
        missing = [column for column in ("latitude", "longitude") if column not in table.columns]
        if missing:
            raise KeyError(f"{name} is missing required coordinate columns: {missing}")
    if lookup_province_column not in province_lookup_table.columns:
        raise KeyError(f"province_lookup_table is missing required province column: {lookup_province_column}")

    features = feature_table.reset_index(drop=True).copy()
    lookup = province_lookup_table.reset_index(drop=True).copy()

    features["_latitude_join_key"] = pd.to_numeric(features["latitude"], errors="coerce").round(coordinate_precision)
    features["_longitude_join_key"] = pd.to_numeric(features["longitude"], errors="coerce").round(coordinate_precision)
    lookup["_latitude_join_key"] = pd.to_numeric(lookup["latitude"], errors="coerce").round(coordinate_precision)
    lookup["_longitude_join_key"] = pd.to_numeric(lookup["longitude"], errors="coerce").round(coordinate_precision)
    lookup[lookup_province_column] = lookup[lookup_province_column].map(_normalize_text)
    lookup = lookup[
        lookup["_latitude_join_key"].notna() & lookup["_longitude_join_key"].notna() & lookup[lookup_province_column].ne("")
    ].copy()

    if lookup.empty:
        raise ValueError("province_lookup_table contains no usable latitude/longitude to province mappings")

    conflict_counts = (
        lookup.groupby(["_latitude_join_key", "_longitude_join_key"])[lookup_province_column].nunique(dropna=True).rename("province_count")
    )
    conflicting = conflict_counts[conflict_counts > 1]
    if not conflicting.empty:
        raise ValueError("province_lookup_table contains conflicting province assignments for the same rounded coordinates")

    province_payload = lookup.loc[:, ["_latitude_join_key", "_longitude_join_key", lookup_province_column]].drop_duplicates(
        subset=["_latitude_join_key", "_longitude_join_key"],
        keep="last",
    )
    province_payload = province_payload.rename(columns={lookup_province_column: "_lookup_province_name"})
    enriched = features.merge(
        province_payload,
        on=["_latitude_join_key", "_longitude_join_key"],
        how="left",
        validate="m:1",
    )

    if province_column in enriched.columns and not overwrite_existing:
        existing = enriched[province_column].map(_normalize_text)
        enriched[province_column] = existing.where(existing.ne(""), enriched["_lookup_province_name"])
    else:
        enriched[province_column] = enriched["_lookup_province_name"]

    return enriched.drop(columns=["_latitude_join_key", "_longitude_join_key", "_lookup_province_name"]).reset_index(drop=True)


def aggregate_flash_flood_features_to_province_day(
    feature_table: Any,
    *,
    province_column: str = "province_name",
):
    """Aggregate a flash-flood feature table into province-day samples."""

    _require_pandas()
    if not isinstance(feature_table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for feature_table, got {type(feature_table)!r}")
    if "date" not in feature_table.columns:
        raise KeyError("feature_table requires a 'date' column")
    if province_column not in feature_table.columns:
        raise KeyError(f"feature_table requires a province column: {province_column}")

    working = feature_table.reset_index(drop=True).copy()
    if "hazard_type" in working.columns:
        working = working[working["hazard_type"].astype(str).str.lower() == "flash_flood"].copy()
    if working.empty:
        raise ValueError("feature_table has no flash_flood rows to aggregate")

    working["date"] = _normalize_date(working["date"])
    if working["date"].isna().any():
        raise ValueError("feature_table contains invalid 'date' values")
    working[province_column] = working[province_column].map(_normalize_text)
    working = working[working[province_column].ne("")].copy()
    if working.empty:
        raise ValueError(f"feature_table has no rows with a usable {province_column}")

    group_columns = ["date", province_column]
    numeric_columns = province_day_numeric_feature_columns(working)
    if not numeric_columns:
        raise ValueError("feature_table has no numeric feature columns available for province-day aggregation")

    aggregated = working.groupby(group_columns, dropna=False)[numeric_columns].mean().reset_index()
    aggregated.insert(1, "hazard_type", "flash_flood")
    counts = working.groupby(group_columns, dropna=False).size().reset_index(name="grid_cell_count")
    aggregated = aggregated.merge(counts, on=group_columns, how="left", validate="1:1")

    if "source_status" in working.columns:
        degraded = (
            working.assign(_is_degraded=working["source_status"].astype(str).str.lower().eq("degraded").astype(int))
            .groupby(group_columns, dropna=False)["_is_degraded"]
            .sum()
            .reset_index(name="degraded_grid_cell_count")
        )
        aggregated = aggregated.merge(degraded, on=group_columns, how="left", validate="1:1")

    first_columns = ["date", "hazard_type", province_column, "grid_cell_count"]
    if "degraded_grid_cell_count" in aggregated.columns:
        first_columns.append("degraded_grid_cell_count")
    ordered_columns = first_columns + [column for column in aggregated.columns if column not in first_columns]
    return aggregated.loc[:, ordered_columns].reset_index(drop=True)
