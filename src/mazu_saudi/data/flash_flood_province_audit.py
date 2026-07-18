"""Audit helpers for flash-flood province lookup coverage."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from mazu_saudi.data.flash_flood_province_lookup import (
    _build_bbox_bucket_index,
    _candidate_boundary_indices,
    _candidate_boundary_indices_with_radius,
    _bbox_contains,
    _point_to_bbox_distance,
    _parse_boundary_polygons,
    _point_to_any_polygon_distance,
    _point_in_any_polygon,
    _prepare_polygons,
    _prepared_polygons_bbox,
)

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for flash-flood province lookup audits")


def _normalize_lookup_table(lookup_table: Any, *, coordinate_precision: int):
    _require_pandas()
    if not isinstance(lookup_table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for lookup_table, got {type(lookup_table)!r}")
    missing = [column for column in ("latitude", "longitude", "match_status") if column not in lookup_table.columns]
    if missing:
        raise KeyError(f"lookup_table is missing required columns: {missing}")
    normalized = lookup_table.copy()
    normalized["latitude"] = pd.to_numeric(normalized["latitude"], errors="coerce").round(coordinate_precision)
    normalized["longitude"] = pd.to_numeric(normalized["longitude"], errors="coerce").round(coordinate_precision)
    normalized["match_status"] = normalized["match_status"].fillna("").astype(str)
    normalized = normalized.dropna(subset=["latitude", "longitude"]).drop_duplicates(subset=["latitude", "longitude"]).reset_index(drop=True)
    return normalized


def _normalize_feature_coordinates(feature_table: Any, *, coordinate_precision: int):
    _require_pandas()
    if not isinstance(feature_table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for feature_table, got {type(feature_table)!r}")
    missing = [column for column in ("latitude", "longitude") if column not in feature_table.columns]
    if missing:
        raise KeyError(f"feature_table is missing required coordinate columns: {missing}")
    columns = ["latitude", "longitude"]
    has_weight = "feature_row_count" in feature_table.columns
    if has_weight:
        columns.append("feature_row_count")
    normalized = feature_table.loc[:, columns].copy()
    normalized["latitude"] = pd.to_numeric(normalized["latitude"], errors="coerce").round(coordinate_precision)
    normalized["longitude"] = pd.to_numeric(normalized["longitude"], errors="coerce").round(coordinate_precision)
    normalized = normalized.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    if has_weight:
        normalized["feature_row_count"] = pd.to_numeric(normalized["feature_row_count"], errors="coerce").fillna(0).astype(int)
        normalized = (
            normalized.groupby(["latitude", "longitude"], dropna=False, as_index=False)["feature_row_count"]
            .sum()
            .reset_index(drop=True)
        )
    return normalized


def _prepare_boundaries(
    boundary_table: Any,
    *,
    boundary_province_column: str,
    geometry_column: str,
    geometry_format: str,
):
    _require_pandas()
    if not isinstance(boundary_table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for boundary_table, got {type(boundary_table)!r}")
    if boundary_province_column not in boundary_table.columns:
        raise KeyError(f"boundary_table is missing province column: {boundary_province_column}")

    boundaries = boundary_table.copy().reset_index(drop=True)
    prepared: list[dict[str, Any]] = []
    for record in boundaries.to_dict(orient="records"):
        polygons = _parse_boundary_polygons(record, geometry_column=geometry_column, geometry_format=geometry_format)
        prepared_polygons = _prepare_polygons(polygons)
        prepared.append(
            {
                "province_name": str(record.get(boundary_province_column, "")).strip(),
                "polygons": prepared_polygons,
                "bbox": _prepared_polygons_bbox(prepared_polygons),
            }
        )
    if not prepared:
        raise ValueError("boundary_table contains no usable rows")
    global_bbox = (
        min(boundary["bbox"][0] for boundary in prepared),
        max(boundary["bbox"][1] for boundary in prepared),
        min(boundary["bbox"][2] for boundary in prepared),
        max(boundary["bbox"][3] for boundary in prepared),
    )
    bbox_bucket_index = _build_bbox_bucket_index(prepared)
    return prepared, global_bbox, bbox_bucket_index


def _coordinate_bbox(table) -> dict[str, float] | None:
    if table.empty:
        return None
    return {
        "min_latitude": float(table["latitude"].min()),
        "max_latitude": float(table["latitude"].max()),
        "min_longitude": float(table["longitude"].min()),
        "max_longitude": float(table["longitude"].max()),
    }


def _bin_start(value: float, *, bin_size: float) -> float:
    return math.floor(value / bin_size) * bin_size


def _classify_boundary_gap(distance_degrees: float) -> str:
    if distance_degrees <= 0.05:
        return "within_0_05_degrees"
    if distance_degrees <= 0.1:
        return "within_0_10_degrees"
    if distance_degrees <= 0.25:
        return "within_0_25_degrees"
    if distance_degrees <= 0.5:
        return "within_0_50_degrees"
    return "beyond_0_50_degrees"


def _nearest_boundary_gap(
    latitude: float,
    longitude: float,
    prepared_boundaries: list[dict[str, Any]],
    bbox_bucket_index: dict[tuple[int, int], list[int]],
    *,
    candidate_indices: list[int] | None = None,
    candidate_boundaries: list[dict[str, Any]] | None = None,
    point_to_any_polygon_distance: Callable[..., float] = _point_to_any_polygon_distance,
) -> float:
    if candidate_indices is None:
        candidate_indices = _candidate_boundary_indices(latitude, longitude, bbox_bucket_index)
    if candidate_boundaries is None:
        if candidate_indices:
            candidate_boundaries = [prepared_boundaries[index] for index in candidate_indices]
        else:
            nearby_indices = _candidate_boundary_indices_with_radius(latitude, longitude, bbox_bucket_index, radius=1)
            candidate_boundaries = [prepared_boundaries[index] for index in nearby_indices]
    if not candidate_boundaries:
        return float(
            min(
                _point_to_bbox_distance(latitude, longitude, boundary["bbox"])
                for boundary in prepared_boundaries
            )
        )
    if len(candidate_boundaries) == 1:
        boundary = candidate_boundaries[0]
        lower_bound_distance = _point_to_bbox_distance(latitude, longitude, boundary["bbox"])
        if lower_bound_distance > 0.5:
            return float(lower_bound_distance)
        return float(
            point_to_any_polygon_distance(
                latitude,
                longitude,
                boundary["polygons"],
                max_distance=0.5,
            )
        )
    candidate_distances = sorted(
        (
            _point_to_bbox_distance(latitude, longitude, boundary["bbox"]),
            boundary["polygons"],
        )
        for boundary in candidate_boundaries
    )
    lower_bound_distance = candidate_distances[0][0]
    if lower_bound_distance > 0.5:
        return float(lower_bound_distance)

    distance = math.inf
    for bbox_distance, polygons in candidate_distances:
        if bbox_distance > max(distance, 0.5):
            break
        exact_distance = point_to_any_polygon_distance(
            latitude,
            longitude,
            polygons,
            max_distance=max(distance, 0.5),
        )
        if exact_distance < distance:
            distance = exact_distance
    if math.isinf(distance):
        return float(lower_bound_distance)
    return float(distance)


def _quantile(values: list[int], probability: float) -> float:
    if not values:
        return 0.0
    if probability <= 0:
        return float(values[0])
    if probability >= 1:
        return float(values[-1])
    position = (len(values) - 1) * probability
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(values[lower_index])
    lower_value = values[lower_index]
    upper_value = values[upper_index]
    weight = position - lower_index
    return float(lower_value + (upper_value - lower_value) * weight)


def _candidate_count_stats(candidate_counts: list[int]) -> dict[str, float | int]:
    if not candidate_counts:
        return {
            "min": 0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0,
            "mean": 0.0,
        }
    ordered = sorted(candidate_counts)
    return {
        "min": int(ordered[0]),
        "p50": _quantile(ordered, 0.5),
        "p95": _quantile(ordered, 0.95),
        "max": int(ordered[-1]),
        "mean": float(sum(ordered) / len(ordered)),
    }


def _wrap_counter(function: Callable[..., Any], counters: dict[str, int], key: str) -> Callable[..., Any]:
    def wrapped(*args, **kwargs):
        counters[key] = counters.get(key, 0) + 1
        return function(*args, **kwargs)

    return wrapped


def audit_flash_flood_province_lookup(
    lookup_table: Any,
    *,
    feature_table: Any | None = None,
    boundary_table: Any | None = None,
    boundary_province_column: str = "province_name",
    geometry_column: str = "geometry",
    geometry_format: str = "geojson",
    coordinate_precision: int = 4,
    bin_size_degrees: float = 1.0,
    top_n: int = 10,
    include_runtime_stats: bool = False,
) -> dict[str, Any]:
    """Summarize where province lookup coverage fails and how concentrated it is."""

    if bin_size_degrees <= 0:
        raise ValueError("bin_size_degrees must be positive")
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    lookup = _normalize_lookup_table(lookup_table, coordinate_precision=coordinate_precision)
    unmatched = lookup.loc[lookup["match_status"].str.lower() != "matched", ["latitude", "longitude", "match_status"]].copy()

    summary: dict[str, Any] = {
        "unique_coordinate_rows": int(len(lookup)),
        "matched_coordinate_rows": int((lookup["match_status"].str.lower() == "matched").sum()),
        "unmatched_coordinate_rows": int(len(unmatched)),
        "unmatched_coordinate_bbox": _coordinate_bbox(unmatched),
    }

    if boundary_table is not None:
        prepared_boundaries, global_bbox, bbox_bucket_index = _prepare_boundaries(
            boundary_table,
            boundary_province_column=boundary_province_column,
            geometry_column=geometry_column,
            geometry_format=geometry_format,
        )
        runtime_counters: dict[str, int] = {}
        point_in_any_polygon = _point_in_any_polygon
        point_to_any_polygon_distance = _point_to_any_polygon_distance
        if include_runtime_stats:
            point_in_any_polygon = _wrap_counter(point_in_any_polygon, runtime_counters, "point_in_any_polygon_calls")
            point_to_any_polygon_distance = _wrap_counter(
                point_to_any_polygon_distance,
                runtime_counters,
                "point_to_any_polygon_distance_calls",
            )
        classifications: list[str] = []
        boundary_gap_distances: list[float | None] = []
        boundary_gap_bands: list[str | None] = []
        candidate_count_distribution: list[int] = []
        inside_bbox_rows = 0
        inside_bbox_zero_candidate_rows = 0
        inside_bbox_broad_scan_rows = 0
        candidate_boundary_checks = 0
        for row in unmatched.to_dict(orient="records"):
            latitude = float(row["latitude"])
            longitude = float(row["longitude"])
            if not _bbox_contains(latitude, longitude, global_bbox):
                classifications.append("outside_boundary_bbox")
                boundary_gap_distances.append(None)
                boundary_gap_bands.append(None)
                continue
            inside_bbox_rows += 1
            candidate_indices = _candidate_boundary_indices(latitude, longitude, bbox_bucket_index)
            candidate_count_distribution.append(len(candidate_indices))
            if not candidate_indices:
                inside_bbox_zero_candidate_rows += 1
            if not candidate_indices:
                inside_bbox_broad_scan_rows += 1
                candidate_boundaries = []
                inside_polygon = False
            else:
                candidate_boundaries = [prepared_boundaries[index] for index in candidate_indices]
                candidate_boundary_checks += len(candidate_boundaries)
                inside_polygon = any(
                    point_in_any_polygon(latitude, longitude, boundary["polygons"]) for boundary in candidate_boundaries
                )
            if inside_polygon:
                classifications.append("inside_polygon_but_unmatched")
                boundary_gap_distances.append(0.0)
                boundary_gap_bands.append("within_0_05_degrees")
            else:
                classifications.append("inside_boundary_bbox_outside_polygon")
                distance = _nearest_boundary_gap(
                    latitude,
                    longitude,
                    prepared_boundaries,
                    bbox_bucket_index,
                    candidate_indices=candidate_indices,
                    candidate_boundaries=candidate_boundaries,
                    point_to_any_polygon_distance=point_to_any_polygon_distance,
                )
                boundary_gap_distances.append(float(distance))
                boundary_gap_bands.append(_classify_boundary_gap(float(distance)))
        unmatched["boundary_classification"] = classifications
        unmatched["boundary_gap_degrees"] = boundary_gap_distances
        unmatched["boundary_gap_band"] = boundary_gap_bands
        summary["boundary_bbox"] = {
            "min_latitude": float(global_bbox[0]),
            "max_latitude": float(global_bbox[1]),
            "min_longitude": float(global_bbox[2]),
            "max_longitude": float(global_bbox[3]),
        }
        summary["unmatched_boundary_classification_counts"] = {
            str(key): int(value) for key, value in unmatched["boundary_classification"].value_counts(dropna=False).to_dict().items()
        }
        inside_bbox_unmatched = unmatched.loc[
            unmatched["boundary_classification"] == "inside_boundary_bbox_outside_polygon",
            ["boundary_gap_degrees", "boundary_gap_band"],
        ].copy()
        if not inside_bbox_unmatched.empty:
            summary["inside_bbox_boundary_gap_band_counts"] = {
                str(key): int(value)
                for key, value in inside_bbox_unmatched["boundary_gap_band"].value_counts(dropna=False).to_dict().items()
            }
            summary["inside_bbox_boundary_gap_stats_degrees"] = {
                "min": float(inside_bbox_unmatched["boundary_gap_degrees"].min()),
                "median": float(inside_bbox_unmatched["boundary_gap_degrees"].median()),
                "mean": float(inside_bbox_unmatched["boundary_gap_degrees"].mean()),
                "max": float(inside_bbox_unmatched["boundary_gap_degrees"].max()),
                "note": "values beyond 0.5 degrees may use bbox lower bounds instead of exact polygon distance",
            }
        if include_runtime_stats:
            summary["runtime_stats"] = {
                "inside_global_bbox_unmatched_rows": int(inside_bbox_rows),
                "outside_global_bbox_unmatched_rows": int(len(unmatched) - inside_bbox_rows),
                "inside_global_bbox_zero_candidate_rows": int(inside_bbox_zero_candidate_rows),
                "inside_global_bbox_nonzero_candidate_rows": int(inside_bbox_rows - inside_bbox_zero_candidate_rows),
                "inside_global_bbox_broad_scan_rows": int(inside_bbox_broad_scan_rows),
                "candidate_boundary_checks": int(candidate_boundary_checks),
                "candidate_count_stats": _candidate_count_stats(candidate_count_distribution),
                "point_in_any_polygon_calls": int(runtime_counters.get("point_in_any_polygon_calls", 0)),
                "point_to_any_polygon_distance_calls": int(runtime_counters.get("point_to_any_polygon_distance_calls", 0)),
            }

    if feature_table is not None:
        features = _normalize_feature_coordinates(feature_table, coordinate_precision=coordinate_precision)
        if "feature_row_count" in features.columns:
            feature_counts = features.copy()
        else:
            feature_counts = (
                features.groupby(["latitude", "longitude"], dropna=False)
                .size()
                .reset_index(name="feature_row_count")
            )
        lookup_counts = lookup.merge(feature_counts, on=["latitude", "longitude"], how="left")
        lookup_counts["feature_row_count"] = lookup_counts["feature_row_count"].fillna(0).astype(int)

        matched_feature_rows = int(lookup_counts.loc[lookup_counts["match_status"].str.lower() == "matched", "feature_row_count"].sum())
        unmatched_feature_rows = int(lookup_counts.loc[lookup_counts["match_status"].str.lower() != "matched", "feature_row_count"].sum())
        summary["feature_row_count"] = int(len(features))
        summary["matched_feature_rows"] = matched_feature_rows
        summary["unmatched_feature_rows"] = unmatched_feature_rows

        unmatched_counts = lookup_counts.loc[lookup_counts["match_status"].str.lower() != "matched", ["latitude", "longitude", "feature_row_count"]].copy()
        if "boundary_classification" in unmatched.columns:
            unmatched_counts = unmatched_counts.merge(
                unmatched.loc[:, ["latitude", "longitude", "boundary_classification", "boundary_gap_degrees", "boundary_gap_band"]],
                on=["latitude", "longitude"],
                how="left",
            )
        unmatched_counts["lat_bin_start"] = unmatched_counts["latitude"].map(lambda value: _bin_start(float(value), bin_size=bin_size_degrees))
        unmatched_counts["lon_bin_start"] = unmatched_counts["longitude"].map(lambda value: _bin_start(float(value), bin_size=bin_size_degrees))

        hot_coordinates = unmatched_counts.sort_values(
            ["feature_row_count", "latitude", "longitude"],
            ascending=[False, True, True],
        ).head(top_n)
        summary["top_unmatched_coordinates"] = hot_coordinates.to_dict(orient="records")

        hotspot_bins = (
            unmatched_counts.groupby(["lat_bin_start", "lon_bin_start"], dropna=False)
            .agg(
                unique_coordinate_rows=("latitude", "size"),
                feature_row_count=("feature_row_count", "sum"),
                min_latitude=("latitude", "min"),
                max_latitude=("latitude", "max"),
                min_longitude=("longitude", "min"),
                max_longitude=("longitude", "max"),
            )
            .reset_index()
            .sort_values(["feature_row_count", "unique_coordinate_rows", "lat_bin_start", "lon_bin_start"], ascending=[False, False, True, True])
            .head(top_n)
        )
        summary["top_unmatched_coordinate_bins"] = hotspot_bins.to_dict(orient="records")
        if "boundary_gap_band" in unmatched_counts.columns:
            weighted_gap_bands = (
                unmatched_counts.loc[unmatched_counts["boundary_gap_band"].notna(), ["boundary_gap_band", "feature_row_count"]]
                .groupby("boundary_gap_band", dropna=False)["feature_row_count"]
                .sum()
                .sort_values(ascending=False)
            )
            summary["weighted_inside_bbox_boundary_gap_band_feature_rows"] = {
                str(key): int(value) for key, value in weighted_gap_bands.to_dict().items()
            }

    return summary
