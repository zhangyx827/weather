"""Map flash-flood event rows to trainable grid-day or province-day labels."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any

import numpy as np

from mazu_saudi.config import FlashFloodLabelMappingConfig
from mazu_saudi.data.flash_flood_labels import expand_flash_flood_events_to_daily_table

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


_CONFIDENCE_LEVELS = {"low": 1, "medium": 2, "high": 3}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _canonicalize_location_text(value: Any) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _normalize_date_column(series: Any):
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def _normalize_province_series(series: Any, config: FlashFloodLabelMappingConfig):
    return series.map(lambda value: config.location_to_province.get(_normalize_text(value), _normalize_text(value)))


def _confidence_at_least(value: str, minimum: str) -> bool:
    return _CONFIDENCE_LEVELS.get(_normalize_text(value), 0) >= _CONFIDENCE_LEVELS.get(_normalize_text(minimum), 0)


def _split_wkt_groups(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_ring_text(text: str) -> list[tuple[float, float]]:
    coordinates: list[tuple[float, float]] = []
    for token in text.split(","):
        pieces = [piece for piece in token.strip().split() if piece]
        if len(pieces) < 2:
            continue
        lon = float(pieces[0])
        lat = float(pieces[1])
        coordinates.append((lat, lon))
    if len(coordinates) < 3:
        raise ValueError("WKT ring requires at least three coordinate pairs")
    return coordinates


def _parse_polygon_text(text: str) -> list[list[tuple[float, float]]]:
    inner = text.strip()
    if not inner.startswith("(") or not inner.endswith(")"):
        raise ValueError("Invalid WKT polygon text")
    ring_groups = _split_wkt_groups(inner[1:-1].strip())
    rings: list[list[tuple[float, float]]] = []
    for group in ring_groups:
        normalized = group.strip()
        if normalized.startswith("(") and normalized.endswith(")"):
            normalized = normalized[1:-1].strip()
        rings.append(_parse_ring_text(normalized))
    if not rings:
        raise ValueError("WKT polygon contains no rings")
    return rings


def _parse_wkt_polygons(value: Any) -> list[list[list[tuple[float, float]]]]:
    text = _normalize_text(value)
    if not text:
        return []
    if text.startswith("polygon"):
        prefix = "polygon"
        return [_parse_polygon_text(text[len(prefix):].strip())]
    if text.startswith("multipolygon"):
        prefix = "multipolygon"
        inner = text[len(prefix):].strip()
        if not inner.startswith("(") or not inner.endswith(")"):
            raise ValueError("Invalid WKT multipolygon text")
        polygon_groups = _split_wkt_groups(inner[1:-1].strip())
        return [_parse_polygon_text(group) for group in polygon_groups]
    raise ValueError(f"Unsupported WKT geometry type: {value}")


def _point_in_ring(latitude: float, longitude: float, ring: list[tuple[float, float]]) -> bool:
    inside = False
    for index, (lat1, lon1) in enumerate(ring):
        lat2, lon2 = ring[(index + 1) % len(ring)]
        intersects = (lat1 > latitude) != (lat2 > latitude)
        if not intersects:
            continue
        denominator = lat2 - lat1
        if abs(denominator) < 1e-12:
            continue
        cross_lon = lon1 + (latitude - lat1) * (lon2 - lon1) / denominator
        if cross_lon > longitude:
            inside = not inside
    return inside


def _point_in_polygon(latitude: float, longitude: float, polygon: list[list[tuple[float, float]]]) -> bool:
    if not polygon:
        return False
    if not _point_in_ring(latitude, longitude, polygon[0]):
        return False
    return not any(_point_in_ring(latitude, longitude, ring) for ring in polygon[1:])


def _point_in_wkt_geometry(latitude: float, longitude: float, geometry_wkt: Any) -> bool:
    try:
        polygons = _parse_wkt_polygons(geometry_wkt)
    except Exception:
        return False
    return any(_point_in_polygon(latitude, longitude, polygon) for polygon in polygons)


def _has_usable_geometry(event: Any) -> bool:
    geometry = _normalize_text(event.get("geometry_wkt"))
    if not geometry:
        return False
    try:
        return bool(_parse_wkt_polygons(geometry))
    except Exception:
        return False


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def _haversine_km_vectorized(latitudes: Any, longitudes: Any, lat2: float, lon2: float):
    radius_km = 6371.0
    latitudes_rad = np.radians(latitudes.astype(float))
    longitudes_rad = np.radians(longitudes.astype(float))
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - latitudes_rad
    dlon = lon2_rad - longitudes_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(latitudes_rad) * math.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(a))


def _resolved_province_name(row: Any, config: FlashFloodLabelMappingConfig) -> str:
    provinces = _resolved_province_names(row, config)
    return provinces[0] if provinces else ""


def _resolved_province_names(row: Any, config: FlashFloodLabelMappingConfig) -> tuple[str, ...]:
    alias_patterns: list[tuple[str, str]] = []
    for alias, province in config.location_to_province.items():
        canonical_alias = _canonicalize_location_text(alias)
        if canonical_alias:
            alias_patterns.append((canonical_alias, province))
    alias_patterns.sort(key=lambda item: len(item[0]), reverse=True)

    ordered_provinces: list[str] = []
    seen_provinces: set[str] = set()
    for key in ("province_name", "admin1_name", "region_name", "location_name"):
        value = _normalize_text(row.get(key))
        if not value:
            continue
        direct = config.location_to_province.get(value)
        if direct and direct not in seen_provinces:
            ordered_provinces.append(direct)
            seen_provinces.add(direct)
        canonical_value = _canonicalize_location_text(value)
        if not canonical_value:
            continue
        padded_value = f" {canonical_value} "
        matches: list[tuple[int, str]] = []
        for canonical_alias, province in alias_patterns:
            alias_token = f" {canonical_alias} "
            position = padded_value.find(alias_token)
            if position >= 0:
                matches.append((position, province))
        for _, province in sorted(matches, key=lambda item: item[0]):
            if province not in seen_provinces:
                ordered_provinces.append(province)
                seen_provinces.add(province)
    return tuple(ordered_provinces)


def _event_mapping_mode(event: Any, config: FlashFloodLabelMappingConfig) -> str:
    validation_status = _normalize_text(event.get("validation_status"))
    spatial_confidence = _normalize_text(event.get("spatial_confidence"))
    if validation_status not in config.positive_validation_statuses:
        return "unsupported"
    if not _confidence_at_least(spatial_confidence, config.min_spatial_confidence):
        return "uncertain"
    if _has_usable_geometry(event):
        return "geometry_wkt"
    latitude = event.get("latitude")
    longitude = event.get("longitude")
    if pd.notna(latitude) and pd.notna(longitude):
        return "point_buffer"
    if config.province_fallback_enabled and _resolved_province_names(event, config):
        return "province_day"
    return "uncertain"


def _sample_matches_event(sample: Any, event: Any, config: FlashFloodLabelMappingConfig) -> tuple[bool, str | None]:
    mode = _event_mapping_mode(event, config)
    if mode == "geometry_wkt":
        sample_lat = sample.get("latitude")
        sample_lon = sample.get("longitude")
        if pd.notna(sample_lat) and pd.notna(sample_lon):
            return (_point_in_wkt_geometry(float(sample_lat), float(sample_lon), event.get("geometry_wkt")), "geometry_wkt")
        return (False, None)
    if mode == "point_buffer":
        sample_lat = sample.get("latitude")
        sample_lon = sample.get("longitude")
        if pd.isna(sample_lat) or pd.isna(sample_lon):
            sample_province = _resolved_province_name(sample, config)
            event_provinces = set(_resolved_province_names(event, config))
            return (sample_province != "" and sample_province in event_provinces, "province_day")
        distance_km = _haversine_km(float(sample_lat), float(sample_lon), float(event["latitude"]), float(event["longitude"]))
        return (distance_km <= config.point_buffer_km, "point_buffer")
    if mode == "province_day":
        sample_province = _resolved_province_name(sample, config)
        event_provinces = set(_resolved_province_names(event, config))
        return (sample_province != "" and sample_province in event_provinces, "province_day")
    return (False, None)


def _supports_event_day_negative(sample_columns: Any, day_events: list[dict[str, Any]], config: FlashFloodLabelMappingConfig) -> bool:
    if not config.emit_event_day_negatives:
        return False
    if not day_events:
        return True
    has_grid = "latitude" in sample_columns and "longitude" in sample_columns
    has_province = any(column in sample_columns for column in ("province_name", "admin1_name", "region_name", "location_name"))
    supported_modes = set()
    if has_grid:
        supported_modes.update({"point_buffer", "geometry_wkt"})
    if has_province:
        supported_modes.update({"point_buffer", "province_day"})
    if not supported_modes:
        return False
    return all(_event_mapping_mode(event, config) in supported_modes for event in day_events)


def _default_provenance_json(
    *,
    date_value: str,
    day_events: list[dict[str, Any]],
    day_event_mapping_modes: list[str],
    config: FlashFloodLabelMappingConfig,
) -> str:
    return json.dumps(
        {
            "date": date_value,
            "event_count_for_day": len(day_events),
            "matched_event_ids": [],
            "matched_location_names": [],
            "mapping_modes": [],
            "day_event_mapping_modes": day_event_mapping_modes,
            "matched_geometry_wkts": [],
            "point_buffer_km": config.point_buffer_km,
            "province_fallback_enabled": config.province_fallback_enabled,
            "emit_event_day_negatives": config.emit_event_day_negatives,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def build_flash_flood_training_labels(
    samples: Any,
    event_daily_table: Any | None = None,
    config: FlashFloodLabelMappingConfig | None = None,
):
    """Attach conservative flash-flood labels to a sample table.

    Required sample columns:
    - ``date``
    Optional spatial columns:
    - ``latitude`` and ``longitude`` for grid-day mapping
    - ``province_name`` / ``admin1_name`` / ``region_name`` for province-day fallback
    """

    if pd is None:
        raise RuntimeError("pandas is required for flash-flood training-label mapping")
    if not isinstance(samples, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(samples)!r}")
    if "date" not in samples.columns:
        raise KeyError("flash-flood label mapping requires a 'date' column in the sample table")

    active_config = config or FlashFloodLabelMappingConfig()
    events = event_daily_table if event_daily_table is not None else expand_flash_flood_events_to_daily_table()
    if not isinstance(events, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for event_daily_table, got {type(events)!r}")
    if "date" not in events.columns:
        raise KeyError("flash-flood event table requires a 'date' column")

    normalized_samples = samples.copy()
    normalized_samples["date"] = _normalize_date_column(normalized_samples["date"])
    if normalized_samples["date"].isna().any():
        raise ValueError("sample table contains invalid 'date' values")

    normalized_events = events.copy()
    normalized_events["date"] = _normalize_date_column(normalized_events["date"])
    if normalized_events["date"].isna().any():
        raise ValueError("event table contains invalid 'date' values")

    events_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in normalized_events.to_dict(orient="records"):
        events_by_date[record["date"]].append(record)

    sample_columns = normalized_samples.columns
    event_dates = set(events_by_date)
    has_event_mask = normalized_samples["date"].isin(event_dates)

    normalized_samples["hazard_type"] = "flash_flood"
    normalized_samples["label"] = np.where(has_event_mask.to_numpy(), np.nan, 0.0)
    normalized_samples["label_status"] = pd.Series(
        np.where(has_event_mask.to_numpy(), "uncertain", "negative"),
        index=normalized_samples.index,
        dtype="object",
    )
    normalized_samples["label_source_mode"] = pd.Series(
        np.where(has_event_mask.to_numpy(), "event_day_unresolved", "no_event_day"),
        index=normalized_samples.index,
        dtype="object",
    )
    normalized_samples["matched_event_ids"] = pd.Series([""] * len(normalized_samples), index=normalized_samples.index, dtype="object")
    normalized_samples["label_provenance"] = pd.Series([""] * len(normalized_samples), index=normalized_samples.index, dtype="object")

    default_no_event_provenance = {
        date_value: _default_provenance_json(
            date_value=date_value,
            day_events=[],
            day_event_mapping_modes=[],
            config=active_config,
        )
        for date_value in normalized_samples.loc[~has_event_mask, "date"].drop_duplicates().tolist()
    }
    normalized_samples.loc[~has_event_mask, "label_provenance"] = (
        normalized_samples.loc[~has_event_mask, "date"].map(default_no_event_provenance).to_numpy()
    )

    if has_event_mask.any():
        event_samples = normalized_samples.loc[has_event_mask].copy()
        event_sample_dates = event_samples["date"].tolist()
        matched_lists: list[list[dict[str, Any]]] = [[] for _ in range(len(event_samples))]
        mode_lists: list[list[str]] = [[] for _ in range(len(event_samples))]

        province_series = None
        province_column = next((column for column in ("province_name", "admin1_name", "region_name", "location_name") if column in event_samples.columns), None)
        if province_column is not None:
            province_series = _normalize_province_series(event_samples[province_column], active_config)

        latitudes = None
        longitudes = None
        if "latitude" in event_samples.columns and "longitude" in event_samples.columns:
            latitudes = pd.to_numeric(event_samples["latitude"], errors="coerce")
            longitudes = pd.to_numeric(event_samples["longitude"], errors="coerce")

        for date_value, day_events in events_by_date.items():
            day_mask = event_samples["date"] == date_value
            if not day_mask.any():
                continue
            day_positions = np.flatnonzero(day_mask.to_numpy())
            day_event_mapping_modes = [_event_mapping_mode(event, active_config) for event in day_events]

            for event in day_events:
                mode = _event_mapping_mode(event, active_config)
                if mode not in {"geometry_wkt", "point_buffer", "province_day"}:
                    continue
                matched_mask = np.zeros(len(day_positions), dtype=bool)
                matched_mode = mode
                day_rows = event_samples.iloc[day_positions]

                if mode == "geometry_wkt" and latitudes is not None and longitudes is not None:
                    day_lat = latitudes.iloc[day_positions]
                    day_lon = longitudes.iloc[day_positions]
                    valid = day_lat.notna() & day_lon.notna()
                    if valid.any():
                        matched_values = [
                            _point_in_wkt_geometry(float(lat), float(lon), event.get("geometry_wkt"))
                            for lat, lon in zip(day_lat.loc[valid].tolist(), day_lon.loc[valid].tolist())
                        ]
                        matched_mask[np.flatnonzero(valid.to_numpy())] = np.asarray(matched_values, dtype=bool)
                elif mode == "point_buffer":
                    if latitudes is not None and longitudes is not None:
                        day_lat = latitudes.iloc[day_positions]
                        day_lon = longitudes.iloc[day_positions]
                        valid = day_lat.notna() & day_lon.notna()
                        if valid.any():
                            distances = _haversine_km_vectorized(
                                day_lat.loc[valid].to_numpy(),
                                day_lon.loc[valid].to_numpy(),
                                float(event["latitude"]),
                                float(event["longitude"]),
                            )
                            matched_mask[np.flatnonzero(valid.to_numpy())] = distances <= active_config.point_buffer_km
                        if (~valid).any() and province_series is not None:
                            event_provinces = set(_resolved_province_names(event, active_config))
                            if event_provinces:
                                invalid_positions = np.flatnonzero((~valid).to_numpy())
                                matched_mask[invalid_positions] = (
                                    province_series.iloc[day_positions].iloc[invalid_positions].isin(event_provinces)
                                ).to_numpy()
                                if matched_mask[invalid_positions].any():
                                    matched_mode = "province_day"
                    elif province_series is not None:
                        event_provinces = set(_resolved_province_names(event, active_config))
                        if event_provinces:
                            matched_mask = province_series.iloc[day_positions].isin(event_provinces).to_numpy()
                            if matched_mask.any():
                                matched_mode = "province_day"
                elif mode == "province_day" and province_series is not None:
                    event_provinces = set(_resolved_province_names(event, active_config))
                    if event_provinces:
                        matched_mask = province_series.iloc[day_positions].isin(event_provinces).to_numpy()

                for local_pos in np.flatnonzero(matched_mask):
                    absolute_pos = int(day_positions[local_pos])
                    matched_lists[absolute_pos].append(event)
                    mode_lists[absolute_pos].append(matched_mode)

            allow_event_day_negatives = _supports_event_day_negative(sample_columns, day_events, active_config) and all(
                mode in {"point_buffer", "province_day", "geometry_wkt"} for mode in day_event_mapping_modes
            )
            for absolute_pos in day_positions.tolist():
                matches = matched_lists[absolute_pos]
                modes = mode_lists[absolute_pos]
                if matches:
                    event_samples.iat[absolute_pos, event_samples.columns.get_loc("label")] = 1.0
                    event_samples.iat[absolute_pos, event_samples.columns.get_loc("label_status")] = "positive"
                    event_samples.iat[absolute_pos, event_samples.columns.get_loc("label_source_mode")] = ",".join(sorted(set(modes))) or "point_buffer"
                    event_samples.iat[absolute_pos, event_samples.columns.get_loc("matched_event_ids")] = ",".join(
                        event["event_id"] for event in matches
                    )
                elif allow_event_day_negatives:
                    event_samples.iat[absolute_pos, event_samples.columns.get_loc("label")] = 0.0
                    event_samples.iat[absolute_pos, event_samples.columns.get_loc("label_status")] = "negative"
                    event_samples.iat[absolute_pos, event_samples.columns.get_loc("label_source_mode")] = "outside_event_footprint"

        provenance_values: list[str] = []
        for date_value, matches, modes in zip(event_sample_dates, matched_lists, mode_lists):
            day_events = events_by_date.get(date_value, [])
            day_event_mapping_modes = [_event_mapping_mode(event, active_config) for event in day_events]
            provenance_values.append(
                json.dumps(
                    {
                        "date": date_value,
                        "event_count_for_day": len(day_events),
                        "matched_event_ids": [event["event_id"] for event in matches],
                        "matched_location_names": [event.get("location_name", "") for event in matches],
                        "mapping_modes": sorted(set(modes)),
                        "day_event_mapping_modes": day_event_mapping_modes,
                        "matched_geometry_wkts": [event.get("geometry_wkt") for event in matches if _normalize_text(event.get("geometry_wkt"))],
                        "point_buffer_km": active_config.point_buffer_km,
                        "province_fallback_enabled": active_config.province_fallback_enabled,
                        "emit_event_day_negatives": active_config.emit_event_day_negatives,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        event_samples["label_provenance"] = provenance_values
        for column in ("label", "label_status", "label_source_mode", "matched_event_ids", "label_provenance"):
            normalized_samples.loc[has_event_mask, column] = event_samples[column].to_numpy()
    return normalized_samples
