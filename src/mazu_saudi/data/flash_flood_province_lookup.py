"""Build reusable latitude/longitude to province lookup tables for flash-flood features."""

from __future__ import annotations

import json
import math
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for flash-flood province lookup assembly")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _split_groups(text: str) -> list[str]:
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
    ring_groups = _split_groups(inner[1:-1].strip())
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
        return [_parse_polygon_text(text[len("polygon"):].strip())]
    if text.startswith("multipolygon"):
        inner = text[len("multipolygon"):].strip()
        if not inner.startswith("(") or not inner.endswith(")"):
            raise ValueError("Invalid WKT multipolygon text")
        polygon_groups = _split_groups(inner[1:-1].strip())
        return [_parse_polygon_text(group) for group in polygon_groups]
    raise ValueError(f"Unsupported WKT geometry type: {value}")


def _parse_geojson_rings(rings: Any) -> list[list[tuple[float, float]]]:
    parsed: list[list[tuple[float, float]]] = []
    for ring in rings or []:
        coordinates: list[tuple[float, float]] = []
        for point in ring or []:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            lon = float(point[0])
            lat = float(point[1])
            coordinates.append((lat, lon))
        if len(coordinates) < 3:
            raise ValueError("GeoJSON ring requires at least three coordinate pairs")
        parsed.append(coordinates)
    if not parsed:
        raise ValueError("GeoJSON polygon contains no rings")
    return parsed


def _parse_geojson_polygons(value: Any) -> list[list[list[tuple[float, float]]]]:
    geometry = value
    if isinstance(geometry, str):
        geometry = json.loads(geometry)
    if not isinstance(geometry, dict):
        raise ValueError("GeoJSON geometry must be a dict or JSON string")
    geometry_type = _normalize_text(geometry.get("type"))
    coordinates = geometry.get("coordinates")
    if geometry_type == "polygon":
        return [_parse_geojson_rings(coordinates)]
    if geometry_type == "multipolygon":
        return [_parse_geojson_rings(polygon) for polygon in coordinates or []]
    raise ValueError(f"Unsupported GeoJSON geometry type: {geometry.get('type')}")


def _parse_boundary_polygons(boundary: Any, *, geometry_column: str, geometry_format: str) -> list[list[list[tuple[float, float]]]]:
    if geometry_column not in boundary:
        raise KeyError(f"boundary table is missing geometry column: {geometry_column}")
    geometry_value = boundary.get(geometry_column)
    if geometry_format == "geojson":
        return _parse_geojson_polygons(geometry_value)
    if geometry_format == "wkt":
        return _parse_wkt_polygons(geometry_value)
    raise ValueError(f"Unsupported geometry format: {geometry_format}")


def _point_on_segment(
    latitude: float,
    longitude: float,
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    *,
    tolerance: float = 1e-9,
) -> bool:
    segment_lat = lat2 - lat1
    segment_lon = lon2 - lon1
    squared_length = segment_lat * segment_lat + segment_lon * segment_lon
    if squared_length <= tolerance * tolerance:
        return math.hypot(latitude - lat1, longitude - lon1) <= tolerance
    point_lat = latitude - lat1
    point_lon = longitude - lon1
    cross = segment_lat * point_lon - segment_lon * point_lat
    if abs(cross) > tolerance:
        return False
    dot = point_lat * segment_lat + point_lon * segment_lon
    if dot < -tolerance:
        return False
    if dot - squared_length > tolerance:
        return False
    return True


def _prepared_point_on_ring_boundary(
    latitude: float,
    longitude: float,
    prepared_ring: dict[str, Any],
    *,
    tolerance: float = 1e-9,
) -> bool:
    min_lat, max_lat, min_lon, max_lon = prepared_ring["bbox"]
    if latitude < min_lat - tolerance or latitude > max_lat + tolerance or longitude < min_lon - tolerance or longitude > max_lon + tolerance:
        return False
    for lat1, lon1, lat2, lon2, segment_bbox, *_segment_metrics in prepared_ring["segment_records"]:
        seg_min_lat, seg_max_lat, seg_min_lon, seg_max_lon = segment_bbox
        if (
            latitude < seg_min_lat - tolerance
            or latitude > seg_max_lat + tolerance
            or longitude < seg_min_lon - tolerance
            or longitude > seg_max_lon + tolerance
        ):
            continue
        if _point_on_segment(latitude, longitude, lat1, lon1, lat2, lon2, tolerance=tolerance):
            return True
    return False


def _point_on_ring_boundary(latitude: float, longitude: float, ring: list[tuple[float, float]], *, tolerance: float = 1e-9) -> bool:
    return _prepared_point_on_ring_boundary(latitude, longitude, _prepare_ring(ring), tolerance=tolerance)


def _prepared_point_in_ring(latitude: float, longitude: float, prepared_ring: dict[str, Any]) -> bool:
    min_lat, max_lat, min_lon, max_lon = prepared_ring["bbox"]
    if latitude < min_lat or latitude > max_lat or longitude < min_lon or longitude > max_lon:
        return False
    inside = False
    for lat1, lon1, lat2, lon2, segment_bbox, *_segment_metrics in prepared_ring["segment_records"]:
        seg_min_lat, seg_max_lat, _, seg_max_lon = segment_bbox
        if latitude <= seg_min_lat or latitude > seg_max_lat or seg_max_lon <= longitude:
            continue
        denominator = lat2 - lat1
        if abs(denominator) < 1e-12:
            continue
        cross_lon = lon1 + (latitude - lat1) * (lon2 - lon1) / denominator
        if cross_lon > longitude:
            inside = not inside
    return inside


def _point_in_ring(latitude: float, longitude: float, ring: list[tuple[float, float]]) -> bool:
    return _prepared_point_in_ring(latitude, longitude, _prepare_ring(ring))


def _prepared_point_in_polygon(latitude: float, longitude: float, prepared_polygon: dict[str, Any]) -> bool:
    if not prepared_polygon["rings"]:
        return False
    if not _bbox_contains(latitude, longitude, prepared_polygon["bbox"]):
        return False
    outer_ring = prepared_polygon["rings"][0]
    outer_inside = _prepared_point_in_ring(latitude, longitude, outer_ring)
    if not outer_inside:
        if _point_to_bbox_distance(latitude, longitude, outer_ring["bbox"]) > 1e-9:
            return False
        return _prepared_point_on_ring_boundary(latitude, longitude, outer_ring)
    for ring in prepared_polygon["rings"][1:]:
        hole_inside = _prepared_point_in_ring(latitude, longitude, ring)
        if not hole_inside:
            if _point_to_bbox_distance(latitude, longitude, ring["bbox"]) > 1e-9:
                continue
            if _prepared_point_on_ring_boundary(latitude, longitude, ring):
                return True
            continue
        if _prepared_point_on_ring_boundary(latitude, longitude, ring):
            return True
        return False
    return True


def _point_in_polygon(latitude: float, longitude: float, polygon: list[list[tuple[float, float]]]) -> bool:
    return _prepared_point_in_polygon(latitude, longitude, _prepare_polygon(polygon))


def _point_in_any_polygon(latitude: float, longitude: float, polygons: list[list[list[tuple[float, float]]]]) -> bool:
    return any(
        _bbox_contains(latitude, longitude, polygon["bbox"]) and _prepared_point_in_polygon(latitude, longitude, polygon)
        for polygon in polygons
    )


def _point_to_segment_distance(latitude: float, longitude: float, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return math.sqrt(_point_to_segment_distance_squared(latitude, longitude, lat1, lon1, lat2, lon2))


def _point_to_segment_distance_squared(
    latitude: float,
    longitude: float,
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    segment_lat = lat2 - lat1
    segment_lon = lon2 - lon1
    squared_length = segment_lat * segment_lat + segment_lon * segment_lon
    if squared_length <= 0:
        latitude_delta = latitude - lat1
        longitude_delta = longitude - lon1
        return latitude_delta * latitude_delta + longitude_delta * longitude_delta
    projection = ((latitude - lat1) * segment_lat + (longitude - lon1) * segment_lon) / squared_length
    projection = max(0.0, min(1.0, projection))
    projected_lat = lat1 + projection * segment_lat
    projected_lon = lon1 + projection * segment_lon
    latitude_delta = latitude - projected_lat
    longitude_delta = longitude - projected_lon
    return latitude_delta * latitude_delta + longitude_delta * longitude_delta


def _point_to_prepared_segment_distance_squared(
    latitude: float,
    longitude: float,
    lat1: float,
    lon1: float,
    segment_lat: float,
    segment_lon: float,
    squared_length: float,
) -> float:
    if squared_length <= 0:
        latitude_delta = latitude - lat1
        longitude_delta = longitude - lon1
        return latitude_delta * latitude_delta + longitude_delta * longitude_delta
    projection = ((latitude - lat1) * segment_lat + (longitude - lon1) * segment_lon) / squared_length
    projection = max(0.0, min(1.0, projection))
    projected_lat = lat1 + projection * segment_lat
    projected_lon = lon1 + projection * segment_lon
    latitude_delta = latitude - projected_lat
    longitude_delta = longitude - projected_lon
    return latitude_delta * latitude_delta + longitude_delta * longitude_delta


def _point_to_ring_distance(
    latitude: float,
    longitude: float,
    ring: list[tuple[float, float]],
    *,
    max_distance: float = math.inf,
) -> float:
    prepared_ring = _prepare_ring(ring)
    return _prepared_point_to_ring_distance(latitude, longitude, prepared_ring, max_distance=max_distance)


def _prepared_point_to_ring_distance(
    latitude: float,
    longitude: float,
    prepared_ring: dict[str, Any],
    *,
    max_distance: float = math.inf,
) -> float:
    best_distance_squared = math.inf if math.isinf(max_distance) else max_distance * max_distance
    for lat1, lon1, _lat2, _lon2, segment_bbox, segment_lat, segment_lon, squared_length in prepared_ring["segment_records"]:
        bbox_distance_squared = _point_to_bbox_distance_squared(latitude, longitude, segment_bbox)
        if bbox_distance_squared >= best_distance_squared:
            continue
        distance_squared = _point_to_prepared_segment_distance_squared(
            latitude,
            longitude,
            lat1,
            lon1,
            segment_lat,
            segment_lon,
            squared_length,
        )
        if distance_squared < best_distance_squared:
            best_distance_squared = distance_squared
    return math.sqrt(best_distance_squared)


def _point_to_polygon_distance(
    latitude: float,
    longitude: float,
    polygon: list[list[tuple[float, float]]],
    *,
    max_distance: float = math.inf,
) -> float:
    prepared_polygon = _prepare_polygon(polygon)
    return _prepared_point_to_polygon_distance(latitude, longitude, prepared_polygon, max_distance=max_distance)


def _prepared_point_to_polygon_distance(
    latitude: float,
    longitude: float,
    prepared_polygon: dict[str, Any],
    *,
    max_distance: float = math.inf,
) -> float:
    if _prepared_point_in_polygon(latitude, longitude, prepared_polygon):
        return 0.0
    best_distance_squared = math.inf if math.isinf(max_distance) else max_distance * max_distance
    if _point_to_bbox_distance_squared(latitude, longitude, prepared_polygon["bbox"]) >= best_distance_squared:
        return max_distance
    rings = prepared_polygon["rings"]
    if len(rings) == 1:
        return _prepared_point_to_ring_distance(latitude, longitude, rings[0], max_distance=max_distance)
    candidate_rings = sorted(
        (
            _point_to_bbox_distance_squared(latitude, longitude, ring["bbox"]),
            ring,
        )
        for ring in rings
    )
    best_distance = max_distance
    for bbox_distance_squared, ring in candidate_rings:
        if bbox_distance_squared >= best_distance_squared:
            break
        distance = _prepared_point_to_ring_distance(latitude, longitude, ring, max_distance=best_distance)
        if distance < best_distance:
            best_distance = distance
            best_distance_squared = distance * distance
    return best_distance


def _point_to_any_polygon_distance(
    latitude: float,
    longitude: float,
    polygons: list[list[list[tuple[float, float]]]],
    *,
    max_distance: float = math.inf,
) -> float:
    if not polygons:
        return math.inf
    if len(polygons) == 1:
        return _prepared_point_to_polygon_distance(latitude, longitude, polygons[0], max_distance=max_distance)
    best_distance_squared = math.inf if math.isinf(max_distance) else max_distance * max_distance
    candidate_distances = sorted(
        (
            _point_to_bbox_distance_squared(latitude, longitude, polygon["bbox"]),
            polygon,
        )
        for polygon in polygons
    )
    best_distance = max_distance
    for bbox_distance_squared, polygon in candidate_distances:
        if bbox_distance_squared >= best_distance_squared:
            break
        exact_distance = _prepared_point_to_polygon_distance(latitude, longitude, polygon, max_distance=best_distance)
        if exact_distance < best_distance:
            best_distance = exact_distance
            best_distance_squared = exact_distance * exact_distance
    return best_distance


def _ring_bbox(ring: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    latitudes = [latitude for latitude, _ in ring]
    longitudes = [longitude for _, longitude in ring]
    return (min(latitudes), max(latitudes), min(longitudes), max(longitudes))


def _prepare_ring(ring: Any) -> dict[str, Any]:
    if isinstance(ring, dict) and "segments" in ring and "points" in ring and "bbox" in ring:
        return ring
    if not isinstance(ring, list) or len(ring) < 3:
        raise ValueError("Ring requires at least three coordinate pairs")
    points = [(float(latitude), float(longitude)) for latitude, longitude in ring]
    segments = [
        (lat1, lon1, lat2, lon2)
        for index, (lat1, lon1) in enumerate(points)
        for lat2, lon2 in [points[(index + 1) % len(points)]]
    ]
    segment_records = []
    for lat1, lon1, lat2, lon2 in segments:
        segment_lat = lat2 - lat1
        segment_lon = lon2 - lon1
        segment_records.append(
            (
                lat1,
                lon1,
                lat2,
                lon2,
                (min(lat1, lat2), max(lat1, lat2), min(lon1, lon2), max(lon1, lon2)),
                segment_lat,
                segment_lon,
                segment_lat * segment_lat + segment_lon * segment_lon,
            )
        )
    return {
        "points": points,
        "segments": segments,
        "segment_records": segment_records,
        "bbox": _ring_bbox(points),
    }


def _prepare_polygon(polygon: Any) -> dict[str, Any]:
    if isinstance(polygon, dict) and "rings" in polygon and "bbox" in polygon:
        return polygon
    if not isinstance(polygon, list) or not polygon:
        raise ValueError("Polygon requires at least one ring")
    rings = [_prepare_ring(ring) for ring in polygon]
    return {
        "rings": rings,
        "bbox": (
            min(ring["bbox"][0] for ring in rings),
            max(ring["bbox"][1] for ring in rings),
            min(ring["bbox"][2] for ring in rings),
            max(ring["bbox"][3] for ring in rings),
        ),
    }


def _prepare_polygons(polygons: list[list[list[tuple[float, float]]]]) -> list[dict[str, Any]]:
    return [_prepare_polygon(polygon) for polygon in polygons]


def _prepared_polygons_bbox(polygons: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    return (
        min(polygon["bbox"][0] for polygon in polygons),
        max(polygon["bbox"][1] for polygon in polygons),
        min(polygon["bbox"][2] for polygon in polygons),
        max(polygon["bbox"][3] for polygon in polygons),
    )


def _polygon_bbox(polygon: list[list[tuple[float, float]]]) -> tuple[float, float, float, float]:
    prepared_polygon = _prepare_polygon(polygon)
    return prepared_polygon["bbox"]


def _polygons_bbox(polygons: list[list[list[tuple[float, float]]]]) -> tuple[float, float, float, float]:
    return _prepared_polygons_bbox(_prepare_polygons(polygons))


def _bbox_contains(latitude: float, longitude: float, bbox: tuple[float, float, float, float]) -> bool:
    min_lat, max_lat, min_lon, max_lon = bbox
    return min_lat <= latitude <= max_lat and min_lon <= longitude <= max_lon


def _point_to_bbox_distance(latitude: float, longitude: float, bbox: tuple[float, float, float, float]) -> float:
    return math.sqrt(_point_to_bbox_distance_squared(latitude, longitude, bbox))


def _point_to_bbox_distance_squared(latitude: float, longitude: float, bbox: tuple[float, float, float, float]) -> float:
    min_lat, max_lat, min_lon, max_lon = bbox
    latitude_gap = 0.0
    longitude_gap = 0.0
    if latitude < min_lat:
        latitude_gap = min_lat - latitude
    elif latitude > max_lat:
        latitude_gap = latitude - max_lat
    if longitude < min_lon:
        longitude_gap = min_lon - longitude
    elif longitude > max_lon:
        longitude_gap = longitude - max_lon
    return latitude_gap * latitude_gap + longitude_gap * longitude_gap


def _bucket_span(min_value: float, max_value: float, *, bucket_size: float) -> range:
    start = math.floor(min_value / bucket_size)
    stop = math.floor(max_value / bucket_size)
    return range(start, stop + 1)


def _build_bbox_bucket_index(
    prepared_boundaries: list[dict[str, Any]],
    *,
    bucket_size_degrees: float = 1.0,
) -> dict[tuple[int, int], list[int]]:
    if bucket_size_degrees <= 0:
        raise ValueError("bucket_size_degrees must be positive")
    index: dict[tuple[int, int], list[int]] = {}
    for boundary_index, boundary in enumerate(prepared_boundaries):
        min_lat, max_lat, min_lon, max_lon = boundary["bbox"]
        for lat_bucket in _bucket_span(min_lat, max_lat, bucket_size=bucket_size_degrees):
            for lon_bucket in _bucket_span(min_lon, max_lon, bucket_size=bucket_size_degrees):
                index.setdefault((lat_bucket, lon_bucket), []).append(boundary_index)
    return index


def _candidate_boundary_indices(
    latitude: float,
    longitude: float,
    bbox_bucket_index: dict[tuple[int, int], list[int]],
    *,
    bucket_size_degrees: float = 1.0,
) -> list[int]:
    lat_bucket = math.floor(latitude / bucket_size_degrees)
    lon_bucket = math.floor(longitude / bucket_size_degrees)
    return bbox_bucket_index.get((lat_bucket, lon_bucket), [])


def _candidate_boundary_indices_with_radius(
    latitude: float,
    longitude: float,
    bbox_bucket_index: dict[tuple[int, int], list[int]],
    *,
    bucket_size_degrees: float = 1.0,
    radius: int = 1,
) -> list[int]:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    lat_bucket = math.floor(latitude / bucket_size_degrees)
    lon_bucket = math.floor(longitude / bucket_size_degrees)
    if radius == 0:
        return bbox_bucket_index.get((lat_bucket, lon_bucket), [])
    indices: list[int] = []
    seen: set[int] = set()
    for lat_offset in range(-radius, radius + 1):
        for lon_offset in range(-radius, radius + 1):
            for boundary_index in bbox_bucket_index.get((lat_bucket + lat_offset, lon_bucket + lon_offset), []):
                if boundary_index in seen:
                    continue
                seen.add(boundary_index)
                indices.append(boundary_index)
    return indices


def build_flash_flood_province_lookup(
    feature_table: Any,
    boundary_table: Any,
    *,
    province_column: str = "province_name",
    boundary_province_column: str = "province_name",
    boundary_id_column: str | None = "boundary_id",
    geometry_column: str = "geometry",
    geometry_format: str = "geojson",
    coordinate_precision: int = 4,
):
    """Map unique feature-table coordinates to province names using polygon boundaries."""

    _require_pandas()
    if not isinstance(feature_table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for feature_table, got {type(feature_table)!r}")
    if not isinstance(boundary_table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for boundary_table, got {type(boundary_table)!r}")
    if boundary_province_column not in boundary_table.columns:
        raise KeyError(f"boundary_table is missing province column: {boundary_province_column}")
    missing = [column for column in ("latitude", "longitude") if column not in feature_table.columns]
    if missing:
        raise KeyError(f"feature_table is missing required coordinate columns: {missing}")

    coordinates = feature_table.loc[:, ["latitude", "longitude"]].copy()
    coordinates["latitude"] = pd.to_numeric(coordinates["latitude"], errors="coerce").round(coordinate_precision)
    coordinates["longitude"] = pd.to_numeric(coordinates["longitude"], errors="coerce").round(coordinate_precision)
    coordinates = coordinates.dropna(subset=["latitude", "longitude"]).drop_duplicates().reset_index(drop=True)
    if coordinates.empty:
        raise ValueError("feature_table contains no usable latitude/longitude rows")

    boundaries = boundary_table.reset_index(drop=True).copy()
    boundaries[boundary_province_column] = boundaries[boundary_province_column].map(_normalize_text)
    boundaries = boundaries[boundaries[boundary_province_column].ne("")].copy()
    if boundaries.empty:
        raise ValueError("boundary_table contains no usable province rows")

    prepared_boundaries: list[dict[str, Any]] = []
    for record in boundaries.to_dict(orient="records"):
        polygons = _parse_boundary_polygons(record, geometry_column=geometry_column, geometry_format=geometry_format)
        prepared_polygons = _prepare_polygons(polygons)
        prepared_boundaries.append(
            {
                "province_name": record[boundary_province_column],
                "boundary_id": record.get(boundary_id_column) if boundary_id_column else None,
                "polygons": prepared_polygons,
                "bbox": _prepared_polygons_bbox(prepared_polygons),
            }
        )
    bbox_bucket_index = _build_bbox_bucket_index(prepared_boundaries)

    records: list[dict[str, Any]] = []
    for coordinate in coordinates.to_dict(orient="records"):
        latitude = float(coordinate["latitude"])
        longitude = float(coordinate["longitude"])
        candidate_boundaries = [
            prepared_boundaries[index]
            for index in _candidate_boundary_indices(latitude, longitude, bbox_bucket_index)
            if _bbox_contains(latitude, longitude, prepared_boundaries[index]["bbox"])
        ]
        matches = [
            boundary for boundary in candidate_boundaries if _point_in_any_polygon(latitude, longitude, boundary["polygons"])
        ]
        matched_provinces = sorted({boundary["province_name"] for boundary in matches})
        if len(matched_provinces) > 1:
            raise ValueError(
                "boundary_table contains overlapping province geometries for at least one feature-table coordinate"
            )
        province_name = matched_provinces[0] if matched_provinces else None
        matched_boundary_ids = sorted(
            {str(boundary["boundary_id"]) for boundary in matches if boundary.get("boundary_id") not in (None, "")}
        )
        records.append(
            {
                "latitude": latitude,
                "longitude": longitude,
                province_column: province_name,
                "match_status": "matched" if province_name else "unmatched",
                "matched_boundary_ids": ",".join(matched_boundary_ids),
            }
        )
    return pd.DataFrame.from_records(records)
