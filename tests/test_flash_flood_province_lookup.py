from __future__ import annotations

import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from mazu_saudi.data import build_flash_flood_province_lookup
from mazu_saudi.data.flash_flood_province_lookup import _prepare_ring, _prepared_point_to_ring_distance

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_flash_flood_province_lookup.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("build_flash_flood_province_lookup", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_flash_flood_province_lookup_maps_unique_coordinates_from_geojson_boundaries():
    features = pd.DataFrame(
        [
            {"latitude": 21.50, "longitude": 39.20, "daily_precip_total": 40.0},
            {"latitude": 21.50, "longitude": 39.20, "daily_precip_total": 55.0},
            {"latitude": 24.71, "longitude": 46.67, "daily_precip_total": 5.0},
            {"latitude": 18.20, "longitude": 42.50, "daily_precip_total": 1.0},
        ]
    )
    boundaries = pd.DataFrame(
        [
            {
                "boundary_id": "makkah-1",
                "province_name": "Makkah",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[39.0, 21.3], [39.4, 21.3], [39.4, 21.7], [39.0, 21.7], [39.0, 21.3]]],
                },
            },
            {
                "boundary_id": "riyadh-1",
                "province_name": "Riyadh",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[46.5, 24.5], [46.9, 24.5], [46.9, 24.9], [46.5, 24.9], [46.5, 24.5]]],
                },
            },
        ]
    )

    lookup = build_flash_flood_province_lookup(features, boundaries)

    assert lookup.columns.tolist() == ["latitude", "longitude", "province_name", "match_status", "matched_boundary_ids"]
    assert len(lookup) == 3
    assert lookup["province_name"].tolist() == ["makkah", "riyadh", None]
    assert lookup["match_status"].tolist() == ["matched", "matched", "unmatched"]
    assert lookup["matched_boundary_ids"].tolist() == ["makkah-1", "riyadh-1", ""]


def test_build_flash_flood_province_lookup_rejects_overlapping_boundaries():
    features = pd.DataFrame([{"latitude": 21.50, "longitude": 39.20}])
    boundaries = pd.DataFrame(
        [
            {
                "boundary_id": "makkah-1",
                "province_name": "Makkah",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[39.0, 21.3], [39.4, 21.3], [39.4, 21.7], [39.0, 21.7], [39.0, 21.3]]],
                },
            },
            {
                "boundary_id": "other-1",
                "province_name": "Other",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[39.1, 21.4], [39.5, 21.4], [39.5, 21.8], [39.1, 21.8], [39.1, 21.4]]],
                },
            },
        ]
    )

    try:
        build_flash_flood_province_lookup(features, boundaries)
    except ValueError as exc:
        assert "overlapping province geometries" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected overlapping boundaries to raise ValueError")


def test_build_flash_flood_province_lookup_supports_wkt_boundaries():
    features = pd.DataFrame(
        [
            {"latitude": 21.50, "longitude": 39.20},
            {"latitude": 24.71, "longitude": 46.67},
        ]
    )
    boundaries = pd.DataFrame(
        [
            {
                "boundary_id": "makkah-1",
                "province_name": "Makkah",
                "geometry_wkt": "POLYGON((39.0 21.3, 39.4 21.3, 39.4 21.7, 39.0 21.7, 39.0 21.3))",
            }
        ]
    )

    lookup = build_flash_flood_province_lookup(features, boundaries, geometry_column="geometry_wkt", geometry_format="wkt")

    assert lookup["province_name"].tolist() == ["makkah", None]
    assert lookup["match_status"].tolist() == ["matched", "unmatched"]


def test_build_flash_flood_province_lookup_treats_boundary_points_as_matched():
    features = pd.DataFrame(
        [
            {"latitude": 21.3, "longitude": 39.2},
            {"latitude": 21.5, "longitude": 39.0},
            {"latitude": 21.7, "longitude": 39.4},
        ]
    )
    boundaries = pd.DataFrame(
        [
            {
                "boundary_id": "makkah-1",
                "province_name": "Makkah",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[39.0, 21.3], [39.4, 21.3], [39.4, 21.7], [39.0, 21.7], [39.0, 21.3]]],
                },
            }
        ]
    )

    lookup = build_flash_flood_province_lookup(features, boundaries)

    assert lookup["province_name"].tolist() == ["makkah", "makkah", "makkah"]
    assert lookup["match_status"].tolist() == ["matched", "matched", "matched"]


def test_build_flash_flood_province_lookup_treats_hole_boundary_points_as_matched():
    features = pd.DataFrame(
        [
            {"latitude": 21.5, "longitude": 39.1},
            {"latitude": 21.5, "longitude": 39.2},
            {"latitude": 21.5, "longitude": 39.3},
        ]
    )
    boundaries = pd.DataFrame(
        [
            {
                "boundary_id": "ring-1",
                "province_name": "Makkah",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[39.0, 21.3], [39.4, 21.3], [39.4, 21.7], [39.0, 21.7], [39.0, 21.3]],
                        [[39.1, 21.4], [39.3, 21.4], [39.3, 21.6], [39.1, 21.6], [39.1, 21.4]],
                    ],
                },
            }
        ]
    )

    lookup = build_flash_flood_province_lookup(features, boundaries)

    assert lookup["province_name"].tolist() == ["makkah", None, "makkah"]
    assert lookup["match_status"].tolist() == ["matched", "unmatched", "matched"]


def test_prepared_point_to_ring_distance_respects_max_distance_cap():
    ring = _prepare_ring([(0.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, 0.0)])

    capped = _prepared_point_to_ring_distance(2.0, 10.0, ring, max_distance=1.5)
    uncapped = _prepared_point_to_ring_distance(2.0, 10.0, ring)

    assert capped == 1.5
    assert uncapped == 6.0


def test_build_flash_flood_province_lookup_script_reads_geojson_and_exports_csv(tmp_path: Path):
    module = _load_script_module()
    features = pd.DataFrame(
        [
            {"date": "2022-12-23", "latitude": 21.5, "longitude": 39.2},
            {"date": "2022-12-23", "latitude": 24.71, "longitude": 46.67},
            {"date": "2022-12-23", "latitude": 18.2, "longitude": 42.5},
        ]
    )
    boundary_payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "id": "makkah-1",
                "type": "Feature",
                "properties": {"province_name": "Makkah"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[39.0, 21.3], [39.4, 21.3], [39.4, 21.7], [39.0, 21.7], [39.0, 21.3]]],
                },
            },
            {
                "id": "riyadh-1",
                "type": "Feature",
                "properties": {"province_name": "Riyadh"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[46.5, 24.5], [46.9, 24.5], [46.9, 24.9], [46.5, 24.9], [46.5, 24.5]]],
                },
            },
        ],
    }
    feature_path = tmp_path / "features.csv"
    boundary_path = tmp_path / "boundaries.geojson"
    output_path = tmp_path / "lookup.csv"
    features.to_csv(feature_path, index=False)
    boundary_path.write_text(json.dumps(boundary_payload), encoding="utf-8")

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        assert module.main(["--features", str(feature_path), "--boundaries", str(boundary_path), "--output", str(output_path)]) == 0

    summary = json.loads(stdout.getvalue())
    exported = pd.read_csv(output_path)

    assert summary["input_rows"] == 3
    assert summary["unique_coordinate_rows"] == 3
    assert summary["matched_coordinate_rows"] == 2
    assert summary["unmatched_coordinate_rows"] == 1
    assert exported["province_name"].fillna("").tolist() == ["makkah", "riyadh", ""]
    assert exported["match_status"].tolist() == ["matched", "matched", "unmatched"]


def test_build_flash_flood_province_lookup_script_streams_parquet_features(tmp_path: Path):
    module = _load_script_module()
    features = pd.DataFrame(
        [
            {"date": "2022-12-23", "latitude": 21.5, "longitude": 39.2},
            {"date": "2022-12-23", "latitude": 21.5, "longitude": 39.2},
            {"date": "2022-12-23", "latitude": 24.71, "longitude": 46.67},
        ]
    )
    boundary_payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "id": "makkah-1",
                "type": "Feature",
                "properties": {"province_name": "Makkah"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[39.0, 21.3], [39.4, 21.3], [39.4, 21.7], [39.0, 21.7], [39.0, 21.3]]],
                },
            },
            {
                "id": "riyadh-1",
                "type": "Feature",
                "properties": {"province_name": "Riyadh"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[46.5, 24.5], [46.9, 24.5], [46.9, 24.9], [46.5, 24.9], [46.5, 24.5]]],
                },
            },
        ],
    }
    feature_path = tmp_path / "features.parquet"
    boundary_path = tmp_path / "boundaries.geojson"
    output_path = tmp_path / "lookup.parquet"
    features.to_parquet(feature_path, index=False)
    boundary_path.write_text(json.dumps(boundary_payload), encoding="utf-8")

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        assert module.main(["--features", str(feature_path), "--boundaries", str(boundary_path), "--output", str(output_path)]) == 0

    summary = json.loads(stdout.getvalue())
    exported = pd.read_parquet(output_path)

    assert summary["input_rows"] == 3
    assert summary["unique_coordinate_rows"] == 2
    assert summary["matched_coordinate_rows"] == 2
    assert exported["province_name"].tolist() == ["makkah", "riyadh"]


def test_build_flash_flood_province_lookup_script_autodetects_geoboundaries_columns(tmp_path: Path):
    module = _load_script_module()
    features = pd.DataFrame(
        [
            {"date": "2022-12-23", "latitude": 21.5, "longitude": 39.2},
            {"date": "2022-12-23", "latitude": 24.71, "longitude": 46.67},
        ]
    )
    boundary_payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"shapeName": "Makkah", "shapeID": "makkah-1"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[39.0, 21.3], [39.4, 21.3], [39.4, 21.7], [39.0, 21.7], [39.0, 21.3]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"shapeName": "Riyadh", "shapeID": "riyadh-1"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[46.5, 24.5], [46.9, 24.5], [46.9, 24.9], [46.5, 24.9], [46.5, 24.5]]],
                },
            },
        ],
    }
    feature_path = tmp_path / "features.csv"
    boundary_path = tmp_path / "boundaries.geojson"
    output_path = tmp_path / "lookup.csv"
    features.to_csv(feature_path, index=False)
    boundary_path.write_text(json.dumps(boundary_payload), encoding="utf-8")

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        assert module.main(["--features", str(feature_path), "--boundaries", str(boundary_path), "--output", str(output_path)]) == 0

    summary = json.loads(stdout.getvalue())
    exported = pd.read_csv(output_path)

    assert summary["boundary_province_column"] == "shapeName"
    assert summary["boundary_id_column"] == "shapeID"
    assert exported["matched_boundary_ids"].tolist() == ["makkah-1", "riyadh-1"]
