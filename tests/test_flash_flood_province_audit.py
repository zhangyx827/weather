from __future__ import annotations

import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from mazu_saudi.data import audit_flash_flood_province_lookup


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "audit_flash_flood_province_lookup.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("audit_flash_flood_province_lookup", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_audit_flash_flood_province_lookup_summarizes_unmatched_hotspots():
    lookup = pd.DataFrame(
        [
            {"latitude": 21.5, "longitude": 39.2, "province_name": "makkah", "match_status": "matched"},
            {"latitude": 19.4, "longitude": 41.1, "province_name": None, "match_status": "unmatched"},
            {"latitude": 30.2, "longitude": 50.4, "province_name": None, "match_status": "unmatched"},
        ]
    )
    features = pd.DataFrame(
        [
            {"latitude": 21.5, "longitude": 39.2},
            {"latitude": 19.4, "longitude": 41.1},
            {"latitude": 19.4, "longitude": 41.1},
            {"latitude": 30.2, "longitude": 50.4},
            {"latitude": 30.2, "longitude": 50.4},
            {"latitude": 30.2, "longitude": 50.4},
        ]
    )
    boundaries = pd.DataFrame(
        [
            {
                "province_name": "Makkah",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[39.0, 21.3], [39.4, 21.3], [39.4, 21.7], [39.0, 21.7], [39.0, 21.3]]],
                },
            },
            {
                "province_name": "Asir",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[41.0, 19.0], [41.5, 19.0], [41.5, 19.3], [41.0, 19.3], [41.0, 19.0]]],
                },
            },
        ]
    )

    summary = audit_flash_flood_province_lookup(lookup, feature_table=features, boundary_table=boundaries, top_n=2)

    assert summary["unmatched_coordinate_rows"] == 2
    assert summary["unmatched_feature_rows"] == 5
    assert summary["unmatched_boundary_classification_counts"] == {
        "inside_boundary_bbox_outside_polygon": 1,
        "outside_boundary_bbox": 1,
    }
    assert summary["inside_bbox_boundary_gap_band_counts"] == {"within_0_10_degrees": 1}
    assert summary["weighted_inside_bbox_boundary_gap_band_feature_rows"] == {"within_0_10_degrees": 2}
    assert summary["top_unmatched_coordinates"][0]["feature_row_count"] == 3
    assert summary["top_unmatched_coordinates"][0]["latitude"] == 30.2
    assert summary["top_unmatched_coordinates"][1]["boundary_gap_band"] == "within_0_10_degrees"


def test_audit_flash_flood_province_lookup_runtime_stats_identify_zero_candidate_rows():
    lookup = pd.DataFrame(
        [
            {"latitude": 21.5, "longitude": 39.2, "province_name": "makkah", "match_status": "matched"},
            {"latitude": 20.2, "longitude": 40.2, "province_name": None, "match_status": "unmatched"},
            {"latitude": 19.4, "longitude": 41.1, "province_name": None, "match_status": "unmatched"},
            {"latitude": 30.2, "longitude": 50.4, "province_name": None, "match_status": "unmatched"},
        ]
    )
    boundaries = pd.DataFrame(
        [
            {
                "province_name": "Makkah",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[39.0, 21.3], [39.4, 21.3], [39.4, 21.7], [39.0, 21.7], [39.0, 21.3]]],
                },
            },
            {
                "province_name": "Asir",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[41.0, 19.0], [41.5, 19.0], [41.5, 19.3], [41.0, 19.3], [41.0, 19.0]]],
                },
            },
        ]
    )

    summary = audit_flash_flood_province_lookup(
        lookup,
        boundary_table=boundaries,
        top_n=2,
        include_runtime_stats=True,
    )

    assert summary["runtime_stats"]["inside_global_bbox_unmatched_rows"] == 2
    assert summary["runtime_stats"]["outside_global_bbox_unmatched_rows"] == 1
    assert summary["runtime_stats"]["inside_global_bbox_zero_candidate_rows"] == 1
    assert summary["runtime_stats"]["inside_global_bbox_nonzero_candidate_rows"] == 1
    assert summary["runtime_stats"]["inside_global_bbox_broad_scan_rows"] == 1
    assert summary["runtime_stats"]["candidate_boundary_checks"] == 1
    assert summary["runtime_stats"]["candidate_count_stats"]["max"] == 1
    assert summary["runtime_stats"]["point_in_any_polygon_calls"] == 1
    assert summary["runtime_stats"]["point_to_any_polygon_distance_calls"] == 1


def test_audit_flash_flood_province_lookup_script_reports_json_summary(tmp_path: Path):
    module = _load_script_module()
    lookup = pd.DataFrame(
        [
            {"latitude": 21.5, "longitude": 39.2, "province_name": "makkah", "match_status": "matched"},
            {"latitude": 19.4, "longitude": 41.1, "province_name": None, "match_status": "unmatched"},
        ]
    )
    features = pd.DataFrame(
        [
            {"latitude": 21.5, "longitude": 39.2},
            {"latitude": 19.4, "longitude": 41.1},
            {"latitude": 19.4, "longitude": 41.1},
        ]
    )
    lookup_path = tmp_path / "lookup.parquet"
    feature_path = tmp_path / "features.csv"
    lookup.to_parquet(lookup_path, index=False)
    features.to_csv(feature_path, index=False)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        assert module.main(["--lookup", str(lookup_path), "--features", str(feature_path), "--top-n", "1"]) == 0

    summary = json.loads(stdout.getvalue())
    assert summary["unique_coordinate_rows"] == 2
    assert summary["unmatched_feature_rows"] == 2
    assert len(summary["top_unmatched_coordinates"]) == 1


def test_audit_flash_flood_province_lookup_script_emits_runtime_stats_and_cprofile(tmp_path: Path):
    module = _load_script_module()
    lookup = pd.DataFrame(
        [
            {"latitude": 21.5, "longitude": 39.2, "province_name": "makkah", "match_status": "matched"},
            {"latitude": 19.4, "longitude": 41.1, "province_name": None, "match_status": "unmatched"},
        ]
    )
    boundaries = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"shapeName": "Makkah"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[39.0, 21.3], [39.4, 21.3], [39.4, 21.7], [39.0, 21.7], [39.0, 21.3]]],
                },
            }
        ],
    }
    lookup_path = tmp_path / "lookup.parquet"
    boundary_path = tmp_path / "boundaries.geojson"
    lookup.to_parquet(lookup_path, index=False)
    boundary_path.write_text(json.dumps(boundaries), encoding="utf-8")

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        assert (
            module.main(
                [
                    "--lookup",
                    str(lookup_path),
                    "--boundaries",
                    str(boundary_path),
                    "--runtime-stats",
                    "--cprofile",
                    "--cprofile-top",
                    "3",
                ]
            )
            == 0
        )

    summary = json.loads(stdout.getvalue())
    assert summary["runtime_stats"]["inside_global_bbox_unmatched_rows"] == 0
    assert summary["runtime_stats"]["outside_global_bbox_unmatched_rows"] == 1
    assert summary["cprofile"]["top_n"] == 3
    assert len(summary["cprofile"]["top_functions"]) <= 3
