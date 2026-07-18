from __future__ import annotations

import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from mazu_saudi.data import (
    aggregate_flash_flood_features_to_province_day,
    build_flash_flood_supervised_training_dataset,
    build_flash_flood_training_labels,
    enrich_flash_flood_features_with_province,
    province_day_numeric_feature_columns,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_flash_flood_province_day_feature_table.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("build_flash_flood_province_day_feature_table", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_enrich_flash_flood_features_with_province_uses_coordinate_lookup():
    features = pd.DataFrame(
        [
            {"date": "2022-12-23", "hazard_type": "flash_flood", "latitude": 21.50004, "longitude": 39.20004, "daily_precip_total": 40.0},
            {"date": "2022-12-23", "hazard_type": "flash_flood", "latitude": 24.71004, "longitude": 46.67004, "daily_precip_total": 5.0},
        ]
    )
    lookup = pd.DataFrame(
        [
            {"latitude": 21.5, "longitude": 39.2, "province_name": "Makkah"},
            {"latitude": 24.71, "longitude": 46.67, "province_name": "Riyadh"},
        ]
    )

    enriched = enrich_flash_flood_features_with_province(features, lookup, coordinate_precision=2)

    assert enriched["province_name"].tolist() == ["makkah", "riyadh"]


def test_enrich_flash_flood_features_with_province_rejects_conflicting_lookup_rows():
    features = pd.DataFrame([{"date": "2022-12-23", "latitude": 21.5, "longitude": 39.2, "daily_precip_total": 40.0}])
    lookup = pd.DataFrame(
        [
            {"latitude": 21.5, "longitude": 39.2, "province_name": "Makkah"},
            {"latitude": 21.5, "longitude": 39.2, "province_name": "Riyadh"},
        ]
    )

    try:
        enrich_flash_flood_features_with_province(features, lookup)
    except ValueError as exc:
        assert "conflicting province assignments" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected conflicting province lookup to raise ValueError")


def test_aggregate_flash_flood_features_to_province_day_means_numeric_features():
    features = pd.DataFrame(
        [
            {
                "date": "2022-12-23",
                "hazard_type": "flash_flood",
                "latitude": 21.5,
                "longitude": 39.2,
                "province_name": "Makkah",
                "daily_precip_total": 40.0,
                "cape": 1200.0,
                "source_status": "primary",
            },
            {
                "date": "2022-12-23",
                "hazard_type": "flash_flood",
                "latitude": 21.6,
                "longitude": 39.3,
                "province_name": "Makkah",
                "daily_precip_total": 60.0,
                "cape": 1800.0,
                "source_status": "degraded",
            },
        ]
    )

    aggregated = aggregate_flash_flood_features_to_province_day(features)

    assert aggregated.columns.tolist()[:5] == [
        "date",
        "hazard_type",
        "province_name",
        "grid_cell_count",
        "degraded_grid_cell_count",
    ]
    assert "latitude" not in aggregated.columns
    assert "longitude" not in aggregated.columns
    assert aggregated.loc[0, "province_name"] == "makkah"
    assert aggregated.loc[0, "grid_cell_count"] == 2
    assert aggregated.loc[0, "degraded_grid_cell_count"] == 1
    assert aggregated.loc[0, "daily_precip_total"] == 50.0
    assert aggregated.loc[0, "cape"] == 1500.0


def test_province_day_numeric_feature_columns_excludes_source_provenance_fields():
    features = pd.DataFrame(
        [
            {
                "date": "2022-12-23",
                "hazard_type": "flash_flood",
                "latitude": 21.5,
                "longitude": 39.2,
                "province_name": "Makkah",
                "daily_precip_total": 40.0,
                "cape": 1200.0,
                "source_mtime_ns": 123,
                "source_mtime_us": 456,
                "source_size_bytes": 789,
            }
        ]
    )

    numeric_columns = province_day_numeric_feature_columns(features)
    aggregated = aggregate_flash_flood_features_to_province_day(features)

    assert numeric_columns == ["daily_precip_total", "cape"]
    assert "source_mtime_ns" not in aggregated.columns
    assert "source_mtime_us" not in aggregated.columns
    assert "source_size_bytes" not in aggregated.columns


def test_province_day_features_support_text_only_flash_flood_labels():
    features = pd.DataFrame(
        [
            {"date": "2022-12-23", "hazard_type": "flash_flood", "latitude": 21.5, "longitude": 39.2, "daily_precip_total": 40.0},
            {"date": "2022-12-23", "hazard_type": "flash_flood", "latitude": 21.6, "longitude": 39.3, "daily_precip_total": 60.0},
            {"date": "2022-12-23", "hazard_type": "flash_flood", "latitude": 24.71, "longitude": 46.67, "daily_precip_total": 5.0},
        ]
    )
    lookup = pd.DataFrame(
        [
            {"latitude": 21.5, "longitude": 39.2, "province_name": "Makkah"},
            {"latitude": 21.6, "longitude": 39.3, "province_name": "Makkah"},
            {"latitude": 24.71, "longitude": 46.67, "province_name": "Riyadh"},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "ff_mecca_text_20221223",
                "hazard_type": "flash_flood",
                "date": "2022-12-23",
                "location_name": "Mecca",
                "country_code": "SAU",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": None,
                "spatial_confidence": "medium",
                "temporal_confidence": "high",
                "source_name": "test",
                "source_url": "",
                "source_record_id": "verified-1",
                "validation_status": "verified",
                "label_status": "positive",
                "notes": "",
            }
        ]
    )

    province_features = aggregate_flash_flood_features_to_province_day(
        enrich_flash_flood_features_with_province(features, lookup)
    )
    labels = build_flash_flood_training_labels(province_features, event_daily_table=events)
    supervised = build_flash_flood_supervised_training_dataset(province_features, labels, drop_uncertain=False)

    assert supervised["training_join_mode"].nunique() == 1
    assert supervised["training_join_mode"].iloc[0] == "province_day:province_name"
    status_by_province = dict(zip(supervised["province_name"], supervised["label_status"]))
    assert status_by_province["makkah"] == "positive"
    assert status_by_province["riyadh"] == "uncertain"


def test_province_day_features_record_province_day_mode_for_point_events_without_coordinates():
    features = pd.DataFrame(
        [
            {"date": "2022-11-24", "hazard_type": "flash_flood", "province_name": "Makkah Region", "daily_precip_total": 40.0},
            {"date": "2022-11-24", "hazard_type": "flash_flood", "province_name": "Riyadh Region", "daily_precip_total": 5.0},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "ff_jeddah_20221124",
                "hazard_type": "flash_flood",
                "date": "2022-11-24",
                "location_name": "Jeddah",
                "country_code": "SAU",
                "latitude": 21.4858,
                "longitude": 39.1925,
                "geometry_wkt": None,
                "spatial_confidence": "high",
                "temporal_confidence": "high",
                "source_name": "test",
                "source_url": "",
                "source_record_id": "verified-1",
                "validation_status": "verified",
                "label_status": "positive",
                "notes": "",
            }
        ]
    )

    labels = build_flash_flood_training_labels(features, event_daily_table=events)

    assert labels["label_status"].tolist() == ["positive", "uncertain"]
    assert labels["label_source_mode"].tolist() == ["province_day", "event_day_unresolved"]


def test_build_flash_flood_province_day_feature_table_script_enriches_and_exports_csv(tmp_path: Path):
    module = _load_script_module()
    features = pd.DataFrame(
        [
            {"date": "2022-12-23", "hazard_type": "flash_flood", "latitude": 21.5, "longitude": 39.2, "daily_precip_total": 40.0},
            {"date": "2022-12-23", "hazard_type": "flash_flood", "latitude": 21.6, "longitude": 39.3, "daily_precip_total": 60.0},
            {"date": "2022-12-23", "hazard_type": "flash_flood", "latitude": 24.71, "longitude": 46.67, "daily_precip_total": 5.0},
        ]
    )
    lookup = pd.DataFrame(
        [
            {"latitude": 21.5, "longitude": 39.2, "province_name": "Makkah"},
            {"latitude": 21.6, "longitude": 39.3, "province_name": "Makkah"},
            {"latitude": 24.71, "longitude": 46.67, "province_name": "Riyadh"},
        ]
    )
    feature_path = tmp_path / "features.csv"
    lookup_path = tmp_path / "lookup.csv"
    output_path = tmp_path / "province_day.csv"
    features.to_csv(feature_path, index=False)
    lookup.to_csv(lookup_path, index=False)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        assert (
            module.main(
                [
                    "--features",
                    str(feature_path),
                    "--province-lookup",
                    str(lookup_path),
                    "--output",
                    str(output_path),
                ]
            )
            == 0
        )

    summary = json.loads(stdout.getvalue())
    exported = pd.read_csv(output_path)

    assert summary["input_rows"] == 3
    assert summary["province_ready_rows"] == 3
    assert summary["province_day_rows"] == 2
    assert exported["province_name"].tolist() == ["makkah", "riyadh"]
    assert exported["grid_cell_count"].tolist() == [2, 1]


def test_build_flash_flood_province_day_feature_table_script_streams_parquet(tmp_path: Path):
    module = _load_script_module()
    features = pd.DataFrame(
        [
            {"date": "2022-12-23", "hazard_type": "flash_flood", "latitude": 21.5, "longitude": 39.2, "daily_precip_total": 40.0},
            {"date": "2022-12-23", "hazard_type": "flash_flood", "latitude": 21.6, "longitude": 39.3, "daily_precip_total": 60.0},
            {"date": "2022-12-23", "hazard_type": "flash_flood", "latitude": 24.71, "longitude": 46.67, "daily_precip_total": 5.0},
        ]
    )
    lookup = pd.DataFrame(
        [
            {"latitude": 21.5, "longitude": 39.2, "province_name": "Makkah"},
            {"latitude": 21.6, "longitude": 39.3, "province_name": "Makkah"},
            {"latitude": 24.71, "longitude": 46.67, "province_name": "Riyadh"},
        ]
    )
    feature_path = tmp_path / "features.parquet"
    lookup_path = tmp_path / "lookup.csv"
    output_path = tmp_path / "province_day.parquet"
    features.to_parquet(feature_path, index=False)
    lookup.to_csv(lookup_path, index=False)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        assert (
            module.main(
                [
                    "--features",
                    str(feature_path),
                    "--province-lookup",
                    str(lookup_path),
                    "--batch-rows",
                    "2",
                    "--output",
                    str(output_path),
                ]
            )
            == 0
        )

    summary = json.loads(stdout.getvalue())
    exported = pd.read_parquet(output_path)

    assert summary["input_rows"] == 3
    assert summary["province_ready_rows"] == 3
    assert summary["province_day_rows"] == 2
    assert exported["province_name"].tolist() == ["makkah", "riyadh"]
    assert exported["grid_cell_count"].tolist() == [2, 1]
