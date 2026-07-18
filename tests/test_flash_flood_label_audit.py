from __future__ import annotations

import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from mazu_saudi.data import audit_flash_flood_province_day_labels


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "audit_flash_flood_province_day_labels.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("audit_flash_flood_province_day_labels", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_audit_flash_flood_province_day_labels_summarizes_unresolved_and_positive_rows():
    labels = pd.DataFrame(
        [
            {
                "date": "2022-12-23",
                "province_name": "makkah",
                "label_status": "positive",
                "label_source_mode": "province_day",
                "matched_event_ids": "ff_mecca_text_20221223",
                "label_provenance": json.dumps({"date": "2022-12-23", "day_event_mapping_modes": ["province_day"]}),
            },
            {
                "date": "2022-12-23",
                "province_name": "riyadh",
                "label_status": "uncertain",
                "label_source_mode": "event_day_unresolved",
                "matched_event_ids": "",
                "label_provenance": json.dumps({"date": "2022-12-23", "event_count_for_day": 1, "day_event_mapping_modes": ["province_day"]}),
            },
            {
                "date": "2022-11-24",
                "province_name": "eastern province",
                "label_status": "uncertain",
                "label_source_mode": "event_day_unresolved",
                "matched_event_ids": "",
                "label_provenance": json.dumps({"date": "2022-11-24", "event_count_for_day": 1, "day_event_mapping_modes": ["point_buffer"]}),
            },
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "ff_mecca_text_20221223",
                "hazard_type": "flash_flood",
                "date": "2022-12-23",
                "location_name": "Mecca",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": None,
                "spatial_confidence": "medium",
                "validation_status": "verified",
            },
            {
                "event_id": "ff_jeddah_20221124",
                "hazard_type": "flash_flood",
                "date": "2022-11-24",
                "location_name": "Jeddah",
                "latitude": 21.4858,
                "longitude": 39.1925,
                "geometry_wkt": None,
                "spatial_confidence": "high",
                "validation_status": "verified",
            },
        ]
    )

    summary = audit_flash_flood_province_day_labels(labels, event_daily_table=events, top_n=5)

    assert summary["positive_rows"] == 1
    assert summary["unresolved_rows"] == 2
    assert summary["unresolved_day_event_mapping_mode_counts"] == {"point_buffer": 1, "province_day": 1}
    assert summary["unresolved_candidate_bucket_counts"] == {"policy_conservative": 2}
    assert summary["top_positive_event_ids"] == [{"event_id": "ff_mecca_text_20221223", "rows": 1}]
    assert summary["top_unresolved_candidate_events"][0]["event_id"] in {"ff_jeddah_20221124", "ff_mecca_text_20221223"}
    assert summary["top_unresolved_candidate_events_by_bucket"]["policy_conservative"][0]["event_id"] in {
        "ff_jeddah_20221124",
        "ff_mecca_text_20221223",
    }


def test_audit_flash_flood_province_day_labels_script_reports_json_summary(tmp_path: Path):
    module = _load_script_module()
    labels = pd.DataFrame(
        [
            {
                "date": "2022-12-23",
                "province_name": "riyadh",
                "label_status": "uncertain",
                "label_source_mode": "event_day_unresolved",
                "matched_event_ids": "",
                "label_provenance": json.dumps({"date": "2022-12-23", "event_count_for_day": 1, "day_event_mapping_modes": ["province_day"]}),
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "ff_mecca_text_20221223",
                "hazard_type": "flash_flood",
                "date": "2022-12-23",
                "location_name": "Mecca",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": None,
                "spatial_confidence": "medium",
                "validation_status": "verified",
            }
        ]
    )
    label_path = tmp_path / "labels.parquet"
    events_path = tmp_path / "events.csv"
    labels.to_parquet(label_path, index=False)
    events.to_csv(events_path, index=False)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        assert module.main(["--labels", str(label_path), "--events-daily", str(events_path), "--top-n", "1"]) == 0

    summary = json.loads(stdout.getvalue())
    assert summary["unresolved_rows"] == 1
    assert summary["top_unresolved_dates"] == [{"date": "2022-12-23", "rows": 1}]


def test_audit_flash_flood_province_day_labels_marks_source_too_vague_candidates():
    labels = pd.DataFrame(
        [
            {
                "date": "2015-10-20",
                "province_name": "riyadh",
                "label_status": "uncertain",
                "label_source_mode": "event_day_unresolved",
                "matched_event_ids": "",
                "label_provenance": json.dumps({"date": "2015-10-20", "event_count_for_day": 1, "day_event_mapping_modes": ["uncertain"]}),
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "web-2015-10-multi-01",
                "hazard_type": "flash_flood",
                "date": "2015-10-20",
                "location_name": "Saudi Arabia (multiple provinces)",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": None,
                "spatial_confidence": "medium",
                "validation_status": "verified",
            }
        ]
    )

    summary = audit_flash_flood_province_day_labels(labels, event_daily_table=events, top_n=5)

    assert summary["unresolved_candidate_bucket_counts"] == {"source_too_vague": 1}
    assert summary["top_unresolved_candidate_events_by_bucket"]["source_too_vague"][0]["event_id"] == "web-2015-10-multi-01"


def test_audit_flash_flood_province_day_labels_classifies_locality_tail_as_manual_candidates():
    labels = pd.DataFrame(
        [
            {
                "date": "2019-02-09",
                "province_name": "al madinah region",
                "label_status": "uncertain",
                "label_source_mode": "event_day_unresolved",
                "matched_event_ids": "",
                "label_provenance": json.dumps({"date": "2019-02-09", "event_count_for_day": 1, "day_event_mapping_modes": ["province_day"]}),
            },
            {
                "date": "2021-11-20",
                "province_name": "tabuk region",
                "label_status": "uncertain",
                "label_source_mode": "event_day_unresolved",
                "matched_event_ids": "",
                "label_provenance": json.dumps({"date": "2021-11-20", "event_count_for_day": 1, "day_event_mapping_modes": ["province_day"]}),
            },
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "extreme_weather_flash_flood_madinah_20190208",
                "hazard_type": "flash_flood",
                "date": "2019-02-09",
                "location_name": "Madinah Region and Al-Ula",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": None,
                "spatial_confidence": "medium",
                "validation_status": "verified",
            },
            {
                "event_id": "extreme_weather_flash_flood_tabuk_20211120",
                "hazard_type": "flash_flood",
                "date": "2021-11-20",
                "location_name": "Duba and Umluj coastal routes",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": None,
                "spatial_confidence": "medium",
                "validation_status": "verified",
            },
        ]
    )

    summary = audit_flash_flood_province_day_labels(labels, event_daily_table=events, top_n=5)

    assert summary["unresolved_candidate_bucket_counts"] == {
        "manual_annotation_candidate": 1,
        "policy_conservative": 1,
    }
    assert summary["top_unresolved_candidate_events_by_bucket"]["manual_annotation_candidate"][0]["event_id"] == (
        "extreme_weather_flash_flood_tabuk_20211120"
    )


def test_audit_flash_flood_province_day_labels_ignores_nan_geometry_sources():
    labels = pd.DataFrame(
        [
            {
                "date": "2022-12-23",
                "province_name": "riyadh",
                "label_status": "uncertain",
                "label_source_mode": "event_day_unresolved",
                "matched_event_ids": "",
                "label_provenance": json.dumps({"date": "2022-12-23", "event_count_for_day": 1, "day_event_mapping_modes": ["uncertain"]}),
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "ff_nan_geometry",
                "hazard_type": "flash_flood",
                "date": "2022-12-23",
                "location_name": "Unknown",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": float("nan"),
                "spatial_confidence": "medium",
                "validation_status": "verified",
            }
        ]
    )

    summary = audit_flash_flood_province_day_labels(labels, event_daily_table=events, top_n=5)

    assert summary["top_unresolved_candidate_events"][0]["day_geometry_event_count"] == 0


def test_audit_flash_flood_province_day_labels_counts_derived_point_buffer_geometry_evidence():
    labels = pd.DataFrame(
        [
            {
                "date": "2022-11-24",
                "province_name": "makkah",
                "label_status": "uncertain",
                "label_source_mode": "event_day_unresolved",
                "matched_event_ids": "",
                "label_provenance": json.dumps({"date": "2022-11-24", "event_count_for_day": 1, "day_event_mapping_modes": ["point_buffer"]}),
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "ff_jeddah_20221124",
                "hazard_type": "flash_flood",
                "date": "2022-11-24",
                "location_name": "Jeddah",
                "latitude": 21.4858,
                "longitude": 39.1925,
                "geometry_wkt": None,
                "geometry_source": "derived_point_buffer",
                "spatial_confidence": "high",
                "validation_status": "verified",
            }
        ]
    )

    summary = audit_flash_flood_province_day_labels(labels, event_daily_table=events, top_n=5)

    assert summary["top_unresolved_candidate_events"][0]["day_geometry_event_count"] == 1


def test_audit_flash_flood_province_day_labels_separates_mixed_supported_and_unsupported_days():
    labels = pd.DataFrame(
        [
            {
                "date": "2022-11-24",
                "province_name": "makkah",
                "label_status": "uncertain",
                "label_source_mode": "event_day_unresolved",
                "matched_event_ids": "",
                "label_provenance": json.dumps(
                    {
                        "date": "2022-11-24",
                        "event_count_for_day": 2,
                        "day_event_mapping_modes": ["point_buffer", "uncertain"],
                        "day_supported_event_count": 1,
                        "day_unsupported_event_count": 1,
                        "day_has_supported_event": True,
                    }
                ),
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "ff_jeddah_20221124",
                "hazard_type": "flash_flood",
                "date": "2022-11-24",
                "location_name": "Jeddah",
                "latitude": 21.4858,
                "longitude": 39.1925,
                "geometry_wkt": None,
                "geometry_source": "derived_point_buffer",
                "spatial_confidence": "high",
                "validation_status": "verified",
            },
            {
                "event_id": "ff_mixed_unsupported_20221124",
                "hazard_type": "flash_flood",
                "date": "2022-11-24",
                "location_name": "Saudi Arabia (multiple provinces)",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": None,
                "geometry_source": "",
                "spatial_confidence": "medium",
                "validation_status": "verified",
            },
        ]
    )

    summary = audit_flash_flood_province_day_labels(labels, event_daily_table=events, top_n=5)

    assert summary["unresolved_candidate_bucket_counts"]["mixed_supported_and_unsupported"] == 2
    assert summary["unresolved_day_category_counts"]["mixed_supported_and_unsupported"] == 2
    assert summary["top_unresolved_candidate_events"][0]["day_event_category"] == "mixed_supported_and_unsupported"
