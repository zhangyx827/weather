from __future__ import annotations

import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from mazu_saudi.config import FlashFloodLabelMappingConfig
from mazu_saudi.data import FlashFloodEvent, build_flash_flood_training_labels

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_flash_flood_training_labels.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("build_flash_flood_training_labels", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_flash_flood_training_labels_marks_point_buffer_match():
    samples = pd.DataFrame(
        [
            {"date": "2022-11-24", "latitude": 21.49, "longitude": 39.20},
            {"date": "2022-11-24", "latitude": 24.71, "longitude": 46.67},
            {"date": "2022-11-25", "latitude": 24.71, "longitude": 46.67},
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
                "source_record_id": "seed",
                "validation_status": "verified",
                "label_status": "positive",
                "notes": "",
            }
        ]
    )

    labeled = build_flash_flood_training_labels(samples, event_daily_table=events)

    assert labeled["label_status"].tolist() == ["positive", "uncertain", "negative"]
    assert labeled["matched_event_ids"].tolist() == ["ff_jeddah_20221124", "", ""]
    provenance = json.loads(labeled.loc[0, "label_provenance"])
    assert provenance["point_buffer_km"] == 25.0


def test_build_flash_flood_training_labels_supports_province_day_fallback():
    samples = pd.DataFrame(
        [
            {"date": "2022-12-23", "province_name": "Makkah"},
            {"date": "2022-12-23", "province_name": "Riyadh"},
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
                "source_record_id": "seed",
                "validation_status": "verified",
                "label_status": "positive",
                "notes": "",
            }
        ]
    )

    labeled = build_flash_flood_training_labels(samples, event_daily_table=events)

    assert labeled["label_status"].tolist() == ["positive", "uncertain"]
    assert labeled["label_source_mode"].tolist() == ["province_day", "event_day_unresolved"]


def test_build_flash_flood_training_labels_normalizes_boundary_style_province_names():
    samples = pd.DataFrame(
        [
            {"date": "2022-12-23", "province_name": "Makkah Region"},
            {"date": "2022-12-23", "province_name": "Riyadh Region"},
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
                "source_record_id": "seed",
                "validation_status": "verified",
                "label_status": "positive",
                "notes": "",
            }
        ]
    )

    labeled = build_flash_flood_training_labels(samples, event_daily_table=events)

    assert labeled["label_status"].tolist() == ["positive", "uncertain"]
    assert labeled["matched_event_ids"].tolist() == ["ff_mecca_text_20221223", ""]


def test_build_flash_flood_training_labels_matches_multi_province_text_event():
    samples = pd.DataFrame(
        [
            {"date": "2022-08-15", "province_name": "Jazan Region"},
            {"date": "2022-08-15", "province_name": "Asir"},
            {"date": "2022-08-15", "province_name": "Al Bahah Region"},
            {"date": "2022-08-15", "province_name": "Riyadh"},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "ff_multi_20220815",
                "hazard_type": "flash_flood",
                "date": "2022-08-15",
                "location_name": "Jazan, Asir, and Al-Baha",
                "country_code": "SAU",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": None,
                "spatial_confidence": "medium",
                "temporal_confidence": "high",
                "source_name": "test",
                "source_url": "",
                "source_record_id": "verified-multi",
                "validation_status": "verified",
                "label_status": "positive",
                "notes": "",
            }
        ]
    )

    labeled = build_flash_flood_training_labels(samples, event_daily_table=events)

    assert labeled["label_status"].tolist() == ["positive", "positive", "positive", "uncertain"]
    assert labeled["label_source_mode"].tolist() == ["province_day", "province_day", "province_day", "event_day_unresolved"]


def test_build_flash_flood_training_labels_normalizes_governorate_and_parenthetical_locations():
    samples = pd.DataFrame(
        [
            {"date": "2024-08-03", "province_name": "Jazan Region"},
            {"date": "2015-10-26", "province_name": "Northern Borders Region"},
            {"date": "2021-01-09", "province_name": "Hayel Region"},
            {"date": "2021-01-09", "province_name": "Riyadh Region"},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "ff_jazan_local_20240803",
                "hazard_type": "flash_flood",
                "date": "2024-08-03",
                "location_name": "Jazan (Sabya-Abu Arish bridge, Wadi Bin Abdullah)",
                "country_code": "SAU",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": None,
                "spatial_confidence": "medium",
                "temporal_confidence": "high",
                "source_name": "test",
                "source_url": "",
                "source_record_id": "verified-jazan",
                "validation_status": "verified",
                "label_status": "positive",
                "notes": "",
            },
            {
                "event_id": "ff_turaif_20151026",
                "hazard_type": "flash_flood",
                "date": "2015-10-26",
                "location_name": "Turaif governorate, Northern Border Province",
                "country_code": "SAU",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": None,
                "spatial_confidence": "medium",
                "temporal_confidence": "high",
                "source_name": "test",
                "source_url": "",
                "source_record_id": "verified-turaif",
                "validation_status": "verified",
                "label_status": "positive",
                "notes": "",
            },
            {
                "event_id": "ff_hail_20210109",
                "hazard_type": "flash_flood",
                "date": "2021-01-09",
                "location_name": "Ha'il region",
                "country_code": "SAU",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": None,
                "spatial_confidence": "medium",
                "temporal_confidence": "high",
                "source_name": "test",
                "source_url": "",
                "source_record_id": "verified-hail",
                "validation_status": "verified",
                "label_status": "positive",
                "notes": "",
            },
        ]
    )

    labeled = build_flash_flood_training_labels(samples, event_daily_table=events)

    assert labeled["label_status"].tolist() == ["positive", "positive", "positive", "uncertain"]
    assert labeled["matched_event_ids"].tolist() == [
        "ff_jazan_local_20240803",
        "ff_turaif_20151026",
        "ff_hail_20210109",
        "",
    ]


def test_build_flash_flood_training_labels_resolves_common_locality_aliases():
    samples = pd.DataFrame(
        [
            {"date": "2019-02-09", "province_name": "Al Ula Region"},
            {"date": "2021-11-20", "province_name": "Duba"},
            {"date": "2022-01-01", "province_name": "Rumah"},
            {"date": "2021-04-17", "province_name": "Abha"},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "ff_al_ula_20190209",
                "hazard_type": "flash_flood",
                "date": "2019-02-09",
                "location_name": "Fadhlan valley, west of Al-Ula governorate",
                "country_code": "SAU",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": None,
                "spatial_confidence": "medium",
                "temporal_confidence": "high",
                "source_name": "test",
                "source_url": "",
                "source_record_id": "verified-alula",
                "validation_status": "verified",
                "label_status": "positive",
                "notes": "",
            },
            {
                "event_id": "ff_tabuk_20211120",
                "hazard_type": "flash_flood",
                "date": "2021-11-20",
                "location_name": "Duba and Umluj coastal routes",
                "country_code": "SAU",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": None,
                "spatial_confidence": "medium",
                "temporal_confidence": "high",
                "source_name": "test",
                "source_url": "",
                "source_record_id": "verified-tabuk",
                "validation_status": "verified",
                "label_status": "positive",
                "notes": "",
            },
            {
                "event_id": "ff_riyadh_20220101",
                "hazard_type": "flash_flood",
                "date": "2022-01-01",
                "location_name": "Eastern Province (Dammam), Rumah, Huraymila",
                "country_code": "SAU",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": None,
                "spatial_confidence": "medium",
                "temporal_confidence": "high",
                "source_name": "test",
                "source_url": "",
                "source_record_id": "verified-riyadh",
                "validation_status": "verified",
                "label_status": "positive",
                "notes": "",
            },
            {
                "event_id": "ff_asir_20210417",
                "hazard_type": "flash_flood",
                "date": "2021-04-17",
                "location_name": "Abha and Khamis Mushait",
                "country_code": "SAU",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": None,
                "spatial_confidence": "medium",
                "temporal_confidence": "high",
                "source_name": "test",
                "source_url": "",
                "source_record_id": "verified-asir",
                "validation_status": "verified",
                "label_status": "positive",
                "notes": "",
            },
        ]
    )

    labeled = build_flash_flood_training_labels(samples, event_daily_table=events)

    assert labeled["label_status"].tolist() == ["positive", "positive", "positive", "positive"]
    assert labeled["matched_event_ids"].tolist() == [
        "ff_al_ula_20190209",
        "ff_tabuk_20211120",
        "ff_riyadh_20220101",
        "ff_asir_20210417",
    ]


def test_build_flash_flood_training_labels_can_emit_event_day_negative_when_enabled():
    samples = pd.DataFrame([{"date": "2022-11-24", "latitude": 24.71, "longitude": 46.67}])
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
                "source_record_id": "seed",
                "validation_status": "verified",
                "label_status": "positive",
                "notes": "",
            }
        ]
    )
    config = FlashFloodLabelMappingConfig(emit_event_day_negatives=True)

    labeled = build_flash_flood_training_labels(samples, event_daily_table=events, config=config)

    assert labeled.loc[0, "label_status"] == "negative"
    assert labeled.loc[0, "label_source_mode"] == "outside_event_footprint"


def test_build_flash_flood_training_labels_matches_geometry_wkt_polygon():
    samples = pd.DataFrame(
        [
            {"date": "2022-11-24", "latitude": 21.50, "longitude": 39.20},
            {"date": "2022-11-24", "latitude": 24.71, "longitude": 46.67},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "ff_jeddah_polygon_20221124",
                "hazard_type": "flash_flood",
                "date": "2022-11-24",
                "location_name": "Jeddah",
                "country_code": "SAU",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": "POLYGON((39.10 21.40, 39.30 21.40, 39.30 21.60, 39.10 21.60, 39.10 21.40))",
                "spatial_confidence": "high",
                "temporal_confidence": "high",
                "source_name": "test",
                "source_url": "",
                "source_record_id": "polygon-seed",
                "validation_status": "verified",
                "label_status": "positive",
                "notes": "",
            }
        ]
    )

    labeled = build_flash_flood_training_labels(samples, event_daily_table=events)

    assert labeled["label_status"].tolist() == ["positive", "uncertain"]
    assert labeled["label_source_mode"].tolist() == ["geometry_wkt", "event_day_unresolved"]
    provenance = json.loads(labeled.loc[0, "label_provenance"])
    assert provenance["matched_geometry_wkts"] == [events.loc[0, "geometry_wkt"]]


def test_build_flash_flood_training_labels_emits_geometry_based_negative_when_enabled():
    samples = pd.DataFrame([{"date": "2022-11-24", "latitude": 24.71, "longitude": 46.67}])
    events = pd.DataFrame(
        [
            {
                "event_id": "ff_jeddah_polygon_20221124",
                "hazard_type": "flash_flood",
                "date": "2022-11-24",
                "location_name": "Jeddah",
                "country_code": "SAU",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": "POLYGON((39.10 21.40, 39.30 21.40, 39.30 21.60, 39.10 21.60, 39.10 21.40))",
                "spatial_confidence": "high",
                "temporal_confidence": "high",
                "source_name": "test",
                "source_url": "",
                "source_record_id": "polygon-seed",
                "validation_status": "verified",
                "label_status": "positive",
                "notes": "",
            }
        ]
    )
    config = FlashFloodLabelMappingConfig(emit_event_day_negatives=True)

    labeled = build_flash_flood_training_labels(samples, event_daily_table=events, config=config)

    assert labeled.loc[0, "label_status"] == "negative"
    assert labeled.loc[0, "label_source_mode"] == "outside_event_footprint"


def test_build_flash_flood_training_labels_script_exports_csv(tmp_path: Path):
    module = _load_script_module()
    samples = pd.DataFrame(
        [
            {"date": "2022-11-24", "latitude": 21.49, "longitude": 39.20},
            {"date": "2022-11-25", "latitude": 24.71, "longitude": 46.67},
        ]
    )
    sample_path = tmp_path / "samples.csv"
    output_path = tmp_path / "labels.csv"
    samples.to_csv(sample_path, index=False)

    original_default = module.DEFAULT_VERIFIED_DAILY_EVENTS
    module.DEFAULT_VERIFIED_DAILY_EVENTS = tmp_path / "missing_verified_daily.csv"
    try:
        assert module.main(["--samples", str(sample_path), "--output", str(output_path)]) == 0
    finally:
        module.DEFAULT_VERIFIED_DAILY_EVENTS = original_default

    exported = pd.read_csv(output_path)
    assert "label_status" in exported.columns
    assert "label_provenance" in exported.columns
    assert set(exported["label_status"]) == {"positive", "negative"}


def test_build_flash_flood_training_labels_script_reports_geometry_audit(tmp_path: Path):
    module = _load_script_module()
    samples = pd.DataFrame(
        [
            {"date": "2022-11-24", "latitude": 21.50, "longitude": 39.20},
            {"date": "2022-11-24", "latitude": 24.71, "longitude": 46.67},
        ]
    )
    sample_path = tmp_path / "samples.csv"
    events_path = tmp_path / "events.csv"
    output_path = tmp_path / "labels.csv"
    samples.to_csv(sample_path, index=False)
    pd.DataFrame(
        [
            {
                "event_id": "ff_jeddah_polygon_20221124",
                "hazard_type": "flash_flood",
                "date": "2022-11-24",
                "location_name": "Jeddah",
                "country_code": "SAU",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": "POLYGON((39.10 21.40, 39.30 21.40, 39.30 21.60, 39.10 21.60, 39.10 21.40))",
                "spatial_confidence": "high",
                "temporal_confidence": "high",
                "source_name": "test",
                "source_url": "",
                "source_record_id": "polygon-seed",
                "validation_status": "verified",
                "label_status": "positive",
                "notes": "",
            }
        ]
    ).to_csv(events_path, index=False)

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        assert module.main(["--samples", str(sample_path), "--events-daily", str(events_path), "--output", str(output_path)]) == 0

    summary = json.loads(stdout.getvalue())
    assert summary["geometry_positive_rows"] == 1
    assert summary["rows_with_matched_event_ids"] == 1
    assert summary["label_source_mode_counts"]["geometry_wkt"] == 1
    assert summary["supervision_quality"]["status"] == "warning"
    assert "high_uncertain_fraction" in summary["supervision_quality"]["warnings"]


def test_build_flash_flood_training_labels_script_prefers_verified_daily_chain_by_default(tmp_path: Path):
    module = _load_script_module()
    samples = pd.DataFrame(
        [
            {"date": "2024-08-03", "province_name": "jazan region"},
            {"date": "2024-08-03", "province_name": "riyadh region"},
        ]
    )
    sample_path = tmp_path / "samples.csv"
    output_path = tmp_path / "labels.parquet"
    default_events_path = tmp_path / "flash_flood_events_verified_combined_daily.csv"
    samples.to_csv(sample_path, index=False)
    pd.DataFrame(
        [
            {
                "event_id": "ff_jazan_verified_20240803",
                "hazard_type": "flash_flood",
                "date": "2024-08-03",
                "location_name": "Jazan",
                "country_code": "SAU",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": None,
                "spatial_confidence": "medium",
                "temporal_confidence": "high",
                "source_name": "verified_chain",
                "source_url": "",
                "source_record_id": "verified-chain-001",
                "validation_status": "verified",
                "label_status": "positive",
                "notes": "",
            }
        ]
    ).to_csv(default_events_path, index=False)

    original_default = module.DEFAULT_VERIFIED_DAILY_EVENTS
    module.DEFAULT_VERIFIED_DAILY_EVENTS = default_events_path
    try:
        assert module.main(["--samples", str(sample_path), "--output", str(output_path)]) == 0
    finally:
        module.DEFAULT_VERIFIED_DAILY_EVENTS = original_default

    labeled = pd.read_parquet(output_path)
    assert labeled["label_status"].tolist() == ["positive", "uncertain"]
    assert labeled["matched_event_ids"].tolist() == ["ff_jazan_verified_20240803", ""]


def test_build_flash_flood_training_labels_script_streams_parquet_and_reopens_output(tmp_path: Path):
    module = _load_script_module()
    samples = pd.DataFrame(
        [
            {"date": "2022-11-24", "latitude": 21.49, "longitude": 39.20},
            {"date": "2022-11-24", "latitude": 24.71, "longitude": 46.67},
            {"date": "2022-11-25", "latitude": 24.71, "longitude": 46.67},
        ]
    )
    sample_path = tmp_path / "samples.parquet"
    output_path = tmp_path / "labels.parquet"
    samples.to_parquet(sample_path, index=False)

    stdout = io.StringIO()
    original_default = module.DEFAULT_VERIFIED_DAILY_EVENTS
    module.DEFAULT_VERIFIED_DAILY_EVENTS = tmp_path / "missing_verified_daily.csv"
    try:
        with redirect_stdout(stdout):
            assert module.main(["--samples", str(sample_path), "--output", str(output_path), "--batch-rows", "1"]) == 0
    finally:
        module.DEFAULT_VERIFIED_DAILY_EVENTS = original_default

    summary = json.loads(stdout.getvalue())
    exported = pd.read_parquet(output_path)

    assert len(exported) == 3
    assert set(exported["label_status"]) == {"positive", "uncertain", "negative"}
    assert summary["rows"] == 3
    assert summary["rows_with_matched_event_ids"] == 1
