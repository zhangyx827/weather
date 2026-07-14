from __future__ import annotations

import importlib.util
import json
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

    assert module.main(["--samples", str(sample_path), "--output", str(output_path)]) == 0

    exported = pd.read_csv(output_path)
    assert "label_status" in exported.columns
    assert "label_provenance" in exported.columns
    assert set(exported["label_status"]) == {"positive", "negative"}
