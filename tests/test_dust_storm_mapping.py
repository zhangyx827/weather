from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from mazu_saudi.config import DustStormLabelMappingConfig
from mazu_saudi.data import build_dust_storm_training_labels

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_dust_storm_training_labels.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("build_dust_storm_training_labels", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_dust_storm_training_labels_marks_resolved_region_match():
    samples = pd.DataFrame(
        [
            {"date": "2025-05-04", "province_name": "Qassim"},
            {"date": "2025-05-04", "province_name": "Madinah"},
            {"date": "2025-05-06", "province_name": "Madinah"},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "dust_20250504_qassim_riyadh",
                "hazard_type": "dust_storm",
                "date": "2025-05-04",
                "location_name": "Qassim and Riyadh",
                "validation_status": "verified",
            }
        ]
    )

    labeled = build_dust_storm_training_labels(samples, events)

    assert labeled["label_status"].tolist() == ["positive", "negative", "negative"]
    assert labeled["label_source_mode"].tolist() == ["region_day_text", "outside_event_regions", "no_event_day"]
    assert labeled["matched_event_ids"].tolist() == ["dust_20250504_qassim_riyadh", "", ""]
    provenance = json.loads(labeled.loc[0, "label_provenance"])
    assert provenance["sample_region_id"] == "qassim"


def test_build_dust_storm_training_labels_can_disable_event_day_negatives():
    samples = pd.DataFrame([{"date": "2025-05-04", "province_name": "Madinah"}])
    events = pd.DataFrame(
        [
            {
                "event_id": "dust_20250504_qassim_riyadh",
                "hazard_type": "dust_storm",
                "date": "2025-05-04",
                "location_name": "Qassim and Riyadh",
                "validation_status": "verified",
            }
        ]
    )

    labeled = build_dust_storm_training_labels(
        samples,
        events,
        config=DustStormLabelMappingConfig(emit_event_day_negatives=False),
    )

    assert labeled.loc[0, "label_status"] == "uncertain"
    assert labeled.loc[0, "label_source_mode"] == "event_day_unresolved"


def test_build_dust_storm_training_labels_script_exports_csv(tmp_path: Path):
    module = _load_script_module()
    samples = pd.DataFrame(
        [
            {"date": "2025-05-04", "province_name": "Qassim"},
            {"date": "2025-05-06", "province_name": "Madinah"},
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
