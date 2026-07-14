from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from mazu_saudi.data import (
    expand_flash_flood_events_to_daily_records,
    flash_flood_event_table_from_sources,
    flash_flood_event_records,
    merge_flash_flood_event_sources,
    seed_flash_flood_events,
    standardize_flash_flood_event_records,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_flash_flood_event_table.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("build_flash_flood_event_table", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_seed_flash_flood_events_match_handoff_count():
    events = seed_flash_flood_events()
    records = flash_flood_event_records(events)

    assert len(events) == 6
    assert len(records) == 6
    assert {record["location_name"] for record in records} == {"Jeddah", "Mecca"}
    assert all(record["hazard_type"] == "flash_flood" for record in records)
    assert all(record["validation_status"] == "seed" for record in records)


def test_flash_flood_daily_expansion_is_inclusive():
    rows = expand_flash_flood_events_to_daily_records()

    assert len(rows) == 6
    assert all(row["label_status"] == "positive" for row in rows)
    assert rows[0]["date"] == "2009-11-25"
    assert rows[-1]["date"] == "2022-12-23"


def test_build_flash_flood_event_table_script_exports_csv(tmp_path: Path):
    module = _load_script_module()
    output = tmp_path / "flash_flood_events.csv"
    daily_output = tmp_path / "flash_flood_events_daily.csv"

    assert module.main(["--output", str(output), "--daily-output", str(daily_output)]) == 0

    events = pd.read_csv(output)
    daily = pd.read_csv(daily_output)
    assert len(events) == 6
    assert len(daily) == 6
    assert "event_id" in events.columns
    assert "label_status" in daily.columns


def test_standardize_flash_flood_event_records_preserves_verified_provenance():
    events = standardize_flash_flood_event_records(
        [
            {
                "record_id": "emdat-001",
                "date": "2022-11-24",
                "location": "Jeddah",
                "lat": 21.49,
                "lon": 39.19,
                "source_url": "https://example.test/event/1",
                "notes": "verified case study",
            }
        ],
        source_name="emdat",
    )

    assert len(events) == 1
    assert events[0].source_name == "emdat"
    assert events[0].source_record_id == "emdat-001"
    assert events[0].validation_status == "verified"
    assert events[0].location_name == "Jeddah"


def test_merge_flash_flood_event_sources_prefers_verified_duplicate():
    verified_events = standardize_flash_flood_event_records(
        [
            {
                "event_id": "ff_jeddah_verified_20221124",
                "record_id": "emdat-001",
                "date": "2022-11-24",
                "location": "Jeddah",
                "lat": 21.4858,
                "lon": 39.1925,
            }
        ],
        source_name="emdat",
    )

    merged = merge_flash_flood_event_sources(
        seed_events=seed_flash_flood_events(),
        verified_events=verified_events,
    )
    matching = [event for event in merged if event.start_date.isoformat() == "2022-11-24" and event.location_name == "Jeddah"]

    assert len(matching) == 1
    assert matching[0].validation_status == "verified"
    assert matching[0].source_name == "emdat"


def test_flash_flood_event_table_from_sources_combines_seed_and_verified():
    table = flash_flood_event_table_from_sources(
        [
            {
                "record_id": "emdat-002",
                "date": "2023-01-05",
                "location": "Taif",
                "lat": 21.2703,
                "lon": 40.4158,
            }
        ],
        source_name="emdat",
    )

    assert len(table) == 7
    assert set(table["validation_status"]) == {"seed", "verified"}
    assert "Taif" in set(table["location_name"])
