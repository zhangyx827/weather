from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from mazu_saudi.data import (
    FlashFloodEvent,
    expand_flash_flood_events_to_daily_records,
    flash_flood_event_table_from_sources,
    flash_flood_event_records,
    merge_flash_flood_event_sources,
    seed_flash_flood_events,
    standardize_flash_flood_event_records,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_flash_flood_event_table.py"
VERIFIED_SCRIPT_PATH = ROOT / "scripts" / "build_verified_flash_flood_event_table.py"
DEFAULT_VERIFIED_INPUTS = [
    ROOT / "data" / "raw" / "flash_flood_verified" / "user_leads_2025_flash_flood_events.csv",
    ROOT / "data" / "raw" / "flash_flood_verified" / "web_verified_events_2024_2026-07-16.csv",
    ROOT / "data" / "raw" / "flash_flood_verified" / "web_verified_events_2026-07-14.csv",
]


def _load_script_module():
    spec = importlib.util.spec_from_file_location("build_flash_flood_event_table", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_verified_script_module():
    spec = importlib.util.spec_from_file_location("build_verified_flash_flood_event_table", VERIFIED_SCRIPT_PATH)
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


def test_merge_flash_flood_event_sources_ignores_coordinate_drift_for_same_location_duplicate():
    verified_events = standardize_flash_flood_event_records(
        [
            {
                "event_id": "ff_jeddah_verified_20221124_shifted",
                "record_id": "emdat-001-shifted",
                "date": "2022-11-24",
                "location": "Jeddah",
                "lat": 21.62,
                "lon": 39.33,
                "source_url": "https://example.test/jeddah-shifted",
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
    assert matching[0].event_id == "ff_jeddah_verified_20221124_shifted"
    assert matching[0].validation_status == "verified"
    assert matching[0].source_record_id == "emdat-001-shifted"


def test_merge_flash_flood_event_sources_uses_coordinates_when_location_missing():
    seed_event = FlashFloodEvent(
        event_id="seed_no_location",
        hazard_type="flash_flood",
        start_date=pd.Timestamp("2022-11-24").date(),
        end_date=pd.Timestamp("2022-11-24").date(),
        location_name="",
        latitude=21.49,
        longitude=39.19,
        source_record_id="seed-001",
        validation_status="seed",
    )
    verified_event = FlashFloodEvent(
        event_id="verified_no_location",
        hazard_type="flash_flood",
        start_date=pd.Timestamp("2022-11-24").date(),
        end_date=pd.Timestamp("2022-11-24").date(),
        location_name="",
        latitude=21.491,
        longitude=39.191,
        source_name="emdat",
        source_record_id="verified-001",
        validation_status="verified",
    )

    merged = merge_flash_flood_event_sources(seed_events=[seed_event], verified_events=[verified_event])

    assert len(merged) == 1
    assert merged[0].event_id == "verified_no_location"
    assert merged[0].validation_status == "verified"


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


def test_build_verified_flash_flood_event_table_script_merges_seed_and_verified(tmp_path: Path):
    module = _load_verified_script_module()
    verified_input = tmp_path / "verified_events.csv"
    output = tmp_path / "flash_flood_events_verified_combined.csv"
    daily_output = tmp_path / "flash_flood_events_verified_combined_daily.csv"
    summary_output = tmp_path / "flash_flood_events_verified_summary.json"
    pd.DataFrame(
        [
            {
                "record_id": "emdat-001",
                "date": "2022-11-24",
                "location": "Jeddah",
                "lat": 21.4858,
                "lon": 39.1925,
                "source_url": "https://example.test/event/1",
            },
            {
                "record_id": "emdat-002",
                "date": "2023-01-05",
                "location": "Taif",
                "lat": 21.2703,
                "lon": 40.4158,
                "source_url": "https://example.test/event/2",
            },
        ]
    ).to_csv(verified_input, index=False)

    assert module.main(
        [
            "--verified-input",
            str(verified_input),
            "--source-name",
            "emdat",
            "--output",
            str(output),
            "--daily-output",
            str(daily_output),
            "--summary-output",
            str(summary_output),
        ]
    ) == 0

    events = pd.read_csv(output)
    daily = pd.read_csv(daily_output)
    summary = json.loads(summary_output.read_text(encoding="utf-8"))

    assert len(events) == 7
    assert len(daily) == 7
    assert set(events["validation_status"]) == {"seed", "verified"}
    assert "Taif" in set(events["location_name"])
    jeddah_20221124 = events[(events["location_name"] == "Jeddah") & (events["start_date"] == "2022-11-24")]
    assert len(jeddah_20221124) == 1
    assert jeddah_20221124.iloc[0]["source_name"] == "emdat"
    assert jeddah_20221124.iloc[0]["validation_status"] == "verified"
    assert summary["verified_rows"] == 2
    assert summary["combined_rows"] == 7
    assert summary["daily_rows"] == 7
    assert summary["validation_status_counts"] == {"seed": 5, "verified": 2}
    assert summary["source_name_counts"]["emdat"] == 2
    assert summary["provenance_field_coverage"]["source_name_non_empty"] == 7
    assert summary["provenance_field_coverage"]["source_url_non_empty"] == 2
    assert summary["provenance_field_coverage"]["source_record_id_non_empty"] == 7
    assert summary["provenance_field_coverage"]["validation_status_non_empty"] == 7


def test_build_verified_flash_flood_event_table_script_uses_bundled_real_verified_input_by_default(tmp_path: Path):
    module = _load_verified_script_module()
    output = tmp_path / "flash_flood_events_verified_combined.csv"
    daily_output = tmp_path / "flash_flood_events_verified_combined_daily.csv"
    summary_output = tmp_path / "flash_flood_events_verified_summary.json"

    assert all(path.exists() for path in DEFAULT_VERIFIED_INPUTS)
    assert module.main(
        [
            "--source-name",
            "web_verified",
            "--output",
            str(output),
            "--daily-output",
            str(daily_output),
            "--summary-output",
            str(summary_output),
        ]
    ) == 0

    events = pd.read_csv(output)
    daily = pd.read_csv(daily_output)
    summary = json.loads(summary_output.read_text(encoding="utf-8"))

    assert len(events) == 14
    assert len(daily) == 18
    assert set(events["validation_status"]) == {"verified"}
    assert "Dammam" in set(events["location_name"])
    assert "Eastern Province" in set(events["location_name"])
    assert "Jeddah" in set(events["location_name"])
    assert "Mecca" in set(events["location_name"])
    assert [Path(path).name for path in summary["verified_inputs"]] == [path.name for path in DEFAULT_VERIFIED_INPUTS]
    assert summary["verified_rows"] == 14
    assert summary["combined_rows"] == 14
    assert summary["daily_rows"] == 18
    assert summary["validation_status_counts"] == {"verified": 14}
    assert summary["source_name_counts"] == {
        "user_session_handoff": 5,
        "web_verified": 7,
        "web_verified_2024": 2,
    }
    assert summary["provenance_field_coverage"]["source_name_non_empty"] == 14
    assert summary["provenance_field_coverage"]["source_url_non_empty"] == 9
    assert summary["provenance_field_coverage"]["source_record_id_non_empty"] == 14
    assert summary["provenance_field_coverage"]["validation_status_non_empty"] == 14
