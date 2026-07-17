from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from mazu_saudi.data.dust_storm_event_sources import (
    expand_dust_storm_events_to_daily_records,
    standardize_dust_storm_event_records,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_dust_storm_event_table.py"
DEFAULT_INPUT = ROOT / "data" / "raw" / "dust_storm_verified" / "user_leads_2025_dust_events.csv"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("build_dust_storm_event_table", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_standardize_dust_storm_records_preserves_verified_status():
    events = standardize_dust_storm_event_records(
        [
            {
                "record_id": "lead-001",
                "start_date": "2025-05-04",
                "end_date": "2025-05-05",
                "location": "Qassim and Riyadh",
                "validation_status": "verified",
                "severity": "severe",
                "notes": "User-provided lead",
            }
        ],
        source_name="user_session_handoff",
    )

    assert len(events) == 1
    assert events[0].hazard_type == "dust_storm"
    assert events[0].validation_status == "verified"
    assert events[0].source_name == "user_session_handoff"
    assert events[0].severity == "severe"


def test_expand_dust_storm_events_to_daily_records_is_inclusive():
    events = standardize_dust_storm_event_records(
        [
            {
                "record_id": "lead-001",
                "start_date": "2025-05-16",
                "end_date": "2025-05-19",
                "location": "Rafha, Hafar Al-Batin, Dammam",
            }
        ],
        source_name="user_session_handoff",
    )

    rows = expand_dust_storm_events_to_daily_records(events)

    assert len(rows) == 4
    assert rows[0]["date"] == "2025-05-16"
    assert rows[-1]["date"] == "2025-05-19"
    assert all(row["label_status"] == "positive" for row in rows)


def test_build_dust_storm_event_table_script_uses_bundled_user_leads_by_default(tmp_path: Path):
    module = _load_script_module()
    output = tmp_path / "dust_storm_events.csv"
    daily_output = tmp_path / "dust_storm_events_daily.csv"
    summary_output = tmp_path / "dust_storm_events_summary.json"

    assert DEFAULT_INPUT.exists()
    assert module.main(
        [
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

    assert len(events) == 3
    assert len(daily) == 12
    assert set(events["validation_status"]) == {"verified"}
    assert summary["rows"] == 3
    assert summary["verified_rows"] == 3
    assert summary["daily_rows"] == 12
    assert summary["validation_status_counts"] == {"verified": 3}
    assert summary["source_name_counts"] == {"user_session_handoff": 3}
