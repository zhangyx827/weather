from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from mazu_saudi.config import FlashFloodLabelMappingConfig
from mazu_saudi.data import (
    FlashFloodEvent,
    build_flash_flood_training_labels,
    expand_flash_flood_events_to_daily_records,
    flash_flood_event_table_from_sources,
    flash_flood_event_records,
    merge_flash_flood_event_sources,
    seed_flash_flood_events,
    standardize_flash_flood_event_records,
)
from mazu_saudi.data.flash_flood_audit import (
    count_flash_flood_boundary_grounded_positive_rows,
    count_flash_flood_explicit_geometry_positive_rows,
    count_flash_flood_geometry_backed_positive_rows,
    summarize_flash_flood_geometry_backed_positive_rows,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_flash_flood_event_table.py"
VERIFIED_SCRIPT_PATH = ROOT / "scripts" / "build_verified_flash_flood_event_table.py"
DEFAULT_VERIFIED_INPUTS = [
    path
    for path in sorted((ROOT / "data" / "raw" / "flash_flood_verified").iterdir())
    if path.is_file() and path.suffix.lower() in {".csv", ".json", ".parquet"} and not path.name.startswith("sample_")
]
DEFAULT_CROSS_HAZARD_VERIFIED_INPUT = (
    ROOT / "data" / "raw" / "extreme_weather_verified" / "verified_extreme_weather_inventory.csv"
)
if DEFAULT_CROSS_HAZARD_VERIFIED_INPUT.exists():
    DEFAULT_VERIFIED_INPUTS.append(DEFAULT_CROSS_HAZARD_VERIFIED_INPUT)
DEFAULT_VERIFIED_INPUTS = sorted(dict.fromkeys(DEFAULT_VERIFIED_INPUTS))


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


def test_build_flash_flood_training_labels_summarizes_geometry_sources_by_type():
    samples = pd.DataFrame(
        [
            {"date": "2022-11-24", "latitude": 21.49, "longitude": 39.20},
            {"date": "2022-11-25", "latitude": 21.49, "longitude": 39.20},
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
            },
            {
                "event_id": "ff_jeddah_poly_20221125",
                "hazard_type": "flash_flood",
                "date": "2022-11-25",
                "location_name": "Jeddah",
                "country_code": "SAU",
                "latitude": None,
                "longitude": None,
                "geometry_wkt": "POLYGON((39.10 21.40, 39.30 21.40, 39.30 21.60, 39.10 21.60, 39.10 21.40))",
                "spatial_confidence": "high",
                "temporal_confidence": "high",
                "source_name": "test",
                "source_url": "",
                "source_record_id": "seed",
                "validation_status": "verified",
                "label_status": "positive",
                "notes": "",
            },
        ]
    )

    labeled = build_flash_flood_training_labels(
        samples,
        event_daily_table=events,
        config=FlashFloodLabelMappingConfig(emit_event_day_negatives=False),
    )

    assert count_flash_flood_geometry_backed_positive_rows(labeled) == 2
    assert summarize_flash_flood_geometry_backed_positive_rows(labeled) == {
        "derived_point_buffer": 1,
        "source_geometry": 1,
    }


def test_boundary_grounded_positive_rows_are_tracked_separately_from_explicit_geometry():
    labels = pd.DataFrame(
        [
            {
                "date": "2022-11-24",
                "label_status": "positive",
                "label_source_mode": "province_day",
                "matched_event_ids": "ff_boundary",
                "label_provenance": json.dumps(
                    {
                        "date": "2022-11-24",
                        "matched_event_ids": ["ff_boundary"],
                        "matched_geometry_sources": ["province_boundary"],
                        "matched_geometry_wkts": [],
                    }
                ),
            },
            {
                "date": "2022-11-24",
                "label_status": "positive",
                "label_source_mode": "point_buffer",
                "matched_event_ids": "ff_point",
                "label_provenance": json.dumps(
                    {
                        "date": "2022-11-24",
                        "matched_event_ids": ["ff_point"],
                        "matched_geometry_sources": ["derived_point_buffer"],
                        "matched_geometry_wkts": [],
                    }
                ),
            },
        ]
    )

    assert count_flash_flood_geometry_backed_positive_rows(labels) == 2
    assert count_flash_flood_boundary_grounded_positive_rows(labels) == 1
    assert count_flash_flood_explicit_geometry_positive_rows(labels) == 1
    assert summarize_flash_flood_geometry_backed_positive_rows(labels) == {
        "derived_point_buffer": 1,
        "province_boundary": 1,
    }


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
                "geometry_wkt": "POLYGON((39.10 21.40, 39.30 21.40, 39.30 21.60, 39.10 21.60, 39.10 21.40))",
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
    assert events[0].geometry_wkt.startswith("POLYGON((")


def test_standardize_flash_flood_event_records_treats_nan_geometry_as_missing():
    events = standardize_flash_flood_event_records(
        [
            {
                "record_id": "emdat-002",
                "date": "2022-11-24",
                "location": "Jeddah",
                "lat": 21.49,
                "lon": 39.19,
                "geometry_wkt": float("nan"),
                "source_url": "https://example.test/event/2",
                "notes": "verified case study",
            }
        ],
        source_name="emdat",
    )

    assert len(events) == 1
    assert events[0].geometry_wkt is None


def test_standardize_flash_flood_event_records_skips_non_flash_flood_rows():
    events = standardize_flash_flood_event_records(
        [
            {
                "record_id": "heat-001",
                "hazard_type": "extreme_heat",
                "date": "2024-06-14",
                "location": "Mecca",
            },
            {
                "record_id": "flood-001",
                "hazard_type": "flash_flood",
                "date": "2024-08-03",
                "location": "Jazan province",
            },
        ],
        source_name="cross_hazard_inventory",
    )

    assert len(events) == 1
    assert events[0].source_record_id == "flood-001"
    assert events[0].hazard_type == "flash_flood"


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


def test_merge_flash_flood_event_sources_prefers_more_specific_verified_spatial_data():
    seed_event = FlashFloodEvent(
        event_id="seed_text_only",
        hazard_type="flash_flood",
        start_date=pd.Timestamp("2022-11-24").date(),
        end_date=pd.Timestamp("2022-11-24").date(),
        location_name="Jeddah",
        validation_status="seed",
    )
    verified_event = FlashFloodEvent(
        event_id="verified_point",
        hazard_type="flash_flood",
        start_date=pd.Timestamp("2022-11-24").date(),
        end_date=pd.Timestamp("2022-11-24").date(),
        location_name="Jeddah",
        latitude=21.4858,
        longitude=39.1925,
        source_name="emdat",
        source_record_id="verified-point-001",
        validation_status="verified",
    )

    merged = merge_flash_flood_event_sources(seed_events=[seed_event], verified_events=[verified_event])

    assert len(merged) == 1
    assert merged[0].event_id == "verified_point"
    assert merged[0].latitude == 21.4858
    assert merged[0].longitude == 39.1925
    assert merged[0].source_name == "emdat"


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
    assert summary["spatial_mode_counts"] == {
        "geometry_wkt_rows": 7,
        "point_rows": 7,
        "text_only_rows": 0,
    }
    assert summary["geometry_provenance_counts"] == {"derived_point_buffer": 7}
    assert daily["geometry_source"].fillna("").astype(str).value_counts().to_dict() == {"derived_point_buffer": 7}
    assert summary["daily_label_source_mode_counts"] == {}
    assert summary["provenance_field_coverage"]["source_name_non_empty"] == 7
    assert summary["provenance_field_coverage"]["source_url_non_empty"] == 2
    assert summary["provenance_field_coverage"]["source_record_id_non_empty"] == 7
    assert summary["provenance_field_coverage"]["validation_status_non_empty"] == 7
    assert summary["summary_output"] == str(summary_output)


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

    expected_verified_rows = sum(len(pd.read_parquet(path)) if path.suffix == ".parquet" else len(pd.read_json(path)) if path.suffix == ".json" else len(pd.read_csv(path)) for path in DEFAULT_VERIFIED_INPUTS)
    if DEFAULT_CROSS_HAZARD_VERIFIED_INPUT in DEFAULT_VERIFIED_INPUTS:
        cross_hazard_rows = pd.read_csv(DEFAULT_CROSS_HAZARD_VERIFIED_INPUT)
        expected_verified_rows -= len(cross_hazard_rows)
        expected_verified_rows += int(cross_hazard_rows["hazard_type"].astype(str).str.lower().eq("flash_flood").sum())

    assert len(events) == summary["combined_rows"]
    assert len(daily) == summary["daily_rows"]
    assert set(events["validation_status"]) == {"verified"}
    assert [Path(path).name for path in summary["verified_inputs"]] == [path.name for path in DEFAULT_VERIFIED_INPUTS]
    assert summary["verified_rows"] == expected_verified_rows
    assert summary["combined_rows"] == len(events)
    assert summary["daily_rows"] == len(daily)
    assert summary["validation_status_counts"] == {"verified": len(events)}
    assert summary["spatial_mode_counts"]["geometry_wkt_rows"] >= 0
    assert summary["spatial_mode_counts"]["point_rows"] >= 0
    assert summary["spatial_mode_counts"]["text_only_rows"] >= 0
    assert summary["daily_label_source_mode_counts"] in ({}, {"positive": len(daily)})
    assert summary["provenance_field_coverage"]["source_name_non_empty"] == len(events)
    assert summary["provenance_field_coverage"]["source_url_non_empty"] <= len(events)
    assert summary["provenance_field_coverage"]["source_record_id_non_empty"] == len(events)
    assert summary["provenance_field_coverage"]["validation_status_non_empty"] == len(events)
    assert summary["summary_output"] == str(summary_output)


def test_build_verified_flash_flood_event_table_script_filters_cross_hazard_inventory_rows(tmp_path: Path):
    module = _load_verified_script_module()
    verified_input = tmp_path / "cross_hazard_inventory.csv"
    output = tmp_path / "flash_flood_events_verified_combined.csv"
    daily_output = tmp_path / "flash_flood_events_verified_combined_daily.csv"
    summary_output = tmp_path / "flash_flood_events_verified_summary.json"
    pd.DataFrame(
        [
            {
                "record_id": "heat-001",
                "hazard_type": "extreme_heat",
                "date": "2024-06-14",
                "location": "Mecca",
                "source_url": "https://example.test/heat",
            },
            {
                "record_id": "flood-001",
                "hazard_type": "flash_flood",
                "date": "2024-08-03",
                "location": "Jazan province",
                "source_url": "https://example.test/flood",
            },
        ]
    ).to_csv(verified_input, index=False)

    assert module.main(
        [
            "--verified-input",
            str(verified_input),
            "--verified-only",
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

    assert events["hazard_type"].tolist() == ["flash_flood"]
    assert events["source_record_id"].tolist() == ["flood-001"]
    assert daily["event_id"].tolist() == [events.loc[0, "event_id"]]
    assert summary["verified_rows"] == 1


def test_build_verified_flash_flood_event_table_script_reports_geometry_rows(tmp_path: Path):
    module = _load_verified_script_module()
    verified_input = tmp_path / "verified_events.csv"
    output = tmp_path / "flash_flood_events_verified_combined.csv"
    daily_output = tmp_path / "flash_flood_events_verified_combined_daily.csv"
    summary_output = tmp_path / "flash_flood_events_verified_summary.json"
    pd.DataFrame(
        [
            {
                "record_id": "poly-001",
                "date": "2022-11-24",
                "location": "Jeddah",
                "geometry_wkt": "POLYGON((39.10 21.40, 39.30 21.40, 39.30 21.60, 39.10 21.60, 39.10 21.40))",
                "source_url": "https://example.test/event/poly",
            }
        ]
    ).to_csv(verified_input, index=False)

    assert module.main(
        [
            "--verified-input",
            str(verified_input),
            "--source-name",
            "emdat",
            "--verified-only",
            "--output",
            str(output),
            "--daily-output",
            str(daily_output),
            "--summary-output",
            str(summary_output),
        ]
    ) == 0

    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert summary["spatial_mode_counts"] == {
        "geometry_wkt_rows": 1,
        "point_rows": 0,
        "text_only_rows": 0,
    }
    assert summary["geometry_provenance_counts"] == {"source_geometry": 1}
    assert summary["summary_output"] == str(summary_output)
