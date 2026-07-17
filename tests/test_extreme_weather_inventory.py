from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "data" / "raw" / "extreme_weather_verified" / "verified_extreme_weather_inventory.csv"


def test_extreme_weather_inventory_preserves_cross_hazard_verified_rows():
    with INVENTORY_PATH.open(newline="", encoding="utf-8") as handle:
        frame = list(csv.DictReader(handle))

    assert len(frame) == 68
    assert {row["validation_status"] for row in frame} == {"verified"}
    assert {row["hazard_type"] for row in frame} == {"dust_storm", "extreme_heat", "flash_flood", "snowstorm", "tropical_cyclone"}
    assert sum(1 for row in frame if row["hazard_type"] == "flash_flood") == 58
    assert sum(1 for row in frame if row["hazard_type"] == "dust_storm") == 5
    assert sum(1 for row in frame if row["hazard_type"] == "extreme_heat") == 2
    assert sum(1 for row in frame if row["hazard_type"] == "tropical_cyclone") == 1
    assert sum(1 for row in frame if row["hazard_type"] == "snowstorm") == 2
    assert "user_session_handoff" in {row["source_name"] for row in frame}
    assert "web_verified_extreme_heat" in {row["source_name"] for row in frame}
    assert "web_verified_dust" in {row["source_name"] for row in frame}
    assert {
        "web-2015-riyadh-02",
        "web-2019-hafar-01",
        "web-2021-central-01",
        "web-2024-aljawf-01",
        "web-2024-central-01",
        "web-2024-western-03",
    }.issubset({row["record_id"] for row in frame})


def test_hazard_specific_verified_raw_files_exist_for_external_heat_and_dust_rows():
    heat_path = ROOT / "data" / "raw" / "extreme_heat_verified" / "web_verified_hajj_heat_events_2023_2024.csv"
    dust_path = ROOT / "data" / "raw" / "dust_storm_verified" / "web_verified_dust_events_2015_2026-07-16.csv"

    with heat_path.open(newline="", encoding="utf-8") as handle:
        heat = list(csv.DictReader(handle))
    with dust_path.open(newline="", encoding="utf-8") as handle:
        dust = list(csv.DictReader(handle))

    assert [row["event_id"] for row in heat] == ["extreme_heat_hajj_20230626", "extreme_heat_hajj_20240614"]
    assert [row["event_id"] for row in dust] == ["dust_web_riyadh_20150401"]
    assert {row["validation_status"] for row in heat} == {"verified"}
    assert {row["validation_status"] for row in dust} == {"verified"}
    assert next(row["source_url"] for row in heat if row["event_id"] == "extreme_heat_hajj_20240614").startswith("https://apnews.com/article/")
