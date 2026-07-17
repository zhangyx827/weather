from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "data" / "raw" / "extreme_weather_verified" / "verified_extreme_weather_inventory_2015_2025.csv"


def test_extreme_weather_inventory_preserves_cross_hazard_verified_rows():
    frame = pd.read_csv(INVENTORY_PATH)

    assert len(frame) == 29
    assert set(frame["validation_status"]) == {"verified"}
    assert set(frame["hazard_type"]) == {"dust_storm", "extreme_heat", "flash_flood", "snowstorm", "tropical_cyclone"}
    assert (frame["hazard_type"] == "flash_flood").sum() == 21
    assert (frame["hazard_type"] == "dust_storm").sum() == 4
    assert (frame["hazard_type"] == "extreme_heat").sum() == 2
    assert (frame["hazard_type"] == "tropical_cyclone").sum() == 1
    assert (frame["hazard_type"] == "snowstorm").sum() == 1
    assert "user_session_handoff" in set(frame["source_name"])
    assert "web_verified_extreme_heat" in set(frame["source_name"])
    assert "web_verified_dust" in set(frame["source_name"])


def test_hazard_specific_verified_raw_files_exist_for_external_heat_and_dust_rows():
    heat_path = ROOT / "data" / "raw" / "extreme_heat_verified" / "web_verified_hajj_heat_events_2023_2024.csv"
    dust_path = ROOT / "data" / "raw" / "dust_storm_verified" / "web_verified_dust_events_2015_2026-07-16.csv"

    heat = pd.read_csv(heat_path)
    dust = pd.read_csv(dust_path)

    assert list(heat["event_id"]) == ["extreme_heat_hajj_20230626", "extreme_heat_hajj_20240614"]
    assert list(dust["event_id"]) == ["dust_web_riyadh_20150401"]
    assert set(heat["validation_status"]) == {"verified"}
    assert set(dust["validation_status"]) == {"verified"}
    assert heat.loc[heat["event_id"] == "extreme_heat_hajj_20240614", "source_url"].item().startswith("https://apnews.com/article/")
