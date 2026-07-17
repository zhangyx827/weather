# Extreme-Weather Verification Matrix

Verification date: `2026-07-16`

This matrix separates three statuses:

- `external_verified`: directly backed in-repo by an external citation URL
- `preserved_user_verified`: preserved in-repo as user-confirmed factual rows with `validation_status=verified`, but not yet upgraded here with a stronger external citation
- `bundled_web_verified`: older web-compiled or previously bundled verified rows already preserved in the repository

## Extreme Heat and Dust Storms

| Date | Location | Hazard | Status | In-repo path | Verification note |
| --- | --- | --- | --- | --- | --- |
| 2015-04-01 to 2015-04-02 | Riyadh | `dust_storm` | `external_verified` | `data/raw/dust_storm_verified/web_verified_dust_events_2015_2026-07-16.csv` | Riyadh climate summary states a massive dust storm suspended classes and cancelled hundreds of flights. |
| 2023-06-26 to 2023-07-01 | Mecca, Mina, Arafat | `extreme_heat` | `external_verified` | `data/raw/extreme_heat_verified/web_verified_hajj_heat_events_2023_2024.csv` | TIME reported Hajj 2023 dates, temperatures above 44C, and more than 8,400 treated for heat illness. |
| 2024-06-14 to 2024-06-19 | Mecca, Mina | `extreme_heat` | `external_verified` | `data/raw/extreme_heat_verified/web_verified_hajj_heat_events_2023_2024.csv` | AP reported the Saudi Health Ministry cautioned that temperatures at holy sites could reach 48C. |
| 2025-05-04 to 2025-05-05 | Qassim and Riyadh | `dust_storm` | `preserved_user_verified` | `data/raw/dust_storm_verified/user_leads_2025_dust_events.csv` | Preserved as verified user-confirmed haboob fact with explicit provenance fields. |
| 2025-05-16 to 2025-05-19 | Rafha, Hafar Al-Batin, Dammam | `dust_storm` | `preserved_user_verified` | `data/raw/dust_storm_verified/user_leads_2025_dust_events.csv` | Preserved as verified user-confirmed sustained dust episode fact. |
| 2025-06-30 to 2025-07-05 | Eastern Province, Hijaz, Madinah, Riyadh | `dust_storm` | `preserved_user_verified` | `data/raw/dust_storm_verified/user_leads_2025_dust_events.csv` | Preserved as verified user-confirmed sustained dust plus heat co-occurrence fact. |
| 2021-03-12 to 2021-03-13 | Riyadh and Eastern Province | `dust_storm` | `external_verified` | `data/raw/dust_storm_verified/web_verified_dust_events_2015_2026-07-16.csv` | Arab News coverage confirms the March 2021 regional dust storm and reduced-visibility transport disruption. |

## Flash Floods

| Date | Location | Hazard | Status | In-repo path | Verification note |
| --- | --- | --- | --- | --- | --- |
| 2009-11-25 | Jeddah | `flash_flood` | `bundled_web_verified` | `data/raw/flash_flood_verified/web_verified_events_2026-07-14.csv` | Older bundled verified Jeddah flood row retained in repo. |
| 2011-01-26 | Jeddah | `flash_flood` | `bundled_web_verified` | `data/raw/flash_flood_verified/web_verified_events_2026-07-14.csv` | Older bundled verified Jeddah flood row retained in repo. |
| 2015-09-11 | Mecca | `flash_flood` | `external_verified` | `data/raw/flash_flood_verified/web_verified_events_2026-07-14.csv` | BBC reporting covers the Grand Mosque crane collapse storm and the torrential rain event. |
| 2015-11-17 | Jeddah | `flash_flood` | `bundled_web_verified` | `data/raw/flash_flood_verified/web_verified_events_2026-07-14.csv` | Older bundled verified Jeddah flood row retained in repo. |
| 2017-11-21 | Jeddah | `flash_flood` | `bundled_web_verified` | `data/raw/flash_flood_verified/web_verified_events_2026-07-14.csv` | Older bundled verified Jeddah flood row retained in repo. |
| 2019-04-13 to 2019-04-17 | Riyadh, Qassim, Hail, and Asir | `flash_flood` | `external_verified` | `data/raw/extreme_weather_verified/user_verified_extreme_weather_events_2013_2023.csv` | Floodlist summary documents widespread spring storm flooding across central and northern Saudi Arabia. |
| 2020-11-25 to 2020-11-26 | Jeddah and Makkah Province | `flash_flood` | `external_verified` | `data/raw/flash_flood_verified/web_verified_events_2026-07-14.csv` | Floodlist coverage describes torrential winter rains, gridlocked roads, and school suspensions in Jeddah. |
| 2021-02-17 to 2021-02-18 | Tabuk (Jabal Al-Lawz) | `snowstorm` | `external_verified` | `data/raw/extreme_weather_verified/user_verified_extreme_weather_events_2013_2023.csv` | Al Arabiya coverage records the Tabuk snowfall and associated cold alerts. |
| 2022-11-24 | Jeddah | `flash_flood` | `bundled_web_verified` | `data/raw/flash_flood_verified/web_verified_events_2026-07-14.csv` | Older bundled verified flood row retained in repo. |
| 2022-12-23 | Mecca | `flash_flood` | `bundled_web_verified` | `data/raw/flash_flood_verified/web_verified_events_2026-07-14.csv` | Older bundled verified flood row retained in repo. |
| 2023-01-03 | Jeddah | `flash_flood` | `bundled_web_verified` | `data/raw/flash_flood_verified/web_verified_events_2026-07-14.csv` | Older bundled verified flood row retained in repo. |
| 2024-04-17 | Eastern Province | `flash_flood` | `bundled_web_verified` | `data/raw/flash_flood_verified/web_verified_events_2024_2026-07-16.csv` | Preserved verified 2024 Gulf storm impact row. |
| 2024-04-17 | Dammam | `flash_flood` | `bundled_web_verified` | `data/raw/flash_flood_verified/web_verified_events_2024_2026-07-16.csv` | Preserved verified 2024 Gulf storm impact row. |
| 2024-08-24 to 2024-08-31 | Madinah, Jazan, and Asir | `flash_flood` | `external_verified` | `data/raw/flash_flood_verified/web_verified_events_2024_2026-07-16.csv` | Al Arabiya coverage documents late-summer flooding across Medina and the southwestern regions. |
| 2025-01-06 to 2025-01-07 | Makkah and Jeddah coastal areas | `flash_flood` | `preserved_user_verified` | `data/raw/flash_flood_verified/user_leads_2025_flash_flood_events.csv` | Preserved as verified user-confirmed red-warning and flood-compound-disaster fact. |
| 2025-03-06 to 2025-03-07 | Hail and Buraidah | `flash_flood` | `preserved_user_verified` | `data/raw/flash_flood_verified/user_leads_2025_flash_flood_events.csv` | Preserved as verified user-confirmed short-duration mountain and valley flood fact. |
| 2025-08-14 | Taif | `flash_flood` | `preserved_user_verified` | `data/raw/flash_flood_verified/user_leads_2025_flash_flood_events.csv` | Preserved as verified user-confirmed hail plus urban flood fact. |
| 2025-08-27 to 2025-08-28 | Asir, Jizan, Najran | `flash_flood` | `preserved_user_verified` | `data/raw/flash_flood_verified/user_leads_2025_flash_flood_events.csv` | Preserved as verified user-confirmed overnight southwest mountain flood fact. |
| 2025-12-09 to 2025-12-10 | Jeddah and Madinah Province | `flash_flood` | `preserved_user_verified` | `data/raw/flash_flood_verified/user_leads_2025_flash_flood_events.csv` | Preserved as verified user-confirmed severe year-end flood fact. |

## Notes

- The companion inventory file `data/raw/extreme_weather_verified/verified_extreme_weather_inventory.csv` currently contains 68 verified rows.
- I did not upgrade the 2025 user-confirmed dust and flash-flood rows to `external_verified` in this patch because I did not have enough strong direct-source evidence for every detailed claim in the user text.
- The repository now has hazard-specific raw files for the externally cited 2015 Riyadh dust event and the 2023-2024 Hajj heat events, instead of only carrying them inside the cross-hazard inventory.
