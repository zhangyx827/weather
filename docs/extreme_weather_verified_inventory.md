# Extreme-Weather Verified Inventory

This repository now includes a cross-hazard verified fact inventory:

- `data/raw/extreme_weather_verified/verified_extreme_weather_inventory_2013_2025.csv`

Purpose:

- keep one discoverable table for preserved Saudi extreme-weather facts across hazards
- retain `hazard_type`, dates, place names, source links, and validation status
- keep provenance in `source_file` when a row was carried forward from an existing verified source
- merge the user-verified 2013-2023 rows into the broader verified inventory without duplicating rows

Current hazards represented:

- `flash_flood`
- `dust_storm`
- `extreme_heat`
- `tropical_cyclone`
- `snowstorm`

Hazard-specific raw files now exist for the externally cited non-flood rows as well:

- `data/raw/dust_storm_verified/web_verified_dust_events_2015_2026-07-16.csv`
- `data/raw/extreme_heat_verified/web_verified_hajj_heat_events_2023_2024.csv`

Important scope notes:

- This file is an inventory layer, not a replacement for the hazard-specific ingestion paths under `data/raw/flash_flood_verified/` and `data/raw/dust_storm_verified/`.
- User-provided 2025 rows remain `validation_status=verified`.
- Web-compiled rows preserve direct source URLs and conservative notes. They should be upgraded later when stronger primary or official citations are available.
- Rows should stay in this inventory only when the cited source supports the event fact directly enough for `validation_status=verified`. Search phrases and forecast-only mentions are not enough on their own.
- The inventory now also carries user-confirmed 2013-2023 historical rows with explicit source URLs supplied during verification; the standalone source CSV has been retired.
- `docs/extreme_weather_verification_matrix_2026-07-16.md` is the current ledger that separates external-citation rows from preserved user-confirmed rows.
