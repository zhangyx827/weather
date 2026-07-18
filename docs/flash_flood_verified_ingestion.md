# Flash-Flood Verified Ingestion

This repository now includes a minimal verified flash-flood source contract and a reproducible ingestion path.

Bundled real verified inputs used by default:

- `data/raw/flash_flood_verified/user_leads_2025_flash_flood_events.csv`
- `data/raw/flash_flood_verified/web_verified_events_2024_2026-07-16.csv`
- `data/raw/flash_flood_verified/web_verified_events_2026-07-14.csv`

Sample verified input:

- `data/raw/flash_flood_verified/sample_verified_events.csv`

Supported input columns:

- `record_id` or `source_record_id`: stable upstream row identifier
- `date` or `start_date`: event start date in ISO format
- `end_date`: optional end date, defaults to start date
- `location` or `location_name`: human-readable location name
- `lat` or `latitude`: optional latitude
- `lon` or `longitude`: optional longitude
- `geometry_wkt`: optional polygon or multipolygon footprint used for geometry-backed mapping
- `source_url`: optional upstream citation URL
- `source_name`: optional upstream source family or file-specific provenance name
- `validation_status`: optional, defaults to `verified`
- `notes`: optional operator notes

Run the verified ingestion script:

```bash
python3 scripts/build_verified_flash_flood_event_table.py
```

Artifacts:

- combined event table: seed + verified rows with verified provenance preferred on duplicates
- daily expansion table: inclusive day-level event expansion for label joins
- summary JSON: row counts plus provenance coverage, geometry-vs-point-vs-text spatial coverage, and daily `label_source_mode` counts

Notes:

- Running the script with no arguments scans `data/raw/flash_flood_verified/` and ingests every bundled non-sample csv/json/parquet file, then writes outputs under `data/processed/real_flash_flood_chain/`.
- Use repeated `--verified-input` flags to override the default scan with a specific set of verified files.
- Pass `--verified-input data/raw/flash_flood_verified/sample_verified_events.csv --source-name sample_verified` to exercise the ingestion contract with the sample file instead.
- `--verified-only` skips built-in seed events and exports only standardized verified rows.
- Duplicate merge identity is based on hazard, date range, country, and either `geometry_wkt`, normalized location name, or coordinates as a fallback when location is missing.
- Small latitude/longitude differences do not block duplicate folding when two rows describe the same named location and date range.
- The current bundled 2024 web-verified file preserves two explicit April 17, 2024 Saudi flood-impact rows: `Eastern Province` and `Dammam`.
- The sample file is a contract example, not a production verified baseline.
