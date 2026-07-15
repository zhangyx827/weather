# Flash-Flood Verified Ingestion

This repository now includes a minimal verified flash-flood source contract and a reproducible ingestion path.

Sample verified input:

- `data/raw/flash_flood_verified/sample_verified_events.csv`

Supported input columns:

- `record_id` or `source_record_id`: stable upstream row identifier
- `date` or `start_date`: event start date in ISO format
- `end_date`: optional end date, defaults to start date
- `location` or `location_name`: human-readable location name
- `lat` or `latitude`: optional latitude
- `lon` or `longitude`: optional longitude
- `source_url`: optional upstream citation URL
- `validation_status`: optional, defaults to `verified`
- `notes`: optional operator notes

Run the verified ingestion script:

```bash
python3 scripts/build_verified_flash_flood_event_table.py \
  --verified-input data/raw/flash_flood_verified/sample_verified_events.csv \
  --source-name sample_verified \
  --output data/processed/labels/flash_flood_events_verified_combined.csv \
  --daily-output data/processed/labels/flash_flood_events_verified_combined_daily.csv \
  --summary-output data/processed/labels/flash_flood_events_verified_summary.json
```

Artifacts:

- combined event table: seed + verified rows with verified provenance preferred on duplicates
- daily expansion table: inclusive day-level event expansion for label joins
- summary JSON: row counts plus `source_name`, `source_url`, `source_record_id`, and `validation_status` coverage

Notes:

- `--verified-only` skips built-in seed events and exports only standardized verified rows.
- Duplicate merge identity is based on hazard, date range, country, and either `geometry_wkt`, normalized location name, or coordinates as a fallback when location is missing.
- Small latitude/longitude differences do not block duplicate folding when two rows describe the same named location and date range.
- The sample file is a contract example, not a production verified baseline.
