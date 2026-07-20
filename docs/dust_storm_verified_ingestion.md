# Dust-Storm Event Ingestion

This repo now uses one integrated raw source file for dust-storm facts.

Default source file:

- `data/raw/verified_dust_storm.csv`

This file is the single user-editable intake point. Append new dust-storm facts there, then rebuild the derived event and training tables.

Current status:

- rows are normalized as `dust_storm`
- provenance defaults to `source_name=user_session_handoff`
- `validation_status` is `verified`
- rows are treated as user-confirmed event facts

Build the normalized event table and inclusive daily table:

```bash
python3 scripts/build_dust_storm_event_table.py
```

Outputs by default:

- `data/processed/real_dust_storm_chain/dust_storm_events_2025_verified.csv`
- `data/processed/real_dust_storm_chain/dust_storm_events_2025_verified_daily.csv`
- `data/processed/real_dust_storm_chain/dust_storm_events_2025_verified_summary.json`

Build downstream label artifacts from those verified facts:

```bash
python3 scripts/build_dust_storm_training_labels.py --samples /path/to/dust_samples.csv
python3 scripts/build_dust_storm_supervised_training_table.py --features /path/to/dust_features.csv --labels /path/to/dust_labels.csv
python3 scripts/build_dust_storm_supervised_training_table.py --labels /path/to/dust_labels.csv
```

The supervised-training script now has two supported paths:

- explicit `--features` input for prebuilt dust feature tables
- default processed-indicator discovery from `data/processed/lightgbm_indicators_nc/` when `--features` is omitted

Current label policy:

- matching resolved province or region text becomes `label_status=positive`
- no-event days become `label_status=negative`
- non-matching regions on event days become `label_status=negative` only when the event coverage is resolved
- unresolved event-day coverage stays `label_status=uncertain`

Expected raw columns:

- `record_id`
- `event_id`
- `start_date`
- `end_date`
- `location_name`
- `source_name`
- `source_url`
- `validation_status`
- `spatial_confidence`
- `temporal_confidence`
- `severity`
- `notes`

Recommended additional columns for the integrated CSV:

- `hazard_type`
- `country_code`
- `latitude`
- `longitude`
- `geometry_wkt`
- `source_record_id`

Notes:

- Keep `hazard_type` fixed to `dust_storm`.
- Use one row per event, not one row per article snippet.
- If you know only a province or city, leave geometry empty and let the downstream join use region-day labels.
- Do not create separate `clean` and `verified` raw files. Keep the single raw CSV and use `validation_status` inside the row.

The current bundled file is treated in-repo as a factual verified event source from the user handoff. Additional source-link enrichment can still be added later without downgrading these rows.
