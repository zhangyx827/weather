# Dust-Storm Event Ingestion

This repo now preserves the user-provided 2025 Saudi dust-storm factual event list as a raw source file instead of leaving it only in chat.

Default source file:

- `data/raw/dust_storm_verified/user_leads_2025_dust_events.csv`

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
```

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
- `location`
- `source_name`
- `source_url`
- `validation_status`
- `spatial_confidence`
- `temporal_confidence`
- `severity`
- `notes`

The current bundled file is treated in-repo as a factual verified event source from the user handoff. Additional source-link enrichment can still be added later without downgrading these rows.
