# MAZU Saudi Early Warning Algorithm MVP

Runnable first-version prototype for a Saudi-localized MAZU multi-hazard early warning agent system.

Architecture layers:

1. Data resource layer: sample JSON and forecast provider interfaces.
2. Physical indicator layer: VPD, heat index, placeholder PWAT/IVT/CAPE, and screening scores.
3. AI forecast background layer: Aurora, GenCast, AIFS, and mock provider interfaces.
4. Multi-hazard risk model layer: five rule-based models with replaceable model interfaces.
5. Disaster knowledge graph layer: RDF-style triples and query services with rdflib integration when installed.
6. MAZU agent service layer: workflow nodes, industry briefings, and FastAPI API.

Run demo:

```bash
python3 examples/demo_saudi_warning.py
```

Run the other examples:

```bash
python3 examples/run_batch_risk_scan.py
python3 examples/run_api_client_demo.py
python3 examples/export_kg_ttl.py
```

Train the Layer-4 LightGBM models:

```bash
./scripts/run_train_layer4_lightgbm.sh --source data/processed/layer4_training_tables/flash_flood_training.parquet --source-format indicator-parquet --hazard-type flash_flood --model-dir models/layer4
```

If you want to call the Python entrypoint directly, use:

```bash
python3 examples/train_layer4_lightgbm.py --source /path/to/dataset --source-format auto --hazard-type extreme_heat --model-dir models/layer4
```

Convert the bundled NIS SRTM elevation tiles into a model-ready Saudi grid:

```bash
python3 scripts/convert_nis_srtm.py data/raw/nis data/output/nis
```

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Inspect current repo capabilities:

```bash
python3 scripts/capability_report.py
```

STCast 2024 data cadence and stats-window contract:

```bash
cat docs/stcast_2024_contract.md
```

Verified flash-flood source ingestion contract:

```bash
cat docs/flash_flood_verified_ingestion.md
python3 scripts/build_verified_flash_flood_event_table.py
python3 scripts/build_flash_flood_training_labels.py --samples /path/to/flash_flood_samples.csv
python3 scripts/build_flash_flood_supervised_training_table.py --features /path/to/flash_flood_features.parquet --labels /path/to/flash_flood_labels.parquet
```

The default flash-flood verified build now ingests all bundled non-sample files under `data/raw/flash_flood_verified/`, including the preserved 2025 user facts and two explicit 2024 web-verified rows for `Eastern Province` and `Dammam`. The verified-event summary now reports geometry-backed vs point/text-only coverage, and the label/training-table builders report label-source-mode and matched-event audit counts.

2025 dust-storm verified ingestion contract:

```bash
cat docs/dust_storm_verified_ingestion.md
python3 scripts/build_dust_storm_event_table.py
python3 scripts/build_dust_storm_training_labels.py --samples /path/to/dust_samples.csv
python3 scripts/build_dust_storm_supervised_training_table.py --features /path/to/dust_features.csv --labels /path/to/dust_labels.csv
```

Run API after installing dependencies:

```bash
python3 -m pip install -r requirements.txt
uvicorn mazu_saudi.api.app:app --app-dir src --reload
```

API checks:

```bash
PYTHONPATH=src python3 -m pytest -q tests/test_api.py
```
