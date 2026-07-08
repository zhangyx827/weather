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

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
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
