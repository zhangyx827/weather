# Repository Guidelines

## Project Structure & Module Organization
Core Python code lives under `src/mazu_saudi/`. Key modules are split by responsibility: `api/` for FastAPI endpoints, `agent/` for workflow and briefing assembly, `indicators/` for physical indicator generation, `risk/` for rule and ML-backed hazard models, `kg/` for knowledge-graph utilities, and `data/` for I/O helpers. Tests live in `tests/` as `test_*.py` files. Examples are in `examples/`, one-off data/build scripts are in `scripts/`, and larger datasets or generated artifacts are stored under `data/` and `models/`.

## Build, Test, and Development Commands
Install the package and common dev tools with `python3 -m pip install -e ".[dev]"`. Add forecast or ML extras only when needed: `python3 -m pip install -e ".[forecast,ml]"`.

Run the main checks with:
- `PYTHONPATH=src python3 -m pytest -q` for the primary test suite.
- `PYTHONPATH=src python3 -m unittest discover -s tests` for compatibility with the README workflow.
- `uvicorn mazu_saudi.api.app:app --app-dir src --reload` to start the local API.
- `python3 examples/demo_saudi_warning.py` to exercise the end-to-end demo pipeline.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, module-level docstrings, type hints on public functions, and small focused modules. Use `snake_case` for files, functions, and variables; use `PascalCase` for Pydantic models and other classes such as `RiskScanRequest`. Keep imports explicit and grouped, and prefer straightforward control flow over framework-heavy abstractions. No formatter or linter config is checked in, so keep changes consistent with surrounding code.

## Testing Guidelines
Add or update tests in `tests/` whenever behavior changes. Name new files `test_<feature>.py` and test functions `test_<scenario>()`. Favor fast, deterministic unit tests that import from `src/` without relying on large local datasets unless the test is explicitly data-pipeline coverage.

## Commit & Pull Request Guidelines
Recent history mixes Conventional Commit prefixes (`feat:`) with short free-form messages (`engineering enhance`, `real data`). Prefer concise, imperative subjects; use a prefix like `feat:`, `fix:`, or `test:` when it improves clarity. Pull requests should summarize the behavior change, list test coverage, note any data/model artifacts touched, and include request or response examples when API behavior changes.

## Data & Configuration Notes
Treat files under `data/raw/` and large NetCDF outputs as inputs or generated artifacts, not hand-edited source. Keep reusable configuration in `src/mazu_saudi/config/`, and avoid hard-coding absolute paths in examples, scripts, or tests.

## Merge Discipline
When a user says "merge into" a file, do not assume they want row removal or deduplication. Confirm the merge direction from the source content, and preserve the existing file unless the user explicitly asks for a reduction or replacement.

## Architecture Evolution Notes

### Runtime Physical Grounding
The indicator pipeline now follows a registry-driven multi-source pattern: a primary source provides the model-facing value, and secondary sources stay outside the feature vector as runtime evidence. Keep that separation intact. The LightGBM and rule models should continue to consume stable primary-source features only; auxiliary sources are allowed to affect confidence, uncertainty language, QA, and failover decisions, but must not be merged into the training or inference feature layer unless the model is explicitly retrained for that distribution.

When extending `build_manifest.json`, `indicator_evidence`, or downstream workflow payloads, prefer additive provenance fields over in-place value rewriting. New runtime grounding metadata should be traceable per indicator family and per source family. At minimum, use explicit source IDs, validation status, and comparable numeric deltas instead of free-form notes.

### Grounding Gap Contract
Planned runtime grounding work should standardize a `grounding_gap` structure for indicators that have a strong bypass source, for example ERA5 vs GPM for precipitation and ERA5 vs MUR/OISST for SST. The intended contract is:
- primary source determines the numeric feature value used by models
- bypass source determines a comparable delta, confidence hint, or uncertainty flag
- the delta must be persisted in either `build_manifest.json`, `indicator_evidence`, or both, with enough metadata to reconstruct which source pair was compared

Prefer a structured payload such as source pair, comparison timestamp, units, absolute difference, and summary statistics over a single unlabeled scalar.

### Agent Consumption Rules
`BriefingNode` and adjacent workflow nodes should treat grounding metadata as required context for high-impact wording. If a grounding gap crosses a physical threshold, the agent should inject an uncertainty or verification note rather than silently discarding the bypass source. This is a runtime communication rule, not a feature-engineering rule.

Keep the policy data-driven:
- thresholds should live in configuration, not hard-coded prompt text
- wording should be deterministic enough to test
- the workflow should degrade gracefully when grounding metadata is missing

### Failover and Degradation
Primary-source failure handling should follow a zero-retrain failover path. If the configured primary dataset for an indicator family is missing, unreadable, or invalid, the runtime may promote a validated secondary source to temporary primary status for that run. Any such promotion must emit explicit degradation metadata, for example `source_status: degraded`, the promoted source ID, and the reason for fallback.

Do not hide failover behind silent substitution. The returned indicator value, the manifest, and any graph persistence layer should all preserve the fact that degradation occurred.

### Drift and Attribution Monitoring
Bypass sources are also part of the monitoring surface. When adding drift checks, prefer simple, inspectable time-series signals first:
- daily or per-run grounding-gap history by indicator family
- alert thresholds for persistent directional drift
- optional consistency checks between large source disagreement and stable model conclusions

SHAP or other attribution-stability checks are useful only if they stay interpretable and cheap enough for routine execution. Keep them as monitoring outputs, not gating logic inside the core inference path, unless a clear operational rule is defined.

### Persistence and Localization
Any grounding gaps, failover events, and drift alerts that matter operationally should be persistable to the knowledge-graph layer with enough provenance to reconstruct the active source relationship for a historical warning. When adding localization or station-based correction hooks, keep them as explicit post-model calibration interfaces. They may adjust reported interpretation, but they should not silently mutate the raw model feature distribution.
