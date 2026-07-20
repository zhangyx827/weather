"""Export the default MAZU Saudi knowledge graph as Turtle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.kg import HazardKnowledgeGraph  # noqa: E402
from mazu_saudi.schemas import GridCell, HazardRisk, RiskLevel  # noqa: E402


def _add_demo_runtime(graph: HazardKnowledgeGraph) -> None:
    """Populate a small deterministic runtime graph for deployment validation."""

    risk = HazardRisk(
        hazard_type="flash_flood",
        risk_probability=0.82,
        risk_level=RiskLevel.HIGH,
        contributing_factors=["precipitation_mm"],
        grid=GridCell(id="demo-riyadh", lat=24.7136, lon=46.6753, region="Riyadh"),
        indicator_evidence={"precipitation_mm": 42.0},
        model_name="demo_runtime",
        model_version="validation",
    )
    graph.add_risk_evidence(risk)
    graph.add_grounding_gap_evidence(
        risk,
        source_metadata={
            "source_status": "validated",
            "resolved_sources": {
                "precipitation": {"dataset_id": "era5_demo", "role": "primary"},
            },
        },
        grounding_gap={
            "precipitation": {
                "source_pair": ["era5_demo", "gpm_demo"],
                "comparison_time": "2026-07-20T00:00:00Z",
                "units": "mm",
                "absolute_difference": 1.5,
            }
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instances",
        action="store_true",
        help="Export runtime instance triples instead of the ontology seed.",
    )
    parser.add_argument(
        "--demo-runtime",
        action="store_true",
        help="Add one deterministic risk and grounding record for validation.",
    )
    parser.add_argument("--output", type=Path, help="Write Turtle to this file.")
    args = parser.parse_args()

    graph = HazardKnowledgeGraph()
    if args.demo_runtime:
        _add_demo_runtime(graph)
    payload = graph.export_instances() if args.instances else graph.to_ttl()
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
