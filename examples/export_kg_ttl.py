"""Export the default MAZU Saudi hazard knowledge graph as Turtle."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.kg import HazardKnowledgeGraph  # noqa: E402


def main() -> None:
    graph = HazardKnowledgeGraph()
    print(graph.to_ttl())


if __name__ == "__main__":
    main()
