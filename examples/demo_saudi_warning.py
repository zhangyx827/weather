"""Run the MAZU Saudi MVP warning pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.agent import run_demo_pipeline  # noqa: E402


def main() -> None:
    """Run the demo and print structured warning output."""

    result = run_demo_pipeline()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
