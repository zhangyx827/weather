"""Run a direct Strands warning-generation smoke test."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.agent.strands import StrandsError, StrandsWarningAgent
from mazu_saudi.agent.workflow import load_sample_features
from mazu_saudi.config import StrandsSettings


def main() -> int:
    os.environ.setdefault("MAZU_STRANDS_ENABLED", "true")
    settings = StrandsSettings.from_env()
    try:
        result = StrandsWarningAgent(settings=settings).execute(
            {
                "features": load_sample_features(),
                "industries": ["meteorology"],
                "language": "zh",
            }
        )
    except StrandsError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(result.output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
