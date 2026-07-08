"""Run a batch risk scan using the standard JSON feature interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.data import read_json_features  # noqa: E402
from mazu_saudi.risk import all_default_models  # noqa: E402


def main() -> None:
    features = read_json_features(ROOT / "examples" / "sample_features.json")
    feature_list = features if isinstance(features, list) else [features]
    risks = []
    for item in feature_list:
        for model in all_default_models():
            risks.append(model.predict_one(item).to_dict())
    print(json.dumps({"count": len(feature_list), "risks": risks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
