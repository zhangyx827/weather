"""Optional ML-backed model adapter contracts.

This module deliberately avoids importing LightGBM, XGBoost, or SHAP unless the
caller asks for a specific backend. Tests and demos use the fallback adapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class OptionalMLAdapter:
    """Stable adapter contract for future LightGBM/XGBoost/SHAP models."""

    backend = "fallback_stub"

    def __init__(self) -> None:
        self.trained = False
        self.metadata: dict[str, Any] = {"backend": self.backend}

    def train(self, dataset: Any) -> dict[str, Any]:
        """Train or register a dataset.

        Fallback behavior records sample count only. Real implementations should
        fit the estimator and return training metrics.
        """

        self.trained = True
        sample_count = len(dataset) if hasattr(dataset, "__len__") else None
        self.metadata.update({"trained": True, "sample_count": sample_count})
        return {"status": "trained_stub", "sample_count": sample_count}

    def save_model(self, path: str | Path) -> None:
        """Persist lightweight adapter metadata or a real model artifact."""

        Path(path).write_text(json.dumps({"backend": self.backend, "metadata": self.metadata}, indent=2), encoding="utf-8")

    def load_model(self, path: str | Path) -> "OptionalMLAdapter":
        """Load lightweight adapter metadata or a real model artifact."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.metadata = dict(payload.get("metadata", {}))
        self.trained = bool(self.metadata.get("trained"))
        return self

    def predict_proba(self, features: Any) -> float:
        """Return a probability-like score for one feature vector."""

        return 0.0

    def shap_explain(self, features: Any) -> dict[str, Any]:
        """Return SHAP-like explanation payload.

        Fallback marks SHAP unavailable while preserving response shape.
        """

        return {"available": False, "backend": self.backend, "values": {}, "reason": "optional SHAP dependency not installed"}


class LightGBMAdapter(OptionalMLAdapter):
    """LightGBM adapter scaffold, active only when lightgbm is installed."""

    backend = "lightgbm"

    def __init__(self) -> None:
        super().__init__()
        try:
            import lightgbm as lgb
        except Exception as exc:  # pragma: no cover - optional package
            raise RuntimeError("lightgbm is not installed") from exc
        self.lgb = lgb


class XGBoostAdapter(OptionalMLAdapter):
    """XGBoost adapter scaffold, active only when xgboost is installed."""

    backend = "xgboost"

    def __init__(self) -> None:
        super().__init__()
        try:
            import xgboost as xgb
        except Exception as exc:  # pragma: no cover - optional package
            raise RuntimeError("xgboost is not installed") from exc
        self.xgb = xgb


def create_ml_adapter(backend: str = "fallback") -> OptionalMLAdapter:
    """Create an optional ML adapter by backend name."""

    normalized = backend.lower()
    if normalized in {"lightgbm", "lgbm"}:
        return LightGBMAdapter()
    if normalized in {"xgboost", "xgb"}:
        return XGBoostAdapter()
    return OptionalMLAdapter()
