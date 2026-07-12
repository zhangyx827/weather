"""Optional ML-backed model adapter contracts.

This module deliberately avoids importing LightGBM, XGBoost, or SHAP unless the
caller asks for a specific backend. Tests and demos use the fallback adapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


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
        self.booster = None

    def load_model(self, path: str | Path) -> "LightGBMAdapter":
        self.booster = self.lgb.Booster(model_file=str(path))
        self.metadata.update(
            {
                "trained": True,
                "model_path": str(path),
                "feature_names": list(self.booster.feature_name()),
            }
        )
        self.trained = True
        return self

    def predict_proba(self, features: Any) -> float:
        if self.booster is None:
            return 0.0
        array = np.asarray(features, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        prediction = np.asarray(self.booster.predict(array), dtype=np.float32).reshape(-1)
        if prediction.size == 0:
            return 0.0
        return float(np.clip(prediction[0], 0.0, 1.0))

    def shap_explain(self, features: Any) -> dict[str, Any]:
        if self.booster is None:
            return super().shap_explain(features)
        array = np.asarray(features, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        contrib = np.asarray(self.booster.predict(array, pred_contrib=True), dtype=np.float32)
        row = contrib[0].reshape(-1)
        feature_names = list(self.booster.feature_name())
        values = {name: float(row[index]) for index, name in enumerate(feature_names)}
        base_value = float(row[-1]) if row.size == len(feature_names) + 1 else 0.0
        ranked = sorted(values.items(), key=lambda item: abs(item[1]), reverse=True)
        return {
            "available": True,
            "backend": self.backend,
            "base_value": base_value,
            "values": values,
            "top_features": [{"feature": name, "contribution": contribution} for name, contribution in ranked[:3]],
        }


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
