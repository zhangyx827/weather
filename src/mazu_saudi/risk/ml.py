"""Optional ML-backed model adapter contracts.

This module deliberately avoids importing LightGBM, XGBoost, or SHAP unless the
caller asks for a specific backend. Tests and demos use the fallback adapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _metadata_sidecar_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_name(f"{target.name}.metadata.json")


def _normalize_split_groups(raw_groups: Any, expected_rows: int) -> np.ndarray:
    groups = np.asarray(raw_groups, dtype=object).reshape(-1)
    if groups.shape[0] != expected_rows:
        raise ValueError(f"Split-group count {groups.shape[0]} does not match feature row count {expected_rows}")
    normalized: list[str] = []
    for index, value in enumerate(groups):
        text = "" if value is None else str(value).strip()
        if not text or text.lower() == "nan":
            text = f"__row_{index}"
        normalized.append(text)
    return np.asarray(normalized, dtype=object)


def _coerce_group_split_indices(
    split_groups: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if split_groups.size < 2:
        return None
    unique_groups = np.unique(split_groups)
    if unique_groups.size < 2:
        return None

    validation_fraction = min(max(validation_fraction, 0.05), 0.5)
    target_valid_rows = max(1, min(split_groups.size - 1, int(split_groups.size * validation_fraction)))
    rng = np.random.default_rng(seed)
    shuffled_groups = unique_groups.copy()
    rng.shuffle(shuffled_groups)

    valid_mask = np.zeros(split_groups.shape[0], dtype=bool)
    for group in shuffled_groups:
        group_mask = split_groups == group
        if not group_mask.any():
            continue
        candidate_mask = valid_mask | group_mask
        remaining_rows = (~candidate_mask).sum()
        if remaining_rows == 0:
            continue
        valid_mask = candidate_mask
        if int(valid_mask.sum()) >= target_valid_rows:
            break

    valid_idx = np.flatnonzero(valid_mask)
    train_idx = np.flatnonzero(~valid_mask)
    if valid_idx.size == 0 or train_idx.size == 0:
        return None
    return train_idx.astype(np.int32), valid_idx.astype(np.int32)


def _coerce_training_arrays(dataset: Any) -> tuple[np.ndarray, np.ndarray, list[str] | None, np.ndarray | None]:
    """Normalize common training payload shapes into dense arrays."""

    feature_names = None
    features = None
    labels = None
    split_groups = None

    if isinstance(dataset, dict):
        features = dataset.get("features", dataset.get("X"))
        labels = dataset.get("labels", dataset.get("y", dataset.get("target")))
        split_groups = dataset.get("split_groups")
        raw_feature_names = dataset.get("feature_names")
        if raw_feature_names is not None:
            feature_names = [str(name) for name in raw_feature_names]
    elif isinstance(dataset, (tuple, list)) and len(dataset) == 2:
        features, labels = dataset
    else:
        features = getattr(dataset, "features", getattr(dataset, "X", None))
        labels = getattr(dataset, "labels", getattr(dataset, "y", getattr(dataset, "target", None)))
        raw_feature_names = getattr(dataset, "feature_names", None)
        if raw_feature_names is not None:
            feature_names = [str(name) for name in raw_feature_names]

    if features is None or labels is None:
        raise TypeError("Training dataset must provide features and labels")

    feature_array = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.float32).reshape(-1)
    if feature_array.ndim == 1:
        feature_array = feature_array.reshape(-1, 1)
    if feature_array.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {feature_array.shape}")
    if feature_array.shape[0] != label_array.shape[0]:
        raise ValueError(
            f"Feature row count {feature_array.shape[0]} does not match label count {label_array.shape[0]}"
        )
    if feature_names is not None and len(feature_names) != feature_array.shape[1]:
        raise ValueError(
            f"Feature name count {len(feature_names)} does not match feature width {feature_array.shape[1]}"
        )
    if feature_names is None:
        feature_names = [f"feature_{index}" for index in range(feature_array.shape[1])]
    normalized_groups = None if split_groups is None else _normalize_split_groups(split_groups, feature_array.shape[0])
    return feature_array, label_array, feature_names, normalized_groups


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

    def train(
        self,
        dataset: Any,
        *,
        validation_fraction: float = 0.2,
        seed: int = 42,
        num_boost_round: int = 100,
        early_stopping_rounds: int = 10,
        objective: str | None = None,
        metric: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        features, labels, feature_names, split_groups = _coerce_training_arrays(dataset)
        indices = np.arange(features.shape[0], dtype=np.int32)
        split_strategy = "random_row_shuffle"
        group_count = None
        validation_group_count = None

        if indices.size >= 10 and validation_fraction > 0.0:
            grouped_split = None
            if split_groups is not None:
                grouped_split = _coerce_group_split_indices(split_groups, validation_fraction=validation_fraction, seed=seed)
            if grouped_split is not None:
                train_idx, valid_idx = grouped_split
                split_strategy = "group_shuffle"
                group_count = int(np.unique(split_groups).size)
                validation_group_count = int(np.unique(split_groups[valid_idx]).size)
            else:
                rng = np.random.default_rng(seed)
                rng.shuffle(indices)
                validation_fraction = min(max(validation_fraction, 0.05), 0.5)
                split = max(1, min(indices.size - 1, int(indices.size * (1.0 - validation_fraction))))
                train_idx = indices[:split]
                valid_idx = indices[split:]
        else:
            train_idx = indices
            valid_idx = np.asarray([], dtype=np.int32)
            split_strategy = "train_only"

        unique_labels = {float(value) for value in np.unique(labels[np.isfinite(labels)])}
        is_binary = unique_labels.issubset({0.0, 1.0}) and len(unique_labels) >= 1
        resolved_objective = objective or ("binary" if is_binary else "regression")
        resolved_metric = metric or ("binary_logloss" if resolved_objective == "binary" else "rmse")
        training_params = {
            "boosting_type": "gbdt",
            "objective": resolved_objective,
            "metric": resolved_metric,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": max(1, min(20, max(1, features.shape[0] // 10))),
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
        }
        if params:
            training_params.update(params)
        train_set = self.lgb.Dataset(
            features[train_idx],
            label=labels[train_idx],
            feature_name=feature_names,
            free_raw_data=False,
        )
        valid_sets = []
        valid_names = []
        callbacks = []
        if valid_idx.size:
            valid_sets.append(
                self.lgb.Dataset(
                    features[valid_idx],
                    label=labels[valid_idx],
                    feature_name=feature_names,
                    free_raw_data=False,
                )
            )
            valid_names.append("validation")
            callbacks.append(self.lgb.early_stopping(early_stopping_rounds, verbose=False))

        self.booster = self.lgb.train(
            training_params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets or None,
            valid_names=valid_names or None,
            callbacks=callbacks,
        )
        self.trained = True
        best_iteration = int(getattr(self.booster, "best_iteration", 0) or 0)
        best_score = getattr(self.booster, "best_score", {}) or {}
        validation_scores = best_score.get("validation", {})
        validation_metric = None
        if resolved_metric in validation_scores:
            validation_metric = float(validation_scores[resolved_metric])
        elif validation_scores:
            first_metric_name, first_metric_value = next(iter(validation_scores.items()))
            resolved_metric = str(first_metric_name)
            validation_metric = float(first_metric_value)
        training_summary = {
            "status": "trained",
            "backend": self.backend,
            "sample_count": int(features.shape[0]),
            "feature_count": int(features.shape[1]),
            "feature_names": list(feature_names),
            "objective": training_params["objective"],
            "metric": resolved_metric,
            "best_iteration": best_iteration,
            "train_rows": int(train_idx.size),
            "validation_rows": int(valid_idx.size),
            "validation_metric": validation_metric,
            "split_strategy": split_strategy,
        }
        if group_count is not None:
            training_summary["split_group_count"] = group_count
        if validation_group_count is not None:
            training_summary["validation_group_count"] = validation_group_count
        self.metadata.update(
            {
                "backend": self.backend,
                "trained": True,
                "training_params": training_params,
                **training_summary,
            }
        )
        return training_summary

    def load_model(self, path: str | Path) -> "LightGBMAdapter":
        self.booster = self.lgb.Booster(model_file=str(path))
        metadata_path = _metadata_sidecar_path(path)
        if metadata_path.exists():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            loaded_metadata = dict(payload.get("metadata", payload))
        else:
            loaded_metadata = {}
        feature_names = list(self.booster.feature_name())
        self.metadata.update(
            {
                **loaded_metadata,
                "backend": self.backend,
                "trained": True,
                "model_path": str(path),
                "feature_names": feature_names,
            }
        )
        self.trained = True
        return self

    def save_model(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.booster is None:
            raise RuntimeError("No LightGBM booster is loaded or trained")
        self.booster.save_model(str(target))
        self.metadata.update(
            {
                "backend": self.backend,
                "trained": True,
                "model_path": str(target),
                "feature_names": list(self.booster.feature_name()),
            }
        )
        _metadata_sidecar_path(target).write_text(
            json.dumps({"backend": self.backend, "metadata": self.metadata}, indent=2),
            encoding="utf-8",
        )

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
