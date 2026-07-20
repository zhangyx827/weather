"""Grid-based Layer-4 risk inference for forecast background fields."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from .layer4_features import (
    _dataset_feature_fields,
    enhancement_feature_names_for_hazard,
    evidence_feature_names_for_hazard,
    feature_matrix_from_dataset,
    feature_names_for_hazard,
    required_feature_names_for_hazard,
)
from .model_paths import REPO_ROOT, resolve_layer4_model_path

try:
    import xarray as xr
except Exception:  # pragma: no cover - optional dependency
    xr = None


def _levels_from_probability(probability: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float32)
    return np.where(
        p >= 0.75,
        3,
        np.where(
            p >= 0.50,
            2,
            np.where(
                p >= 0.20,
                1,
                0,
            ),
        ),
    ).astype(np.int8)


class LightGBMLayer4Model:
    """Layer-4 grid inference backed by LightGBM booster files."""

    def __init__(
        self,
        extreme_heat_model_path: str | Path | None = None,
        dry_heat_model_path: str | Path | None = None,
        flash_flood_model_path: str | Path | None = None,
        *,
        extreme_heat_model: Any | None = None,
        dry_heat_model: Any | None = None,
        flash_flood_model: Any | None = None,
    ) -> None:
        self.extreme_heat_model_path = self._resolve_model_path(
            explicit=extreme_heat_model_path,
            env_key="MAZU_LAYER4_EXTREME_HEAT_MODEL",
            default_name="extreme_heat.txt",
            allow_missing=extreme_heat_model is not None,
        )
        self.dry_heat_model_path = self._resolve_model_path(
            explicit=dry_heat_model_path,
            env_key="MAZU_LAYER4_DRY_HEAT_MODEL",
            default_name="dry_heat_stress.txt",
            allow_missing=dry_heat_model is not None,
        )
        self.flash_flood_model_path = self._resolve_model_path(
            explicit=flash_flood_model_path,
            env_key="MAZU_LAYER4_FLASH_FLOOD_MODEL",
            default_name="flash_flood.txt",
            allow_missing=True,
        )
        self.extreme_heat_model = extreme_heat_model or self._load_booster(self.extreme_heat_model_path)
        self.dry_heat_model = dry_heat_model or self._load_booster(self.dry_heat_model_path)
        self.flash_flood_model = flash_flood_model or self._load_optional_booster(self.flash_flood_model_path)

    @staticmethod
    def _resolve_model_path(
        explicit: str | Path | None,
        env_key: str,
        default_name: str,
        *,
        allow_missing: bool,
    ) -> Path | None:
        hazard_type = {
            "MAZU_LAYER4_EXTREME_HEAT_MODEL": "extreme_heat",
            "MAZU_LAYER4_DRY_HEAT_MODEL": "dry_heat_agriculture",
            "MAZU_LAYER4_FLASH_FLOOD_MODEL": "flash_flood",
        }.get(env_key)
        if hazard_type is None:
            raise ValueError(f"Unsupported Layer-4 model env key: {env_key}")
        return resolve_layer4_model_path(
            hazard_type,
            explicit=explicit,
            env_key=env_key,
            default_name=default_name,
            allow_missing=allow_missing,
        )

    @staticmethod
    def _import_lightgbm() -> Any:
        try:
            import lightgbm as lgb
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "lightgbm is required for Layer-4 grid inference. "
                "Install it with `pip install lightgbm` and provide trained model files."
            ) from exc
        return lgb

    def _load_booster(self, path: Path | None) -> Any:
        if path is None:
            raise RuntimeError("Layer-4 model path is missing")
        lgb = self._import_lightgbm()
        try:
            return lgb.Booster(model_file=str(path))
        except Exception as exc:
            raise RuntimeError(f"Unable to load LightGBM model from {path}: {exc}") from exc

    def _load_optional_booster(self, path: Path | None) -> Any | None:
        if path is None:
            return None
        return self._load_booster(path)

    @staticmethod
    def _display_path(path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    def _feature_matrix(self, dataset: Any) -> tuple[np.ndarray, tuple[int, ...]]:
        return feature_matrix_from_dataset(dataset, hazard_type="extreme_heat")

    @staticmethod
    def _model_feature_names(model: Any, hazard_type: str) -> tuple[str, ...]:
        feature_names_attr = getattr(model, "feature_name", None)
        if callable(feature_names_attr):
            names = tuple(str(name) for name in feature_names_attr() if str(name))
            if names:
                return names
        return feature_names_for_hazard(hazard_type)

    def _aligned_feature_matrix(self, dataset: Any, hazard_type: str, model: Any) -> tuple[np.ndarray, tuple[int, ...]]:
        model_feature_names = self._model_feature_names(model, hazard_type)
        if model_feature_names == feature_names_for_hazard(hazard_type):
            return feature_matrix_from_dataset(dataset, hazard_type=hazard_type)

        fields, shape, _ = _dataset_feature_fields(dataset, hazard_type, include_evidence_only=True)
        matrix = np.column_stack([fields[name].reshape(-1) for name in model_feature_names]).astype(np.float32)
        required_indexes = [index for index, name in enumerate(model_feature_names) if name not in enhancement_feature_names_for_hazard(hazard_type)]
        if required_indexes and not np.all(np.isfinite(matrix[:, required_indexes]), axis=1).any():
            raise ValueError(
                f"Layer-4 dataset has no valid cells for {hazard_type} using trained model features {model_feature_names}"
            )
        return matrix, shape

    @staticmethod
    def _predict_probability(model: Any, features: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        prediction = np.asarray(model.predict(features))
        if prediction.ndim == 2:
            if prediction.shape[1] == 2:
                prediction = prediction[:, 1]
            elif prediction.shape[1] == 1:
                prediction = prediction[:, 0]
            else:
                raise ValueError(f"Expected binary LightGBM probabilities, got prediction shape {prediction.shape}")
        if prediction.size != features.shape[0]:
            raise ValueError(
                f"Layer-4 prediction size {prediction.size} does not match feature rows {features.shape[0]}"
            )
        return np.clip(prediction.reshape(shape), 0.0, 1.0)

    def predict_fields(self, dataset: Any) -> Any:
        if xr is None:
            raise RuntimeError("xarray is required for Layer-4 prediction fields")
        if not hasattr(dataset, "data_vars"):
            raise TypeError(f"Expected xarray.Dataset-like input, got {type(dataset)!r}")

        ds = dataset
        first_var = next(iter(ds.data_vars))
        source_dims = ds[first_var].dims
        data_vars: dict[str, Any] = {}
        shape: tuple[int, ...] | None = None
        dims: tuple[str, ...] | None = None
        skipped_hazards: list[dict[str, str]] = []

        def _dims_for_shape(current_shape: tuple[int, ...]) -> tuple[str, ...]:
            if len(source_dims) != len(current_shape):
                return tuple(dim for dim in source_dims if ds[first_var].sizes.get(dim) != 1)
            return source_dims

        def _predict_hazard(
            *,
            hazard_type: str,
            model: Any,
            probability_name: str,
            level_name: str,
        ) -> None:
            nonlocal shape, dims
            try:
                features, current_shape = self._aligned_feature_matrix(ds, hazard_type, model)
            except (KeyError, ValueError) as exc:
                skipped_hazards.append({"hazard_type": hazard_type, "reason": str(exc)})
                return
            if shape is None:
                shape = current_shape
                dims = _dims_for_shape(current_shape)
            elif current_shape != shape:
                raise ValueError(f"{hazard_type} feature shape {current_shape} does not match {shape}")
            probability = self._predict_probability(model, features, shape)
            assert dims is not None
            data_vars[probability_name] = (dims, np.asarray(probability, dtype=np.float32), {"units": "1"})
            data_vars[level_name] = (dims, _levels_from_probability(probability), {"units": "class"})

        _predict_hazard(
            hazard_type="extreme_heat",
            model=self.extreme_heat_model,
            probability_name="ExtremeHeat_Risk_Prob",
            level_name="ExtremeHeat_Risk_Level",
        )
        _predict_hazard(
            hazard_type="dry_heat_agriculture",
            model=self.dry_heat_model,
            probability_name="DryHeatStress_Risk_Prob",
            level_name="DryHeatStress_Risk_Level",
        )

        attrs = {
            "model_family": "lightgbm",
            "model_name": "LightGBMLayer4Model",
            "extreme_heat_model": self._display_path(self.extreme_heat_model_path),
            "dry_heat_model": self._display_path(self.dry_heat_model_path),
            "flash_flood_model": self._display_path(self.flash_flood_model_path),
            "feature_names_extreme_heat": ",".join(feature_names_for_hazard("extreme_heat")),
            "feature_names_dry_heat_agriculture": ",".join(feature_names_for_hazard("dry_heat_agriculture")),
            "required_core_features_extreme_heat": ",".join(required_feature_names_for_hazard("extreme_heat")),
            "required_core_features_dry_heat_agriculture": ",".join(required_feature_names_for_hazard("dry_heat_agriculture")),
            "optional_enhancement_features_extreme_heat": ",".join(enhancement_feature_names_for_hazard("extreme_heat")),
            "optional_enhancement_features_dry_heat_agriculture": ",".join(enhancement_feature_names_for_hazard("dry_heat_agriculture")),
            "evidence_only_features_extreme_heat": ",".join(evidence_feature_names_for_hazard("extreme_heat")),
            "evidence_only_features_dry_heat_agriculture": ",".join(evidence_feature_names_for_hazard("dry_heat_agriculture")),
            "feature_source_contract_version": "layer4_v2",
        }
        if self.flash_flood_model is not None:
            _predict_hazard(
                hazard_type="flash_flood",
                model=self.flash_flood_model,
                probability_name="FlashFlood_Risk_Prob",
                level_name="FlashFlood_Risk_Level",
            )
            attrs["feature_names_flash_flood"] = ",".join(feature_names_for_hazard("flash_flood"))
            attrs["required_core_features_flash_flood"] = ",".join(required_feature_names_for_hazard("flash_flood"))
            attrs["optional_enhancement_features_flash_flood"] = ",".join(enhancement_feature_names_for_hazard("flash_flood"))
            attrs["evidence_only_features_flash_flood"] = ",".join(evidence_feature_names_for_hazard("flash_flood"))

        if not data_vars or shape is None or dims is None:
            reasons = "; ".join(f"{item['hazard_type']}: {item['reason']}" for item in skipped_hazards) or "no supported hazard features"
            raise ValueError(f"Layer-4 dataset produced no risk fields: {reasons}")
        if skipped_hazards:
            attrs["skipped_hazards_json"] = json.dumps(skipped_hazards, ensure_ascii=False, sort_keys=True)

        return xr.Dataset(
            data_vars=data_vars,
            coords={
                name: (
                    ds.coords[name].isel({name: 0}, drop=True)
                    if name in ds.coords and ds.coords[name].sizes.get(name, 0) == 1 and name not in dims
                    else ds.coords[name]
                )
                for name in ds.coords
                if name in dims or name not in ds.dims or ds.coords[name].sizes.get(name, 0) != 1
            },
            attrs=attrs,
        )


def predict_layer4_risk_fields(
    dataset: Any,
    *,
    extreme_heat_model_path: str | Path | None = None,
    dry_heat_model_path: str | Path | None = None,
    flash_flood_model_path: str | Path | None = None,
) -> Any:
    """Run the standard Layer-4 grid inference entrypoint."""

    model = LightGBMLayer4Model(
        extreme_heat_model_path=extreme_heat_model_path,
        dry_heat_model_path=dry_heat_model_path,
        flash_flood_model_path=flash_flood_model_path,
    )
    return model.predict_fields(dataset)
