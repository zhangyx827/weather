"""Grid-based Layer-4 risk inference for forecast background fields."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .layer4_features import feature_matrix_from_dataset, feature_names_for_hazard

try:
    import xarray as xr
except Exception:  # pragma: no cover - optional dependency
    xr = None


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LAYER4_MODEL_DIR = REPO_ROOT / "models" / "layer4"


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
        raw_path = explicit or os.environ.get(env_key) or (DEFAULT_LAYER4_MODEL_DIR / default_name)
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.exists():
            return path
        if allow_missing:
            return None
        raise FileNotFoundError(
            f"Layer-4 LightGBM model file not found: {path}. "
            f"Set {env_key} to a trained LightGBM booster file."
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
        features, shape = self._feature_matrix(ds)
        first_var = next(iter(ds.data_vars))
        source_dims = ds[first_var].dims
        if len(source_dims) != len(shape):
            dims = tuple(dim for dim in source_dims if ds[first_var].sizes.get(dim) != 1)
        else:
            dims = source_dims

        extreme_heat_prob = self._predict_probability(self.extreme_heat_model, features, shape)
        dry_features, dry_shape = feature_matrix_from_dataset(ds, hazard_type="dry_heat_agriculture")
        if dry_shape != shape:
            raise ValueError(f"Dry-heat feature shape {dry_shape} does not match extreme-heat feature shape {shape}")
        dry_heat_prob = self._predict_probability(self.dry_heat_model, dry_features, shape)

        data_vars = {
            "ExtremeHeat_Risk_Prob": (dims, np.asarray(extreme_heat_prob, dtype=np.float32), {"units": "1"}),
            "ExtremeHeat_Risk_Level": (dims, _levels_from_probability(extreme_heat_prob), {"units": "class"}),
            "DryHeatStress_Risk_Prob": (dims, np.asarray(dry_heat_prob, dtype=np.float32), {"units": "1"}),
            "DryHeatStress_Risk_Level": (dims, _levels_from_probability(dry_heat_prob), {"units": "class"}),
        }
        attrs = {
            "model_family": "lightgbm",
            "model_name": "LightGBMLayer4Model",
            "extreme_heat_model": self._display_path(self.extreme_heat_model_path),
            "dry_heat_model": self._display_path(self.dry_heat_model_path),
            "flash_flood_model": self._display_path(self.flash_flood_model_path),
            "feature_names_extreme_heat": ",".join(feature_names_for_hazard("extreme_heat")),
            "feature_names_dry_heat_agriculture": ",".join(feature_names_for_hazard("dry_heat_agriculture")),
            "feature_source_contract_version": "layer4_v2",
        }
        if self.flash_flood_model is not None:
            flood_features, flood_shape = feature_matrix_from_dataset(ds, hazard_type="flash_flood")
            if flood_shape != shape:
                raise ValueError(f"Flash-flood feature shape {flood_shape} does not match extreme-heat feature shape {shape}")
            flash_flood_prob = self._predict_probability(self.flash_flood_model, flood_features, shape)
            data_vars.update(
                {
                    "FlashFlood_Risk_Prob": (dims, np.asarray(flash_flood_prob, dtype=np.float32), {"units": "1"}),
                    "FlashFlood_Risk_Level": (dims, _levels_from_probability(flash_flood_prob), {"units": "class"}),
                }
            )
            attrs["feature_names_flash_flood"] = ",".join(feature_names_for_hazard("flash_flood"))

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
