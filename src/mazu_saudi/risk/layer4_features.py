"""Shared Layer-4 feature preparation for training and inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

try:
    import xarray as xr
except Exception:  # pragma: no cover - optional dependency
    xr = None

DEFAULT_HAZARD_TYPE = "extreme_heat"

@dataclass(frozen=True)
class HazardFeatureSchema:
    required_core_features: tuple[str, ...]
    optional_enhancement_features: tuple[str, ...] = ()
    evidence_only_features: tuple[str, ...] = ()

    @property
    def model_feature_names(self) -> tuple[str, ...]:
        return self.required_core_features + self.optional_enhancement_features

    @property
    def all_feature_names(self) -> tuple[str, ...]:
        return self.model_feature_names + self.evidence_only_features


LAYER4_FEATURE_SCHEMAS: dict[str, HazardFeatureSchema] = {
    "extreme_heat": HazardFeatureSchema(
        required_core_features=(
            "temp_c",
            "tmax_c",
            "tmin_c",
            "heat_index_c",
            "vpd_kpa",
            "wind_speed_mps",
            "relative_humidity_percent",
        ),
        optional_enhancement_features=("sst_celsius",),
        evidence_only_features=(
            "t2m_anomaly_c",
            "tmax_anomaly_c",
            "heatwave_day_flag",
            "heatwave_duration_days",
        ),
    ),
    "dry_heat_agriculture": HazardFeatureSchema(
        required_core_features=(
            "temp_c",
            "tmax_c",
            "heat_index_c",
            "vpd_kpa",
            "wind_speed_mps",
            "relative_humidity_percent",
        ),
        evidence_only_features=(
            "t2m_anomaly_c",
            "heatwave_day_flag",
            "heatwave_duration_days",
        ),
    ),
    "flash_flood": HazardFeatureSchema(
        required_core_features=(
            "daily_precip_total",
            "daily_convective_precip",
            "daily_large_scale_precip",
            "cape",
            "pwat",
            "ivt",
            "wind850_speed",
            "wind_shear_850_200",
            "flash_flood_risk",
        ),
        evidence_only_features=("daily_precip_anomaly",),
    ),
}

LAYER4_FEATURE_NAMES = LAYER4_FEATURE_SCHEMAS[DEFAULT_HAZARD_TYPE].model_feature_names

_FRAME_ALIASES: dict[str, tuple[str, ...]] = {
    "temp_c": ("temp_c", "t2m_c"),
    "tmax_c": ("tmax_c",),
    "tmin_c": ("tmin_c",),
    "vpd_kpa": ("vpd_kpa",),
    "heat_index_c": ("heat_index_c",),
    "wind_speed_mps": ("wind_speed_mps", "wind10_speed"),
    "relative_humidity_percent": ("relative_humidity_percent", "rh2m", "rh_percent"),
    "sst_celsius": ("sst_celsius",),
    "daily_precip_total": ("daily_precip_total", "gpm_daily_precip"),
    "daily_convective_precip": ("daily_convective_precip",),
    "daily_large_scale_precip": ("daily_large_scale_precip",),
    "cape": ("cape",),
    "pwat": ("pwat", "pwat_mm"),
    "ivt": ("ivt", "ivt_kg_m_s"),
    "wind850_speed": ("wind850_speed",),
    "wind_shear_850_200": ("wind_shear_850_200",),
    "flash_flood_risk": ("flash_flood_risk",),
    "daily_precip_anomaly": ("daily_precip_anomaly",),
    "t2m_anomaly_c": ("t2m_anomaly_c",),
    "tmax_anomaly_c": ("tmax_anomaly_c",),
    "heatwave_day_flag": ("heatwave_day_flag",),
    "heatwave_duration_days": ("heatwave_duration_days",),
}

def feature_schema_for_hazard(hazard_type: str) -> HazardFeatureSchema:
    normalized = hazard_type.strip().lower()
    if normalized not in LAYER4_FEATURE_SCHEMAS:
        raise ValueError(f"Unsupported Layer-4 hazard type: {hazard_type}")
    return LAYER4_FEATURE_SCHEMAS[normalized]


def feature_names_for_hazard(hazard_type: str) -> tuple[str, ...]:
    return feature_schema_for_hazard(hazard_type).model_feature_names


def all_feature_names_for_hazard(hazard_type: str) -> tuple[str, ...]:
    return feature_schema_for_hazard(hazard_type).all_feature_names


def required_feature_names_for_hazard(hazard_type: str) -> tuple[str, ...]:
    return feature_schema_for_hazard(hazard_type).required_core_features


def enhancement_feature_names_for_hazard(hazard_type: str) -> tuple[str, ...]:
    return feature_schema_for_hazard(hazard_type).optional_enhancement_features


def evidence_feature_names_for_hazard(hazard_type: str) -> tuple[str, ...]:
    return feature_schema_for_hazard(hazard_type).evidence_only_features


def optional_feature_names_for_hazard(hazard_type: str) -> tuple[str, ...]:
    return enhancement_feature_names_for_hazard(hazard_type)


def _to_numpy(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _sanitize_feature_array(values: Any, name: str) -> np.ndarray:
    arr = _to_numpy(values)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    if name in {"relative_humidity_percent"}:
        arr = np.clip(arr, 0.0, 100.0)
    elif name == "vpd_kpa":
        arr = np.clip(arr, 0.0, 15.0)
    elif name in {"temp_c", "tmax_c", "tmin_c", "heat_index_c", "sst_celsius"}:
        arr = np.where((arr >= -40.0) & (arr <= 80.0), arr, np.nan)
    elif name in {"wind_speed_mps", "wind850_speed", "wind_shear_850_200"}:
        arr = np.where((arr >= 0.0) & (arr <= 120.0), arr, np.nan)
    elif name in {"daily_precip_total", "daily_convective_precip", "daily_large_scale_precip", "daily_precip_anomaly"}:
        arr = np.where(np.abs(arr) <= 2000.0, arr, np.nan)
    elif name == "cape":
        arr = np.where((arr >= 0.0) & (arr <= 10000.0), arr, np.nan)
    elif name in {"pwat", "ivt"}:
        arr = np.where((arr >= 0.0) & (arr <= 2000.0), arr, np.nan)
    elif name == "flash_flood_risk":
        arr = np.where((arr >= 0.0) & (arr <= 10.0), arr, np.nan)
    elif name in {"heatwave_day_flag"}:
        arr = np.where((arr >= 0.0) & (arr <= 1.0), arr, np.nan)
    elif name in {"heatwave_duration_days"}:
        arr = np.where((arr >= 0.0) & (arr <= 365.0), arr, np.nan)
    return arr


def _first_present(mapping: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(names[0])


def _normalize_dataarray_shape(values: Any) -> Any:
    if xr is None or not hasattr(values, "dims"):
        return values
    da = values
    rename: dict[str, str] = {}
    if "lat" in da.dims and "latitude" not in da.dims:
        rename["lat"] = "latitude"
    if "lon" in da.dims and "longitude" not in da.dims:
        rename["lon"] = "longitude"
    if rename:
        da = da.rename(rename)
    for dim in list(da.dims):
        if dim in {"latitude", "longitude"}:
            continue
        da = da.isel({dim: 0}, drop=True)
    return da


def _dataset_feature_array(dataset: Any, target_name: str, aliases: tuple[str, ...]) -> np.ndarray:
    ds = dataset
    for name in aliases:
        if name in ds.data_vars:
            values = _normalize_dataarray_shape(ds[name])
            return _sanitize_feature_array(values.values if hasattr(values, "values") else values, target_name)
    if target_name == "temp_c" and "t2m" in ds.data_vars:
        values = _normalize_dataarray_shape(ds["t2m"])
        return _sanitize_feature_array(values.values - 273.15, target_name)
    raise KeyError(aliases[0])


def prepare_feature_frame(table: Any, hazard_type: str = DEFAULT_HAZARD_TYPE, *, include_evidence_only: bool = False) -> Any:
    """Return a table-like object with canonical Layer-4 feature names."""

    if pd is None:
        raise RuntimeError("pandas is required for Layer-4 table preparation")
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(table)!r}")

    columns = all_feature_names_for_hazard(hazard_type) if include_evidence_only else feature_names_for_hazard(hazard_type)
    fill_missing = set(optional_feature_names_for_hazard(hazard_type)) | set(evidence_feature_names_for_hazard(hazard_type))
    data: dict[str, Any] = {}
    for target_name in columns:
        aliases = _FRAME_ALIASES[target_name]
        try:
            series = _first_present(table, aliases)
        except KeyError:
            if target_name in fill_missing:
                data[target_name] = np.full(len(table), np.nan, dtype=np.float32)
                continue
            raise
        data[target_name] = _sanitize_feature_array(series, target_name)
    frame = pd.DataFrame(data)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    required = list(required_feature_names_for_hazard(hazard_type))
    frame = frame.dropna(subset=required)
    if frame.empty:
        raise ValueError("Layer-4 feature table has no valid rows after sanitization")
    return frame


def feature_matrix_from_frame(table: Any, hazard_type: str = DEFAULT_HAZARD_TYPE) -> np.ndarray:
    frame = prepare_feature_frame(table, hazard_type=hazard_type)
    columns = list(feature_names_for_hazard(hazard_type))
    return frame.loc[:, columns].to_numpy(dtype=np.float32)


def _normalize_dataset(dataset: Any) -> Any:
    if xr is None:
        return dataset
    if not hasattr(dataset, "data_vars"):
        raise TypeError(f"Expected xarray.Dataset-like input, got {type(dataset)!r}")
    ds = dataset
    rename: dict[str, str] = {}
    if "lat" in ds.dims and "latitude" not in ds.dims:
        rename["lat"] = "latitude"
    if "lon" in ds.dims and "longitude" not in ds.dims:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)
    return ds


def _dataset_feature_fields(
    dataset: Any,
    hazard_type: str,
    *,
    include_evidence_only: bool = False,
) -> tuple[dict[str, np.ndarray], tuple[int, ...], tuple[str, ...]]:
    ds = _normalize_dataset(dataset)
    fields: dict[str, np.ndarray] = {}
    shape: tuple[int, ...] | None = None
    columns = all_feature_names_for_hazard(hazard_type) if include_evidence_only else feature_names_for_hazard(hazard_type)
    fill_missing = set(optional_feature_names_for_hazard(hazard_type)) | set(evidence_feature_names_for_hazard(hazard_type))
    for target_name in columns:
        aliases = _FRAME_ALIASES[target_name]
        try:
            arr = _dataset_feature_array(ds, target_name, aliases)
        except KeyError:
            if target_name in fill_missing:
                if shape is None:
                    continue
                arr = np.full(shape, np.nan, dtype=np.float32)
            else:
                raise
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            if target_name in fill_missing:
                arr = np.full(shape, np.nan, dtype=np.float32)
            else:
                raise ValueError(f"Layer-4 feature {target_name!r} has shape {arr.shape}, expected {shape}")
        fields[target_name] = arr

    if shape is None:
        raise ValueError("Layer-4 dataset has no usable feature fields")
    for target_name in columns:
        fields.setdefault(target_name, np.full(shape, np.nan, dtype=np.float32))
    return fields, shape, columns


def feature_frame_from_dataset(dataset: Any, hazard_type: str = DEFAULT_HAZARD_TYPE, *, include_evidence_only: bool = True) -> Any:
    """Build a flattened pandas DataFrame from an indicator Dataset."""

    if pd is None:
        raise RuntimeError("pandas is required for Layer-4 dataset frame preparation")
    if xr is None:
        raise RuntimeError("xarray is required for Layer-4 dataset frame preparation")

    ds = _normalize_dataset(dataset)
    fields, shape, columns = _dataset_feature_fields(ds, hazard_type, include_evidence_only=include_evidence_only)
    matrix = np.column_stack([fields[name].reshape(-1) for name in columns]).astype(np.float32)
    required = list(required_feature_names_for_hazard(hazard_type))
    required_indexes = [columns.index(name) for name in required]
    valid_mask = np.all(np.isfinite(matrix[:, required_indexes]), axis=1)
    if not valid_mask.any():
        raise ValueError("Layer-4 dataset has no valid cells after sanitization")

    frame_data: dict[str, Any] = {}
    if "latitude" in ds.coords and "longitude" in ds.coords and len(shape) == 2:
        lat_values = np.asarray(ds.coords["latitude"].values, dtype=np.float32)
        lon_values = np.asarray(ds.coords["longitude"].values, dtype=np.float32)
        lat_grid, lon_grid = np.meshgrid(lat_values, lon_values, indexing="ij")
        frame_data["latitude"] = lat_grid.reshape(-1)[valid_mask]
        frame_data["longitude"] = lon_grid.reshape(-1)[valid_mask]
    for name in columns:
        frame_data[name] = fields[name].reshape(-1)[valid_mask]
    return pd.DataFrame(frame_data)


def feature_matrix_from_dataset(dataset: Any, hazard_type: str = DEFAULT_HAZARD_TYPE) -> tuple[np.ndarray, tuple[int, ...]]:
    """Build canonical Layer-4 features from an xarray Dataset."""

    if xr is None:
        raise RuntimeError("xarray is required for Layer-4 dataset preparation")
    fields, shape, columns = _dataset_feature_fields(dataset, hazard_type)

    matrix = np.column_stack([fields[name].reshape(-1) for name in columns]).astype(np.float32)
    required = list(required_feature_names_for_hazard(hazard_type))
    required_indexes = [columns.index(name) for name in required]
    if not np.all(np.isfinite(matrix[:, required_indexes]), axis=1).any():
        raise ValueError("Layer-4 dataset has no valid cells after sanitization")
    return matrix, shape
