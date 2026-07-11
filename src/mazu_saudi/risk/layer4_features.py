"""Shared Layer-4 feature preparation for training and inference."""

from __future__ import annotations

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

LAYER4_FEATURE_NAMES = (
    "temp_c",
    "vpd_kpa",
    "heat_index_c",
    "wind_speed_mps",
    "relative_humidity_percent",
)

_FRAME_ALIASES = {
    "temp_c": ("temp_c", "t2m_c"),
    "vpd_kpa": ("vpd_kpa",),
    "heat_index_c": ("heat_index_c",),
    "wind_speed_mps": ("wind_speed_mps", "wind10_speed"),
    "relative_humidity_percent": ("relative_humidity_percent", "rh2m"),
}


def _first_present(mapping: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(names[0])


def _to_numpy(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _sanitize_feature_array(values: Any, name: str) -> np.ndarray:
    arr = _to_numpy(values)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    if name == "relative_humidity_percent":
        arr = np.clip(arr, 0.0, 100.0)
    elif name == "vpd_kpa":
        arr = np.clip(arr, 0.0, 15.0)
    elif name in {"temp_c", "heat_index_c"}:
        arr = np.where((arr >= -40.0) & (arr <= 80.0), arr, np.nan)
    elif name == "wind_speed_mps":
        arr = np.where((arr >= 0.0) & (arr <= 80.0), arr, np.nan)
    return arr


def _dataset_feature_array(dataset: Any, target_name: str, aliases: tuple[str, ...]) -> np.ndarray:
    for name in aliases:
        if name in dataset.data_vars:
            return _sanitize_feature_array(dataset[name].values, target_name)
    if target_name == "temp_c" and "t2m" in dataset.data_vars:
        return _sanitize_feature_array(dataset["t2m"].values - 273.15, target_name)
    raise KeyError(aliases[0])


def prepare_feature_frame(table: Any) -> Any:
    """Return a table-like object with canonical Layer-4 feature names."""

    if pd is None:
        raise RuntimeError("pandas is required for Layer-4 table preparation")
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(table)!r}")

    data: dict[str, Any] = {}
    for target_name, aliases in _FRAME_ALIASES.items():
        series = _first_present(table, aliases)
        data[target_name] = _sanitize_feature_array(series, target_name)
    frame = pd.DataFrame(data)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        raise ValueError("Layer-4 feature table has no valid rows after sanitization")
    return frame


def feature_matrix_from_frame(table: Any) -> np.ndarray:
    frame = prepare_feature_frame(table)
    return frame.loc[:, list(LAYER4_FEATURE_NAMES)].to_numpy(dtype=np.float32)


def feature_matrix_from_dataset(dataset: Any) -> tuple[np.ndarray, tuple[int, ...]]:
    """Build canonical Layer-4 features from an xarray Dataset."""

    if xr is None:
        raise RuntimeError("xarray is required for Layer-4 dataset preparation")
    if not hasattr(dataset, "data_vars"):
        raise TypeError(f"Expected xarray.Dataset-like input, got {type(dataset)!r}")

    ds = dataset
    if "time" in ds.dims and ds.sizes.get("time", 0) == 1:
        ds = ds.isel(time=0, drop=True)

    fields: dict[str, np.ndarray] = {}
    shape: tuple[int, ...] | None = None
    for target_name, aliases in _FRAME_ALIASES.items():
        arr = _dataset_feature_array(ds, target_name, aliases)
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            raise ValueError(f"Layer-4 feature {target_name!r} has shape {arr.shape}, expected {shape}")
        fields[target_name] = arr

    assert shape is not None
    matrix = np.column_stack([fields[name].reshape(-1) for name in LAYER4_FEATURE_NAMES]).astype(np.float32)
    if not np.isfinite(matrix).any():
        raise ValueError("Layer-4 dataset has no valid cells after sanitization")
    return matrix, shape
