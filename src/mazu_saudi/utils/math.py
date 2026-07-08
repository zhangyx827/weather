"""Small numeric helpers with optional NumPy support."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def is_missing(value: Any) -> bool:
    """Return True for None or NaN-like scalar values."""

    if value is None:
        return True
    try:
        return bool(math.isnan(value))
    except TypeError:
        return False


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a numeric value."""

    if is_missing(value):
        return low
    return max(low, min(high, float(value)))


def map_values(func, *args):
    """Apply a scalar function to scalars, lists, tuples, NumPy arrays, or xarray DataArrays."""

    if any(_is_xarray_dataarray(arg) for arg in args):
        import xarray as xr

        return xr.apply_ufunc(func, *args, vectorize=True, output_dtypes=[float])
    if any(_is_numpy_array(arg) for arg in args):
        import numpy as np

        vectorized = np.vectorize(func, otypes=[float])
        return vectorized(*args)
    if any(isinstance(arg, (list, tuple)) for arg in args):
        size = max(len(arg) for arg in args if isinstance(arg, (list, tuple)))
        expanded = []
        for arg in args:
            expanded.append(arg if isinstance(arg, (list, tuple)) else [arg] * size)
        return [func(*items) for items in zip(*expanded)]
    return func(*args)


def _is_numpy_array(value: Any) -> bool:
    return value.__class__.__module__.startswith("numpy")


def _is_xarray_dataarray(value: Any) -> bool:
    return value.__class__.__module__.startswith("xarray") and value.__class__.__name__ == "DataArray"


def mean_present(values: Iterable[float | None], default: float = 0.0) -> float:
    """Mean over present scalar values."""

    present = [float(v) for v in values if not is_missing(v)]
    if not present:
        return default
    return sum(present) / len(present)
