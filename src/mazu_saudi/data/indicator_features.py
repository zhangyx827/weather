"""Processed indicator NetCDF readers."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mazu_saudi.data.io import read_netcdf_dataset
from mazu_saudi.schemas import GridCell, IndicatorFieldSet


def read_indicator_dataset(path: str | Path) -> Any:
    """Read a processed indicator NetCDF dataset."""

    return read_netcdf_dataset(path)


def indicator_point_from_dataset(
    dataset: Any,
    latitude: float,
    longitude: float,
    *,
    region: str | None = None,
    source: str | None = None,
) -> IndicatorFieldSet:
    """Select the nearest grid cell from a processed indicator dataset."""

    if "latitude" not in dataset.coords or "longitude" not in dataset.coords:
        raise ValueError("indicator dataset must expose latitude and longitude coordinates")

    point = dataset.sel(latitude=latitude, longitude=longitude, method="nearest")
    lat = float(point["latitude"].values)
    lon = float(point["longitude"].values)
    values: dict[str, float | int | None] = {}
    units: dict[str, str] = {}

    for name, da in point.data_vars.items():
        selected = da.isel(time=0, drop=True) if "time" in da.dims else da
        if selected.size != 1:
            continue
        try:
            numeric = float(selected.values.item())
        except (TypeError, ValueError):
            continue
        values[name] = None if not math.isfinite(numeric) else numeric
        unit = selected.attrs.get("units") or da.attrs.get("units")
        if unit:
            units[name] = str(unit)

    elevation = values.get("orography")
    grid = GridCell(
        id=f"saudi_{lat:.1f}_{lon:.1f}",
        lat=lat,
        lon=lon,
        elevation_m=None if elevation is None else float(elevation),
        region=region,
    )
    return IndicatorFieldSet(
        grid=grid,
        valid_time=_dataset_valid_time(dataset),
        values=values,
        units=units,
        source=source,
    )


def indicator_point_from_netcdf(
    path: str | Path,
    latitude: float,
    longitude: float,
    *,
    region: str | None = None,
) -> IndicatorFieldSet:
    """Read a NetCDF file and return one nearest-neighbor indicator point."""

    path_obj = Path(path)
    dataset = read_indicator_dataset(path_obj)
    try:
        return indicator_point_from_dataset(dataset, latitude, longitude, region=region, source=str(path_obj))
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()


def highest_indicator_point_from_dataset(
    dataset: Any,
    variable: str,
    *,
    region: str | None = None,
    source: str | None = None,
) -> IndicatorFieldSet:
    """Select the grid cell where an indicator variable is largest."""

    if variable not in dataset.data_vars:
        raise KeyError(variable)
    da = dataset[variable]
    if "time" in da.dims:
        da = da.isel(time=0, drop=True)
    idx = da.fillna(float("-inf")).argmax(...)
    lat = float(da["latitude"].isel(latitude=idx["latitude"]).values)
    lon = float(da["longitude"].isel(longitude=idx["longitude"]).values)
    return indicator_point_from_dataset(dataset, lat, lon, region=region, source=source)


def _dataset_valid_time(dataset: Any) -> datetime:
    if "time" not in dataset.coords or dataset.sizes.get("time", 0) == 0:
        return datetime.now(timezone.utc)
    value = dataset["time"].isel(time=0).values
    try:
        import pandas as pd

        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.to_pydatetime()
    except Exception:
        return datetime.now(timezone.utc)
