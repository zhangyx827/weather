"""Standard data I/O and validation helpers.

The functions in this module are intentionally small contracts around common
weather feature formats. They keep examples, providers, workflows, and API
handlers from baking in separate ad-hoc JSON/xarray handling.
"""

from __future__ import annotations

import json
import math
from dataclasses import is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mazu_saudi.schemas import ForecastField, GridCell, MeteorologicalFeatures

SAUDI_BBOX = (16.0, 34.0, 32.0, 56.0)
STANDARD_GRID_RESOLUTION_DEG = 0.1

FEATURE_UNITS = {
    "temp_c": "degC",
    "rh_percent": "%",
    "dewpoint_c": "degC",
    "precip_1h_mm": "mm",
    "precip_6h_mm": "mm",
    "precip_24h_mm": "mm",
    "wind_speed_mps": "m/s",
    "wind_gust_mps": "m/s",
    "soil_moisture_frac": "1",
    "slope_deg": "degree",
    "impervious_frac": "1",
    "vegetation_index": "1",
    "pressure_hpa": "hPa",
    "visibility_km": "km",
    "coastal_distance_km": "km",
    "pwat_mm": "mm",
    "ivt_kg_m_s": "kg m-1 s-1",
    "cape_j_kg": "J/kg",
}


def read_json_features(path: str | Path) -> MeteorologicalFeatures | list[MeteorologicalFeatures]:
    """Read one feature record or a list of records from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [MeteorologicalFeatures.from_dict(item) for item in payload]
    if "features" in payload and isinstance(payload["features"], list):
        return [MeteorologicalFeatures.from_dict(item) for item in payload["features"]]
    return MeteorologicalFeatures.from_dict(payload)


def write_json_features(path: str | Path, features: MeteorologicalFeatures | list[MeteorologicalFeatures]) -> None:
    """Write one feature record or a list of records to JSON."""

    items: Any
    if isinstance(features, list):
        items = [item.to_dict() if hasattr(item, "to_dict") else item for item in features]
    else:
        items = features.to_dict() if hasattr(features, "to_dict") else features
    Path(path).write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def read_netcdf_dataset(path: str | Path) -> Any:
    """Read a NetCDF file as ``xarray.Dataset`` when xarray is installed."""

    try:
        import xarray as xr
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("NetCDF reading requires optional dependency: xarray") from exc
    return xr.open_dataset(path)


def write_netcdf_dataset(path: str | Path, dataset: Any) -> None:
    """Write a basic ``xarray.Dataset`` to NetCDF.

    Contract: ``dataset`` must expose ``to_netcdf(path)``. This keeps the
    adapter usable with xarray without importing xarray at module import time.
    """

    if not hasattr(dataset, "to_netcdf"):
        raise TypeError("NetCDF writer expects an xarray.Dataset-like object with to_netcdf(path)")
    dataset.to_netcdf(path)


def read_zarr_dataset(path: str | Path) -> Any:
    """Read a Zarr store as ``xarray.Dataset`` when xarray/zarr are installed."""

    try:
        import xarray as xr
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("Zarr reading requires optional dependency: xarray") from exc
    return xr.open_zarr(path)


def write_zarr_dataset(path: str | Path, dataset: Any, mode: str = "w") -> None:
    """Write a basic ``xarray.Dataset`` to Zarr using a unified signature."""

    if not hasattr(dataset, "to_zarr"):
        raise TypeError("Zarr writer expects an xarray.Dataset-like object with to_zarr(path, mode=...)")
    dataset.to_zarr(path, mode=mode)


def crop_to_bbox(obj: Any, bbox: tuple[float, float, float, float] = SAUDI_BBOX) -> Any:
    """Crop supported objects to ``(min_lat, min_lon, max_lat, max_lon)``.

    Supported inputs: ``GridCell`` lists, ``MeteorologicalFeatures`` lists,
    ``ForecastField``, and xarray Dataset/DataArray objects with lat/lon coords.
    """

    min_lat, min_lon, max_lat, max_lon = bbox
    if isinstance(obj, ForecastField):
        pairs = [(v, g) for v, g in zip(obj.values, obj.grid) if _inside(g.lat, g.lon, bbox)]
        if not pairs:
            return ForecastField(obj.provider, obj.variable, obj.units, obj.valid_time, [], [], dict(obj.metadata))
        values, grid = zip(*pairs)
        return ForecastField(obj.provider, obj.variable, obj.units, obj.valid_time, list(values), list(grid), dict(obj.metadata))
    if isinstance(obj, list):
        if all(isinstance(item, GridCell) for item in obj):
            return [item for item in obj if _inside(item.lat, item.lon, bbox)]
        if all(isinstance(item, MeteorologicalFeatures) for item in obj):
            return [item for item in obj if _inside(item.grid.lat, item.grid.lon, bbox)]
    if _is_xarray_like(obj):
        lat_name = "lat" if "lat" in obj.coords else "latitude"
        lon_name = "lon" if "lon" in obj.coords else "longitude"
        return obj.sel({lat_name: slice(min_lat, max_lat), lon_name: slice(min_lon, max_lon)})
    raise TypeError(f"Unsupported bbox crop object: {type(obj)!r}")


def crop_to_saudi(obj: Any) -> Any:
    """Crop supported objects to the Saudi operating domain."""

    return crop_to_bbox(obj, SAUDI_BBOX)


def generate_standard_grid(
    bbox: tuple[float, float, float, float] = SAUDI_BBOX,
    resolution_deg: float = STANDARD_GRID_RESOLUTION_DEG,
    prefix: str = "saudi",
) -> list[GridCell]:
    """Generate a 0.1 degree Saudi grid as point cells."""

    min_lat, min_lon, max_lat, max_lon = bbox
    lats = _inclusive_range(min_lat, max_lat, resolution_deg)
    lons = _inclusive_range(min_lon, max_lon, resolution_deg)
    return [
        GridCell(id=f"{prefix}_{lat:.1f}_{lon:.1f}", lat=lat, lon=lon)
        for lat in lats
        for lon in lons
    ]


def check_missing_values(features: MeteorologicalFeatures | list[MeteorologicalFeatures], required_fields: list[str] | None = None) -> dict[str, Any]:
    """Return missing-value diagnostics for required meteorological fields."""

    records = features if isinstance(features, list) else [features]
    fields = required_fields or ["temp_c", "rh_percent", "wind_speed_mps", "precip_1h_mm"]
    missing: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        for field in fields:
            if _is_missing(getattr(record, field, None)):
                missing.append({"index": index, "field": field, "grid_id": record.grid.id})
    return {"ok": not missing, "missing_count": len(missing), "missing": missing}


def validate_units(units: dict[str, str], expected_units: dict[str, str] | None = None) -> dict[str, Any]:
    """Validate unit metadata for feature or forecast variables."""

    expected = expected_units or FEATURE_UNITS
    mismatches = [
        {"field": field, "expected": expected_unit, "actual": units.get(field)}
        for field, expected_unit in expected.items()
        if field in units and units.get(field) != expected_unit
    ]
    missing = [field for field in expected if field not in units]
    return {"ok": not mismatches, "mismatches": mismatches, "missing_unit_fields": missing}


def validate_time_dimension(obj: Any, time_field: str = "valid_time") -> dict[str, Any]:
    """Validate that records or xarray-like objects expose a usable time dimension."""

    if isinstance(obj, MeteorologicalFeatures):
        return {"ok": isinstance(obj.valid_time, datetime), "time_field": time_field, "count": 1}
    if isinstance(obj, list) and all(isinstance(item, MeteorologicalFeatures) for item in obj):
        bad = [item.grid.id for item in obj if not isinstance(item.valid_time, datetime)]
        return {"ok": not bad, "time_field": time_field, "count": len(obj), "invalid_grid_ids": bad}
    if _is_xarray_like(obj):
        has_time = "time" in getattr(obj, "dims", {}) or "time" in getattr(obj, "coords", {})
        count = int(obj.sizes.get("time", 0)) if has_time and hasattr(obj, "sizes") else 0
        return {"ok": has_time, "time_field": "time", "count": count}
    return {"ok": False, "time_field": time_field, "error": f"Unsupported time object: {type(obj)!r}"}


def _inside(lat: float, lon: float, bbox: tuple[float, float, float, float]) -> bool:
    min_lat, min_lon, max_lat, max_lon = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def _inclusive_range(start: float, stop: float, step: float) -> list[float]:
    size = int(round((stop - start) / step))
    return [round(start + i * step, 10) for i in range(size + 1)]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(value))
    except TypeError:
        return False


def _is_xarray_like(obj: Any) -> bool:
    return hasattr(obj, "coords") and hasattr(obj, "sel") and not is_dataclass(obj)
