"""Standard data I/O and validation helpers.

The functions in this module are intentionally small contracts around common
weather feature formats. They keep examples, providers, workflows, and API
handlers from baking in separate ad-hoc JSON/xarray handling.
"""

from __future__ import annotations

import json
import math
import tempfile
import zipfile
from dataclasses import is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mazu_saudi.schemas import ForecastField, GridCell, MeteorologicalFeatures
from mazu_saudi.indicators import (
    compute_dewpoint_depression,
    compute_extreme_precip_flags,
    compute_heat_index_c,
    compute_ivt_placeholder,
    compute_pwat_placeholder,
    compute_relative_humidity_from_dewpoint,
    compute_vpd_kpa,
)

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

    path_obj = Path(path)
    engines = ("netcdf4", "h5netcdf", "scipy")
    if zipfile.is_zipfile(path_obj):
        return _read_zipped_netcdf_dataset(path_obj, engines)
    last_error: Exception | None = None

    for engine in engines:
        try:
            return xr.open_dataset(path_obj, engine=engine)
        except Exception as exc:  # pragma: no cover - backend availability varies by env
            last_error = exc

    raise RuntimeError(
        f"Unable to read NetCDF file {path_obj}; tried engines {engines}. "
        f"Last error: {last_error}"
    ) from last_error


def _read_zipped_netcdf_dataset(path_obj: Path, engines: tuple[str, ...]) -> Any:
    """Read a CDS ZIP response containing one or more NetCDF members."""

    try:
        import xarray as xr
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("NetCDF reading requires optional dependency: xarray") from exc

    last_error: Exception | None = None
    datasets = []

    with tempfile.TemporaryDirectory() as tmpdir:
        extracted_paths: list[Path] = []
        with zipfile.ZipFile(path_obj) as archive:
            member_names = [name for name in archive.namelist() if not name.endswith("/")]
            if not member_names:
                raise RuntimeError(f"ZIP archive {path_obj} does not contain any files")
            for member_name in member_names:
                target_path = Path(tmpdir) / Path(member_name).name
                target_path.write_bytes(archive.read(member_name))
                extracted_paths.append(target_path)

        for extracted_path in extracted_paths:
            for engine in engines:
                try:
                    dataset = xr.open_dataset(extracted_path, engine=engine)
                    datasets.append(dataset.load())
                    dataset.close()
                    break
                except Exception as exc:  # pragma: no cover - backend availability varies by env
                    last_error = exc
            else:
                continue

        if not datasets:
            raise RuntimeError(
                f"Unable to read NetCDF members from ZIP archive {path_obj}; tried engines {engines}. "
                f"Last error: {last_error}"
            ) from last_error

        if len(datasets) == 1:
            return datasets[0]
        return xr.merge(datasets, compat="override")


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
        lat_slice = _coord_slice(obj.coords[lat_name], min_lat, max_lat)
        lon_slice = _coord_slice(obj.coords[lon_name], min_lon, max_lon)
        return obj.sel({lat_name: lat_slice, lon_name: lon_slice})
    raise TypeError(f"Unsupported bbox crop object: {type(obj)!r}")


def crop_to_saudi(obj: Any) -> Any:
    """Crop supported objects to the Saudi operating domain."""

    return crop_to_bbox(obj, SAUDI_BBOX)


def compute_daily_precipitation_statistics(
    dataset: Any,
    precip_var: str = "precipitation",
    heavy_threshold_mm: float = 25.0,
    extreme_threshold_mm: float = 50.0,
    daily_freq: str = "1D",
) -> Any:
    """Aggregate sub-daily precipitation to daily totals and threshold flags.

    Contract: ``dataset`` is an xarray Dataset or DataArray with a ``time``
    coordinate. The precipitation variable is expected in millimetres per time
    step unless its units metadata is metres, in which case it is converted.
    """

    data_array = _xarray_dataarray(dataset, precip_var)
    if "time" not in getattr(data_array, "coords", {}):
        raise ValueError("Daily precipitation statistics require a time coordinate")

    precip_mm = _precip_to_mm(data_array)
    daily_total = precip_mm.resample(time=daily_freq).sum(skipna=True)
    daily_max = precip_mm.resample(time=daily_freq).max(skipna=True)
    heavy_flag = compute_extreme_precip_flags(
        daily_total,
        heavy_threshold_mm=heavy_threshold_mm,
        extreme_threshold_mm=extreme_threshold_mm,
    )

    return _xarray_dataset(
        {
            "daily_precip_total_mm": daily_total,
            "daily_precip_max_step_mm": daily_max,
            "daily_precip_extreme_flag": heavy_flag,
        }
    )


def derive_xarray_physical_indicators(
    dataset: Any,
    variable_map: dict[str, str] | None = None,
) -> Any:
    """Derive common physical indicators from an xarray Dataset.

    Available outputs depend on source variables:
    ``rh_percent``, ``dewpoint_depression_c``, ``vpd_kpa``, ``heat_index_c``,
    ``pwat_mm`` and ``ivt_kg_m_s``.
    """

    if not hasattr(dataset, "data_vars"):
        raise TypeError("Physical indicator derivation expects an xarray.Dataset-like object")

    names = {
        "temp_c": "temp_c",
        "dewpoint_c": "dewpoint_c",
        "rh_percent": "rh_percent",
        "pressure_hpa": "pressure_hpa",
        "wind_speed_mps": "wind_speed_mps",
        "u_wind_mps": "u_wind_mps",
        "v_wind_mps": "v_wind_mps",
    }
    if variable_map:
        names.update(variable_map)

    outputs: dict[str, Any] = {}
    temp = _optional_data_var(dataset, names["temp_c"])
    dewpoint = _optional_data_var(dataset, names["dewpoint_c"])
    rh = _optional_data_var(dataset, names["rh_percent"])
    pressure = _optional_data_var(dataset, names["pressure_hpa"])
    wind = _optional_data_var(dataset, names["wind_speed_mps"])
    u_wind = _optional_data_var(dataset, names["u_wind_mps"])
    v_wind = _optional_data_var(dataset, names["v_wind_mps"])

    if rh is None and temp is not None and dewpoint is not None:
        rh = compute_relative_humidity_from_dewpoint(temp, dewpoint)
        outputs["rh_percent"] = rh
    if temp is not None and dewpoint is not None:
        outputs["dewpoint_depression_c"] = compute_dewpoint_depression(temp, dewpoint)
    if temp is not None and rh is not None:
        outputs["vpd_kpa"] = compute_vpd_kpa(temp, rh)
        outputs["heat_index_c"] = compute_heat_index_c(temp, rh)
        outputs["pwat_mm"] = compute_pwat_placeholder(temp, rh, pressure)
    if wind is None and u_wind is not None and v_wind is not None:
        wind = (u_wind**2 + v_wind**2) ** 0.5
        outputs["wind_speed_mps"] = wind
    if wind is not None and "pwat_mm" in outputs:
        outputs["ivt_kg_m_s"] = compute_ivt_placeholder(wind, outputs["pwat_mm"])

    return _xarray_dataset(outputs)


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


def _coord_slice(coord: Any, start: float, stop: float) -> slice:
    values = getattr(coord, "values", coord)
    if len(values) == 0:
        return slice(start, stop)
    first = float(values[0])
    last = float(values[-1])
    if first > last:
        return slice(stop, start)
    return slice(start, stop)


def _xarray_dataarray(dataset: Any, variable: str) -> Any:
    if hasattr(dataset, "data_vars"):
        if variable not in dataset:
            raise KeyError(f"Variable {variable!r} not found in dataset")
        return dataset[variable]
    if dataset.__class__.__name__ == "DataArray":
        return dataset
    raise TypeError("Expected an xarray.Dataset or xarray.DataArray")


def _xarray_dataset(data_vars: dict[str, Any]) -> Any:
    try:
        import xarray as xr
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("xarray.Dataset creation requires optional dependency: xarray") from exc
    return xr.Dataset(data_vars)


def _optional_data_var(dataset: Any, name: str) -> Any | None:
    return dataset[name] if name in dataset else None


def _precip_to_mm(data_array: Any) -> Any:
    units = str(getattr(data_array, "attrs", {}).get("units", "")).lower()
    if units in {"m", "meter", "meters", "metre", "metres"}:
        converted = data_array * 1000.0
        converted.attrs["units"] = "mm"
        return converted
    return data_array
