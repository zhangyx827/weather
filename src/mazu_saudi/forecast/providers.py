"""Forecast provider abstraction and mock implementations."""

from __future__ import annotations

import json
import math
import tempfile
import zipfile
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mazu_saudi.data import crop_to_saudi as data_crop_to_saudi
from mazu_saudi.data import read_netcdf_dataset, read_zarr_dataset
from mazu_saudi.schemas import ForecastField, GridCell

SAUDI_BBOX = (16.0, 34.0, 32.0, 56.0)

VARIABLE_ALIASES = {
    "2t": "temp_c",
    "t2m": "temp_c",
    "temperature": "temp_c",
    "relative_humidity": "rh_percent",
    "rh": "rh_percent",
    "10u": "wind_u_mps",
    "10v": "wind_v_mps",
    "wind": "wind_speed_mps",
    "u10": "wind_u_mps",
    "v10": "wind_v_mps",
    "d2m": "dewpoint_c",
    "sp": "pressure_hpa",
    "cape": "cape_j_kg",
    "tp": "precip_1h_mm",
    "precipitation": "precip_1h_mm",
    "precip": "precip_1h_mm",
    "precip_24h": "precip_24h_mm",
    "mswep_precipitation": "precip_24h_mm",
}

ERA5_VARIABLES = {
    "temp_c": ("data_stream-oper_stepType-instant.nc", "t2m", "degC"),
    "dewpoint_c": ("data_stream-oper_stepType-instant.nc", "d2m", "degC"),
    "wind_u_mps": ("data_stream-oper_stepType-instant.nc", "u10", "m/s"),
    "wind_v_mps": ("data_stream-oper_stepType-instant.nc", "v10", "m/s"),
    "pressure_hpa": ("data_stream-oper_stepType-instant.nc", "sp", "hPa"),
    "cape_j_kg": ("data_stream-oper_stepType-instant.nc", "cape", "J/kg"),
    "era5_precip_1h_mm": ("data_stream-oper_stepType-accum.nc", "tp", "mm"),
}


class BaseForecastProvider(ABC):
    """Unified forecast provider interface."""

    name = "base"
    dataset: Any = None

    @abstractmethod
    def fetch(self, variable: str, valid_time: datetime | None = None, bbox: tuple[float, float, float, float] = SAUDI_BBOX) -> ForecastField:
        """Fetch or create a standard forecast field."""

    def load_from_local(self, path: str | Path) -> "BaseForecastProvider":
        """Load a local JSON/NetCDF/Zarr forecast dataset into the provider."""

        path_obj = Path(path)
        if path_obj.suffix.lower() == ".json":
            self.dataset = json.loads(path_obj.read_text(encoding="utf-8"))
        elif path_obj.suffix.lower() in {".nc", ".netcdf"}:
            self.dataset = read_netcdf_dataset(path_obj)
        else:
            self.dataset = read_zarr_dataset(path_obj)
        return self

    def get_forecast(
        self,
        issue_time: datetime,
        lead_hours: int | list[int],
        bbox: tuple[float, float, float, float] = SAUDI_BBOX,
        variables: list[str] | None = None,
    ) -> dict[str, ForecastField]:
        """Return forecast fields keyed by ``variable:+lead_hour``."""

        leads = lead_hours if isinstance(lead_hours, list) else [lead_hours]
        selected = variables or ["temp_c", "rh_percent", "wind_speed_mps", "precip_1h_mm"]
        fields = {}
        for lead in leads:
            valid = issue_time + timedelta(hours=int(lead))
            for variable in selected:
                normalized = self.normalize_variables(variable)
                fields[f"{normalized}:+{int(lead)}h"] = self.fetch(normalized, valid_time=valid, bbox=bbox)
        return fields

    def normalize_variables(self, variable: str) -> str:
        """Normalize provider-specific variable names to MAZU names."""

        return VARIABLE_ALIASES.get(variable, variable)

    def crop_to_saudi(self, field: ForecastField) -> ForecastField:
        """Crop a forecast field to the Saudi operating domain."""

        return data_crop_to_saudi(field)

    def resample_to_grid(self, field: ForecastField, grid: list[GridCell] | None = None) -> ForecastField:
        """Placeholder for future conservative/bilinear grid remapping."""

        metadata = dict(field.metadata)
        metadata.update({"resample_status": "placeholder", "target_grid_size": len(grid or field.grid)})
        return ForecastField(field.provider, field.variable, field.units, field.valid_time, list(field.values), list(field.grid), metadata)

    def crop_to_bbox(self, field: ForecastField, bbox: tuple[float, float, float, float] = SAUDI_BBOX) -> ForecastField:
        """Return points inside ``(min_lat, min_lon, max_lat, max_lon)``."""

        min_lat, min_lon, max_lat, max_lon = bbox
        pairs = [(v, g) for v, g in zip(field.values, field.grid) if min_lat <= g.lat <= max_lat and min_lon <= g.lon <= max_lon]
        if not pairs:
            return ForecastField(field.provider, field.variable, field.units, field.valid_time, [], [], dict(field.metadata))
        values, grid = zip(*pairs)
        return ForecastField(field.provider, field.variable, field.units, field.valid_time, list(values), list(grid), dict(field.metadata))


class MockForecastProvider(BaseForecastProvider):
    """Deterministic mock forecast provider for demos and tests."""

    name = "mock"

    def fetch(self, variable: str, valid_time: datetime | None = None, bbox: tuple[float, float, float, float] = SAUDI_BBOX) -> ForecastField:
        valid = valid_time or datetime.now(timezone.utc)
        variable = self.normalize_variables(variable)
        grid = [
            GridCell(id="riyadh", lat=24.7136, lon=46.6753, elevation_m=612, region="Riyadh"),
            GridCell(id="jeddah", lat=21.4858, lon=39.1925, elevation_m=12, region="Makkah"),
            GridCell(id="dammam", lat=26.4207, lon=50.0888, elevation_m=10, region="Eastern Province"),
        ]
        defaults = {
            "temp_c": [45.0, 38.0, 41.0],
            "rh_percent": [18.0, 72.0, 64.0],
            "wind_speed_mps": [8.0, 10.0, 14.0],
            "precip_1h_mm": [2.0, 12.0, 5.0],
        }
        values = defaults.get(variable, [0.0 for _ in grid])
        units = {"temp_c": "degC", "rh_percent": "%", "wind_speed_mps": "m/s", "precip_1h_mm": "mm"}.get(variable, "unknown")
        field = ForecastField(
            self.name,
            variable,
            units,
            valid,
            values,
            grid,
            {
                "source": "deterministic_mock",
                "structure": {"dims": ["time", "lat", "lon"], "time": [valid.isoformat()], "lat": [g.lat for g in grid], "lon": [g.lon for g in grid]},
            },
        )
        return self.crop_to_bbox(field, bbox)


class JSONForecastProvider(BaseForecastProvider):
    """Read a local JSON forecast sample and return a standard field."""

    name = "json"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def fetch(self, variable: str, valid_time: datetime | None = None, bbox: tuple[float, float, float, float] = SAUDI_BBOX) -> ForecastField:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        valid = valid_time or datetime.fromisoformat(payload.get("valid_time", datetime.now(timezone.utc).isoformat()))
        grid = [GridCell(**item) for item in payload["grid"]]
        normalized = self.normalize_variables(variable)
        values = payload["variables"][normalized]["values"]
        units = payload["variables"][normalized].get("units", "unknown")
        metadata = {"source_path": str(self.path), "structure": payload.get("structure", {"dims": ["time", "lat", "lon"]})}
        field = ForecastField(self.name, normalized, units, valid, values, grid, metadata)
        return self.crop_to_bbox(field, bbox)


class ERA5MSWEPForecastProvider(BaseForecastProvider):
    """Read real ERA5 single-level fields and MSWEP precipitation files.

    This provider uses ERA5 for atmospheric state variables and MSWEP for
    precipitation. It returns historical/reanalysis fields through the forecast
    provider contract so the rest of the pipeline can run against real gridded
    inputs.
    """

    name = "era5_mswep"

    def __init__(self, era5_dir: str | Path = "era5_single_levels_2025", precip_dir: str | Path = "precip"):
        self.era5_dir = Path(era5_dir)
        self.precip_dir = Path(precip_dir)
        self._zip_cache: dict[tuple[Path, str], tuple[Any, tempfile.TemporaryDirectory[str]]] = {}

    def close(self) -> None:
        """Close cached ERA5 datasets and remove extracted zip members."""

        for dataset, temp_dir in self._zip_cache.values():
            dataset.close()
            temp_dir.cleanup()
        self._zip_cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def fetch(self, variable: str, valid_time: datetime | None = None, bbox: tuple[float, float, float, float] = SAUDI_BBOX) -> ForecastField:
        valid = _utc_naive(valid_time or datetime.now(timezone.utc))
        normalized = self.normalize_variables(variable)
        if normalized in {"precip_1h_mm", "precip_24h_mm"}:
            return self._fetch_mswep_precip(normalized, valid, bbox)
        if normalized == "rh_percent":
            return self._fetch_era5_relative_humidity(valid, bbox)
        if normalized == "wind_speed_mps":
            return self._fetch_era5_wind_speed(valid, bbox)
        if normalized in ERA5_VARIABLES:
            return self._fetch_era5_variable(normalized, valid, bbox)
        raise ValueError(f"Unsupported real forecast variable: {variable!r} normalized to {normalized!r}")

    def _fetch_era5_variable(self, variable: str, valid: datetime, bbox: tuple[float, float, float, float]) -> ForecastField:
        member_name, source_var, units = ERA5_VARIABLES[variable]
        data_array = self._era5_data_array(valid, member_name, source_var)
        selected = _select_xarray_time(data_array, "valid_time", valid)
        converted = _convert_era5_units(selected, source_var)
        return _field_from_data_array(
            provider=self.name,
            variable=variable,
            units=units,
            valid_time=valid.replace(tzinfo=timezone.utc),
            data_array=converted,
            bbox=bbox,
            metadata={
                "source": "ERA5 single levels",
                "source_variable": source_var,
                "source_path": str(_era5_month_path(self.era5_dir, valid)),
                "zip_member": member_name,
            },
        )

    def _fetch_era5_relative_humidity(self, valid: datetime, bbox: tuple[float, float, float, float]) -> ForecastField:
        member_name = "data_stream-oper_stepType-instant.nc"
        temp_c = _select_xarray_time(self._era5_data_array(valid, member_name, "t2m"), "valid_time", valid) - 273.15
        dewpoint_c = _select_xarray_time(self._era5_data_array(valid, member_name, "d2m"), "valid_time", valid) - 273.15
        saturation = np.exp((17.625 * temp_c) / (243.04 + temp_c))
        vapor = np.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))
        rh = (100.0 * vapor / saturation).clip(min=0.0, max=100.0)
        return _field_from_data_array(
            provider=self.name,
            variable="rh_percent",
            units="%",
            valid_time=valid.replace(tzinfo=timezone.utc),
            data_array=rh,
            bbox=bbox,
            metadata={
                "source": "ERA5 single levels",
                "source_variable": "relative humidity derived from t2m and d2m",
                "source_path": str(_era5_month_path(self.era5_dir, valid)),
                "zip_member": member_name,
            },
        )

    def _fetch_era5_wind_speed(self, valid: datetime, bbox: tuple[float, float, float, float]) -> ForecastField:
        member_name = "data_stream-oper_stepType-instant.nc"
        u = _select_xarray_time(self._era5_data_array(valid, member_name, "u10"), "valid_time", valid)
        v = _select_xarray_time(self._era5_data_array(valid, member_name, "v10"), "valid_time", valid)
        speed = (u**2 + v**2) ** 0.5
        return _field_from_data_array(
            provider=self.name,
            variable="wind_speed_mps",
            units="m/s",
            valid_time=valid.replace(tzinfo=timezone.utc),
            data_array=speed,
            bbox=bbox,
            metadata={
                "source": "ERA5 single levels",
                "source_variable": "sqrt(u10^2 + v10^2)",
                "source_path": str(_era5_month_path(self.era5_dir, valid)),
                "zip_member": member_name,
            },
        )

    def _fetch_mswep_precip(self, variable: str, valid: datetime, bbox: tuple[float, float, float, float]) -> ForecastField:
        dataset = read_netcdf_dataset(_mswep_day_path(self.precip_dir, valid))
        try:
            data_array = _select_xarray_time(dataset["precipitation"], "time", valid)
            if variable == "precip_1h_mm":
                data_array = data_array / 24.0
                derivation = "MSWEP daily mm/d divided by 24"
            else:
                derivation = "MSWEP daily mm/d as 24h total"
            return _field_from_data_array(
                provider=self.name,
                variable=variable,
                units="mm",
                valid_time=valid.replace(tzinfo=timezone.utc),
                data_array=data_array,
                bbox=bbox,
                metadata={
                    "source": "MSWEP",
                    "source_variable": "precipitation",
                    "source_units": dataset["precipitation"].attrs.get("units", "mm/d"),
                    "source_path": str(_mswep_day_path(self.precip_dir, valid)),
                    "derivation": derivation,
                },
            )
        finally:
            dataset.close()

    def _era5_data_array(self, valid: datetime, member_name: str, source_var: str) -> Any:
        month_path = _era5_month_path(self.era5_dir, valid)
        dataset = self._open_era5_zip_member(month_path, member_name)
        return dataset[source_var]

    def _open_era5_zip_member(self, month_path: Path, member_name: str) -> Any:
        cache_key = (month_path, member_name)
        cached = self._zip_cache.get(cache_key)
        if cached is not None:
            return cached[0]
        if not month_path.exists():
            raise FileNotFoundError(f"ERA5 month file not found: {month_path}")
        temp_dir = tempfile.TemporaryDirectory()
        extracted_path = Path(temp_dir.name) / member_name
        with zipfile.ZipFile(month_path) as archive:
            extracted_path.write_bytes(archive.read(member_name))
        dataset = read_netcdf_dataset(extracted_path)
        self._zip_cache[cache_key] = (dataset, temp_dir)
        return dataset


class AuroraForecastProvider(MockForecastProvider):
    """Aurora integration placeholder with the standard provider interface."""

    name = "aurora_placeholder"


class GenCastForecastProvider(MockForecastProvider):
    """GenCast integration placeholder with the standard provider interface."""

    name = "gencast_placeholder"

    def fetch(self, variable: str, valid_time: datetime | None = None, bbox: tuple[float, float, float, float] = SAUDI_BBOX) -> ForecastField:
        field = super().fetch(variable, valid_time, bbox)
        member_values = []
        for offset in (-0.15, 0.0, 0.12, 0.25):
            member_values.append([float(value) * (1.0 + offset) for value in field.values])
        metadata = dict(field.metadata)
        metadata.update({"member_count": len(member_values), "member_values": member_values, "structure": {"dims": ["member", "time", "lat", "lon"]}})
        return ForecastField(field.provider, field.variable, field.units, field.valid_time, field.values, field.grid, metadata)

    def member_count(self, field: ForecastField) -> int:
        """Return ensemble member count from metadata."""

        return int(field.metadata.get("member_count", len(field.metadata.get("member_values", [])) or 1))

    def ensemble_mean(self, field: ForecastField) -> list[float]:
        """Compute pointwise ensemble mean."""

        members = field.metadata.get("member_values") or [field.values]
        return [sum(values) / len(values) for values in zip(*members)]

    def ensemble_spread(self, field: ForecastField) -> list[float]:
        """Compute pointwise ensemble standard deviation."""

        means = self.ensemble_mean(field)
        members = field.metadata.get("member_values") or [field.values]
        spread = []
        for index, mean in enumerate(means):
            variance = sum((member[index] - mean) ** 2 for member in members) / len(members)
            spread.append(variance ** 0.5)
        return spread

    def exceedance_probability(self, field: ForecastField, threshold: float, variable: str | None = None) -> list[float]:
        """Compute pointwise probability that ensemble members exceed threshold."""

        if variable is not None and self.normalize_variables(variable) != field.variable:
            raise ValueError(f"field variable {field.variable!r} does not match requested {variable!r}")
        members = field.metadata.get("member_values") or [field.values]
        return [sum(1 for member in members if member[index] >= threshold) / len(members) for index in range(len(field.values))]


class AIFSBenchmarkProvider(MockForecastProvider):
    """AIFS benchmark integration placeholder with the standard provider interface."""

    name = "aifs_benchmark_placeholder"


def _utc_naive(value: datetime) -> datetime:
    """Return a timezone-naive UTC datetime for xarray/NumPy selection."""

    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _era5_month_path(era5_dir: Path, valid: datetime) -> Path:
    return era5_dir / f"era5_single_levels_{valid.year}_{valid.month:02d}.nc"


def _mswep_day_path(precip_dir: Path, valid: datetime) -> Path:
    day_of_year = valid.timetuple().tm_yday
    path = precip_dir / f"{valid.year}{day_of_year:03d}.nc"
    if not path.exists():
        raise FileNotFoundError(f"MSWEP daily precipitation file not found: {path}")
    return path


def _select_xarray_time(data_array: Any, coord_name: str, valid: datetime) -> Any:
    import pandas as pd

    target = np.datetime64(valid)
    times = data_array[coord_name].values
    if len(times) == 1:
        selected = data_array.isel({coord_name: 0})
    else:
        selected = data_array.sel({coord_name: target}, method="nearest")

    selected_value = pd.Timestamp(selected[coord_name].values).to_pydatetime().replace(tzinfo=None)
    max_delta = timedelta(hours=12) if coord_name == "valid_time" else timedelta(days=1)
    if abs(selected_value - valid) > max_delta:
        raise ValueError(f"Nearest {coord_name} {selected_value.isoformat()} is too far from requested {valid.isoformat()}")
    return selected


def _convert_era5_units(data_array: Any, source_var: str) -> Any:
    if source_var in {"t2m", "d2m", "mx2t", "mn2t"}:
        return data_array - 273.15
    if source_var == "sp":
        return data_array / 100.0
    if source_var == "tp":
        return data_array * 1000.0
    return data_array


def _field_from_data_array(
    provider: str,
    variable: str,
    units: str,
    valid_time: datetime,
    data_array: Any,
    bbox: tuple[float, float, float, float],
    metadata: dict[str, Any],
) -> ForecastField:
    cropped = _crop_xarray_to_bbox(data_array, bbox)
    loaded = cropped.load()
    lat_name = "lat" if "lat" in loaded.coords else "latitude"
    lon_name = "lon" if "lon" in loaded.coords else "longitude"
    lat_values = [float(value) for value in loaded[lat_name].values]
    lon_values = [float(value) for value in loaded[lon_name].values]
    values_array = np.asarray(loaded.values, dtype=float)
    values: list[float] = []
    grid: list[GridCell] = []
    for lat_index, lat in enumerate(lat_values):
        for lon_index, lon in enumerate(lon_values):
            value = float(values_array[lat_index, lon_index])
            if math.isnan(value):
                continue
            values.append(value)
            grid.append(GridCell(id=f"{variable}_{lat:.3f}_{lon:.3f}", lat=lat, lon=lon))
    structure = {
        "dims": [lat_name, lon_name],
        lat_name: lat_values,
        lon_name: lon_values,
        "point_count": len(values),
    }
    field_metadata = dict(metadata)
    field_metadata["structure"] = structure
    return ForecastField(provider, variable, units, valid_time, values, grid, field_metadata)


def _crop_xarray_to_bbox(data_array: Any, bbox: tuple[float, float, float, float]) -> Any:
    min_lat, min_lon, max_lat, max_lon = bbox
    lat_name = "lat" if "lat" in data_array.coords else "latitude"
    lon_name = "lon" if "lon" in data_array.coords else "longitude"
    lat_coord = data_array[lat_name]
    lon_coord = data_array[lon_name]
    lat_slice = slice(max_lat, min_lat) if float(lat_coord[0]) > float(lat_coord[-1]) else slice(min_lat, max_lat)
    lon_slice = slice(max_lon, min_lon) if float(lon_coord[0]) > float(lon_coord[-1]) else slice(min_lon, max_lon)
    return data_array.sel({lat_name: lat_slice, lon_name: lon_slice})


def load_netcdf_or_zarr_placeholder(path: str | Path, variable: str) -> dict[str, Any]:
    """Placeholder for future xarray-backed NetCDF/Zarr loading."""

    path_obj = Path(path)
    try:
        dataset = read_netcdf_dataset(path_obj) if path_obj.suffix.lower() in {".nc", ".netcdf"} else read_zarr_dataset(path_obj)
    except Exception as exc:
        return {"path": str(path), "variable": variable, "status": "xarray_loader_unavailable", "error": str(exc)}
    return {"path": str(path), "variable": variable, "status": "loaded", "dims": dict(getattr(dataset, "sizes", {}))}
