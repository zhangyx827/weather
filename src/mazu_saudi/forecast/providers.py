"""Forecast provider abstraction and mock implementations."""

from __future__ import annotations

import json
import math
import re
import tempfile
import zipfile
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mazu_saudi.data import crop_to_saudi as data_crop_to_saudi
from mazu_saudi.data import read_netcdf_dataset, read_zarr_dataset
from mazu_saudi.indicators import (
    compute_cape_placeholder,
    compute_flash_flood_screening_score,
    compute_heat_index_c,
    compute_ivt_placeholder,
    compute_pwat_placeholder,
    compute_vpd_kpa,
)
from mazu_saudi.schemas import ForecastField, GridCell
from mazu_saudi.utils.math import is_missing

try:
    import xarray as xr
except Exception:  # pragma: no cover - optional dependency
    xr = None

SAUDI_BBOX = (16.0, 34.0, 32.0, 56.0)
DEFAULT_HEAT_CLIMATOLOGY_DIR = Path(__file__).resolve().parents[3] / "data" / "processed" / "lightgbm_indicators_nc"

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
    "msl": "pressure_hpa",
    "pressure": "pressure_hpa",
    "tcwv": "total_column_water_mm",
    "pwat_mm": "pwat",
    "ivt_kg_m_s": "ivt",
}

ERA5_VARIABLES = {
    "temp_c": ("data_stream-oper_stepType-instant.nc", "t2m", "degC"),
    "tmax_c": ("data_stream-oper_stepType-max.nc", "mx2t", "degC"),
    "tmin_c": ("data_stream-oper_stepType-max.nc", "mn2t", "degC"),
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
    provider_role = "deterministic"
    provider_status = "ready"
    source_status = "normal"

    def __init__(self, heat_climatology_dir: str | Path | None = DEFAULT_HEAT_CLIMATOLOGY_DIR) -> None:
        self.heat_climatology_dir = Path(heat_climatology_dir) if heat_climatology_dir is not None else None

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
        selected = variables or [
            "temp_c",
            "tmax_c",
            "tmin_c",
            "rh_percent",
            "wind_speed_mps",
            "pressure_hpa",
            "cape_j_kg",
            "precip_1h_mm",
            "precip_24h_mm",
        ]
        fields = {}
        for lead in leads:
            valid = issue_time + timedelta(hours=int(lead))
            for variable in selected:
                normalized = self.normalize_variables(variable)
                fields[f"{normalized}:+{int(lead)}h"] = self.fetch(normalized, valid_time=valid, bbox=bbox)
        return fields

    def forecast_dataset(
        self,
        issue_time: datetime,
        lead_hours: int | list[int],
        bbox: tuple[float, float, float, float] = SAUDI_BBOX,
        variables: list[str] | None = None,
    ) -> Any:
        """Return a Layer-4-compatible xarray Dataset for one provider run."""

        if xr is None:
            raise RuntimeError("xarray is required for forecast dataset export")
        selected = list(variables) if variables is not None else None
        if selected is None:
            selected = [
                "temp_c",
                "tmax_c",
                "tmin_c",
                "rh_percent",
                "wind_speed_mps",
                "pressure_hpa",
                "cape_j_kg",
                "precip_1h_mm",
                "precip_24h_mm",
            ]
            for extra in self.runtime_dataset_variables():
                if extra not in selected:
                    selected.append(extra)
        fields = self.get_forecast(issue_time, lead_hours, bbox=bbox, variables=selected)
        return forecast_fields_to_dataset(fields, provider_metadata=self.build_runtime_metadata())

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
            return ForecastField(
                field.provider,
                field.variable,
                field.units,
                field.valid_time,
                [],
                [],
                dict(field.metadata),
                provider_role=field.provider_role,
                provider_status=field.provider_status,
                source_status=field.source_status,
                degradation_metadata=dict(field.degradation_metadata),
            )
        values, grid = zip(*pairs)
        return ForecastField(
            field.provider,
            field.variable,
            field.units,
            field.valid_time,
            list(values),
            list(grid),
            dict(field.metadata),
            provider_role=field.provider_role,
            provider_status=field.provider_status,
            source_status=field.source_status,
            degradation_metadata=dict(field.degradation_metadata),
        )

    def build_runtime_metadata(self) -> dict[str, Any]:
        return {
            "provider_role": self.provider_role,
            "provider_status": self.provider_status,
            "source_status": self.source_status,
            "heat_climatology": _heat_climatology_summary(self.heat_climatology_dir),
        }

    def runtime_dataset_variables(self) -> list[str]:
        """Optional extra variables to fetch for dataset export."""

        return []


class MockForecastProvider(BaseForecastProvider):
    """Deterministic mock forecast provider for demos and tests."""

    name = "mock"
    provider_role = "deterministic_fallback"
    provider_status = "degraded_mock"
    source_status = "degraded"

    def runtime_dataset_variables(self) -> list[str]:
        return ["pwat", "ivt", "wind850_speed", "wind_shear_850_200"]

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
            "tmax_c": [47.0, 40.0, 43.0],
            "tmin_c": [31.0, 29.0, 30.0],
            "rh_percent": [18.0, 72.0, 64.0],
            "wind_speed_mps": [8.0, 10.0, 14.0],
            "pressure_hpa": [1008.0, 1006.0, 1007.0],
            "cape_j_kg": [120.0, 900.0, 650.0],
            "precip_1h_mm": [2.0, 12.0, 5.0],
            "precip_24h_mm": [8.0, 48.0, 20.0],
            "pwat": [2.9, 4.9, 4.1],
            "ivt": [66.7, 122.0, 132.1],
            "wind850_speed": [9.2, 11.5, 16.1],
            "wind_shear_850_200": [4.4, 9.7, 11.2],
        }
        values = defaults.get(variable, [0.0 for _ in grid])
        units = {
            "temp_c": "degC",
            "tmax_c": "degC",
            "tmin_c": "degC",
            "rh_percent": "%",
            "wind_speed_mps": "m/s",
            "pressure_hpa": "hPa",
            "cape_j_kg": "J/kg",
            "precip_1h_mm": "mm",
            "precip_24h_mm": "mm",
            "pwat": "mm",
            "ivt": "kg m-1 s-1",
            "wind850_speed": "m/s",
            "wind_shear_850_200": "m/s",
        }.get(variable, "unknown")
        field = ForecastField(
            self.name,
            variable,
            units,
            valid,
            values,
            grid,
            {
                "source": "deterministic_mock",
                **self.build_runtime_metadata(),
                "structure": {"dims": ["time", "lat", "lon"], "time": [valid.isoformat()], "lat": [g.lat for g in grid], "lon": [g.lon for g in grid]},
            },
            provider_role=self.provider_role,
            provider_status=self.provider_status,
            source_status=self.source_status,
            degradation_metadata={"reason": "mock_provider_used", "provider": self.name},
        )
        return self.crop_to_bbox(field, bbox)


class JSONForecastProvider(BaseForecastProvider):
    """Read a local JSON forecast sample and return a standard field."""

    name = "json"

    def __init__(self, path: str | Path):
        super().__init__()
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
    provider_role = "reanalysis_background"
    provider_status = "ready"
    source_status = "normal"

    def __init__(
        self,
        era5_dir: str | Path = "era5_single_levels_2025",
        precip_dir: str | Path = "precip",
        pressure_dir: str | Path | None = None,
        heat_climatology_dir: str | Path | None = DEFAULT_HEAT_CLIMATOLOGY_DIR,
    ):
        super().__init__(heat_climatology_dir=heat_climatology_dir)
        self.era5_dir = Path(era5_dir)
        self.precip_dir = Path(precip_dir)
        self.pressure_dir = Path(pressure_dir) if pressure_dir is not None else None
        self._zip_cache: dict[tuple[Path, str], tuple[Any, tempfile.TemporaryDirectory[str]]] = {}
        self._direct_era5_cache: dict[Path, Any] = {}
        self._pressure_cache: dict[tuple[Path, str], Any] = {}

    def runtime_dataset_variables(self) -> list[str]:
        if self.pressure_dir is None:
            return []
        return ["pwat", "ivt", "wind850_speed", "wind_shear_850_200"]

    def forecast_dataset(
        self,
        issue_time: datetime,
        lead_hours: int | list[int],
        bbox: tuple[float, float, float, float] = SAUDI_BBOX,
        variables: list[str] | None = None,
    ) -> Any:
        selected = list(variables) if variables is not None else None
        if selected is None:
            selected = [
                "temp_c",
                "tmax_c",
                "tmin_c",
                "rh_percent",
                "wind_speed_mps",
                "pressure_hpa",
                "cape_j_kg",
                "precip_1h_mm",
                "precip_24h_mm",
            ]
            for extra in self.runtime_dataset_variables():
                if extra not in selected:
                    selected.append(extra)
            leads = lead_hours if isinstance(lead_hours, list) else [lead_hours]
            valid_times = [_utc_naive(issue_time + timedelta(hours=int(lead))) for lead in leads]
            if self._supports_direct_precip_partition(valid_times):
                selected.extend(["daily_convective_precip", "daily_large_scale_precip"])
        fields = self.get_forecast(issue_time, lead_hours, bbox=bbox, variables=selected)
        return forecast_fields_to_dataset(fields, provider_metadata=self.build_runtime_metadata())

    def close(self) -> None:
        """Close cached ERA5 datasets and remove extracted zip members."""

        for dataset, temp_dir in self._zip_cache.values():
            dataset.close()
            temp_dir.cleanup()
        self._zip_cache.clear()
        for dataset in self._direct_era5_cache.values():
            dataset.close()
        self._direct_era5_cache.clear()
        for dataset in self._pressure_cache.values():
            dataset.close()
        self._pressure_cache.clear()

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
        if normalized in {"daily_convective_precip", "daily_large_scale_precip"}:
            return self._fetch_era5_precip_partition(normalized, valid, bbox)
        if normalized == "pwat":
            return self._fetch_pressure_derived(valid, bbox, "pwat")
        if normalized == "ivt":
            return self._fetch_pressure_derived(valid, bbox, "ivt")
        if normalized == "wind850_speed":
            return self._fetch_pressure_derived(valid, bbox, "wind850_speed")
        if normalized == "wind_shear_850_200":
            return self._fetch_pressure_derived(valid, bbox, "wind_shear_850_200")
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

    def _fetch_era5_precip_partition(self, variable: str, valid: datetime, bbox: tuple[float, float, float, float]) -> ForecastField:
        member_name = "data_stream-oper_stepType-accum.nc"
        total = _select_xarray_time(self._era5_data_array(valid, member_name, "tp"), "valid_time", valid)
        if not self._era5_member_has_variable(valid, member_name, "cp"):
            raise ValueError(f"{self.name} does not have direct convective precipitation for {valid.date().isoformat()}")
        convective = _select_xarray_time(self._era5_data_array(valid, member_name, "cp"), "valid_time", valid)
        total_mm = _convert_era5_units(total, "tp")
        convective_mm = _convert_era5_units(convective, "cp").clip(min=0.0)
        if variable == "daily_convective_precip":
            data_array = convective_mm.rename("daily_convective_precip")
            source_variable = "cp"
            derivation = "ERA5 convective precipitation accumulated over step"
        else:
            data_array = (total_mm - convective_mm).clip(min=0.0).rename("daily_large_scale_precip")
            source_variable = "tp-cp"
            derivation = "ERA5 total precipitation minus direct convective precipitation"
        return _field_from_data_array(
            provider=self.name,
            variable=variable,
            units="mm",
            valid_time=valid.replace(tzinfo=timezone.utc),
            data_array=data_array,
            bbox=bbox,
            metadata={
                "source": "ERA5 single levels",
                "source_variable": source_variable,
                "source_path": str(_era5_month_path(self.era5_dir, valid)),
                "zip_member": member_name,
                "derivation": derivation,
            },
        )

    def _era5_data_array(self, valid: datetime, member_name: str, source_var: str) -> Any:
        month_path = _era5_month_path(self.era5_dir, valid)
        if month_path.exists() and zipfile.is_zipfile(month_path):
            dataset = self._open_era5_zip_member(month_path, member_name)
            return dataset[source_var]
        direct_dataset = self._open_era5_direct_dataset(month_path)
        if direct_dataset is not None and source_var in direct_dataset.data_vars:
            return direct_dataset[source_var]
        dataset = self._open_era5_zip_member(_era5_supplement_path(self.era5_dir, valid), member_name)
        return dataset[source_var]

    def _open_era5_direct_dataset(self, month_path: Path) -> Any | None:
        if not month_path.exists() or zipfile.is_zipfile(month_path):
            return None
        cached = self._direct_era5_cache.get(month_path)
        if cached is not None:
            return cached
        dataset = read_netcdf_dataset(month_path)
        self._direct_era5_cache[month_path] = dataset
        return dataset

    def _open_era5_zip_member(self, month_path: Path, member_name: str) -> Any:
        cache_key = (month_path, member_name)
        cached = self._zip_cache.get(cache_key)
        if cached is not None:
            return cached[0]
        if not month_path.exists():
            raise FileNotFoundError(f"ERA5 month file not found: {month_path}")
        if not zipfile.is_zipfile(month_path):
            raise FileNotFoundError(f"ERA5 archive member {member_name} not available in non-archive file: {month_path}")
        temp_dir = tempfile.TemporaryDirectory()
        extracted_path = Path(temp_dir.name) / member_name
        with zipfile.ZipFile(month_path) as archive:
            extracted_path.write_bytes(archive.read(member_name))
        dataset = read_netcdf_dataset(extracted_path)
        self._zip_cache[cache_key] = (dataset, temp_dir)
        return dataset

    def _era5_member_has_variable(self, valid: datetime, member_name: str, source_var: str) -> bool:
        month_path = _era5_month_path(self.era5_dir, valid)
        if month_path.exists() and zipfile.is_zipfile(month_path):
            try:
                dataset = self._open_era5_zip_member(month_path, member_name)
            except FileNotFoundError:
                return False
            return source_var in dataset.data_vars
        direct_dataset = self._open_era5_direct_dataset(month_path)
        if direct_dataset is not None and source_var in direct_dataset.data_vars:
            return True
        try:
            dataset = self._open_era5_zip_member(_era5_supplement_path(self.era5_dir, valid), member_name)
        except FileNotFoundError:
            return False
        return source_var in dataset.data_vars

    def _supports_direct_precip_partition(self, valid_times: list[datetime]) -> bool:
        member_name = "data_stream-oper_stepType-accum.nc"
        return bool(valid_times) and all(self._era5_member_has_variable(valid, member_name, "cp") for valid in valid_times)

    def _fetch_pressure_derived(self, valid: datetime, bbox: tuple[float, float, float, float], variable: str) -> ForecastField:
        if self.pressure_dir is None:
            raise ValueError(f"{self.name} does not have a pressure-level directory configured for {variable}")
        q = self._pressure_data_array(valid, "specific_humidity", "q")
        u = self._pressure_data_array(valid, "u_component_of_wind", "u")
        v = self._pressure_data_array(valid, "v_component_of_wind", "v")
        q = _select_xarray_time(q, "valid_time", valid)
        u = _select_xarray_time(u, "valid_time", valid)
        v = _select_xarray_time(v, "valid_time", valid)
        q = _normalize_pressure_levels(q)
        u = _normalize_pressure_levels(u)
        v = _normalize_pressure_levels(v)
        if variable == "pwat":
            data_array = _pressure_weighted_water(q)
            units = "mm"
            source_variable = "specific humidity profile"
        elif variable == "ivt":
            data_array = _integrated_vapor_transport(q, u, v)
            units = "kg m-1 s-1"
            source_variable = "specific humidity + wind profile"
        elif variable == "wind850_speed":
            data_array = np.hypot(_select_pressure_level(u, 850), _select_pressure_level(v, 850)).rename("wind850_speed")
            units = "m/s"
            source_variable = "u850/v850"
        elif variable == "wind_shear_850_200":
            data_array = np.hypot(
                _select_pressure_level(u, 200) - _select_pressure_level(u, 850),
                _select_pressure_level(v, 200) - _select_pressure_level(v, 850),
            ).rename("wind_shear_850_200")
            units = "m/s"
            source_variable = "u850/v850/u200/v200"
        else:
            raise ValueError(f"Unsupported pressure-derived variable: {variable}")
        return _field_from_data_array(
            provider=self.name,
            variable=variable,
            units=units,
            valid_time=valid.replace(tzinfo=timezone.utc),
            data_array=data_array,
            bbox=bbox,
            metadata={
                "source": "ERA5 pressure levels",
                "source_variable": source_variable,
                "source_path": str(self._pressure_month_path(valid, "specific_humidity")),
            },
        )

    def _pressure_month_path(self, valid: datetime, family: str) -> Path:
        assert self.pressure_dir is not None
        suffix = {
            "specific_humidity": "specific_humidity",
            "u_component_of_wind": "u_component_of_wind",
            "v_component_of_wind": "v_component_of_wind",
        }[family]
        path = self.pressure_dir / f"era5_pl_{valid.year}_{valid.month:02d}_{suffix}.nc"
        if not path.exists():
            raise FileNotFoundError(f"ERA5 pressure-level file not found: {path}")
        return path

    def _pressure_data_array(self, valid: datetime, family: str, var_name: str) -> Any:
        path = self._pressure_month_path(valid, family)
        cache_key = (path, var_name)
        cached = self._pressure_cache.get(cache_key)
        if cached is not None:
            return cached[var_name]
        dataset = read_netcdf_dataset(path)
        self._pressure_cache[cache_key] = dataset
        return dataset[var_name]


class AuroraForecastProvider(MockForecastProvider):
    """Aurora integration placeholder with the standard provider interface."""

    name = "aurora"
    provider_role = "primary_deterministic"

    def __init__(self, status: str = "degraded_mock") -> None:
        super().__init__()
        self.provider_status = status
        self.source_status = "normal" if status == "ready" else "degraded"


class GenCastForecastProvider(MockForecastProvider):
    """GenCast integration placeholder with the standard provider interface."""

    name = "gencast"
    provider_role = "secondary_ensemble"

    def fetch(self, variable: str, valid_time: datetime | None = None, bbox: tuple[float, float, float, float] = SAUDI_BBOX) -> ForecastField:
        field = super().fetch(variable, valid_time, bbox)
        member_values = []
        for offset in (-0.15, 0.0, 0.12, 0.25):
            member_values.append([float(value) * (1.0 + offset) for value in field.values])
        metadata = dict(field.metadata)
        metadata.update(
            {
                "provider_role": self.provider_role,
                "provider_status": self.provider_status,
                "member_count": len(member_values),
                "member_values": member_values,
                "ensemble_stats": {
                    "member_count": len(member_values),
                    "mean": [sum(values) / len(values) for values in zip(*member_values)],
                },
                "structure": {"dims": ["member", "time", "lat", "lon"]},
            }
        )
        return ForecastField(
            field.provider,
            field.variable,
            field.units,
            field.valid_time,
            field.values,
            field.grid,
            metadata,
            provider_role=self.provider_role,
            provider_status=self.provider_status,
            source_status=self.source_status,
            degradation_metadata={"reason": "ensemble_placeholder_used", "provider": self.name},
        )

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

    name = "aifs"
    provider_role = "benchmark"

    def fetch(self, variable: str, valid_time: datetime | None = None, bbox: tuple[float, float, float, float] = SAUDI_BBOX) -> ForecastField:
        field = super().fetch(variable, valid_time, bbox)
        metadata = dict(field.metadata)
        metadata.update(
            {
                "provider_role": self.provider_role,
                "provider_status": self.provider_status,
                "benchmark_comparison": {"status": "placeholder", "benchmark_name": "aifs"},
            }
        )
        return ForecastField(
            field.provider,
            field.variable,
            field.units,
            field.valid_time,
            field.values,
            field.grid,
            metadata,
            provider_role=self.provider_role,
            provider_status=self.provider_status,
            source_status=self.source_status,
            degradation_metadata={"reason": "benchmark_placeholder_used", "provider": self.name},
        )


def _utc_naive(value: datetime) -> datetime:
    """Return a timezone-naive UTC datetime for xarray/NumPy selection."""

    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def forecast_fields_to_dataset(fields: dict[str, ForecastField], provider_metadata: dict[str, Any] | None = None) -> Any:
    """Convert standard forecast fields to a Layer-4-compatible Dataset."""

    if xr is None:
        raise RuntimeError("xarray is required for forecast dataset export")
    if not fields:
        raise ValueError("at least one ForecastField is required")

    grouped: dict[datetime, dict[str, ForecastField]] = {}
    latitudes: set[float] = set()
    longitudes: set[float] = set()
    for field in fields.values():
        grouped.setdefault(field.valid_time, {})[field.variable] = field
        for cell in field.grid:
            latitudes.add(float(cell.lat))
            longitudes.add(float(cell.lon))

    times = sorted(grouped.keys())
    sorted_latitudes = sorted(latitudes)
    sorted_longitudes = sorted(longitudes)
    latitude = np.asarray(sorted_latitudes, dtype=np.float32)
    longitude = np.asarray(sorted_longitudes, dtype=np.float32)
    lat_index = {value: idx for idx, value in enumerate(sorted_latitudes)}
    lon_index = {value: idx for idx, value in enumerate(sorted_longitudes)}

    data_vars: dict[str, Any] = {}
    variable_units: dict[str, str] = {}
    for valid_time, field_map in grouped.items():
        for variable, field in field_map.items():
            if variable not in data_vars:
                data_vars[variable] = np.full((len(times), len(latitude), len(longitude)), np.nan, dtype=np.float32)
                variable_units[variable] = field.units
            time_idx = times.index(valid_time)
            for value, cell in zip(field.values, field.grid):
                if is_missing(value):
                    continue
                data_vars[variable][time_idx, lat_index[float(cell.lat)], lon_index[float(cell.lon)]] = float(value)

    original_variables = set(data_vars)
    derived, derivation_metadata = _derived_dataset_fields(
        data_vars,
        valid_times=times,
        provider_metadata=provider_metadata or {},
    )
    data_vars.update(derived)

    attrs = _dataset_runtime_attrs(fields, provider_metadata, original_variables, derivation_metadata)
    ds = xr.Dataset(
        data_vars={
            name: (("time", "latitude", "longitude"), values, {"units": variable_units.get(name, _derived_units(name))})
            for name, values in data_vars.items()
        },
        coords={
            "time": np.asarray(times, dtype="datetime64[ns]"),
            "latitude": latitude,
            "longitude": longitude,
        },
        attrs=attrs,
    )
    return ds


def _dataset_runtime_attrs(
    fields: dict[str, ForecastField],
    provider_metadata: dict[str, Any] | None,
    original_variables: set[str],
    derivation_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    first = next(iter(fields.values()))
    effective_source_status, effective_degradation_metadata = _effective_runtime_degradation(
        source_status=first.source_status,
        degradation_metadata=first.degradation_metadata,
        derivation_metadata=derivation_metadata,
    )
    attrs = {
        "primary_provider": first.provider,
        "provider_role": first.provider_role,
        "provider_status": first.provider_status,
        "source_status": effective_source_status,
        "degradation_metadata_json": json.dumps(effective_degradation_metadata, ensure_ascii=False, sort_keys=True),
    }
    if provider_metadata:
        attrs.update(provider_metadata)
    member_values = first.metadata.get("member_values")
    if member_values is not None:
        attrs["ensemble_member_count"] = int(first.metadata.get("member_count", len(member_values)))
    benchmark = first.metadata.get("benchmark_comparison")
    if benchmark is not None:
        attrs["benchmark_comparison_json"] = json.dumps(benchmark, ensure_ascii=False, sort_keys=True)
    attrs["source_metadata_json"] = json.dumps(
        _forecast_source_metadata(
            primary_provider=attrs["primary_provider"],
            source_status=attrs["source_status"],
            original_variables=original_variables,
            derivation_metadata=derivation_metadata,
            degradation_metadata=effective_degradation_metadata,
        ),
        ensure_ascii=False,
        sort_keys=True,
    )
    return attrs


def _derived_dataset_fields(
    data_vars: dict[str, np.ndarray],
    valid_times: list[datetime] | None = None,
    provider_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    temp = data_vars.get("temp_c")
    tmax = data_vars.get("tmax_c")
    tmin = data_vars.get("tmin_c")
    rh = data_vars.get("rh_percent")
    wind = data_vars.get("wind_speed_mps")
    pressure = data_vars.get("pressure_hpa")
    precip_1h = data_vars.get("precip_1h_mm")
    precip_24h = data_vars.get("precip_24h_mm")
    cape = data_vars.get("cape_j_kg")
    heat_climatology = _heat_climatology_summary_from_metadata(provider_metadata or {})
    times = list(valid_times or [])
    derived: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    if "pwat" in data_vars:
        metadata["pwat"] = {"status": "direct", "method": "provider_field", "source_variables": ["pwat"]}
    if "ivt" in data_vars:
        metadata["ivt"] = {"status": "direct", "method": "provider_field", "source_variables": ["ivt"]}
    if "wind850_speed" in data_vars:
        metadata["wind850_speed"] = {"status": "direct", "method": "provider_field", "source_variables": ["wind850_speed"]}
    if "wind_shear_850_200" in data_vars:
        metadata["wind_shear_850_200"] = {
            "status": "direct",
            "method": "provider_field",
            "source_variables": ["wind_shear_850_200"],
        }
    if "daily_convective_precip" in data_vars:
        metadata["daily_convective_precip"] = {
            "status": "direct",
            "method": "provider_field",
            "source_variables": ["daily_convective_precip"],
        }
    if "daily_large_scale_precip" in data_vars:
        metadata["daily_large_scale_precip"] = {
            "status": "derived",
            "method": "daily_total_minus_convective",
            "source_variables": ["daily_precip_total", "daily_convective_precip"],
        }
    if temp is not None and rh is not None:
        if "relative_humidity_percent" not in data_vars:
            derived["relative_humidity_percent"] = np.clip(np.asarray(rh, dtype=np.float32), 0.0, 100.0)
            metadata["relative_humidity_percent"] = {
                "status": "derived",
                "method": "copy_rh_percent",
                "source_variables": ["rh_percent"],
            }
        heat_index = np.full(temp.shape, np.nan, dtype=np.float32)
        vpd = np.full(temp.shape, np.nan, dtype=np.float32)
        pwat = np.full(temp.shape, np.nan, dtype=np.float32)
        fallback_cape = np.full(temp.shape, np.nan, dtype=np.float32)
        for index in np.ndindex(temp.shape):
            t = temp[index]
            h = rh[index]
            if np.isfinite(t) and np.isfinite(h):
                heat_index[index] = float(compute_heat_index_c(float(t), float(h)))
                vpd[index] = float(compute_vpd_kpa(float(t), float(h)))
                pressure_value = None
                if pressure is not None:
                    pressure_candidate = pressure[index]
                    if np.isfinite(pressure_candidate):
                        pressure_value = float(pressure_candidate)
                pwat[index] = float(compute_pwat_placeholder(float(t), float(h), pressure_value))
                fallback_cape[index] = float(compute_cape_placeholder(float(t), float(h)))
        if "heat_index_c" not in data_vars:
            derived["heat_index_c"] = heat_index
            metadata["heat_index_c"] = {"status": "derived", "method": "heat_index_formula", "source_variables": ["temp_c", "rh_percent"]}
        if "vpd_kpa" not in data_vars:
            derived["vpd_kpa"] = vpd
            metadata["vpd_kpa"] = {"status": "derived", "method": "vpd_formula", "source_variables": ["temp_c", "rh_percent"]}
        if "pwat" not in data_vars:
            derived["pwat"] = pwat
            metadata["pwat"] = {
                "status": "proxy",
                "method": "compute_pwat_placeholder",
                "source_variables": ["temp_c", "rh_percent", "pressure_hpa"],
            }
        if cape is None:
            derived["cape"] = fallback_cape
            metadata["cape"] = {"status": "proxy", "method": "compute_cape_placeholder", "source_variables": ["temp_c", "rh_percent"]}
        heat_index_source = data_vars.get("heat_index_c", derived.get("heat_index_c"))
        if heat_index_source is not None:
            heatwave = np.full(temp.shape, np.nan, dtype=np.float32)
            temp_arr = np.asarray(temp, dtype=np.float32)
            tmax_arr = np.asarray(tmax, dtype=np.float32) if tmax is not None else None
            heat_index_arr = np.asarray(heat_index_source, dtype=np.float32)
            for index in np.ndindex(temp_arr.shape):
                temp_value = temp_arr[index]
                tmax_value = tmax_arr[index] if tmax_arr is not None else np.nan
                heat_index_value = heat_index_arr[index]
                if not (np.isfinite(temp_value) or np.isfinite(tmax_value) or np.isfinite(heat_index_value)):
                    continue
                heatwave[index] = float(
                    (np.isfinite(temp_value) and temp_value >= 40.0)
                    or (np.isfinite(tmax_value) and tmax_value >= 45.0)
                    or (np.isfinite(heat_index_value) and heat_index_value >= 40.0)
                )
            if "heatwave_day_flag" not in data_vars:
                derived["heatwave_day_flag"] = heatwave
                metadata["heatwave_day_flag"] = {
                    "status": "derived_context_only",
                    "method": "single_day_heatwave_thresholds",
                    "source_variables": ["temp_c", "tmax_c", "heat_index_c"],
                    "thresholds": {"temp_c_ge": 40.0, "tmax_c_ge": 45.0, "heat_index_c_ge": 40.0},
                }
            if "heatwave_duration_days" not in data_vars:
                derived["heatwave_duration_days"] = np.where(np.isfinite(heatwave), heatwave, np.nan).astype(np.float32)
                metadata["heatwave_duration_days"] = {
                    "status": "derived_context_only",
                    "method": "single_day_heatwave_duration_proxy",
                    "source_variables": ["heatwave_day_flag"],
                    "duration_horizon_days": 1,
                }
    if temp is not None and "t2m_anomaly_c" not in data_vars:
        temp_arr = np.asarray(temp, dtype=np.float32)
        temp_anomaly, temp_metadata = _heat_climatology_anomaly(
            variable_name="temp_c",
            field_name="t2m_anomaly_c",
            values=temp_arr,
            valid_times=times,
            heat_climatology=heat_climatology,
        )
        derived["t2m_anomaly_c"] = temp_anomaly
        metadata["t2m_anomaly_c"] = temp_metadata
    if tmax is not None and "tmax_anomaly_c" not in data_vars:
        tmax_arr = np.asarray(tmax, dtype=np.float32)
        tmax_anomaly, tmax_metadata = _heat_climatology_anomaly(
            variable_name="tmax_c",
            field_name="tmax_anomaly_c",
            values=tmax_arr,
            valid_times=times,
            heat_climatology=heat_climatology,
        )
        derived["tmax_anomaly_c"] = tmax_anomaly
        metadata["tmax_anomaly_c"] = tmax_metadata
    if wind is not None and "wind_speed_mps" not in derived:
        derived["wind_speed_mps"] = np.asarray(wind, dtype=np.float32)
        metadata["wind_speed_mps"] = {"status": "direct", "method": "provider_field", "source_variables": ["wind_speed_mps"]}
    if cape is not None:
        derived["cape"] = np.asarray(cape, dtype=np.float32)
        metadata["cape"] = {"status": "direct", "method": "copy_cape_j_kg", "source_variables": ["cape_j_kg"]}
    if precip_24h is None and precip_1h is not None:
        precip_24h = np.asarray(precip_1h, dtype=np.float32) * 24.0
        metadata["daily_precip_total"] = {
            "status": "proxy",
            "method": "precip_1h_mm_times_24",
            "source_variables": ["precip_1h_mm"],
        }
    if precip_24h is not None:
        daily_total = np.clip(np.asarray(precip_24h, dtype=np.float32), 0.0, None)
        if "daily_precip_total" not in data_vars:
            derived["daily_precip_total"] = daily_total
            metadata.setdefault(
                "daily_precip_total",
                {"status": "direct", "method": "copy_precip_24h_mm", "source_variables": ["precip_24h_mm"]},
            )
        convective_direct = data_vars.get("daily_convective_precip")
        large_scale_direct = data_vars.get("daily_large_scale_precip")
        if convective_direct is None:
            convective_ratio = np.full(daily_total.shape, 0.55, dtype=np.float32)
            cape_for_ratio = data_vars.get("cape") if "cape" in data_vars else derived.get("cape")
            if cape_for_ratio is not None:
                convective_ratio = np.clip(0.35 + np.asarray(cape_for_ratio, dtype=np.float32) / 2500.0, 0.35, 0.85)
            convective = daily_total * convective_ratio
            derived["daily_convective_precip"] = convective
            metadata["daily_convective_precip"] = {
                "status": "proxy",
                "method": "cape_weighted_convective_ratio",
                "source_variables": ["precip_24h_mm", "cape"],
            }
        if large_scale_direct is None:
            convective_source = convective_direct if convective_direct is not None else derived.get("daily_convective_precip")
            if convective_source is not None:
                derived["daily_large_scale_precip"] = np.clip(daily_total - np.asarray(convective_source, dtype=np.float32), 0.0, None)
                metadata["daily_large_scale_precip"] = {
                    "status": "proxy" if convective_direct is None else "derived",
                    "method": "daily_total_minus_convective",
                    "source_variables": ["daily_precip_total", "daily_convective_precip"],
                }
    if ("pwat" in data_vars or "pwat" in derived) and wind is not None:
        ivt = np.full(np.asarray(wind, dtype=np.float32).shape, np.nan, dtype=np.float32)
        wind_arr = np.asarray(wind, dtype=np.float32)
        pwat_source = data_vars.get("pwat", derived.get("pwat"))
        pwat_arr = np.asarray(pwat_source, dtype=np.float32)
        for index in np.ndindex(wind_arr.shape):
            wind_value = wind_arr[index]
            pwat_value = pwat_arr[index]
            if np.isfinite(wind_value) and np.isfinite(pwat_value):
                ivt[index] = float(compute_ivt_placeholder(float(wind_value), float(pwat_value)))
        if "ivt" not in data_vars:
            derived["ivt"] = ivt
            metadata["ivt"] = {"status": "proxy", "method": "compute_ivt_placeholder", "source_variables": ["wind_speed_mps", "pwat"]}
        wind850 = np.clip(wind_arr * 1.15, 0.0, None)
        if "wind850_speed" not in data_vars:
            derived["wind850_speed"] = wind850
            metadata["wind850_speed"] = {
                "status": "proxy",
                "method": "wind10_scaled_to_850",
                "source_variables": ["wind_speed_mps"],
            }
        if "wind_shear_850_200" not in data_vars:
            shear = np.clip(wind850 * 0.45, 0.0, None)
            cape_for_shear = data_vars.get("cape") if "cape" in data_vars else derived.get("cape")
            if cape_for_shear is not None:
                shear = shear + np.clip(np.asarray(cape_for_shear, dtype=np.float32) / 200.0, 0.0, 20.0)
            derived["wind_shear_850_200"] = shear.astype(np.float32)
            metadata["wind_shear_850_200"] = {
                "status": "proxy",
                "method": "wind850_and_cape_proxy_shear",
                "source_variables": ["wind_speed_mps", "cape"],
            }
    if precip_24h is not None:
        p1 = np.asarray(precip_1h, dtype=np.float32) if precip_1h is not None else np.asarray(precip_24h, dtype=np.float32) / 24.0
        p24 = np.asarray(precip_24h, dtype=np.float32)
        p6 = np.clip(p1 * 6.0, 0.0, p24)
        flash = np.full(p24.shape, np.nan, dtype=np.float32)
        for index in np.ndindex(p24.shape):
            one_hour = p1[index]
            six_hour = p6[index]
            day_total = p24[index]
            if np.isfinite(one_hour) and np.isfinite(six_hour) and np.isfinite(day_total):
                flash[index] = float(compute_flash_flood_screening_score(float(one_hour), float(six_hour), float(day_total)))
        derived["flash_flood_risk"] = flash
        metadata["flash_flood_risk"] = {
            "status": "derived",
            "method": "compute_flash_flood_screening_score",
            "source_variables": ["precip_1h_mm", "precip_24h_mm"],
        }
    return derived, metadata


def _effective_runtime_degradation(
    *,
    source_status: str,
    degradation_metadata: dict[str, Any],
    derivation_metadata: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    effective_status = str(source_status or "normal")
    effective_metadata = dict(degradation_metadata)
    partition_fallback = _flash_flood_precip_partition_fallback(derivation_metadata)
    if partition_fallback is None:
        return effective_status, effective_metadata
    runtime_fallbacks = list(effective_metadata.get("runtime_fallbacks", []))
    runtime_fallbacks.append(partition_fallback)
    effective_metadata["runtime_fallbacks"] = runtime_fallbacks
    if effective_status != "degraded":
        effective_status = "degraded"
    return effective_status, effective_metadata


def _flash_flood_precip_partition_fallback(derivation_metadata: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    convective = derivation_metadata.get("daily_convective_precip", {})
    large_scale = derivation_metadata.get("daily_large_scale_precip", {})
    if str(convective.get("status", "")).lower() != "proxy":
        return None
    return {
        "category": "flash_flood_precip_partition",
        "fallback_reason": "direct_precip_partition_unavailable",
        "fallback_status": "heuristic_proxy_fallback",
        "convective_method": convective.get("method"),
        "large_scale_method": large_scale.get("method"),
        "convective_source_variables": list(convective.get("source_variables", [])),
        "large_scale_source_variables": list(large_scale.get("source_variables", [])),
    }


def _forecast_source_metadata(
    *,
    primary_provider: str,
    source_status: str,
    original_variables: set[str],
    derivation_metadata: dict[str, dict[str, Any]],
    degradation_metadata: dict[str, Any],
) -> dict[str, Any]:
    flash_features = (
        "daily_precip_total",
        "daily_convective_precip",
        "daily_large_scale_precip",
        "cape",
        "pwat",
        "ivt",
        "wind850_speed",
        "wind_shear_850_200",
        "flash_flood_risk",
    )
    feature_status: dict[str, Any] = {}
    proxy_features: list[str] = []
    for name in flash_features:
        entry = derivation_metadata.get(name)
        if entry is None and name in original_variables:
            entry = {"status": "direct", "method": "provider_field", "source_variables": [name]}
        if entry is None:
            continue
        feature_status[name] = dict(entry)
        if str(entry.get("status", "")).lower() == "proxy":
            proxy_features.append(name)
    precip_partition = _flash_flood_precip_partition_metadata(primary_provider, feature_status)
    heat_features = (
        "temp_c",
        "tmax_c",
        "tmin_c",
        "heat_index_c",
        "vpd_kpa",
        "relative_humidity_percent",
        "t2m_anomaly_c",
        "tmax_anomaly_c",
        "heatwave_day_flag",
        "heatwave_duration_days",
    )
    heat_feature_status: dict[str, Any] = {}
    heat_unavailable_features: list[str] = []
    context_only_features: list[str] = []
    heat_comparison_sources: list[str] = []
    for name in heat_features:
        entry = derivation_metadata.get(name)
        if entry is None and name in original_variables:
            entry = {"status": "direct", "method": "provider_field", "source_variables": [name]}
        if entry is None:
            continue
        heat_feature_status[name] = dict(entry)
        normalized_status = str(entry.get("status", "")).lower()
        if normalized_status == "unavailable":
            heat_unavailable_features.append(name)
        if normalized_status == "derived_context_only":
            context_only_features.append(name)
        comparison_source_id = entry.get("comparison_source_id")
        if comparison_source_id:
            comparison_source_id = str(comparison_source_id)
            if comparison_source_id not in heat_comparison_sources:
                heat_comparison_sources.append(comparison_source_id)
    if heat_unavailable_features:
        heat_status = "comparison_not_available"
    elif any(str(heat_feature_status.get(name, {}).get("status", "")).lower() == "derived" for name in ("t2m_anomaly_c", "tmax_anomaly_c")):
        heat_status = "comparison_available"
    else:
        heat_status = "context_only"
    heat_source_pair = [primary_provider]
    if heat_comparison_sources:
        heat_source_pair.extend(heat_comparison_sources)
    return {
        "resolved_sources": {
            "flash_flood_features": {
                "resolved_source": primary_provider,
                "feature_status": feature_status,
            },
            "heat_features": {
                "resolved_source": primary_provider,
                "feature_status": heat_feature_status,
                "comparison_source_ids": heat_comparison_sources,
            },
        },
        "source_status": source_status,
        "primary_source_id": primary_provider,
        "secondary_source_ids": [],
        "grounding_gap": {
            "flash_flood_features": {
                "status": "proxy_present" if proxy_features else "direct_only",
                "proxy_features": proxy_features,
                "feature_status": feature_status,
                "precipitation_partition": precip_partition,
            },
            "heat_features": {
                "status": heat_status,
                "source_pair": heat_source_pair,
                "feature_status": heat_feature_status,
                "missing_comparison_features": heat_unavailable_features,
                "context_only_features": context_only_features,
                "heatwave_context": _heatwave_context_metadata(primary_provider, heat_feature_status),
            },
        },
        "degradation_metadata": dict(degradation_metadata),
    }


def _flash_flood_precip_partition_metadata(primary_provider: str, feature_status: dict[str, dict[str, Any]]) -> dict[str, Any]:
    convective = dict(feature_status.get("daily_convective_precip", {}))
    large_scale = dict(feature_status.get("daily_large_scale_precip", {}))
    convective_status = str(convective.get("status", "missing")).lower()
    large_scale_status = str(large_scale.get("status", "missing")).lower()
    if convective_status == "proxy":
        return {
            "status": "heuristic_proxy_fallback",
            "source_pair": [primary_provider, "cape_proxy"],
            "units": "mm",
            "fallback_reason": "direct_precip_partition_unavailable",
            "convective_method": convective.get("method"),
            "large_scale_method": large_scale.get("method"),
            "convective_status": convective_status,
            "large_scale_status": large_scale_status,
            "convective_source_variables": list(convective.get("source_variables", [])),
            "large_scale_source_variables": list(large_scale.get("source_variables", [])),
        }
    if convective_status == "direct" and large_scale_status == "derived":
        return {
            "status": "same_source_residual",
            "source_pair": [primary_provider, primary_provider],
            "units": "mm",
            "fallback_reason": None,
            "convective_method": convective.get("method"),
            "large_scale_method": large_scale.get("method"),
            "convective_status": convective_status,
            "large_scale_status": large_scale_status,
            "convective_source_variables": list(convective.get("source_variables", [])),
            "large_scale_source_variables": list(large_scale.get("source_variables", [])),
        }
    if convective_status == "direct" and large_scale_status == "direct":
        return {
            "status": "direct_partition",
            "source_pair": [primary_provider, primary_provider],
            "units": "mm",
            "fallback_reason": None,
            "convective_method": convective.get("method"),
            "large_scale_method": large_scale.get("method"),
            "convective_status": convective_status,
            "large_scale_status": large_scale_status,
            "convective_source_variables": list(convective.get("source_variables", [])),
            "large_scale_source_variables": list(large_scale.get("source_variables", [])),
        }
    return {
        "status": "unavailable",
        "source_pair": [primary_provider],
        "units": "mm",
        "fallback_reason": "precip_partition_not_emitted",
        "convective_status": convective_status,
        "large_scale_status": large_scale_status,
        "convective_source_variables": list(convective.get("source_variables", [])),
        "large_scale_source_variables": list(large_scale.get("source_variables", [])),
    }


def _normalize_pressure_levels(data_array: Any) -> Any:
    pressure_dim = "pressure_level" if "pressure_level" in data_array.dims else "level"
    return data_array.rename({pressure_dim: "level"}) if pressure_dim != "level" else data_array


def _select_pressure_level(data_array: Any, level: int) -> Any:
    return data_array.sel(level=level).drop_vars("level", errors="ignore")


def _pressure_weighted_water(q: Any) -> Any:
    gravity = 9.80665
    pressure = q["level"]
    pressure_pa = xr.where(pressure.max(skipna=True) < 2000.0, pressure * 100.0, pressure)
    order = np.argsort(pressure_pa.values)
    q_sorted = q.isel(level=order).assign_coords(level=pressure_pa.isel(level=order))
    pwat = (q_sorted.integrate("level") / gravity).clip(min=0.0).rename("pwat")
    pwat.attrs["units"] = "mm"
    return pwat


def _integrated_vapor_transport(q: Any, u: Any, v: Any) -> Any:
    gravity = 9.80665
    q, u, v = xr.align(q, u, v, join="inner")
    pressure = q["level"]
    pressure_pa = xr.where(pressure.max(skipna=True) < 2000.0, pressure * 100.0, pressure)
    order = np.argsort(pressure_pa.values)
    q = q.isel(level=order).assign_coords(level=pressure_pa.isel(level=order))
    u = u.isel(level=order).assign_coords(level=pressure_pa.isel(level=order))
    v = v.isel(level=order).assign_coords(level=pressure_pa.isel(level=order))
    ivt_u = (q * u).integrate("level") / gravity
    ivt_v = (q * v).integrate("level") / gravity
    ivt = np.hypot(ivt_u, ivt_v).astype("float32").rename("ivt")
    ivt.attrs["units"] = "kg m-1 s-1"
    return ivt


def _derived_units(name: str) -> str:
    return {
        "heat_index_c": "degC",
        "vpd_kpa": "kPa",
        "relative_humidity_percent": "%",
        "wind_speed_mps": "m/s",
        "temp_c": "degC",
        "rh_percent": "%",
        "pressure_hpa": "hPa",
        "cape": "J/kg",
        "pwat": "mm",
        "ivt": "kg m-1 s-1",
        "wind850_speed": "m/s",
        "wind_shear_850_200": "m/s",
        "daily_precip_total": "mm",
        "daily_convective_precip": "mm",
        "daily_large_scale_precip": "mm",
        "flash_flood_risk": "1",
        "t2m_anomaly_c": "degC",
        "tmax_anomaly_c": "degC",
        "heatwave_day_flag": "1",
        "heatwave_duration_days": "days",
    }.get(name, "unknown")


_HEAT_CLIMATOLOGY_CACHE: dict[Path, dict[str, Any]] = {}


def _heat_climatology_summary_from_metadata(provider_metadata: dict[str, Any]) -> dict[str, Any]:
    heat_climatology = provider_metadata.get("heat_climatology")
    if isinstance(heat_climatology, dict):
        return dict(heat_climatology)
    return {}


def _heat_climatology_summary(heat_climatology_dir: Path | None) -> dict[str, Any]:
    if heat_climatology_dir is None:
        return {
            "status": "unavailable",
            "source_id": "historical_indicator_archive",
            "reason": "heat_climatology_dir_not_configured",
        }
    resolved = heat_climatology_dir.resolve()
    cached = _HEAT_CLIMATOLOGY_CACHE.get(resolved)
    if cached is not None:
        return dict(cached)
    if not resolved.exists():
        summary = {
            "status": "unavailable",
            "source_id": "historical_indicator_archive",
            "source_path": str(resolved),
            "reason": "heat_climatology_dir_missing",
        }
        _HEAT_CLIMATOLOGY_CACHE[resolved] = summary
        return dict(summary)
    files = sorted(path for path in resolved.glob("*.nc") if path.is_file())
    if not files:
        summary = {
            "status": "unavailable",
            "source_id": "historical_indicator_archive",
            "source_path": str(resolved),
            "reason": "heat_climatology_dir_empty",
        }
        _HEAT_CLIMATOLOGY_CACHE[resolved] = summary
        return dict(summary)
    temp_sum = 0.0
    temp_count = 0
    tmax_sum = 0.0
    tmax_count = 0
    temp_files = 0
    tmax_files = 0
    temp_by_doy: dict[int, dict[str, float]] = {}
    tmax_by_doy: dict[int, dict[str, float]] = {}
    for path in files:
        dataset = read_netcdf_dataset(path)
        try:
            day_of_year = _heat_climatology_day_of_year(dataset, path)
            temp_array = None
            for candidate in ("temp_c", "t2m_c"):
                if candidate in dataset.data_vars:
                    temp_array = _normalize_dataarray_shape(dataset[candidate])
                    break
            if temp_array is not None:
                values = np.asarray(temp_array.values, dtype=np.float32)
                finite = values[np.isfinite(values)]
                if finite.size:
                    temp_sum += float(finite.sum())
                    temp_count += int(finite.size)
                    temp_files += 1
                    bucket = temp_by_doy.setdefault(day_of_year, {"sum": 0.0, "count": 0.0})
                    bucket["sum"] += float(finite.sum())
                    bucket["count"] += float(finite.size)
            tmax_array = None
            if "tmax_c" in dataset.data_vars:
                tmax_array = _normalize_dataarray_shape(dataset["tmax_c"])
            if tmax_array is not None:
                values = np.asarray(tmax_array.values, dtype=np.float32)
                finite = values[np.isfinite(values)]
                if finite.size:
                    tmax_sum += float(finite.sum())
                    tmax_count += int(finite.size)
                    tmax_files += 1
                    bucket = tmax_by_doy.setdefault(day_of_year, {"sum": 0.0, "count": 0.0})
                    bucket["sum"] += float(finite.sum())
                    bucket["count"] += float(finite.size)
        finally:
            close = getattr(dataset, "close", None)
            if callable(close):
                close()
    if temp_count == 0 or tmax_count == 0:
        summary = {
            "status": "unavailable",
            "source_id": "historical_indicator_archive",
            "source_path": str(resolved),
            "reason": "heat_climatology_fields_missing",
            "sample_count": len(files),
            "temp_files": temp_files,
            "tmax_files": tmax_files,
        }
        _HEAT_CLIMATOLOGY_CACHE[resolved] = summary
        return dict(summary)
    summary = {
        "status": "ready",
        "source_id": "historical_indicator_archive",
        "source_kind": "lightgbm_indicator_archive",
        "source_path": str(resolved),
        "sample_count": len(files),
        "temp_files": temp_files,
        "tmax_files": tmax_files,
        "temp_c_mean": temp_sum / temp_count,
        "tmax_c_mean": tmax_sum / tmax_count,
        "temp_c_count": temp_count,
        "tmax_c_count": tmax_count,
        "day_of_year_baseline": {
            "status": "ready",
            "temp_c_mean_by_doy": {
                str(day): bucket["sum"] / bucket["count"]
                for day, bucket in sorted(temp_by_doy.items())
                if bucket["count"] > 0
            },
            "tmax_c_mean_by_doy": {
                str(day): bucket["sum"] / bucket["count"]
                for day, bucket in sorted(tmax_by_doy.items())
                if bucket["count"] > 0
            },
            "temp_c_count_by_doy": {
                str(day): int(bucket["count"])
                for day, bucket in sorted(temp_by_doy.items())
                if bucket["count"] > 0
            },
            "tmax_c_count_by_doy": {
                str(day): int(bucket["count"])
                for day, bucket in sorted(tmax_by_doy.items())
                if bucket["count"] > 0
            },
            "available_days": sorted(int(day) for day in set(temp_by_doy) | set(tmax_by_doy)),
        },
        "reference_variables": {"temp_c": "t2m_c", "tmax_c": "tmax_c"},
    }
    _HEAT_CLIMATOLOGY_CACHE[resolved] = summary
    return dict(summary)


def _heat_climatology_day_of_year(dataset: Any, path: Path) -> int:
    for candidate in ("time", "valid_time"):
        if candidate in getattr(dataset, "coords", {}):
            coord = dataset[candidate]
            values = np.asarray(coord.values).reshape(-1)
            if values.size:
                raw_value = values[0]
                if np.issubdtype(np.asarray(raw_value).dtype, np.datetime64):
                    iso_date = np.datetime_as_string(raw_value, unit="D")
                    return datetime.fromisoformat(iso_date).timetuple().tm_yday
    match = re.search(r"(\d{8})", path.stem)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d").timetuple().tm_yday
    raise ValueError(f"Could not determine climatology day-of-year for {path}")


def _heat_climatology_anomaly(
    *,
    variable_name: str,
    field_name: str,
    values: np.ndarray,
    valid_times: list[datetime],
    heat_climatology: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    baseline_info = heat_climatology.get("day_of_year_baseline") if isinstance(heat_climatology, dict) else None
    global_baseline = heat_climatology.get(f"{variable_name}_mean") if isinstance(heat_climatology, dict) else None
    global_count = heat_climatology.get(f"{variable_name}_count") if isinstance(heat_climatology, dict) else None
    source_id = heat_climatology.get("source_id", "historical_indicator_archive") if isinstance(heat_climatology, dict) else "historical_indicator_archive"
    source_path = heat_climatology.get("source_path") if isinstance(heat_climatology, dict) else None
    source_kind = heat_climatology.get("source_kind") if isinstance(heat_climatology, dict) else None
    if values.ndim < 3:
        values = np.asarray(values, dtype=np.float32)
    anomaly = np.full(values.shape, np.nan, dtype=np.float32)
    comparison_schedule: list[dict[str, Any]] = []
    day_of_year_count = 0
    fallback_used = 0
    for time_index, valid_time in enumerate(valid_times or []):
        if time_index >= values.shape[0]:
            break
        baseline = _heat_climatology_baseline_for_valid_time(
            baseline_info=baseline_info if isinstance(baseline_info, dict) else {},
            global_baseline=global_baseline,
            global_count=global_count,
            variable_name=variable_name,
            valid_time=valid_time,
        )
        schedule_entry = {"valid_time": valid_time.isoformat(), "day_of_year": valid_time.timetuple().tm_yday}
        if baseline is None:
            schedule_entry["status"] = "unavailable"
            comparison_schedule.append(schedule_entry)
            continue
        schedule_entry.update(
            {
                "status": baseline["status"],
                "method": baseline["method"],
                "comparison_value_c": baseline["comparison_value_c"],
                "comparison_sample_count": baseline["comparison_sample_count"],
                "comparison_day_of_year": baseline.get("comparison_day_of_year"),
            }
        )
        comparison_schedule.append(schedule_entry)
        anomaly[time_index] = (np.asarray(values[time_index], dtype=np.float32) - np.float32(baseline["comparison_value_c"])).astype(np.float32)
        if baseline["status"] == "derived":
            day_of_year_count += 1
        elif baseline["status"] == "derived_fallback":
            fallback_used += 1
    if day_of_year_count == 0 and fallback_used == 0:
        metadata = {
            "status": "unavailable",
            "method": "climatology_comparison_unavailable",
            "source_variables": [field_name],
            "comparison_schedule": comparison_schedule,
        }
        return anomaly, metadata
    method = "day_of_year_archive_mean"
    if fallback_used and day_of_year_count == 0:
        method = "historical_indicator_archive_mean_fallback"
    elif fallback_used:
        method = "day_of_year_archive_mean_with_archive_fallback"
    metadata = {
        "status": "derived",
        "method": method,
        "comparison_source_id": source_id,
        "comparison_source_path": source_path,
        "comparison_source_kind": source_kind,
        "comparison_sample_count": global_count,
        "comparison_value_c": comparison_schedule[0]["comparison_value_c"] if comparison_schedule else None,
        "comparison_schedule": comparison_schedule,
        "source_variables": [field_name],
    }
    if baseline_info and isinstance(baseline_info, dict):
        metadata["comparison_day_of_year_count"] = len(baseline_info.get(f"{variable_name}_mean_by_doy", {}))
    return anomaly, metadata


def _heat_climatology_baseline_for_valid_time(
    *,
    baseline_info: dict[str, Any],
    global_baseline: Any,
    global_count: Any,
    variable_name: str,
    valid_time: datetime,
) -> dict[str, Any] | None:
    day_of_year = valid_time.timetuple().tm_yday
    means_by_doy = baseline_info.get(f"{variable_name}_mean_by_doy", {})
    counts_by_doy = baseline_info.get(f"{variable_name}_count_by_doy", {})
    if isinstance(means_by_doy, dict):
        baseline_value = means_by_doy.get(str(day_of_year))
        if baseline_value is not None and np.isfinite(baseline_value):
            return {
                "status": "derived",
                "method": "day_of_year_archive_mean",
                "comparison_value_c": float(baseline_value),
                "comparison_sample_count": int(counts_by_doy.get(str(day_of_year), 0)),
                "comparison_day_of_year": day_of_year,
            }
    if global_baseline is not None and np.isfinite(global_baseline):
        return {
            "status": "derived_fallback",
            "method": "historical_indicator_archive_mean_fallback",
            "comparison_value_c": float(global_baseline),
            "comparison_sample_count": int(global_count or 0),
            "comparison_day_of_year": day_of_year,
        }
    return None


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


def _heatwave_context_metadata(primary_provider: str, feature_status: dict[str, dict[str, Any]]) -> dict[str, Any]:
    heatwave_flag = dict(feature_status.get("heatwave_day_flag", {}))
    duration = dict(feature_status.get("heatwave_duration_days", {}))
    return {
        "status": "single_day_context_only" if heatwave_flag or duration else "unavailable",
        "source_pair": [primary_provider],
        "units": {"heatwave_day_flag": "1", "heatwave_duration_days": "days"},
        "heatwave_day_flag_status": str(heatwave_flag.get("status", "missing")).lower(),
        "heatwave_duration_days_status": str(duration.get("status", "missing")).lower(),
        "thresholds": dict(heatwave_flag.get("thresholds", {})),
        "duration_horizon_days": duration.get("duration_horizon_days"),
    }


def _era5_month_path(era5_dir: Path, valid: datetime) -> Path:
    return era5_dir / f"era5_single_levels_{valid.year}_{valid.month:02d}.nc"


def _era5_supplement_path(era5_dir: Path, valid: datetime) -> Path:
    return era5_dir / f"era5_single_levels_{valid.year}_{valid.month:02d}_supplement.nc"


def _mswep_day_path(precip_dir: Path, valid: datetime) -> Path:
    day_of_year = valid.timetuple().tm_yday
    candidates = [
        precip_dir / f"{valid.year}{day_of_year:03d}.nc",
        precip_dir / f"GPM_3IMERGDF_{valid.year}{valid.month:02d}{valid.day:02d}.nc4",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"MSWEP daily precipitation file not found. Tried: {', '.join(str(path) for path in candidates)}")


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
    if source_var in {"tp", "cp"}:
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
    lat_name, lon_name = _resolve_spatial_coord_names(cropped, bbox)
    loaded = cropped.transpose(lat_name, lon_name).load()
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
    provider_role = str(field_metadata.get("provider_role", "deterministic"))
    provider_status = str(field_metadata.get("provider_status", "ready"))
    source_status = str(field_metadata.get("source_status", "normal"))
    degradation_metadata = dict(field_metadata.get("degradation_metadata", {}))
    return ForecastField(
        provider,
        variable,
        units,
        valid_time,
        values,
        grid,
        field_metadata,
        provider_role=provider_role,
        provider_status=provider_status,
        source_status=source_status,
        degradation_metadata=degradation_metadata,
    )


def _crop_xarray_to_bbox(data_array: Any, bbox: tuple[float, float, float, float]) -> Any:
    min_lat, min_lon, max_lat, max_lon = bbox
    lat_name, lon_name = _resolve_spatial_coord_names(data_array, bbox)
    lat_coord = data_array[lat_name]
    lon_coord = data_array[lon_name]
    lat_slice = slice(max_lat, min_lat) if float(lat_coord[0]) > float(lat_coord[-1]) else slice(min_lat, max_lat)
    lon_slice = slice(max_lon, min_lon) if float(lon_coord[0]) > float(lon_coord[-1]) else slice(min_lon, max_lon)
    return data_array.sel({lat_name: lat_slice, lon_name: lon_slice})


def _resolve_spatial_coord_names(data_array: Any, bbox: tuple[float, float, float, float] | None = None) -> tuple[str, str]:
    lat_candidate = "lat" if "lat" in data_array.coords else "latitude"
    lon_candidate = "lon" if "lon" in data_array.coords else "longitude"
    if bbox is None:
        return lat_candidate, lon_candidate
    min_lat, min_lon, max_lat, max_lon = bbox
    standard_score = _coord_overlap_score(data_array[lat_candidate], min_lat, max_lat) + _coord_overlap_score(data_array[lon_candidate], min_lon, max_lon)
    swapped_score = _coord_overlap_score(data_array[lat_candidate], min_lon, max_lon) + _coord_overlap_score(data_array[lon_candidate], min_lat, max_lat)
    if swapped_score > standard_score:
        return lon_candidate, lat_candidate
    return lat_candidate, lon_candidate


def _coord_overlap_score(coord: Any, lower: float, upper: float) -> float:
    values = np.asarray(coord.values, dtype=float)
    if values.size == 0:
        return float("-inf")
    coord_min = float(np.nanmin(values))
    coord_max = float(np.nanmax(values))
    overlap = min(coord_max, upper) - max(coord_min, lower)
    return max(overlap, 0.0)


def load_netcdf_or_zarr_placeholder(path: str | Path, variable: str) -> dict[str, Any]:
    """Placeholder for future xarray-backed NetCDF/Zarr loading."""

    path_obj = Path(path)
    try:
        dataset = read_netcdf_dataset(path_obj) if path_obj.suffix.lower() in {".nc", ".netcdf"} else read_zarr_dataset(path_obj)
    except Exception as exc:
        return {"path": str(path), "variable": variable, "status": "xarray_loader_unavailable", "error": str(exc)}
    return {"path": str(path), "variable": variable, "status": "loaded", "dims": dict(getattr(dataset, "sizes", {}))}
