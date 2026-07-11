#!/usr/bin/env python3
"""
MAZU multi-hazard indicator engine for Saudi Arabia.

The module is intentionally data-layout tolerant:
- ERA5 single-level ZIP containers are extracted to a cache directory.
- ERA5 pressure-level files split by variable/month are scanned and merged.
- Common coordinate aliases are normalized before clipping and interpolation.
- All xarray loads use ``chunks="auto"`` by default for Dask-backed execution.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import xarray as xr

InterpMethod = Literal["linear", "nearest"]

LOGGER = logging.getLogger("mazu_indicator_engine")


@dataclass(frozen=True)
class SaudiGridSpec:
    """Standard Saudi Arabia 0.1 degree grid."""

    lat_min: float = 16.0
    lat_max: float = 32.0
    lon_min: float = 34.0
    lon_max: float = 56.0
    resolution: float = 0.1

    @property
    def latitudes(self) -> xr.DataArray:
        values = np.round(
            np.arange(self.lat_min, self.lat_max + self.resolution / 2.0, self.resolution),
            4,
        )
        return xr.DataArray(values, dims=("latitude",), name="latitude")

    @property
    def longitudes(self) -> xr.DataArray:
        values = np.round(
            np.arange(self.lon_min, self.lon_max + self.resolution / 2.0, self.resolution),
            4,
        )
        return xr.DataArray(values, dims=("longitude",), name="longitude")


@dataclass
class DataAligner:
    """Load, sanitize, clip and interpolate gridded weather datasets."""

    grid: SaudiGridSpec = field(default_factory=SaudiGridSpec)
    chunks: str | Mapping[str, int] | None = "auto"
    cache_dir: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "mazu_indicator_cache")
    missing_abs_threshold: float = 1.0e20
    missing_values: tuple[float, ...] = (9999.0, -9999.0, 1.0e36, -1.0e36)
    output_time_chunk: int = 24

    latitude_aliases: tuple[str, ...] = ("latitude", "lat", "y", "nav_lat")
    longitude_aliases: tuple[str, ...] = ("longitude", "lon", "lng", "x", "nav_lon")
    time_aliases: tuple[str, ...] = ("valid_time", "time", "datetime", "date")
    level_aliases: tuple[str, ...] = (
        "pressure_level",
        "level",
        "plev",
        "isobaricInhPa",
        "isobaric",
    )

    def standard_grid_dataset(self) -> xr.Dataset:
        return xr.Dataset(coords={"latitude": self.grid.latitudes, "longitude": self.grid.longitudes})

    def discover_files(
        self,
        directory: str | Path,
        suffixes: Sequence[str] = (".nc", ".nc4", ".grib", ".grb", ".grb2"),
    ) -> list[Path]:
        root = Path(directory)
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]
        return sorted(files)

    def open_dataset(self, path: str | Path) -> xr.Dataset:
        """Open a single dataset with Dask chunks, including ZIP-wrapped NetCDF files."""
        path = Path(path)
        if self._is_zip_file(path):
            datasets = [self.open_dataset(member) for member in self._extract_zip_members(path)]
            if not datasets:
                raise ValueError(f"No NetCDF members found inside ZIP container: {path}")
            return xr.merge(datasets, compat="override", join="outer")
        try:
            return xr.open_dataset(path, chunks=self.chunks)
        except ImportError as exc:
            raise ImportError(
                "Dask is required for chunks='auto'. Install dask or pass chunks=None."
            ) from exc
        except ValueError as exc:
            raise ValueError(f"Unable to open {path} with xarray. Check the file engine/format.") from exc

    def open_mfdataset(self, paths: Sequence[str | Path]) -> xr.Dataset:
        """Open multiple files and combine by coordinates after ZIP expansion."""
        expanded: list[Path] = []
        for path in paths:
            p = Path(path)
            expanded.extend(self._extract_zip_members(p) if self._is_zip_file(p) else [p])
        if not expanded:
            raise ValueError("No input files were provided.")
        try:
            return xr.open_mfdataset(
                expanded,
                chunks=self.chunks,
                combine="by_coords",
                compat="override",
                join="outer",
                coords="minimal",
                data_vars="minimal",
                parallel=True,
            )
        except Exception:
            LOGGER.info("Falling back to explicit xr.merge for %d files.", len(expanded))
            return xr.merge([self.open_dataset(p) for p in expanded], compat="override", join="outer")

    def load_directory(
        self,
        directory: str | Path,
        *,
        variables: Sequence[str] | None = None,
        method: InterpMethod = "linear",
        file_filter: str | None = None,
        interpolate: bool = True,
    ) -> xr.Dataset:
        """Load every data file in a directory, optionally select variables, then align."""
        files = self.discover_files(directory)
        if file_filter:
            files = [p for p in files if file_filter in p.name]
        if not files:
            raise FileNotFoundError(f"No gridded files found under {directory}")
        ds = self.open_mfdataset(files)
        if variables is not None:
            ds = self.select_existing_variables(ds, variables)
        return self.align_dataset(ds, method=method, interpolate=interpolate)

    def load_variables_from_directory(
        self,
        directory: str | Path,
        variables: Sequence[str],
        *,
        method: InterpMethod = "linear",
        file_hints: Mapping[str, Sequence[str]] | None = None,
        file_filter: str | None = None,
        interpolate: bool = True,
    ) -> xr.Dataset:
        """
        Load variables from a directory where each variable may be split into many files.

        ``file_hints`` lets callers map canonical short names to filename fragments, e.g.
        ``{"q": ("specific_humidity",), "u": ("u_component_of_wind",)}``.
        """
        all_files = self.discover_files(directory)
        if file_filter:
            all_files = [p for p in all_files if file_filter in p.name]
        if not all_files:
            raise FileNotFoundError(f"No gridded files found under {directory}")

        pieces: list[xr.Dataset] = []
        for variable in variables:
            candidates = self._candidate_files_for_variable(all_files, variable, file_hints)
            variable_parts: list[xr.Dataset] = []
            for path in candidates:
                try:
                    ds = self.open_dataset(path)
                except Exception as exc:
                    LOGGER.warning("Skipping unreadable file %s: %s", path, exc)
                    continue
                if variable in ds.data_vars:
                    variable_parts.append(ds[[variable]])
            if not variable_parts:
                raise KeyError(f"Variable {variable!r} was not found under {directory}")
            merged = xr.combine_by_coords(
                variable_parts,
                compat="override",
                combine_attrs="drop_conflicts",
                data_vars="minimal",
                coords="minimal",
            )
            pieces.append(merged)

        return self.align_dataset(
            xr.merge(pieces, compat="override", join="outer"),
            method=method,
            interpolate=interpolate,
        )

    def select_existing_variables(self, ds: xr.Dataset, variables: Sequence[str]) -> xr.Dataset:
        existing = [name for name in variables if name in ds.data_vars]
        missing = sorted(set(variables) - set(existing))
        if missing:
            LOGGER.warning("Missing variables ignored: %s", ", ".join(missing))
        if not existing:
            raise KeyError(f"None of the requested variables exist: {variables}")
        return ds[existing]

    def align_dataset(
        self,
        ds: xr.Dataset,
        *,
        method: InterpMethod = "linear",
        interpolate: bool = True,
    ) -> xr.Dataset:
        ds = self.normalize_coordinates(ds)
        ds = self.sanitize_dataset(ds)
        ds = self.clip_to_saudi(ds)
        if interpolate and "latitude" in ds.coords and "longitude" in ds.coords:
            ds = ds.interp(
                latitude=self.grid.latitudes,
                longitude=self.grid.longitudes,
                method=method,
                kwargs={"fill_value": np.nan},
            )
        return self.rechunk_for_processing(ds)

    def rechunk_for_processing(self, ds: xr.Dataset) -> xr.Dataset:
        chunks: dict[str, int] = {}
        if "time" in ds.dims:
            chunks["time"] = self.output_time_chunk
        if "pressure_level" in ds.dims:
            chunks["pressure_level"] = -1
        if "latitude" in ds.dims:
            chunks["latitude"] = len(ds["latitude"])
        if "longitude" in ds.dims:
            chunks["longitude"] = len(ds["longitude"])
        return ds.chunk(chunks) if chunks else ds

    def normalize_coordinates(self, ds: xr.Dataset) -> xr.Dataset:
        rename_map: dict[str, str] = {}
        lat_name = self._find_coord_name(ds, self.latitude_aliases)
        lon_name = self._find_coord_name(ds, self.longitude_aliases)
        time_name = self._find_coord_name(ds, self.time_aliases)
        level_name = self._find_coord_name(ds, self.level_aliases)

        if lat_name and lon_name and self._coords_look_swapped(ds, lat_name, lon_name):
            ds = ds.rename({lat_name: "__mazu_lon__", lon_name: "__mazu_lat__"})
            lat_name, lon_name = "__mazu_lat__", "__mazu_lon__"

        if lat_name and lat_name != "latitude":
            rename_map[lat_name] = "latitude"
        if lon_name and lon_name != "longitude":
            rename_map[lon_name] = "longitude"
        if time_name and time_name != "time":
            rename_map[time_name] = "time"
        if level_name and level_name != "pressure_level":
            rename_map[level_name] = "pressure_level"
        if rename_map:
            ds = ds.rename(rename_map)

        if "longitude" in ds.coords:
            lon = ds["longitude"]
            if float(lon.max(skipna=True)) > 180.0:
                ds = ds.assign_coords(longitude=((lon + 180.0) % 360.0) - 180.0)
            ds = ds.sortby("longitude")
        if "latitude" in ds.coords:
            ds = ds.sortby("latitude")
        if "pressure_level" in ds.coords:
            ds = ds.sortby("pressure_level")
        return ds

    def sanitize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        cleaned: dict[str, xr.DataArray] = {}
        for name, da in ds.data_vars.items():
            if not np.issubdtype(da.dtype, np.number):
                cleaned[name] = da
                continue
            finite = xr.apply_ufunc(np.isfinite, da, dask="allowed")
            valid = finite & (np.abs(da) < self.missing_abs_threshold)
            for value in self.missing_values:
                valid = valid & (da != value)
            cleaned[name] = da.where(valid)
        return xr.Dataset(cleaned, coords=ds.coords, attrs=ds.attrs)

    def clip_to_saudi(self, ds: xr.Dataset) -> xr.Dataset:
        if "latitude" not in ds.coords or "longitude" not in ds.coords:
            return ds
        lat_slice = slice(self.grid.lat_min, self.grid.lat_max)
        lon_slice = slice(self.grid.lon_min, self.grid.lon_max)
        return ds.sel(latitude=lat_slice, longitude=lon_slice)

    def _candidate_files_for_variable(
        self,
        files: Sequence[Path],
        variable: str,
        file_hints: Mapping[str, Sequence[str]] | None,
    ) -> list[Path]:
        hints = tuple(file_hints.get(variable, ()) if file_hints else ())
        lowered_hints = tuple(h.lower() for h in (hints or (variable,)))
        matched = [p for p in files if any(hint in p.name.lower() for hint in lowered_hints)]
        return matched or list(files)

    def _find_coord_name(self, ds: xr.Dataset, aliases: Iterable[str]) -> str | None:
        names = set(ds.coords) | set(ds.dims) | set(ds.variables)
        lower_lookup = {name.lower(): name for name in names}
        for alias in aliases:
            if alias.lower() in lower_lookup:
                return lower_lookup[alias.lower()]
        return None

    def _coords_look_swapped(self, ds: xr.Dataset, lat_name: str, lon_name: str) -> bool:
        try:
            lat = ds[lat_name]
            lon = ds[lon_name]
            lat_overlaps_lat = self._range_overlaps(lat, self.grid.lat_min, self.grid.lat_max)
            lon_overlaps_lon = self._range_overlaps(lon, self.grid.lon_min, self.grid.lon_max)
            lat_overlaps_lon = self._range_overlaps(lat, self.grid.lon_min, self.grid.lon_max)
            lon_overlaps_lat = self._range_overlaps(lon, self.grid.lat_min, self.grid.lat_max)
        except Exception:
            return False
        return (not lat_overlaps_lat or not lon_overlaps_lon) and lat_overlaps_lon and lon_overlaps_lat

    @staticmethod
    def _range_overlaps(values: xr.DataArray, low: float, high: float) -> bool:
        vmin = float(values.min(skipna=True))
        vmax = float(values.max(skipna=True))
        return max(vmin, low) <= min(vmax, high)

    @staticmethod
    def _is_zip_file(path: Path) -> bool:
        return path.is_file() and zipfile.is_zipfile(path)

    def _extract_zip_members(self, path: Path) -> list[Path]:
        target = self.cache_dir / path.stem
        target.mkdir(parents=True, exist_ok=True)
        extracted: list[Path] = []
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if Path(member).suffix.lower() not in {".nc", ".nc4"}:
                    continue
                out = target / Path(member).name
                if not out.exists() or out.stat().st_size == 0:
                    archive.extract(member, target)
                    extracted_path = target / member
                    if extracted_path != out:
                        extracted_path.replace(out)
                extracted.append(out)
        return extracted


class DerivedIndicators:
    """Meteorological indicator calculations using xarray/Dask arrays."""

    GRAVITY: float = 9.80665
    EPSILON: float = 1.0e-6

    @staticmethod
    def saturation_vapor_pressure_kpa(temperature_c: xr.DataArray) -> xr.DataArray:
        return 0.6108 * np.exp((17.27 * temperature_c) / (temperature_c + 237.3))

    @classmethod
    def relative_humidity_from_dewpoint(
        cls,
        temperature_k: xr.DataArray,
        dewpoint_k: xr.DataArray,
    ) -> xr.DataArray:
        temperature_c = temperature_k - 273.15
        dewpoint_c = dewpoint_k - 273.15
        rh = 100.0 * cls.saturation_vapor_pressure_kpa(dewpoint_c) / (
            cls.saturation_vapor_pressure_kpa(temperature_c) + cls.EPSILON
        )
        return rh.clip(min=0.0, max=100.0).rename("relative_humidity")

    @classmethod
    def vpd(
        cls,
        t2m: xr.DataArray,
        d2m: xr.DataArray | None = None,
        relative_humidity: xr.DataArray | None = None,
    ) -> xr.DataArray:
        if relative_humidity is None:
            if d2m is None:
                raise ValueError("Either d2m or relative_humidity is required for VPD.")
            relative_humidity = cls.relative_humidity_from_dewpoint(t2m, d2m)
        temperature_c = t2m - 273.15
        es = cls.saturation_vapor_pressure_kpa(temperature_c)
        ea = es * relative_humidity.clip(min=0.0, max=100.0) / 100.0
        return (es - ea).clip(min=0.0).rename("vpd_kpa")

    @classmethod
    def heat_index(
        cls,
        t2m: xr.DataArray,
        d2m: xr.DataArray | None = None,
        relative_humidity: xr.DataArray | None = None,
    ) -> xr.DataArray:
        """
        Rothfusz heat index in degrees Celsius.

        Formula is applied in Fahrenheit and converted back to Celsius. For
        cooler points where Rothfusz is not valid, the air temperature is used.
        """
        if relative_humidity is None:
            if d2m is None:
                raise ValueError("Either d2m or relative_humidity is required for heat index.")
            relative_humidity = cls.relative_humidity_from_dewpoint(t2m, d2m)

        t_f = (t2m - 273.15) * 9.0 / 5.0 + 32.0
        rh = relative_humidity.clip(min=0.0, max=100.0)
        hi_f = (
            -42.379
            + 2.04901523 * t_f
            + 10.14333127 * rh
            - 0.22475541 * t_f * rh
            - 0.00683783 * t_f**2
            - 0.05481717 * rh**2
            + 0.00122874 * t_f**2 * rh
            + 0.00085282 * t_f * rh**2
            - 0.00000199 * t_f**2 * rh**2
        )

        low_rh_adjust = ((13.0 - rh) / 4.0) * np.sqrt(
            ((17.0 - np.abs(t_f - 95.0)).clip(min=0.0)) / 17.0
        )
        hi_f = xr.where((rh < 13.0) & (t_f >= 80.0) & (t_f <= 112.0), hi_f - low_rh_adjust, hi_f)

        high_rh_adjust = ((rh - 85.0) / 10.0) * ((87.0 - t_f) / 5.0)
        hi_f = xr.where((rh > 85.0) & (t_f >= 80.0) & (t_f <= 87.0), hi_f + high_rh_adjust, hi_f)
        hi_f = xr.where(t_f < 80.0, t_f, hi_f)
        return ((hi_f - 32.0) * 5.0 / 9.0).rename("heat_index_c")

    @classmethod
    def ivt(
        cls,
        q: xr.DataArray,
        u: xr.DataArray,
        v: xr.DataArray,
        *,
        pressure_dim: str = "pressure_level",
    ) -> xr.DataArray:
        """Integrated vapor transport magnitude in kg m-1 s-1."""
        if pressure_dim not in q.dims:
            raise ValueError(f"Pressure dimension {pressure_dim!r} not found in q dimensions {q.dims}")

        q, u, v = xr.align(q, u, v, join="inner")
        pressure = q[pressure_dim]
        pressure_pa = xr.where(pressure.max(skipna=True) < 2000.0, pressure * 100.0, pressure)
        order = np.argsort(pressure_pa.values)
        q = q.isel({pressure_dim: order})
        u = u.isel({pressure_dim: order})
        v = v.isel({pressure_dim: order})
        pressure_pa = pressure_pa.isel({pressure_dim: order})
        q = q.assign_coords({pressure_dim: pressure_pa})
        u = u.assign_coords({pressure_dim: pressure_pa})
        v = v.assign_coords({pressure_dim: pressure_pa})

        flux_u = (q * u).integrate(coord=pressure_dim) / cls.GRAVITY
        flux_v = (q * v).integrate(coord=pressure_dim) / cls.GRAVITY
        return np.hypot(flux_u, flux_v).astype("float32").rename("ivt_kg_m_s")

    @staticmethod
    def hourly_max_wind_speed(u10: xr.DataArray, v10: xr.DataArray) -> xr.DataArray:
        wind = np.hypot(u10, v10).rename("wind10_speed_m_s")
        return wind.resample(time="1h").max().rename("hourly_max_wind10_m_s") if "time" in wind.dims else wind

    @classmethod
    def convective_precip_ratio(cls, cp: xr.DataArray, tp: xr.DataArray) -> xr.DataArray:
        return (cp / (tp + cls.EPSILON)).clip(min=0.0, max=1.0).rename("convective_precip_ratio")


@dataclass
class MazuIndicatorEngine:
    """Business-facing orchestration layer for MAZU indicator generation."""

    root: Path = Path(".")
    aligner: DataAligner = field(default_factory=DataAligner)
    indicators: type[DerivedIndicators] = DerivedIndicators

    pressure_file_hints: Mapping[str, Sequence[str]] = field(
        default_factory=lambda: {
            "q": ("specific_humidity",),
            "u": ("u_component_of_wind",),
            "v": ("v_component_of_wind",),
            "t": ("temperature",),
            "z": ("geopotential",),
        }
    )

    def load_single_levels(self, variables: Sequence[str] | None = None, month: int | None = None) -> xr.Dataset:
        return self.aligner.load_directory(
            self.root / "era5_single_levels_2025",
            variables=variables,
            method="linear",
            file_filter=self._month_filter(month),
        )

    def load_pressure_levels(
        self,
        variables: Sequence[str],
        month: int | None = None,
        *,
        interpolate: bool = True,
    ) -> xr.Dataset:
        return self.aligner.load_variables_from_directory(
            self.root / "era5_pressure_levels_2025",
            variables,
            method="linear",
            file_hints=self.pressure_file_hints,
            file_filter=self._month_filter(month),
            interpolate=interpolate,
        )

    def load_precipitation(self, variables: Sequence[str] | None = None) -> xr.Dataset:
        return self.aligner.load_directory(self.root / "precip", variables=variables, method="nearest")

    def build_core_indicators(
        self,
        *,
        single_levels: xr.Dataset | None = None,
        pressure_levels: xr.Dataset | None = None,
        month: int | None = None,
    ) -> xr.Dataset:
        if single_levels is None:
            single_levels = self.load_single_levels(("t2m", "d2m", "u10", "v10", "tp", "cp"), month=month)
        if pressure_levels is None:
            pressure_levels = self.load_pressure_levels(("q", "u", "v"), month=month, interpolate=False)

        outputs: list[xr.DataArray] = []
        if {"q", "u", "v"}.issubset(pressure_levels.data_vars):
            raw_ivt = self.indicators.ivt(pressure_levels["q"], pressure_levels["u"], pressure_levels["v"])
            aligned_ivt = self.aligner.align_dataset(raw_ivt.to_dataset(), method="linear")["ivt_kg_m_s"]
            outputs.append(aligned_ivt)
        if {"t2m", "d2m"}.issubset(single_levels.data_vars):
            outputs.append(self.indicators.vpd(single_levels["t2m"], d2m=single_levels["d2m"]))
            outputs.append(self.indicators.heat_index(single_levels["t2m"], d2m=single_levels["d2m"]))
        if {"u10", "v10"}.issubset(single_levels.data_vars):
            outputs.append(self.indicators.hourly_max_wind_speed(single_levels["u10"], single_levels["v10"]))
        if {"cp", "tp"}.issubset(single_levels.data_vars):
            outputs.append(self.indicators.convective_precip_ratio(single_levels["cp"], single_levels["tp"]))

        if not outputs:
            raise ValueError("No requested core indicators could be computed from available variables.")
        return self.prepare_output_dataset(xr.merge(outputs, compat="override", join="outer"))

    def write_core_indicators(self, output_path: str | Path, *, month: int | None = None) -> Path:
        ds = self.build_core_indicators(month=month)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if "time" in ds.dims:
            self.write_streaming_netcdf(ds, output)
        else:
            ds.to_netcdf(output, encoding=self.netcdf_encoding(ds))
        return output

    def prepare_output_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        drop_coords = [name for name in ds.coords if name not in set(ds.dims)]
        if drop_coords:
            ds = ds.drop_vars(drop_coords, errors="ignore")
        ds = ds.chunk(
            {
                dim: chunk
                for dim, chunk in {
                    "time": self.aligner.output_time_chunk,
                    "latitude": len(ds["latitude"]) if "latitude" in ds.dims else None,
                    "longitude": len(ds["longitude"]) if "longitude" in ds.dims else None,
                }.items()
                if dim in ds.dims and chunk is not None
            }
        )
        ds.attrs = {
            "title": "MAZU Saudi Arabia derived hazard indicators",
            "spatial_domain": "Saudi Arabia bbox: latitude 16.0-32.0, longitude 34.0-56.0",
            "grid_resolution_degree": self.aligner.grid.resolution,
            "processing_note": "Sanitized missing values, clipped to Saudi bbox, interpolated to standard 0.1 degree grid.",
        }
        variable_attrs = {
            "ivt_kg_m_s": ("Integrated vapor transport", "kg m-1 s-1"),
            "vpd_kpa": ("Vapor pressure deficit", "kPa"),
            "heat_index_c": ("Rothfusz heat index", "degC"),
            "hourly_max_wind10_m_s": ("Hourly maximum 10 m wind speed", "m s-1"),
            "convective_precip_ratio": ("Convective precipitation fraction", "1"),
        }
        for name, (long_name, units) in variable_attrs.items():
            if name in ds:
                ds[name].attrs = {"long_name": long_name, "units": units}
        return ds

    def netcdf_encoding(self, ds: xr.Dataset) -> dict[str, dict[str, object]]:
        encoding: dict[str, dict[str, object]] = {}
        chunk_dims = tuple(dim for dim in ("time", "latitude", "longitude") if dim in ds.dims)
        chunk_sizes = tuple(
            self.aligner.output_time_chunk if dim == "time" else len(ds[dim]) for dim in chunk_dims
        )
        for name, da in ds.data_vars.items():
            if not np.issubdtype(da.dtype, np.number):
                continue
            encoding[name] = {
                "dtype": "float32",
                "zlib": True,
                "complevel": 3,
                "_FillValue": np.float32(np.nan),
            }
            if tuple(da.dims) == chunk_dims:
                encoding[name]["chunksizes"] = chunk_sizes
        return encoding

    def write_streaming_netcdf(self, ds: xr.Dataset, output: Path) -> None:
        """Write a time-indexed dataset chunk by chunk to avoid high memory peaks."""
        try:
            import netCDF4
        except ImportError as exc:
            raise ImportError("netCDF4 is required for streaming NetCDF output.") from exc

        time_values = ds["time"].values
        lat_values = ds["latitude"].values.astype("float64")
        lon_values = ds["longitude"].values.astype("float64")
        time_units = "seconds since 1970-01-01 00:00:00 UTC"
        calendar = "proleptic_gregorian"
        numeric_time = time_values.astype("datetime64[s]").astype("int64")

        with netCDF4.Dataset(output, mode="w", format="NETCDF4") as nc:
            nc.createDimension("time", len(time_values))
            nc.createDimension("latitude", len(lat_values))
            nc.createDimension("longitude", len(lon_values))

            time_var = nc.createVariable("time", "f8", ("time",))
            time_var.units = time_units
            time_var.calendar = calendar
            time_var[:] = numeric_time

            lat_var = nc.createVariable("latitude", "f8", ("latitude",))
            lat_var.units = "degrees_north"
            lat_var[:] = lat_values

            lon_var = nc.createVariable("longitude", "f8", ("longitude",))
            lon_var.units = "degrees_east"
            lon_var[:] = lon_values

            for key, value in ds.attrs.items():
                setattr(nc, key, value)

            variables = {}
            for name, da in ds.data_vars.items():
                var = nc.createVariable(
                    name,
                    "f4",
                    ("time", "latitude", "longitude"),
                    zlib=True,
                    complevel=3,
                    chunksizes=(self.aligner.output_time_chunk, len(lat_values), len(lon_values)),
                    fill_value=np.float32(np.nan),
                )
                for key, value in da.attrs.items():
                    setattr(var, key, value)
                variables[name] = var

            step = self.aligner.output_time_chunk
            for start in range(0, len(time_values), step):
                stop = min(start + step, len(time_values))
                LOGGER.info("Writing time slice %s:%s to %s", start, stop, output)
                block = ds.isel(time=slice(start, stop)).load()
                for name in ds.data_vars:
                    values = np.asarray(block[name].values, dtype="float32")
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=DeprecationWarning)
                        variables[name][start:stop, :, :] = values

    @staticmethod
    def _month_filter(month: int | None) -> str | None:
        if month is None:
            return None
        if not 1 <= month <= 12:
            raise ValueError("month must be between 1 and 12")
        return f"_{month:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MAZU Saudi Arabia gridded indicators.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Raw data root directory.")
    parser.add_argument("--output", type=Path, help="Optional NetCDF output path for core indicators.")
    parser.add_argument("--month", type=int, help="Optional month filter, 1-12. Recommended for large runs.")
    parser.add_argument("--time-chunk", type=int, default=24, help="Hours per streaming write block.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()))
    engine = MazuIndicatorEngine(root=args.root, aligner=DataAligner(output_time_chunk=args.time_chunk))
    if args.output:
        output = engine.write_core_indicators(args.output, month=args.month)
        LOGGER.info("Wrote %s", output)
    else:
        print(engine.build_core_indicators(month=args.month))


if __name__ == "__main__":
    main()
