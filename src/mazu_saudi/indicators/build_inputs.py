"""Build Aurora-ready inputs and LightGBM indicator files from ``data/raw``."""

from __future__ import annotations

import json
import tempfile
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from mazu_saudi.indicators.physical import compute_flash_flood_screening_score

SAUDI_BBOX = (16.0, 34.0, 32.0, 56.0)
STANDARD_RESOLUTION = 0.1
PRESSURE_LEVELS = (925, 850, 700, 500, 300, 200)
AURORA_CADENCE_HOURS = 6

FILL_CANDIDATES = (9999.0, -9999.0, 1.0e20, -1.0e20, 1.0e30, -1.0e30, 9.96921e36)
MISSING_INDICATOR_GROUPS = ("ds1_monthly", "ds10_subdaily")
MIN_RATIO_DENOMINATOR = 1.0e-6


def _standard_latitudes() -> xr.DataArray:
    values = np.round(np.arange(SAUDI_BBOX[0], SAUDI_BBOX[2] + STANDARD_RESOLUTION / 2.0, STANDARD_RESOLUTION), 4)
    return xr.DataArray(values, dims=("latitude",), name="latitude")


def _standard_longitudes() -> xr.DataArray:
    values = np.round(np.arange(SAUDI_BBOX[1], SAUDI_BBOX[3] + STANDARD_RESOLUTION / 2.0, STANDARD_RESOLUTION), 4)
    return xr.DataArray(values, dims=("longitude",), name="longitude")


def _aurora_patch_latitudes() -> xr.DataArray:
    values = _standard_latitudes().values
    size = (len(values) // 4) * 4
    return xr.DataArray(values[:size], dims=("lat",), name="lat")


def _aurora_patch_longitudes() -> xr.DataArray:
    values = _standard_longitudes().values
    size = (len(values) // 4) * 4
    return xr.DataArray(values[:size], dims=("lon",), name="lon")


def _normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename: dict[str, str] = {}
    if "valid_time" in ds.coords:
        rename["valid_time"] = "time"
    if "lat" in ds.coords:
        rename["lat"] = "latitude"
    if "lon" in ds.coords:
        rename["lon"] = "longitude"
    if "pressure_level" in ds.coords:
        rename["pressure_level"] = "level"
    if rename:
        ds = ds.rename(rename)
    if "latitude" in ds.coords and "longitude" in ds.coords:
        lat_min = float(ds["latitude"].min())
        lat_max = float(ds["latitude"].max())
        lon_min = float(ds["longitude"].min())
        lon_max = float(ds["longitude"].max())
        lat_looks_like_lon = SAUDI_BBOX[1] - 5.0 <= lat_min <= SAUDI_BBOX[3] + 5.0 and lat_max <= SAUDI_BBOX[3] + 5.0
        lon_looks_like_lat = SAUDI_BBOX[0] - 5.0 <= lon_min <= SAUDI_BBOX[2] + 5.0 and lon_max <= SAUDI_BBOX[2] + 5.0
        if lat_looks_like_lon and lon_looks_like_lat:
            ds = ds.rename({"latitude": "_tmp_latitude", "longitude": "latitude"})
            ds = ds.rename({"_tmp_latitude": "longitude"})
    if "longitude" in ds.coords and float(ds["longitude"].min()) < 0:
        ds = ds.assign_coords(longitude=(ds["longitude"] % 360)).sortby("longitude")
    return ds


def _sanitize_values(da: xr.DataArray) -> xr.DataArray:
    if not np.issubdtype(da.dtype, np.number):
        return da
    cleaned = da.where(np.isfinite(da))
    cleaned = cleaned.where(np.abs(cleaned) < 1.0e19)
    for candidate in FILL_CANDIDATES:
        cleaned = cleaned.where(cleaned != candidate)
    return cleaned


def _mask_outside_range(da: xr.DataArray, min_value: float | None = None, max_value: float | None = None) -> xr.DataArray:
    if min_value is not None:
        da = da.where(da >= min_value)
    if max_value is not None:
        da = da.where(da <= max_value)
    return da


def _sanitize_dataset(ds: xr.Dataset) -> xr.Dataset:
    return xr.Dataset({name: _sanitize_values(da) for name, da in ds.data_vars.items()}, coords=ds.coords, attrs=ds.attrs)


def _coord_slice(coord: xr.DataArray, start: float, stop: float) -> slice:
    values = coord.values
    if len(values) == 0:
        return slice(start, stop)
    if float(values[0]) > float(values[-1]):
        return slice(stop, start)
    return slice(start, stop)


def _crop_saudi(ds: xr.Dataset) -> xr.Dataset:
    lat_slice = _coord_slice(ds["latitude"], SAUDI_BBOX[0], SAUDI_BBOX[2])
    lon_slice = _coord_slice(ds["longitude"], SAUDI_BBOX[1], SAUDI_BBOX[3])
    return ds.sel(latitude=lat_slice, longitude=lon_slice)


def _align_to_standard_grid(ds: xr.Dataset, method: str = "linear") -> xr.Dataset:
    ds = _normalize_dataset(ds)
    ds = _sanitize_dataset(ds)
    ds = _crop_saudi(ds)
    ds = ds.interp(latitude=_standard_latitudes(), longitude=_standard_longitudes(), method=method)
    if "time" in ds.coords:
        ds = ds.sortby("time")
    return ds.sortby("latitude").sortby("longitude")


def _align_to_aurora_grid(ds: xr.Dataset, method: str = "linear") -> xr.Dataset:
    ds = _normalize_dataset(ds)
    ds = _sanitize_dataset(ds)
    ds = _crop_saudi(ds)
    ds = ds.interp(latitude=_aurora_patch_latitudes().rename({"lat": "latitude"}), longitude=_aurora_patch_longitudes().rename({"lon": "longitude"}), method=method)
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    return ds.sortby("lat", ascending=False).sortby("lon")


def _to_celsius(da: xr.DataArray) -> xr.DataArray:
    units = str(da.attrs.get("units", "")).lower()
    if units in {"k", "kelvin"}:
        result = da - 273.15
    else:
        result = da
    result.attrs["units"] = "degC"
    return result


def _to_hpa(da: xr.DataArray) -> xr.DataArray:
    units = str(da.attrs.get("units", "")).lower()
    if units == "pa":
        result = da / 100.0
    else:
        result = da
    result.attrs["units"] = "hPa"
    return result


def _daily_sum_from_accum(da: xr.DataArray, name: str, units: str = "") -> xr.DataArray:
    diff = da.diff("time", label="upper")
    if diff.sizes.get("time", 0) == 0:
        empty = da.isel(time=slice(0, 0)).rename(name)
        return empty
    diff = diff.clip(min=0.0)
    valid_hours = diff["time"].dt.hour
    reset_values = da.sel(time=diff["time"]).clip(min=0.0)
    daily = diff.where(valid_hours != 0, reset_values).resample(time="1D").sum(skipna=True)
    daily = daily.clip(min=0.0)
    daily = daily.rename(name)
    if units:
        daily.attrs["units"] = units
    return daily


def _daily_mean(da: xr.DataArray, name: str, units: str = "") -> xr.DataArray:
    daily = da.resample(time="1D").mean(skipna=True).rename(name)
    if units:
        daily.attrs["units"] = units
    return daily


def _daily_max(da: xr.DataArray, name: str, units: str = "") -> xr.DataArray:
    daily = da.resample(time="1D").max(skipna=True).rename(name)
    if units:
        daily.attrs["units"] = units
    return daily


def _daily_min(da: xr.DataArray, name: str, units: str = "") -> xr.DataArray:
    daily = da.resample(time="1D").min(skipna=True).rename(name)
    if units:
        daily.attrs["units"] = units
    return daily


def _es_kpa(temp_c: xr.DataArray) -> xr.DataArray:
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def _rh_from_t_td(temp_c: xr.DataArray, dewpoint_c: xr.DataArray) -> xr.DataArray:
    rh = 100.0 * _es_kpa(dewpoint_c) / xr.where(_es_kpa(temp_c) <= 1.0e-6, 1.0e-6, _es_kpa(temp_c))
    return rh.clip(min=0.0, max=100.0).rename("rh2m")


def _heat_index_c(temp_c: xr.DataArray, rh_percent: xr.DataArray) -> xr.DataArray:
    t_f = temp_c * 9.0 / 5.0 + 32.0
    hi_f = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * rh_percent
        - 0.22475541 * t_f * rh_percent
        - 6.83783e-3 * t_f**2
        - 5.481717e-2 * rh_percent**2
        + 1.22874e-3 * t_f**2 * rh_percent
        + 8.5282e-4 * t_f * rh_percent**2
        - 1.99e-6 * t_f**2 * rh_percent**2
    )
    hi_f = xr.where(t_f < 80.0, t_f, hi_f)
    return ((hi_f - 32.0) * 5.0 / 9.0).rename("heat_index_c")


def _safe_ratio(numerator: xr.DataArray, denominator: xr.DataArray, min_denominator: float = MIN_RATIO_DENOMINATOR) -> xr.DataArray:
    return numerator / xr.where(np.abs(denominator) > min_denominator, denominator, np.nan)


def _apparent_temperature_c(temp_c: xr.DataArray, rh_percent: xr.DataArray, wind_speed: xr.DataArray) -> xr.DataArray:
    e = (rh_percent / 100.0) * 6.105 * np.exp((17.27 * temp_c) / (237.7 + temp_c))
    apparent = temp_c + 0.33 * e - 0.7 * wind_speed - 4.0
    return apparent.rename("apparent_temp_c")


def _wet_bulb_proxy_c(temp_c: xr.DataArray, rh_percent: xr.DataArray) -> xr.DataArray:
    rh = rh_percent.clip(min=0.0, max=100.0)
    tw = (
        temp_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(temp_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )
    return tw.rename("wet_bulb_proxy_c")


def _flag(da: xr.DataArray, threshold: float, op: str = ">=") -> xr.DataArray:
    if op == ">=":
        values = xr.where(da >= threshold, 1, 0)
    elif op == ">":
        values = xr.where(da > threshold, 1, 0)
    elif op == "<=":
        values = xr.where(da <= threshold, 1, 0)
    else:
        values = xr.where(da < threshold, 1, 0)
    return values.astype("int8")


def _coriolis_parameter(latitude: xr.DataArray) -> xr.DataArray:
    omega = 7.2921159e-5
    radians = np.deg2rad(latitude)
    return xr.DataArray(2.0 * omega * np.sin(radians), coords=latitude.coords, dims=latitude.dims)


def _gradient_spacing(values: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(values)
    return np.gradient(radians)


def _compute_slope_deg(elevation: xr.DataArray) -> xr.DataArray:
    lat = elevation["latitude"].values
    lon = elevation["longitude"].values
    lat_spacing = _gradient_spacing(lat) * 6371000.0
    lon_spacing = _gradient_spacing(lon) * 6371000.0 * np.cos(np.deg2rad(lat))[:, None]
    dz_dlat = np.gradient(elevation.values, axis=0) / lat_spacing[:, None]
    dz_dlon = np.gradient(elevation.values, axis=1) / lon_spacing
    slope = np.degrees(np.arctan(np.hypot(dz_dlat, dz_dlon)))
    result = xr.DataArray(slope.astype("float32"), coords=elevation.coords, dims=elevation.dims, name="slope_deg")
    result.attrs["units"] = "degree"
    return result


def _ivt_components(q: xr.DataArray, u: xr.DataArray, v: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    gravity = 9.80665
    q, u, v = xr.align(q, u, v, join="inner")
    pressure = q["level"]
    pressure_pa = xr.where(pressure.max(skipna=True) < 2000.0, pressure * 100.0, pressure)
    order = np.argsort(pressure_pa.values)
    q = q.isel(level=order).assign_coords(level=pressure_pa.isel(level=order))
    u = u.isel(level=order).assign_coords(level=pressure_pa.isel(level=order))
    v = v.isel(level=order).assign_coords(level=pressure_pa.isel(level=order))
    ivt_u = ((q * u).integrate("level") / gravity).rename("ivt_u")
    ivt_v = ((q * v).integrate("level") / gravity).rename("ivt_v")
    ivt = np.hypot(ivt_u, ivt_v).rename("ivt")
    for item in (ivt_u, ivt_v, ivt):
        item.attrs["units"] = "kg m-1 s-1"
    return ivt_u, ivt_v, ivt


def _pwat_mm(q: xr.DataArray) -> xr.DataArray:
    gravity = 9.80665
    pressure = q["level"]
    pressure_pa = xr.where(pressure.max(skipna=True) < 2000.0, pressure * 100.0, pressure)
    order = np.argsort(pressure_pa.values)
    q_sorted = q.isel(level=order).assign_coords(level=pressure_pa.isel(level=order))
    pwat = (q_sorted.integrate("level") / gravity).clip(min=0.0).rename("pwat")
    pwat.attrs["units"] = "mm"
    return pwat


def _horizontal_gradients(da: xr.DataArray, lat_name: str = "latitude", lon_name: str = "longitude") -> tuple[xr.DataArray, xr.DataArray]:
    lat = np.deg2rad(da[lat_name].values)
    lon = np.deg2rad(da[lon_name].values)
    earth_radius = 6371000.0
    dy = np.gradient(lat) * earth_radius
    dx = np.gradient(lon) * earth_radius * np.cos(lat)[:, None]
    grad_y = np.gradient(da.values, axis=da.get_axis_num(lat_name)) / dy[:, None]
    grad_x = np.gradient(da.values, axis=da.get_axis_num(lon_name)) / dx
    dims = da.dims
    coords = da.coords
    return (
        xr.DataArray(grad_y.astype("float32"), coords=coords, dims=dims),
        xr.DataArray(grad_x.astype("float32"), coords=coords, dims=dims),
    )


def _select_level(da: xr.DataArray, level: int) -> xr.DataArray:
    return da.sel(level=level).drop_vars("level", errors="ignore")


def _era5_hourly_precip_mm(tp_daily_total: xr.DataArray) -> xr.DataArray:
    result = tp_daily_total.copy()
    units = str(result.attrs.get("units", "")).lower()
    if units in {"m", "meter", "meters", "metre", "metres"}:
        result = result * 1000.0
    else:
        result = result * 1000.0
    result.attrs["units"] = "mm"
    return result


def _as_day_start(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


def _promote_2d_to_time(da: xr.DataArray, time_coord: xr.DataArray) -> xr.DataArray:
    if "time" in da.dims:
        return da
    return da.expand_dims(time=time_coord)


@dataclass
class BuildManifestEntry:
    kind: str
    identifier: str
    status: str
    detail: str = ""


@dataclass
class BuildResult:
    entries: list[BuildManifestEntry] = field(default_factory=list)

    def add(self, kind: str, identifier: str, status: str, detail: str = "") -> None:
        self.entries.append(BuildManifestEntry(kind=kind, identifier=identifier, status=status, detail=detail))

    def write(self, path: Path) -> None:
        payload = [
            {"kind": entry.kind, "identifier": entry.identifier, "status": entry.status, "detail": entry.detail}
            for entry in self.entries
        ]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class RawInputBuilder:
    raw_root: Path
    aurora_out: Path
    indicator_nc_out: Path
    indicator_parquet_out: Path
    aurora_cadence_hours: int = AURORA_CADENCE_HOURS

    def __post_init__(self) -> None:
        self.single_dir = self.raw_root / "era5_single_levels_2025"
        self.pressure_dir = self.raw_root / "era5_pressure_levels_2025"
        self.precip_dir = self.raw_root / "precip"
        self.dust_dir = self.raw_root / "dust"
        self.sst_dir = self.raw_root / "sst"
        self.nis_path = self.raw_root.parent / "output" / "nis" / "nis_elevation_grid.nc"
        self._single_cache: dict[tuple[int, int, str], xr.Dataset] = {}
        self._pressure_cache: dict[tuple[int, int, str], xr.Dataset] = {}
        self._dust_cache: dict[date, xr.Dataset] = {}
        self._gpm_cache: dict[date, xr.Dataset] = {}
        self._sst_cache: dict[str, xr.Dataset] = {}
        self._chirps_cache: xr.Dataset | None = None
        self._static_cache: xr.Dataset | None = None
        self._elevation_cache: xr.Dataset | None = None
        self._daily_precip_cache: dict[date, xr.DataArray] = {}

    def close(self) -> None:
        for dataset in self._single_cache.values():
            dataset.close()
        for dataset in self._pressure_cache.values():
            dataset.close()
        for dataset in self._dust_cache.values():
            dataset.close()
        for dataset in self._gpm_cache.values():
            dataset.close()
        for dataset in self._sst_cache.values():
            dataset.close()
        if self._chirps_cache is not None:
            self._chirps_cache.close()
        if self._static_cache is not None:
            self._static_cache.close()
        if self._elevation_cache is not None:
            self._elevation_cache.close()

    def build(self, start_date: date, end_date: date) -> BuildResult:
        result = BuildResult()
        self.aurora_out.mkdir(parents=True, exist_ok=True)
        self.indicator_nc_out.mkdir(parents=True, exist_ok=True)
        self.indicator_parquet_out.mkdir(parents=True, exist_ok=True)

        indicator_frames: list[Any] = []
        current = start_date
        while current <= end_date:
            try:
                daily = self.build_daily_indicators(current)
                nc_path = self.indicator_nc_out / f"saudi_indicators_{current:%Y%m%d}.nc"
                daily.to_netcdf(nc_path)
                result.add("indicator_nc", current.isoformat(), "ok", str(nc_path))
                indicator_frames.append(self._daily_to_frame(daily, current))
            except Exception as exc:
                result.add("indicator_nc", current.isoformat(), "error", str(exc))
            current += timedelta(days=1)

        if indicator_frames:
            table = self._concat_frames(indicator_frames)
            table_path = self.indicator_parquet_out / f"saudi_indicator_samples_{start_date.year}.parquet"
            self._write_parquet(table, table_path)
            result.add("indicator_table", str(start_date.year), "ok", str(table_path))

        aurora_time = _as_day_start(start_date)
        final_time = _as_day_start(end_date) + timedelta(hours=18)
        while aurora_time <= final_time:
            try:
                ds = self.build_aurora_input(aurora_time)
                path = self.aurora_out / f"aurora_input_{aurora_time:%Y%m%d%H}.nc"
                ds.to_netcdf(path)
                result.add("aurora", aurora_time.isoformat(), "ok", str(path))
            except Exception as exc:
                result.add("aurora", aurora_time.isoformat(), "skipped", str(exc))
            aurora_time += timedelta(hours=self.aurora_cadence_hours)

        result.write(self.aurora_out.parent / "build_manifest.json")
        return result

    def build_aurora_input(self, issue_time: datetime) -> xr.Dataset:
        valid_utc = issue_time.astimezone(timezone.utc).replace(tzinfo=None)
        history = [valid_utc - timedelta(hours=6), valid_utc]
        history_index = np.array(history, dtype="datetime64[ns]")
        start = history_index.min()
        end = history_index.max()
        instant = self._single_member_slice(valid_utc.year, valid_utc.month, "data_stream-oper_stepType-instant.nc", start, end)
        aurora_surface = self._aurora_single_month_slice(valid_utc.year, valid_utc.month, start, end)
        single = xr.merge(
            [
                _align_to_aurora_grid(instant[["t2m", "u10", "v10"]]),
                _align_to_aurora_grid(aurora_surface[["msl"]]),
            ],
            compat="override",
        )
        if not set(history_index).issubset(set(single["time"].values)):
            raise ValueError("missing required single-level history times")
        pressure_parts = []
        for short, long_name in {
            "u": "u_component_of_wind",
            "v": "v_component_of_wind",
            "z": "geopotential",
            "t": "temperature",
            "q": "specific_humidity",
        }.items():
            ds = self._pressure_month_slice(valid_utc.year, valid_utc.month, long_name, start, end)[[short]]
            pressure_parts.append(_align_to_aurora_grid(ds, method="linear"))
        pressure = xr.merge(pressure_parts, compat="override")
        if not set(history_index).issubset(set(pressure["time"].values)):
            raise ValueError("missing required pressure-level history times")

        surf = {
            "surf_2t": single["t2m"].sel(time=history_index),
            "surf_10u": single["u10"].sel(time=history_index),
            "surf_10v": single["v10"].sel(time=history_index),
            "surf_msl": single["msl"].sel(time=history_index),
        }
        static = self._aurora_static()
        atmos = {
            "atmos_z": pressure["z"].sel(time=history_index, level=list(PRESSURE_LEVELS)),
            "atmos_u": pressure["u"].sel(time=history_index, level=list(PRESSURE_LEVELS)),
            "atmos_v": pressure["v"].sel(time=history_index, level=list(PRESSURE_LEVELS)),
            "atmos_t": pressure["t"].sel(time=history_index, level=list(PRESSURE_LEVELS)),
            "atmos_q": pressure["q"].sel(time=history_index, level=list(PRESSURE_LEVELS)),
        }
        data_vars: dict[str, Any] = {}
        for name, da in surf.items():
            data_vars[name] = (("time", "lat", "lon"), da.values.astype("float32"))
        for source_name, output_name in (("lsm", "static_lsm"), ("z", "static_z"), ("slt", "static_slt")):
            data_vars[output_name] = (("lat", "lon"), static[source_name].values.astype("float32"))
        for name, da in atmos.items():
            data_vars[name] = (("time", "level", "lat", "lon"), da.values.astype("float32"))
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": history_index,
                "level": list(PRESSURE_LEVELS),
                "lat": surf["surf_2t"]["lat"].values,
                "lon": surf["surf_2t"]["lon"].values,
            },
            attrs={
                "title": "Aurora direct input file",
                "issue_time": issue_time.isoformat(),
                "source": "ERA5 single levels + pressure levels from data/raw",
                "history_hours": 6,
            },
        )
        return ds

    def build_daily_indicators(self, day: date) -> xr.Dataset:
        single = self._daily_single_level_fields(day)
        pressure = self._daily_pressure_level_fields(day)
        gpm = self._daily_gpm(day)
        dust = self._daily_dust(day)
        sst = self._daily_sst(day)
        elevation = self._elevation()
        chirps_monthly = self._monthly_chirps(day)

        t2m_hourly = _mask_outside_range(_to_celsius(single["t2m"]).rename("t2m_c"), -40.0, 65.0)
        d2m_hourly = _mask_outside_range(_to_celsius(single["d2m"]).rename("d2m_c"), -60.0, 45.0)
        rh_hourly = _mask_outside_range(_rh_from_t_td(t2m_hourly, d2m_hourly), 0.0, 100.0)
        vpd_hourly = _mask_outside_range((_es_kpa(t2m_hourly) * (1.0 - rh_hourly / 100.0)).rename("vpd_kpa"), 0.0, 15.0)
        heat_index_hourly = _heat_index_c(t2m_hourly, rh_hourly)
        wind10_hourly = _mask_outside_range(np.hypot(single["u10"], single["v10"]).rename("wind10_speed"), 0.0, 80.0)
        wind10_hourly.attrs["units"] = "m s-1"
        dewpoint_depression_hourly = (t2m_hourly - d2m_hourly).rename("dewpoint_depression_c")
        apparent_temp_hourly = _mask_outside_range(_apparent_temperature_c(t2m_hourly, rh_hourly, wind10_hourly), -40.0, 75.0)
        wet_bulb_hourly = _mask_outside_range(_wet_bulb_proxy_c(t2m_hourly, rh_hourly), -40.0, 45.0)

        t2m_c = _daily_mean(t2m_hourly, "t2m_c", "degC")
        d2m_c = _daily_mean(d2m_hourly, "d2m_c", "degC")
        rh2m = _daily_mean(rh_hourly, "rh2m", "%")
        vpd = _daily_mean(vpd_hourly, "vpd_kpa", "kPa")
        heat_index = _daily_mean(heat_index_hourly, "heat_index_c", "degC")
        wind10 = _daily_mean(wind10_hourly, "wind10_speed", "m s-1")
        dewpoint_depression = _daily_mean(dewpoint_depression_hourly, "dewpoint_depression_c", "degC")
        apparent_temp = _daily_mean(apparent_temp_hourly, "apparent_temp_c", "degC")
        wet_bulb = _daily_mean(wet_bulb_hourly, "wet_bulb_proxy_c", "degC")

        tp_mm = _era5_hourly_precip_mm(single["tp"])
        cp_mm = _era5_hourly_precip_mm(single["cp"])
        daily_total = _daily_sum_from_accum(tp_mm, "daily_precip_total", "mm")
        convective_total = _daily_sum_from_accum(cp_mm, "daily_convective_precip", "mm")
        nonconvective = (daily_total - convective_total).clip(min=0.0).rename("daily_large_scale_precip")
        conv_ratio = _safe_ratio(convective_total, daily_total.where(daily_total > 0.1)).clip(min=0.0, max=1.0).rename("convective_precip_ratio")

        sw_net = (_daily_mean(single["ssrd"], "ssrd", "J m-2") - 0).rename("sw_net")
        lw_net = (_daily_mean(single["strd"], "strd", "J m-2") - 0).rename("lw_net")
        net_radiation = (sw_net + lw_net).rename("net_radiation")
        sensible_flux = _daily_mean(single["sshf"], "sshf")
        latent_flux = _daily_mean(single["slhf"], "slhf")
        bowen = _safe_ratio(sensible_flux, latent_flux).rename("bowen_ratio")
        radiative_heat_load = (net_radiation + sw_net).rename("radiative_heat_load")

        tmax = _to_celsius(_daily_max(single["mx2t"], "tmax_c"))
        tmin = _to_celsius(_daily_min(single["mn2t"], "tmin_c"))
        dtr = (tmax - tmin).rename("diurnal_temp_range_c")
        hot_day_flag = _flag(tmax, 45.0).rename("hot_day_flag")
        hot_night_flag = _flag(tmin, 30.0).rename("hot_night_flag")
        compound_heat_flag = (_flag(heat_index, 41.0) * _flag(vpd, 3.0)).astype("int8").rename("compound_heat_flag")

        cape = _daily_max(single["cape"], "cape", "J kg-1")
        cin = _daily_min(single["cin"], "cin", "J kg-1")
        tcc = (_daily_mean(single["tcc"], "total_cloud_cover") * 100.0).rename("total_cloud_cover")
        lcc = (_daily_mean(single["lcc"], "low_cloud_cover") * 100.0).rename("low_cloud_cover")
        mcc = (_daily_mean(single["mcc"], "middle_cloud_cover") * 100.0).rename("middle_cloud_cover")
        hcc = (_daily_mean(single["hcc"], "high_cloud_cover") * 100.0).rename("high_cloud_cover")
        surface_pressure = _daily_mean(single["sp"], "surface_pressure", "Pa")

        q = pressure["q"]
        u = pressure["u"]
        v = pressure["v"]
        z = pressure["z"]
        w = pressure["w"]
        d = pressure["d"]
        rh_pl = pressure["r"]
        ivt_u, ivt_v, ivt = _ivt_components(q, u, v)
        wind925 = np.hypot(_select_level(u, 925), _select_level(v, 925)).rename("wind925_speed")
        wind850 = np.hypot(_select_level(u, 850), _select_level(v, 850)).rename("wind850_speed")
        jet300 = np.hypot(_select_level(u, 300), _select_level(v, 300)).rename("jet300_speed")
        jet200 = np.hypot(_select_level(u, 200), _select_level(v, 200)).rename("jet200_speed")
        moisture925 = (_select_level(q, 925) * wind925).rename("moisture_transport925")
        moisture850 = (_select_level(q, 850) * wind850).rename("moisture_transport850")
        shear_850_300 = np.hypot(_select_level(u, 300) - _select_level(u, 850), _select_level(v, 300) - _select_level(v, 850)).rename("wind_shear_850_300")
        shear_850_200 = np.hypot(_select_level(u, 200) - _select_level(u, 850), _select_level(v, 200) - _select_level(v, 850)).rename("wind_shear_850_200")
        vort850_y, vort850_x = _horizontal_gradients(_select_level(v, 850))
        du850_y, du850_x = _horizontal_gradients(_select_level(u, 850))
        relative_vorticity850 = (vort850_x - du850_y).rename("relative_vorticity850")
        divergence850 = (_select_level(d, 850)).rename("divergence850")
        omega700 = _select_level(w, 700).rename("omega700")
        omega500 = _select_level(w, 500).rename("omega500")
        geopotential_height500 = (_select_level(z, 500) / 9.80665).rename("geopotential_height500")
        pwat = _pwat_mm(q)
        ivt_div_y, ivt_div_x = _horizontal_gradients(ivt_v)
        ivt_u_y, ivt_u_x = _horizontal_gradients(ivt_u)
        ivt_divergence = (ivt_u_x + ivt_div_y).rename("ivt_divergence")
        ivt_convergence = (-ivt_divergence).rename("ivt_convergence")
        f850 = _coriolis_parameter(relative_vorticity850["latitude"]).broadcast_like(relative_vorticity850)
        absolute_vorticity850 = (relative_vorticity850 + f850).rename("absolute_vorticity850")
        low_level_jet_flag = _flag(wind850, 12.0).rename("low_level_jet_flag")
        strong_shear_flag = _flag(shear_850_200, 20.0).rename("strong_shear_flag")

        sst_da = sst.rename("sst_celsius")
        dust_aod = dust["DUEXTTAU"].rename("dust_aod")
        dust_column = dust["DUCMASS"].rename("dust_column_mass")
        dust_surface = dust["DUSMASS"].rename("dust_surface_mass")
        dust_risk = (
            xr.where(wind10.isel(time=0, drop=True) >= 8.0, 0.4, 0.0)
            + xr.where(dust_aod >= 0.3, 0.3, 0.0)
            + xr.where(dust_surface >= 5.0e-4, 0.3, 0.0)
        ).clip(max=1.0).rename("dust_risk_proxy")
        strong_wind_flag = _flag(wind10.isel(time=0, drop=True), 10.0).rename("strong_wind_flag")
        dust_emission_flag = (_flag(wind10.isel(time=0, drop=True), 8.0) * _flag(dust_surface, 5.0e-4)).astype("int8").rename("dust_emission_flag")

        gpm_daily = gpm["precipitation"].rename("gpm_daily_precip")
        gpm_diff = (gpm_daily - daily_total.isel(time=0, drop=True)).rename("gpm_era5_precip_diff")
        gpm_ratio = _safe_ratio(gpm_daily, daily_total.isel(time=0, drop=True).where(daily_total.isel(time=0, drop=True) > 0.1)).rename("gpm_era5_precip_ratio")
        heavy_rain_flag = _flag(daily_total.isel(time=0, drop=True), 25.0).rename("heavy_rain_flag")
        extreme_rain_flag = _flag(daily_total.isel(time=0, drop=True), 50.0).rename("extreme_rain_flag")
        gpm_overlap = (_flag(gpm_daily, 10.0) * _flag(daily_total.isel(time=0, drop=True), 10.0)).astype("int8").rename("gpm_era5_heavy_rain_overlap")
        precip_3day = self._daily_precip_window_total(day, 3, current_day_total=daily_total.isel(time=0, drop=True)).rename("precip_3day")
        precip_7day = self._daily_precip_window_total(day, 7, current_day_total=daily_total.isel(time=0, drop=True)).rename("precip_7day")

        elev = elevation["elevation_m"].rename("orography")
        slope = _compute_slope_deg(elev)

        flash = xr.apply_ufunc(
            np.vectorize(compute_flash_flood_screening_score),
            xr.full_like(gpm_daily, np.nan) + daily_total.isel(time=0, drop=True),
            xr.full_like(gpm_daily, np.nan) + daily_total.isel(time=0, drop=True),
            xr.full_like(gpm_daily, np.nan) + gpm_daily,
            slope,
            xr.zeros_like(gpm_daily) + 0.15,
            xr.zeros_like(gpm_daily) + 0.05,
        ).rename("flash_flood_risk")
        monthly_chirps_precip_total = None
        monthly_chirps_precip_mmday = None
        if chirps_monthly is not None:
            days_in_month = float((date(day.year + (day.month == 12), 1 if day.month == 12 else day.month + 1, 1) - date(day.year, day.month, 1)).days)
            monthly_chirps_precip_total = chirps_monthly.rename("monthly_chirps_precip_total")
            monthly_chirps_precip_mmday = (chirps_monthly / days_in_month).rename("monthly_chirps_precip_mmday")
        daily_time = daily_total["time"]
        dust_aod = _promote_2d_to_time(dust_aod, daily_time)
        dust_column = _promote_2d_to_time(dust_column, daily_time)
        dust_surface = _promote_2d_to_time(dust_surface, daily_time)
        dust_risk = _promote_2d_to_time(dust_risk, daily_time)
        gpm_daily = _promote_2d_to_time(gpm_daily, daily_time)
        gpm_diff = _promote_2d_to_time(gpm_diff, daily_time)
        gpm_ratio = _promote_2d_to_time(gpm_ratio, daily_time)
        elev = _promote_2d_to_time(elev, daily_time)
        slope = _promote_2d_to_time(slope, daily_time)
        sst_da = _promote_2d_to_time(sst_da, daily_time)
        flash = _promote_2d_to_time(flash, daily_time)
        strong_wind_flag = _promote_2d_to_time(strong_wind_flag, daily_time)
        dust_emission_flag = _promote_2d_to_time(dust_emission_flag, daily_time)
        heavy_rain_flag = _promote_2d_to_time(heavy_rain_flag, daily_time)
        extreme_rain_flag = _promote_2d_to_time(extreme_rain_flag, daily_time)
        gpm_overlap = _promote_2d_to_time(gpm_overlap, daily_time)
        precip_3day = _promote_2d_to_time(precip_3day, daily_time)
        precip_7day = _promote_2d_to_time(precip_7day, daily_time)
        if monthly_chirps_precip_total is not None and monthly_chirps_precip_mmday is not None:
            monthly_chirps_precip_total = _promote_2d_to_time(monthly_chirps_precip_total, daily_time)
            monthly_chirps_precip_mmday = _promote_2d_to_time(monthly_chirps_precip_mmday, daily_time)

        grid_fields = xr.Dataset(
            {
                "t2m_c": t2m_c.isel(time=0, drop=True),
                "d2m_c": d2m_c.isel(time=0, drop=True),
                "rh2m": rh2m.isel(time=0, drop=True),
                "dewpoint_depression_c": dewpoint_depression.isel(time=0, drop=True),
                "vpd_kpa": vpd.isel(time=0, drop=True),
                "heat_index_c": heat_index.isel(time=0, drop=True),
                "apparent_temp_c": apparent_temp.isel(time=0, drop=True),
                "wet_bulb_proxy_c": wet_bulb.isel(time=0, drop=True),
                "wind10_speed": wind10.isel(time=0, drop=True),
                "tmax_c": tmax.isel(time=0, drop=True),
                "tmin_c": tmin.isel(time=0, drop=True),
                "diurnal_temp_range_c": dtr.isel(time=0, drop=True),
                "hot_day_flag": hot_day_flag.isel(time=0, drop=True),
                "hot_night_flag": hot_night_flag.isel(time=0, drop=True),
                "compound_heat_flag": compound_heat_flag.isel(time=0, drop=True),
                "total_cloud_cover": tcc.isel(time=0, drop=True),
                "low_cloud_cover": lcc.isel(time=0, drop=True),
                "middle_cloud_cover": mcc.isel(time=0, drop=True),
                "high_cloud_cover": hcc.isel(time=0, drop=True),
                "daily_precip_total": daily_total.isel(time=0, drop=True),
                "daily_convective_precip": convective_total.isel(time=0, drop=True),
                "daily_large_scale_precip": nonconvective.isel(time=0, drop=True),
                "convective_precip_ratio": conv_ratio.isel(time=0, drop=True),
                "precip_3day": precip_3day,
                "precip_7day": precip_7day,
                "heavy_rain_flag": heavy_rain_flag,
                "extreme_rain_flag": extreme_rain_flag,
                "sw_net": sw_net.isel(time=0, drop=True),
                "lw_net": lw_net.isel(time=0, drop=True),
                "net_radiation": net_radiation.isel(time=0, drop=True),
                "bowen_ratio": bowen.isel(time=0, drop=True),
                "radiative_heat_load": radiative_heat_load.isel(time=0, drop=True),
                "cape": cape.isel(time=0, drop=True),
                "cin": cin.isel(time=0, drop=True),
                "surface_pressure": surface_pressure.isel(time=0, drop=True),
                "pwat": pwat.isel(time=0, drop=True),
                "ivt_u": ivt_u.isel(time=0, drop=True),
                "ivt_v": ivt_v.isel(time=0, drop=True),
                "ivt": ivt.isel(time=0, drop=True),
                "ivt_divergence": ivt_divergence.isel(time=0, drop=True),
                "ivt_convergence": ivt_convergence.isel(time=0, drop=True),
                "wind925_speed": wind925.isel(time=0, drop=True),
                "wind850_speed": wind850.isel(time=0, drop=True),
                "moisture_transport925": moisture925.isel(time=0, drop=True),
                "moisture_transport850": moisture850.isel(time=0, drop=True),
                "jet300_speed": jet300.isel(time=0, drop=True),
                "jet200_speed": jet200.isel(time=0, drop=True),
                "wind_shear_850_300": shear_850_300.isel(time=0, drop=True),
                "wind_shear_850_200": shear_850_200.isel(time=0, drop=True),
                "relative_vorticity850": relative_vorticity850.isel(time=0, drop=True),
                "absolute_vorticity850": absolute_vorticity850.isel(time=0, drop=True),
                "divergence850": divergence850.isel(time=0, drop=True),
                "omega700": omega700.isel(time=0, drop=True),
                "omega500": omega500.isel(time=0, drop=True),
                "geopotential_height500": geopotential_height500.isel(time=0, drop=True),
                "relative_humidity850": _select_level(rh_pl, 850).isel(time=0, drop=True),
                "low_level_jet_flag": low_level_jet_flag.isel(time=0, drop=True),
                "strong_shear_flag": strong_shear_flag.isel(time=0, drop=True),
                "sst_celsius": sst_da,
                "dust_aod": dust_aod,
                "dust_column_mass": dust_column,
                "dust_surface_mass": dust_surface,
                "dust_risk_proxy": dust_risk,
                "strong_wind_flag": strong_wind_flag,
                "dust_emission_flag": dust_emission_flag,
                "gpm_daily_precip": gpm_daily,
                "gpm_era5_precip_diff": gpm_diff,
                "gpm_era5_precip_ratio": gpm_ratio,
                "gpm_era5_heavy_rain_overlap": gpm_overlap,
                "orography": elev,
                "slope_deg": slope,
                "flash_flood_risk": flash,
            },
            coords={"latitude": t2m_c["latitude"], "longitude": t2m_c["longitude"]},
            attrs={
                "title": "Saudi daily multi-hazard indicators",
                "valid_date": day.isoformat(),
                "source": "ERA5 + GPM + MERRA2 + OISST + SRTM",
                "missing_indicator_groups": ", ".join(MISSING_INDICATOR_GROUPS),
            },
        )
        if monthly_chirps_precip_total is not None and monthly_chirps_precip_mmday is not None:
            grid_fields["monthly_chirps_precip_total"] = monthly_chirps_precip_total
            grid_fields["monthly_chirps_precip_mmday"] = monthly_chirps_precip_mmday
        return grid_fields

    def _aurora_single_levels_for_month(self, year: int, month: int) -> xr.Dataset:
        start = np.datetime64(datetime(year, month, 1))
        end = start + np.timedelta64(32, "D")
        instant = _align_to_aurora_grid(self._single_member_slice(year, month, "data_stream-oper_stepType-instant.nc", start, end)[["t2m", "u10", "v10"]])
        static = _align_to_aurora_grid(self._aurora_single_month(year, month)[["msl"]])
        merged = xr.merge([instant, static], compat="override", join="outer")
        return merged

    def _aurora_pressure_levels_for_month(self, year: int, month: int) -> xr.Dataset:
        names = {
            "u": "u_component_of_wind",
            "v": "v_component_of_wind",
            "z": "geopotential",
            "t": "temperature",
            "q": "specific_humidity",
        }
        parts = []
        for short, long_name in names.items():
            ds = self._pressure_month(year, month, long_name)[[short]]
            parts.append(_align_to_aurora_grid(ds, method="linear"))
        return xr.merge(parts, compat="override")

    def _aurora_static(self) -> xr.Dataset:
        if self._static_cache is None:
            aurora_candidate = self.single_dir / "era5_single_levels_2025_01_aurora.nc"
            static_candidate = self.single_dir / "era5_single_levels_2025_01_static.nc"
            datasets = []
            for path in (aurora_candidate, static_candidate):
                if path.exists():
                    datasets.append(_align_to_aurora_grid(xr.open_dataset(path, engine="netcdf4"), method="nearest"))
            surface_geopotential = _align_to_aurora_grid(
                self._single_member_slice(2025, 1, "data_stream-oper_stepType-instant.nc", np.datetime64("2025-01-01"), np.datetime64("2025-02-01"))[["z"]],
                method="nearest",
            )
            datasets.append(surface_geopotential)
            if not datasets:
                raise FileNotFoundError("missing ERA5 Aurora/static single-level files")
            self._static_cache = xr.merge(datasets, compat="override", join="outer")
        ds = self._static_cache
        if "time" in ds.dims:
            return ds.isel(time=-1, drop=True)
        return ds.isel(valid_time=-1, drop=True) if "valid_time" in ds.dims else ds

    def _daily_single_level_fields(self, day: date) -> xr.Dataset:
        start = np.datetime64(datetime.combine(day, time.min))
        end = start + np.timedelta64(1, "D") - np.timedelta64(1, "ns")
        ds = xr.merge(
            [
                _align_to_standard_grid(self._single_member_slice(day.year, day.month, "data_stream-oper_stepType-instant.nc", start, end)),
                _align_to_standard_grid(self._single_member_slice(day.year, day.month, "data_stream-oper_stepType-accum.nc", start, end)),
                _align_to_standard_grid(self._single_member_slice(day.year, day.month, "data_stream-oper_stepType-max.nc", start, end)),
            ],
            compat="override",
            join="outer",
        )
        subset = ds.sel(time=slice(start, end))
        if subset.sizes.get("time", 0) == 0:
            raise ValueError("no single-level data for requested day")
        return subset

    def _daily_pressure_level_fields(self, day: date) -> xr.Dataset:
        start = np.datetime64(datetime.combine(day, time.min))
        end = start + np.timedelta64(1, "D") - np.timedelta64(1, "ns")
        names = (
            "specific_humidity",
            "u_component_of_wind",
            "v_component_of_wind",
            "geopotential",
            "temperature",
            "vertical_velocity",
            "divergence",
            "relative_vorticity",
        )
        parts = []
        for long_name in names:
            ds = self._pressure_month_slice(day.year, day.month, long_name, start, end)
            parts.append(_align_to_standard_grid(ds, method="linear"))
        merged = xr.merge(parts, compat="override", join="outer")
        subset = merged.sel(time=slice(start, end))
        if subset.sizes.get("time", 0) == 0:
            raise ValueError("no pressure-level data for requested day")
        return subset

    def _daily_gpm(self, day: date) -> xr.Dataset:
        if day not in self._gpm_cache:
            path = self.precip_dir / f"GPM_3IMERGDF_{day:%Y%m%d}.nc4"
            if not path.exists():
                raise FileNotFoundError(path)
            ds = xr.open_dataset(path, engine="netcdf4")
            ds = ds.rename({"lat": "latitude", "lon": "longitude"})
            if "longitude" in ds.coords and "latitude" in ds.coords:
                lon_min = float(ds["longitude"].min())
                lon_max = float(ds["longitude"].max())
                lat_min = float(ds["latitude"].min())
                lat_max = float(ds["latitude"].max())
                if lon_max <= SAUDI_BBOX[2] + 5.0 and lat_min >= SAUDI_BBOX[1] - 5.0:
                    ds = ds.rename({"longitude": "_tmp_longitude", "latitude": "longitude"})
                    ds = ds.rename({"_tmp_longitude": "latitude"})
            ds = _align_to_standard_grid(ds, method="nearest")
            if "time" in ds.dims:
                ds = ds.isel(time=0, drop=True)
            self._gpm_cache[day] = ds
        return self._gpm_cache[day]

    def _daily_dust(self, day: date) -> xr.Dataset:
        if day not in self._dust_cache:
            path = self.dust_dir / f"MERRA2_{day:%Y%m%d}.nc4"
            if not path.exists():
                raise FileNotFoundError(path)
            ds = xr.open_dataset(path, engine="netcdf4")
            ds = _align_to_standard_grid(ds, method="linear")
            ds = ds.resample(time="1D").mean(skipna=True).isel(time=0, drop=True)
            self._dust_cache[day] = ds
        return self._dust_cache[day]

    def _daily_sst(self, day: date) -> xr.DataArray:
        key = "oisst.day.mean.2025.nc"
        if key not in self._sst_cache:
            path = self.sst_dir / key
            if not path.exists():
                path = self.sst_dir / "jplMURSST41_72e4_84a7_e7ac.nc"
            ds = xr.open_dataset(path, engine="netcdf4")
            self._sst_cache[key] = _align_to_standard_grid(ds, method="nearest")
        ds = self._sst_cache[key]
        da_name = "sst" if "sst" in ds.data_vars else "analysed_sst"
        selected = ds[da_name].sel(time=np.datetime64(day.isoformat()), method="nearest").drop_vars("time", errors="ignore")
        selected.attrs["units"] = "degC"
        return selected

    def _monthly_chirps(self, day: date) -> xr.DataArray | None:
        path = self.precip_dir / "chirps-v3.0.2025.monthly.nc"
        if not path.exists():
            return None
        if self._chirps_cache is None:
            self._chirps_cache = _align_to_standard_grid(xr.open_dataset(path, engine="netcdf4"), method="nearest")
        ds = self._chirps_cache
        var_name = "precip" if "precip" in ds.data_vars else list(ds.data_vars)[0]
        selected = ds[var_name].sel(time=np.datetime64(day.replace(day=1).isoformat()), method="nearest").drop_vars("time", errors="ignore")
        selected.attrs["units"] = "mm/month"
        return selected

    def _elevation(self) -> xr.Dataset:
        if self._elevation_cache is None:
            if not self.nis_path.exists():
                raise FileNotFoundError(self.nis_path)
            self._elevation_cache = _align_to_standard_grid(xr.open_dataset(self.nis_path, engine="netcdf4"), method="nearest")
        return self._elevation_cache

    def _daily_precip_total_for_day(self, day: date) -> xr.DataArray:
        if day in self._daily_precip_cache:
            return self._daily_precip_cache[day]
        single = self._daily_single_level_fields(day)
        total = _daily_sum_from_accum(_era5_hourly_precip_mm(single["tp"]), "daily_precip_total", "mm")
        total_2d = total.isel(time=0, drop=True)
        self._daily_precip_cache[day] = total_2d
        return total_2d

    def _daily_precip_window_total(self, day: date, window_days: int, current_day_total: xr.DataArray | None = None) -> xr.DataArray:
        totals = []
        for offset in range(window_days):
            target_day = day - timedelta(days=offset)
            try:
                if offset == 0 and current_day_total is not None:
                    totals.append(current_day_total)
                else:
                    totals.append(self._daily_precip_total_for_day(target_day))
            except Exception:
                continue
        if not totals:
            template = current_day_total if current_day_total is not None else self._daily_precip_total_for_day(day)
            return xr.full_like(template, np.nan)
        return xr.concat(totals, dim="rolling_day").sum("rolling_day", skipna=True)

    def _single_member(self, year: int, month: int, member_name: str) -> xr.Dataset:
        key = (year, month, member_name)
        if key in self._single_cache:
            return self._single_cache[key]
        path = self.single_dir / f"era5_single_levels_{year}_{month:02d}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / member_name
            with zipfile.ZipFile(path) as archive:
                target.write_bytes(archive.read(member_name))
            ds = xr.open_dataset(target, engine="netcdf4").load()
        self._single_cache[key] = _normalize_dataset(ds)
        return self._single_cache[key]

    def _single_member_slice(
        self,
        year: int,
        month: int,
        member_name: str,
        start: np.datetime64,
        end: np.datetime64,
    ) -> xr.Dataset:
        path = self.single_dir / f"era5_single_levels_{year}_{month:02d}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / member_name
            with zipfile.ZipFile(path) as archive:
                target.write_bytes(archive.read(member_name))
            with xr.open_dataset(target, engine="netcdf4") as ds:
                normalized = _normalize_dataset(ds)
                subset = normalized.sel(time=slice(start, end)).load()
        return subset

    def _aurora_single_month(self, year: int, month: int) -> xr.Dataset:
        path = self.single_dir / f"era5_single_levels_{year}_{month:02d}_aurora.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        return _normalize_dataset(xr.open_dataset(path, engine="netcdf4").load())

    def _aurora_single_month_slice(self, year: int, month: int, start: np.datetime64, end: np.datetime64) -> xr.Dataset:
        path = self.single_dir / f"era5_single_levels_{year}_{month:02d}_aurora.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        with xr.open_dataset(path, engine="netcdf4") as ds:
            normalized = _normalize_dataset(ds)
            subset = normalized.sel(time=slice(start, end)).load()
        return subset

    def _pressure_month(self, year: int, month: int, long_name: str) -> xr.Dataset:
        key = (year, month, long_name)
        if key in self._pressure_cache:
            return self._pressure_cache[key]
        path = self.pressure_dir / f"era5_pl_{year}_{month:02d}_{long_name}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        ds = xr.open_dataset(path, engine="netcdf4").load()
        self._pressure_cache[key] = _normalize_dataset(ds)
        return self._pressure_cache[key]

    def _pressure_month_slice(
        self,
        year: int,
        month: int,
        long_name: str,
        start: np.datetime64,
        end: np.datetime64,
    ) -> xr.Dataset:
        path = self.pressure_dir / f"era5_pl_{year}_{month:02d}_{long_name}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        with xr.open_dataset(path, engine="netcdf4") as ds:
            normalized = _normalize_dataset(ds)
            subset = normalized.sel(time=slice(start, end)).load()
        return subset

    @staticmethod
    def _concat_frames(frames: Sequence[Any]) -> Any:
        import pandas as pd

        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _daily_to_frame(ds: xr.Dataset, day: date) -> Any:
        import pandas as pd

        flat = ds.isel(time=0, drop=True).to_dataframe().reset_index()
        flat.insert(0, "date", pd.Timestamp(day))
        return flat

    @staticmethod
    def _write_parquet(table: Any, path: Path) -> None:
        try:
            table.to_parquet(path, index=False)
        except Exception as exc:
            raise RuntimeError(
                "Parquet export requires a pandas parquet engine such as pyarrow or fastparquet"
            ) from exc


def daterange(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)
