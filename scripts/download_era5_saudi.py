#!/usr/bin/env python3
"""Download legacy raw Saudi ERA5 inputs via CDS API.

The Layer-4 training path in this repository now operates on daily feature
tables or daily NetCDF inputs. This script remains a raw-data helper and is no
longer the default prerequisite for daily model training.
"""

from __future__ import annotations

import argparse
import calendar
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import requests
import xarray as xr
from urllib3.exceptions import MaxRetryError

SAUDI_AREA = [33, 34, 16, 57]
PRESSURE_LEVELS = ['50', '100', '150', '250', '400', '600', '1000', '200', '300', '500', '700', '850', '925']
SURFACE_VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "convective_available_potential_energy",
    "convective_inhibition",
    "total_cloud_cover",
    "low_cloud_cover",
    "medium_cloud_cover",
    "high_cloud_cover",
    "total_precipitation",
    "convective_precipitation",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "surface_sensible_heat_flux",
    "surface_latent_heat_flux",
    "maximum_2m_temperature_since_previous_post_processing",
    "minimum_2m_temperature_since_previous_post_processing",
    "geopotential",
]
PRESSURE_VARIABLES = [
    ("u_component_of_wind", "u_component_of_wind"),
    ("v_component_of_wind", "v_component_of_wind"),
    ("geopotential", "geopotential"),
    ("temperature", "temperature"),
    ("specific_humidity", "specific_humidity"),
    ("vertical_velocity", "vertical_velocity"),
    ("divergence", "divergence"),
    # CDS variable name is `vorticity`, while the downstream pipeline expects
    # `era5_pl_<year>_<month>_relative_vorticity.nc`.
    ("vorticity", "relative_vorticity"),
]
SINGLE_LEVEL_SHORT_NAMES = {
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "surface_pressure": "sp",
    "convective_available_potential_energy": "cape",
    "convective_inhibition": "cin",
    "total_cloud_cover": "tcc",
    "low_cloud_cover": "lcc",
    "medium_cloud_cover": "mcc",
    "high_cloud_cover": "hcc",
    "total_precipitation": "tp",
    "convective_precipitation": "cp",
    "surface_solar_radiation_downwards": "ssrd",
    "surface_thermal_radiation_downwards": "strd",
    "surface_sensible_heat_flux": "sshf",
    "surface_latent_heat_flux": "slhf",
    "maximum_2m_temperature_since_previous_post_processing": "mx2t",
    "minimum_2m_temperature_since_previous_post_processing": "mn2t",
    "geopotential": "z",
}
LEGACY_PRESSURE_SUFFIXES = {
    "relative_vorticity": ("relative_vorticity", "vorticity"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download legacy raw Saudi ERA5 input files using CDS API.")
    parser.add_argument("--years", nargs="+", required=True, help="Years to download, e.g. 2023 2024")
    parser.add_argument(
        "--dates",
        nargs="*",
        default=None,
        help="Optional explicit dates used to select which months to backfill in YYYY-MM-DD format. Defaults to all months in the selected years.",
    )
    parser.add_argument("--single-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--pressure-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite existing files instead of skipping them.")
    return parser.parse_args()


def _daily_times() -> list[str]:
    return ["00:00"]


def _month_days(year: str, month_str: str) -> list[str]:
    _, days_in_month = calendar.monthrange(int(year), int(month_str))
    return [f"{day:02d}" for day in range(1, days_in_month + 1)]


def _iter_months(years: list[str], explicit_dates: list[str] | None) -> list[tuple[str, str]]:
    if explicit_dates:
        month_keys = {
            datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m")
            for value in explicit_dates
        }
        return [(value[:4], value[5:7]) for value in sorted(month_keys)]

    all_months: list[tuple[str, str]] = []
    for year_text in years:
        for month in range(1, 13):
            all_months.append((year_text, f"{month:02d}"))
    return all_months


def _candidate_year_dirs(root: Path, prefix: str, year: str) -> list[Path]:
    return [
        root / f"{prefix}_{year}",
        root / f"{prefix}_{year}_6h",
        Path(f"{prefix}_{year}"),
        Path(f"{prefix}_{year}_6h"),
    ]


def _resolve_year_dir(root: Path, prefix: str, year: str) -> Path:
    for candidate in _candidate_year_dirs(root, prefix, year):
        if candidate.exists():
            return candidate
    return root / f"{prefix}_{year}"


def _open_dataset_vars(path: Path) -> set[str]:
    if not path.exists():
        return set()
    suffixes = {suffix.lower() for suffix in path.suffixes}
    if ".zip" in suffixes:
        return _zip_dataset_vars(path)
    if path.suffix.lower() == ".nc":
        try:
            ds = xr.open_dataset(path)
        except ValueError:
            if zipfile.is_zipfile(path):
                return _zip_dataset_vars(path)
            raise
        try:
            return set(ds.data_vars)
        finally:
            ds.close()
    return set()


def _zip_dataset_vars(path: Path) -> set[str]:
    data_vars: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".nc"):
                continue
            with archive.open(name) as member, tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
                tmp.write(member.read())
                tmp.flush()
                ds = xr.open_dataset(tmp.name)
                try:
                    data_vars.update(ds.data_vars)
                finally:
                    ds.close()
    return data_vars


def _rewrite_zip_as_netcdf(path: Path) -> None:
    datasets: list[xr.Dataset] = []
    with zipfile.ZipFile(path) as archive:
        member_names = [name for name in archive.namelist() if name.endswith(".nc")]
        if not member_names:
            raise ValueError(f"{path} is a zip archive but does not contain any .nc members")
        for name in member_names:
            with archive.open(name) as member, tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
                tmp.write(member.read())
                tmp.flush()
                ds = xr.open_dataset(tmp.name)
                try:
                    datasets.append(ds.load())
                finally:
                    ds.close()

    merged = xr.merge(datasets, compat="override", combine_attrs="override") if len(datasets) > 1 else datasets[0]
    with tempfile.NamedTemporaryFile(suffix=".nc", dir=path.parent, delete=False) as tmp_output:
        tmp_output_path = Path(tmp_output.name)
    try:
        merged.to_netcdf(tmp_output_path)
        tmp_output_path.replace(path)
    finally:
        merged.close()
        if tmp_output_path.exists():
            tmp_output_path.unlink(missing_ok=True)


def _single_month_paths(root: Path, year: str, month_str: str) -> tuple[Path, Path | None]:
    single_dir = _resolve_year_dir(root, "era5_single_levels", year)
    main = single_dir / f"era5_single_levels_{year}_{month_str}.nc"
    supplement = single_dir / f"era5_single_levels_{year}_{month_str}_supplement.nc"
    return main, supplement if supplement.exists() else None


def _missing_single_level_variables(root: Path, year: str, month_str: str) -> list[str]:
    main_path, supplement_path = _single_month_paths(root, year, month_str)
    available = _open_dataset_vars(main_path)
    if supplement_path is not None:
        available.update(_open_dataset_vars(supplement_path))
    missing = [
        cds_name
        for cds_name in SURFACE_VARS
        if SINGLE_LEVEL_SHORT_NAMES[cds_name] not in available
    ]
    return missing


def _pressure_suffix_candidates(output_suffix: str) -> tuple[str, ...]:
    return LEGACY_PRESSURE_SUFFIXES.get(output_suffix, (output_suffix,))


def _missing_pressure_variables(root: Path, year: str, month_str: str) -> list[tuple[str, str]]:
    pressure_dir = _resolve_year_dir(root, "era5_pressure_levels", year)
    missing: list[tuple[str, str]] = []
    for cds_variable, output_suffix in PRESSURE_VARIABLES:
        candidates = [
            pressure_dir / f"era5_pl_{year}_{month_str}_{suffix}.nc"
            for suffix in _pressure_suffix_candidates(output_suffix)
        ]
        if any(path.exists() for path in candidates):
            continue
        missing.append((cds_variable, output_suffix))
    return missing


def pressure_target_path(year_dir: Path, year: str, month_str: str, output_suffix: str) -> Path:
    return year_dir / f"era5_pl_{year}_{month_str}_{output_suffix}.nc"


def single_supplement_target_path(root: Path, year: str, month_str: str) -> Path:
    return root / f"era5_single_levels_{year}_{month_str}_supplement_backfill.nc"


def _retrieve_with_retry(client, dataset: str, request: dict[str, object], target: Path, label: str) -> None:
    retries = 5
    backoff = 5
    for attempt in range(retries):
        try:
            # 【新增】如果上次尝试留下了残余/损坏文件，先清理干净，确保全新下载
            if target.exists():
                target.unlink()

            client.retrieve(dataset, request, str(target))
            if zipfile.is_zipfile(target):
                print(f"📦 {label} 返回了 zip，正在自动转换为 NetCDF...")
                _rewrite_zip_as_netcdf(target)
            time.sleep(3)
            return
        # 【修改】将 AssertionError 纳入捕获范围
        except (requests.exceptions.SSLError, MaxRetryError, requests.exceptions.HTTPError, AssertionError) as exc:
            # 【新增】断流或失败时，务必当场清理掉只下载了一半的坏文件，防止干扰后续判断
            if target.exists():
                target.unlink()
                
            if attempt == retries - 1:
                raise exc
            print(f"⚠️ {label} 下载异常或文件大小不匹配 ({type(exc).__name__})，{backoff}秒后重试...")
            time.sleep(backoff)
            backoff *= 2


def _single_level_request(year: str, month_str: str, days: list[str], variables: list[str]) -> dict[str, object]:
    return {
        "product_type": "reanalysis",
        "format": "netcdf",
        "data_format": "netcdf",
        "download_format": "unarchived",
        "variable": variables,
        "year": year,
        "month": month_str,
        "day": days,
        "time": _daily_times(),
        "area": SAUDI_AREA,
    }


def _pressure_level_request(year: str, month_str: str, days: list[str], cds_variable: str) -> dict[str, object]:
    return {
        "product_type": "reanalysis",
        "format": "netcdf",
        "data_format": "netcdf",
        "download_format": "unarchived",
        "variable": [cds_variable],
        "pressure_level": PRESSURE_LEVELS,
        "year": year,
        "month": month_str,
        "day": days,
        "time": _daily_times(),
        "area": SAUDI_AREA,
    }


def main() -> int:
    import cdsapi

    args = parse_args()
    client = cdsapi.Client()

    target_months = _iter_months(args.years, args.dates)

    for year in args.years:
        single_dir = _resolve_year_dir(args.single_root, "era5_single_levels", year)
        pressure_dir = _resolve_year_dir(args.pressure_root, "era5_pressure_levels", year)
        single_dir.mkdir(parents=True, exist_ok=True)
        pressure_dir.mkdir(parents=True, exist_ok=True)

    single_missing_cache: dict[tuple[str, str], list[str]] = {}
    pressure_missing_cache: dict[tuple[str, str], list[tuple[str, str]]] = {}

    for year, month_str in target_months:
        cache_key = (year, month_str)
        single_dir = _resolve_year_dir(args.single_root, "era5_single_levels", year)
        pressure_dir = _resolve_year_dir(args.pressure_root, "era5_pressure_levels", year)
        month_days = _month_days(year, month_str)

        if cache_key not in single_missing_cache:
            single_missing_cache[cache_key] = _missing_single_level_variables(args.single_root, year, month_str)
        if cache_key not in pressure_missing_cache:
            pressure_missing_cache[cache_key] = _missing_pressure_variables(args.pressure_root, year, month_str)

        missing_single_vars = single_missing_cache[cache_key]
        if missing_single_vars:
            single_target = single_supplement_target_path(single_dir, year, month_str)
            if single_target.exists() and not args.overwrite:
                print(f"⏭️  [{year}-{month_str}] 单层缺失变量月补丁已存在，自动跳过。")
            else:
                print(f"📡 [{year}-{month_str}] 正在请求单层缺失变量: {', '.join(missing_single_vars)}")
                _retrieve_with_retry(
                    client,
                    "reanalysis-era5-single-levels",
                    _single_level_request(year, month_str, month_days, missing_single_vars),
                    single_target,
                    f"单层 {year}-{month_str}",
                )
        else:
            print(f"⏭️  [{year}-{month_str}] 单层变量已完整，跳过下载。")

        for cds_variable, output_suffix in pressure_missing_cache[cache_key]:
            pressure_target = pressure_target_path(pressure_dir, year, month_str, output_suffix)
            if pressure_target.exists() and not args.overwrite:
                print(f"⏭️  [{year}-{month_str}] 高空变量 {output_suffix} 月文件已存在，自动跳过。")
                continue

            print(f"📡 [{year}-{month_str}] 正在请求高空缺失变量: {cds_variable} -> {output_suffix}")
            _retrieve_with_retry(
                client,
                "reanalysis-era5-pressure-levels",
                _pressure_level_request(year, month_str, month_days, cds_variable),
                pressure_target,
                f"高空 {cds_variable} {year}-{month_str}",
            )

    print("🎉 所有任务处理完毕！")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
