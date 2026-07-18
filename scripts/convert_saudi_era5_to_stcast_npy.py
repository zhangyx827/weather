#!/usr/bin/env python3
"""Convert Saudi ERA5 NetCDF inputs into STCast's timestamped .npy layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


PRESSURE_LEVELS = [1000.0, 925.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 250.0, 200.0, 150.0, 100.0, 50.0]
PRESSURE_VARIABLES = {
    "z": "geopotential",
    "q": "specific_humidity",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "t": "temperature",
}
SURFACE_VARIABLES = ("t2m", "u10", "v10", "msl")


def _open_monthly_surface(surface_dir: Path, year: int, month: int) -> xr.Dataset:
    path = surface_dir / f"era5_single_levels_{year}_{month:02d}.nc"
    if not path.exists():
        raise FileNotFoundError(f"Missing surface file: {path}")
    return xr.open_dataset(path)


def _open_monthly_pressure(pressure_dir: Path, missing_dir: Path, year: int, month: int, long_name: str) -> xr.DataArray:
    regular_path = pressure_dir / f"era5_pl_{year}_{month:02d}_{long_name}.nc"
    missing_path = missing_dir / f"era5_pl_{year}_{month:02d}_{long_name}_missing.nc"
    if not regular_path.exists():
        raise FileNotFoundError(f"Missing pressure file: {regular_path}")
    with xr.open_dataset(regular_path) as regular_ds:
        regular_var = next(iter(regular_ds.data_vars))
        regular = regular_ds[regular_var]
        regular_levels = regular.pressure_level.values.astype(float).tolist()
        if regular_levels == PRESSURE_LEVELS:
            return regular.load()

        if not missing_path.exists():
            raise FileNotFoundError(
                f"Missing supplemental pressure file for incomplete levels: {missing_path}"
            )

        with xr.open_dataset(missing_path) as missing_ds:
            missing_var = next(iter(missing_ds.data_vars))
            combined = xr.concat([regular, missing_ds[missing_var]], dim="pressure_level")
            combined = combined.sortby("pressure_level", ascending=False)
            levels = combined.pressure_level.values.astype(float).tolist()
            if levels != PRESSURE_LEVELS:
                raise ValueError(f"Unexpected pressure levels for {long_name} {year}-{month:02d}: {levels}")
            return combined.load()


def _sanitize_field(data: xr.DataArray) -> np.ndarray:
    array = data.squeeze(drop=True).transpose("latitude", "longitude").values.astype(np.float32)
    return np.ascontiguousarray(array)


def _write_array(dst_path: Path, array: np.ndarray) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst_path, array)


def convert_month(surface_dir: Path, pressure_dir: Path, missing_dir: Path, output_dir: Path, year: int, month: int) -> int:
    with _open_monthly_surface(surface_dir, year, month) as surface_ds:
        surface_ds = surface_ds.load()
        pressure_by_var = {
            short_name: _open_monthly_pressure(pressure_dir, missing_dir, year, month, long_name)
            for short_name, long_name in PRESSURE_VARIABLES.items()
        }

        timestamps = surface_ds.valid_time.values
        count = 0
        
        for ts_value in timestamps:
            stamp = np.datetime_as_string(ts_value, unit="h")
            date_part, hour_part = stamp.split("T")
            year_part = date_part[:4]
            pressure_base = output_dir / year_part / date_part
            surface_base = output_dir / "single" / year_part / date_part
            time_prefix = f"{hour_part}:00:00-"

            # ==========================================
            # 1. 检查断点续传：判断该时次的所有输出是否已经存在
            # ==========================================
            all_files_exist = True
            
            # 检查气压层文件
            for short_name in PRESSURE_VARIABLES.keys():
                for level in PRESSURE_LEVELS:
                    expected_path = pressure_base / f"{time_prefix}{short_name}-{level}.npy"
                    if not expected_path.exists():
                        all_files_exist = False
                        break
                if not all_files_exist:
                    break
            
            # 检查地面文件
            if all_files_exist:
                for short_name in SURFACE_VARIABLES:
                    expected_path = surface_base / f"{time_prefix}{short_name}.npy"
                    if not expected_path.exists():
                        all_files_exist = False
                        break

            if all_files_exist:
                # 所有目标文件已存在，安全跳过
                count += 1
                continue

            # ==========================================
            # 2. 检查时间对齐：确保该时间戳在所有气压层数据中都存在
            # ==========================================
            missing_pressure_ts = False
            for short_name, pressure_field in pressure_by_var.items():
                if ts_value not in pressure_field.valid_time.values:
                    print(f"Warning: Timestamp {stamp} not found in pressure field for '{short_name}'. Skipping.")
                    missing_pressure_ts = True
                    break
            
            if missing_pressure_ts:
                continue

            # ==========================================
            # 3. 处理并写入文件
            # ==========================================
            # 写入气压层数据 (使用 .sel 确保时间完全对齐)
            for short_name, pressure_field in pressure_by_var.items():
                selected = pressure_field.sel(valid_time=ts_value)
                for level in PRESSURE_LEVELS:
                    level_field = selected.sel(pressure_level=level)
                    array = _sanitize_field(level_field)
                    _write_array(pressure_base / f"{time_prefix}{short_name}-{level}.npy", array)

            # 写入地面数据 (同样使用 .sel 确保安全)
            for short_name in SURFACE_VARIABLES:
                if short_name not in surface_ds:
                    raise ValueError(f"Surface variable {short_name} not found in monthly dataset")
                array = _sanitize_field(surface_ds[short_name].sel(valid_time=ts_value))
                _write_array(surface_base / f"{time_prefix}{short_name}.npy", array)

            count += 1
            
        return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-dir", default="era5_single_levels_2024", type=Path)
    parser.add_argument("--pressure-dir", default="era5_pressure_levels_2024", type=Path)
    parser.add_argument("--missing-pressure-dir", default="era5_pressure_levels_2024_missing", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--year", default=2024, type=int)
    parser.add_argument("--months", nargs="+", type=int, default=list(range(1, 13)))
    args = parser.parse_args()

    total = 0
    for month in args.months:
        total += convert_month(
            surface_dir=args.surface_dir,
            pressure_dir=args.pressure_dir,
            missing_dir=args.missing_pressure_dir,
            output_dir=args.output_dir,
            year=args.year,
            month=month,
        )
    print(f"Converted/Verified {total} timestamps into {args.output_dir}")


if __name__ == "__main__":
    main()