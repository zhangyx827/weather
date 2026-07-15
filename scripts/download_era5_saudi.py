#!/usr/bin/env python3
"""Download legacy raw Saudi ERA5 inputs via CDS API.

The Layer-4 training path in this repository now operates on daily feature
tables or daily NetCDF inputs. This script remains a raw-data helper and is no
longer the default prerequisite for daily model training.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import requests
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
PRESSURE_VARS = [
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
    "temperature",
    "specific_humidity",
    "vertical_velocity",
    "divergence",
    "vorticity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download legacy raw Saudi ERA5 input files using CDS API.")
    parser.add_argument("--years", nargs="+", required=True, help="Years to download, e.g. 2023 2024")
    parser.add_argument("--single-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--pressure-root", type=Path, default=Path("data/raw"))
    # 修改：默认就是跳过，只有传入 --overwrite 时才会强制覆盖下载
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite existing files instead of skipping them.")
    return parser.parse_args()


def _month_days() -> list[str]:
    return [f"{day:02d}" for day in range(1, 32)]


def _hours() -> list[str]:
    return [f"{hour:02d}:00" for hour in range(24)]


def main() -> int:
    import cdsapi

    args = parse_args()
    client = cdsapi.Client()
    
    for year in args.years:
        single_dir = args.single_root / f"era5_single_levels_{year}"
        pressure_dir = args.pressure_root / f"era5_pressure_levels_{year}"
        single_dir.mkdir(parents=True, exist_ok=True)
        pressure_dir.mkdir(parents=True, exist_ok=True)

        for month in range(1, 13):
            month_str = f"{month:02d}"
            
            # ----------------------------------------------------
            # 1. 下载地面/单层变量 (Single-level)
            # ----------------------------------------------------
            single_target = single_dir / f"era5_single_levels_{year}_{month_str}.nc"
            
            # 默认逻辑：如果文件存在且没有指定 --overwrite，则跳过
            if single_target.exists() and not args.overwrite:
                print(f"⏭️  [{year}-{month_str}] 地面单层数据文件已存在，自动跳过。")
            else:
                print(f"📡 [{year}-{month_str}] 正在请求地面单层数据...")
                retries = 5
                backoff = 5
                for attempt in range(retries):
                    try:
                        client.retrieve(
                            "reanalysis-era5-single-levels",
                            {
                                "product_type": "reanalysis",
                                "format": "netcdf",
                                "variable": SURFACE_VARS,
                                "year": year,
                                "month": month_str,
                                "day": _month_days(),
                                "time": _hours(),
                                "area": SAUDI_AREA,
                            },
                            str(single_target),
                        )
                        time.sleep(3)  # 成功后冷却
                        break
                    except (requests.exceptions.SSLError, MaxRetryError, requests.exceptions.HTTPError) as e:
                        if attempt == retries - 1:
                            raise e
                        print(f"⚠️ 地面数据下载异常，{backoff}秒后重试...")
                        time.sleep(backoff)
                        backoff *= 2

            # ----------------------------------------------------
            # 2. 下载高空/气压层变量 (Pressure-level)
            # ----------------------------------------------------
            for variable in PRESSURE_VARS:
                pressure_target = pressure_dir / f"era5_pl_{year}_{month_str}_{variable}.nc"
                
                # 默认逻辑：如果文件存在且没有指定 --overwrite，则跳过
                if pressure_target.exists() and not args.overwrite:
                    print(f"⏭️  [{year}-{month_str}] 高空变量 {variable} 已存在，自动跳过。")
                    continue
                
                print(f"📡 [{year}-{month_str}] 正在请求高空层变量: {variable} ...")
                retries = 5
                backoff = 5
                for attempt in range(retries):
                    try:
                        client.retrieve(
                            "reanalysis-era5-pressure-levels",
                            {
                                "product_type": "reanalysis",
                                "data_format": "netcdf",
                                "variable": [variable],
                                "pressure_level": PRESSURE_LEVELS,
                                "year": year,
                                "month": month_str,
                                "day": _month_days(),
                                "time": _hours(),
                                "area": SAUDI_AREA,
                            },
                            str(pressure_target),
                        )
                        time.sleep(3)  # 成功后冷却
                        break
                    except (requests.exceptions.SSLError, MaxRetryError, requests.exceptions.HTTPError) as e:
                        if attempt == retries - 1:
                            raise e
                        print(f"⚠️ 高空变量 {variable} 下载异常，{backoff}秒后重试...")
                        time.sleep(backoff)
                        backoff *= 2
                        
    print("🎉 所有任务处理完毕！")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
