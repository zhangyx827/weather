#!/usr/bin/env python3
"""Download Saudi ERA5 single-level and pressure-level inputs via CDS API."""

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
    parser = argparse.ArgumentParser(description="Download Saudi ERA5 input files using CDS API.")
    parser.add_argument("--years", nargs="+", required=True, help="Years to download, e.g. 2023 2024")
    parser.add_argument("--single-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--pressure-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--skip-existing", action="store_true")
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
            if not (args.skip_existing and single_target.exists()):
                print(f"[{year}-{month_str}] 正在请求地面单层数据...")
                
                # 增加网络抖动重试机制
                for attempt in range(3):
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
                        break  # 下载成功，跳出重试循环
                    except (requests.exceptions.SSLError, MaxRetryError) as e:
                        if attempt == 2:  # 三次都失败则抛出异常
                            raise e
                        print(f"⚠️ 遇到网络连接异常(SSL/Retry)，5秒后进行第 {attempt + 2} 次重试...")
                        time.sleep(5)

            # ----------------------------------------------------
            # 2. 下载高空/气压层变量 (Pressure-level)
            # ----------------------------------------------------
            # 优化：合并所有变量到单个 NetCDF 文件中请求，避免高频请求触发 CDS 服务器防火墙拦截
            pressure_target = pressure_dir / f"era5_pl_{year}_{month_str}.nc"
            if not (args.skip_existing and pressure_target.exists()):
                print(f"[{year}-{month_str}] 正在合并请求高空气压层数据...")
                
                for attempt in range(3):
                    try:
                        client.retrieve(
                            "reanalysis-era5-pressure-levels",
                            {
                                "product_type": "reanalysis",
                                "data_format": "netcdf",
                                "variable": PRESSURE_VARS,  # 传入完整的变量列表，不再循环
                                "pressure_level": PRESSURE_LEVELS,
                                "year": year,
                                "month": month_str,
                                "day": _month_days(),
                                "time": _hours(),
                                "area": SAUDI_AREA,
                            },
                            str(pressure_target),
                        )
                        break  # 下载成功，跳出重试循环
                    except (requests.exceptions.SSLError, MaxRetryError) as e:
                        if attempt == 2:
                            raise e
                        print(f"⚠️ 遇到网络连接异常(SSL/Retry)，5秒后进行第 {attempt + 2} 次重试...")
                        time.sleep(5)
                        
    print("🎉 所有请求及下载任务已顺利完成！")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())