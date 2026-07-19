#!/usr/bin/env python3
"""Convert ERA5 NetCDF inputs into STCast's timestamped .npy layout.

This wrapper extends the existing single-year Saudi converter so you can:

- select multiple years in one run
- select a subset of months
- optionally build a joint stats directory that matches STCast loaders

The output layout is the same as STCast expects:

- {output_dir}/{YYYY}/{YYYY-MM-DD}/{HH}:00:00-{var}-{level}.npy
- {output_dir}/single/{YYYY}/{YYYY-MM-DD}/{HH}:00:00-{var}.npy
"""

from __future__ import annotations

import argparse
import calendar
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr


DEFAULT_MONTHS = tuple(range(1, 13))
PRESSURE_LEVELS = [1000.0, 925.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 250.0, 200.0, 150.0, 100.0, 50.0]
PRESSURE_VARIABLES = {
    "z": "geopotential",
    "q": "specific_humidity",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "t": "temperature",
}
SURFACE_VARIABLES = ("t2m", "u10", "v10", "msl")


@dataclass(frozen=True)
class MonthWindow:
    year: int
    month: int
    start: datetime
    end: datetime


def _parse_years(args: argparse.Namespace) -> list[int]:
    years: list[int] = []
    if args.years:
        years.extend(args.years)
    if args.year:
        years.extend(args.year)
    if not years:
        raise ValueError("At least one year must be provided via --years or --year")
    return sorted(set(years))


def _parse_months(values: list[int] | None) -> list[int]:
    months = list(values) if values else list(DEFAULT_MONTHS)
    invalid = [month for month in months if month < 1 or month > 12]
    if invalid:
        raise ValueError(f"Months must be in 1..12; got {invalid}")
    return sorted(set(months))


def _month_window(year: int, month: int) -> MonthWindow:
    last_day = calendar.monthrange(year, month)[1]
    start = datetime(year, month, 1, 0, 0)
    end = datetime(year, month, last_day, 18, 0)
    return MonthWindow(year=year, month=month, start=start, end=end)


def _source_pattern(path: Path, year: int, month: int, template: str, long_name: str | None = None) -> Path:
    return path / template.format(
        year=year,
        month=month,
        month02=f"{month:02d}",
        long_name=long_name or "",
    )


def _open_surface_dataset(surface_file: Path) -> xr.Dataset:
    import xarray as xr

    if not surface_file.exists():
        raise FileNotFoundError(f"Missing surface file: {surface_file}")
    return xr.open_dataset(surface_file)


def _open_pressure_data(pressure_file: Path, missing_file: Path | None) -> xr.DataArray:
    import xarray as xr

    if not pressure_file.exists():
        raise FileNotFoundError(f"Missing pressure file: {pressure_file}")
    with xr.open_dataset(pressure_file) as regular_ds:
        regular_var = next(iter(regular_ds.data_vars))
        regular = regular_ds[regular_var]
        regular_levels = regular.pressure_level.values.astype(float).tolist()
        if regular_levels == PRESSURE_LEVELS:
            return regular.load()

        if missing_file is None:
            raise FileNotFoundError(
                f"Missing supplemental pressure file for incomplete levels: {pressure_file.stem}_missing.nc"
            )
        if not missing_file.exists():
            raise FileNotFoundError(f"Missing supplemental pressure file: {missing_file}")

        with xr.open_dataset(missing_file) as missing_ds:
            missing_var = next(iter(missing_ds.data_vars))
            combined = xr.concat([regular, missing_ds[missing_var]], dim="pressure_level")
            combined = combined.sortby("pressure_level", ascending=False)
            levels = combined.pressure_level.values.astype(float).tolist()
            if levels != PRESSURE_LEVELS:
                raise ValueError(f"Unexpected pressure levels for {pressure_file}: {levels}")
            return combined.load()


def _sanitize_field(data: xr.DataArray) -> np.ndarray:
    array = data.squeeze(drop=True)
    # ERA5 pressure files can carry an extra non-spatial axis, usually `time`
    # with one valid slice and one all-NaN fallback slice. Keep the first finite
    # slice so the saved field is always 2D [latitude, longitude].
    for dim in list(array.dims):
        if dim in ("latitude", "longitude"):
            continue
        if array.sizes[dim] == 1:
            array = array.isel({dim: 0}, drop=True)
            continue
        selected = array.isel({dim: 0}, drop=True)
        if not np.isfinite(selected.values).any():
            for idx in range(1, array.sizes[dim]):
                candidate = array.isel({dim: idx}, drop=True)
                if np.isfinite(candidate.values).any():
                    selected = candidate
                    break
        array = selected
    array = array.transpose("latitude", "longitude").values.astype(np.float32)
    return np.ascontiguousarray(array)


def _write_array(dst_path: Path, array: np.ndarray) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst_path, array)


def _convert_month(
    *,
    surface_file: Path,
    pressure_dir: Path,
    missing_pressure_dir: Path,
    year: int,
    month: int,
    surface_template: str,
    pressure_template: str,
    missing_pressure_template: str,
    output_dir: Path,
    overwrite: bool = False,
) -> int:
    with _open_surface_dataset(surface_file) as surface_ds:
        surface_ds = surface_ds.load()
        pressure_by_var = {
            short_name: _open_pressure_data(
                _source_pattern(pressure_dir, year, month, pressure_template, long_name),
                _source_pattern(missing_pressure_dir, year, month, missing_pressure_template, long_name),
            )
            for short_name, long_name in PRESSURE_VARIABLES.items()
        }

        timestamps = surface_ds.valid_time.values
        count = 0
        for ts_idx, ts_value in enumerate(timestamps):
            stamp = np.datetime_as_string(ts_value, unit="h")
            date_part, hour_part = stamp.split("T")
            pressure_base = output_dir / date_part[:4] / date_part
            surface_base = output_dir / "single" / date_part[:4] / date_part
            time_prefix = f"{hour_part}:00:00-"

            all_files_exist = True
            for short_name in PRESSURE_VARIABLES:
                for level in PRESSURE_LEVELS:
                    if not (pressure_base / f"{time_prefix}{short_name}-{level}.npy").exists():
                        all_files_exist = False
                        break
                if not all_files_exist:
                    break
            if all_files_exist:
                for short_name in SURFACE_VARIABLES:
                    if not (surface_base / f"{time_prefix}{short_name}.npy").exists():
                        all_files_exist = False
                        break
            if all_files_exist and not overwrite:
                count += 1
                continue

            for short_name, pressure_field in pressure_by_var.items():
                selected = pressure_field.isel(valid_time=ts_idx)
                for level in PRESSURE_LEVELS:
                    array = _sanitize_field(selected.sel(pressure_level=level))
                    _write_array(pressure_base / f"{time_prefix}{short_name}-{level}.npy", array)

            for short_name in SURFACE_VARIABLES:
                if short_name not in surface_ds:
                    raise ValueError(f"Surface variable {short_name} not found in {surface_file}")
                array = _sanitize_field(surface_ds[short_name].isel(valid_time=ts_idx))
                _write_array(surface_base / f"{time_prefix}{short_name}.npy", array)

            count += 1

        return count


def _build_stats_sources(output_dir: Path, years: list[int], months: list[int]) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    for year in years:
        for month in months:
            window = _month_window(year, month)
            sources.append(
                {
                    "root_dir": output_dir,
                    "train_start": window.start,
                    "train_end": window.end,
                }
            )
    return sources


def convert_years(
    *,
    surface_dir: Path,
    pressure_dir: Path,
    missing_pressure_dir: Path,
    output_dir: Path,
    years: list[int],
    months: list[int],
    surface_template: str,
    pressure_template: str,
    missing_pressure_template: str,
    overwrite: bool = False,
) -> int:
    total = 0
    for year in years:
        for month in months:
            surface_file = _source_pattern(surface_dir, year, month, surface_template)
            total += _convert_month(
                surface_file=surface_file,
                pressure_dir=pressure_dir,
                missing_pressure_dir=missing_pressure_dir,
                year=year,
                month=month,
                surface_template=surface_template,
                pressure_template=pressure_template,
                missing_pressure_template=missing_pressure_template,
                output_dir=output_dir,
                overwrite=overwrite,
            )
    return total


def build_joint_stats(output_dir: Path, stats_dir: Path, years: list[int], months: list[int], step_hours: int) -> None:
    from build_saudi_stcast_stats import build_stats

    sources = _build_stats_sources(output_dir, years, months)
    build_stats(sources, stats_dir, step_hours)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface-dir", required=True, type=Path, help="Directory containing monthly surface NetCDF files.")
    parser.add_argument("--pressure-dir", required=True, type=Path, help="Directory containing monthly pressure NetCDF files.")
    parser.add_argument(
        "--missing-pressure-dir",
        type=Path,
        default=None,
        help="Optional directory containing supplemental pressure NetCDF files for incomplete level sets.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="STCast npy output root.")
    parser.add_argument("--years", nargs="+", type=int, help="Years to convert, for example --years 2021 2022.")
    parser.add_argument(
        "--year",
        action="append",
        type=int,
        help="Repeatable year argument, kept for compatibility with older one-year invocations.",
    )
    parser.add_argument("--months", nargs="+", type=int, default=None, help="Months to convert, defaults to 1..12.")
    parser.add_argument(
        "--surface-template",
        default="era5_single_levels_{year}_{month02}.nc",
        help="Filename template relative to --surface-dir.",
    )
    parser.add_argument(
        "--pressure-template",
        default="era5_pl_{year}_{month02}_{long_name}.nc",
        help="Filename template relative to --pressure-dir.",
    )
    parser.add_argument(
        "--missing-pressure-template",
        default="era5_pl_{year}_{month02}_{long_name}_missing.nc",
        help="Filename template relative to --missing-pressure-dir.",
    )
    parser.add_argument("--build-stats", action="store_true", help="Build a joint STCast stats_dir after conversion.")
    parser.add_argument("--stats-dir", type=Path, help="Destination directory for mean_std.json and mean_std_single.json.")
    parser.add_argument("--stats-step-hours", type=int, default=6, help="Step size used when building joint stats.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing .npy files instead of skipping them.")
    args = parser.parse_args()

    years = _parse_years(args)
    months = _parse_months(args.months)
    missing_pressure_dir = args.missing_pressure_dir or args.pressure_dir

    total = convert_years(
        surface_dir=args.surface_dir,
        pressure_dir=args.pressure_dir,
        missing_pressure_dir=missing_pressure_dir,
        output_dir=args.output_dir,
        years=years,
        months=months,
        surface_template=args.surface_template,
        pressure_template=args.pressure_template,
        missing_pressure_template=args.missing_pressure_template,
        overwrite=args.overwrite,
    )
    print(f"Converted/verified {total} timestamps into {args.output_dir}")

    if args.build_stats:
        if args.stats_dir is None:
            parser.error("--build-stats requires --stats-dir")
        build_joint_stats(
            output_dir=args.output_dir,
            stats_dir=args.stats_dir,
            years=years,
            months=months,
            step_hours=args.stats_step_hours,
        )
        print(f"Wrote joint stats to {args.stats_dir}")


if __name__ == "__main__":
    main()
