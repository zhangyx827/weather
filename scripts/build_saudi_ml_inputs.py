#!/usr/bin/env python3
"""Build Aurora-ready inputs and LightGBM indicator files from ``data/raw``."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.indicators import RawInputBuilder


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def parse_era5_source_spec(value: str) -> tuple[int, Path, Path, Path | None]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            "ERA5 source spec must be YEAR,SINGLE_DIR,PRESSURE_DIR[,MISSING_PRESSURE_DIR]"
        )
    try:
        year = int(parts[0])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ERA5 source year: {parts[0]!r}") from exc
    single_dir = Path(parts[1])
    pressure_dir = Path(parts[2])
    missing_pressure_dir = Path(parts[3]) if len(parts) == 4 and parts[3] else None
    return year, single_dir, pressure_dir, missing_pressure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily Saudi LightGBM indicator files from ERA5 and auxiliary sources.")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--single-dir", type=Path, help="Override the ERA5 single-level directory, e.g. ./era5_single_levels_2024.")
    parser.add_argument("--pressure-dir", type=Path, help="Override the ERA5 pressure-level directory, e.g. ./era5_pressure_levels_2024.")
    parser.add_argument("--missing-pressure-dir", type=Path, help="Optional fallback directory for recovered pressure-level files.")
    parser.add_argument(
        "--era5-source",
        dest="era5_sources",
        action="append",
        type=parse_era5_source_spec,
        help="Repeatable YEAR,SINGLE_DIR,PRESSURE_DIR[,MISSING_PRESSURE_DIR] spec for multi-year builds.",
    )
    parser.add_argument("--aurora-out", type=Path, default=ROOT / "data" / "processed" / "aurora_inputs")
    parser.add_argument("--indicator-nc-out", type=Path, default=ROOT / "data" / "processed" / "lightgbm_indicators_nc")
    parser.add_argument("--indicator-parquet-out", type=Path, default=ROOT / "data" / "processed" / "lightgbm_tables")
    parser.add_argument("--start-date", type=parse_date, required=True)
    parser.add_argument("--end-date", type=parse_date, required=True)
    parser.add_argument("--aurora-cadence-hours", type=int, default=6)
    parser.add_argument(
        "--include-aurora",
        action="store_true",
        help="Also build aurora inputs. Disabled by default because daily LightGBM indicators are the primary output.",
    )
    return parser.parse_args()


def build_year_directory_mappings(
    specs: list[tuple[int, Path, Path, Path | None]] | None,
) -> tuple[dict[int, Path], dict[int, Path], dict[int, Path | None]]:
    single_dirs: dict[int, Path] = {}
    pressure_dirs: dict[int, Path] = {}
    missing_pressure_dirs: dict[int, Path | None] = {}
    for year, single_dir, pressure_dir, missing_pressure_dir in specs or []:
        single_dirs[year] = single_dir
        pressure_dirs[year] = pressure_dir
        missing_pressure_dirs[year] = missing_pressure_dir
    return single_dirs, pressure_dirs, missing_pressure_dirs


def main() -> int:
    args = parse_args()
    single_dirs_by_year, pressure_dirs_by_year, missing_pressure_dirs_by_year = build_year_directory_mappings(args.era5_sources)
    builder = RawInputBuilder(
        raw_root=args.raw_root,
        aurora_out=args.aurora_out,
        indicator_nc_out=args.indicator_nc_out,
        indicator_parquet_out=args.indicator_parquet_out,
        aurora_cadence_hours=args.aurora_cadence_hours,
        single_dir=args.single_dir,
        pressure_dir=args.pressure_dir,
        missing_pressure_dir=args.missing_pressure_dir,
        single_dirs_by_year=single_dirs_by_year,
        pressure_dirs_by_year=pressure_dirs_by_year,
        missing_pressure_dirs_by_year=missing_pressure_dirs_by_year,
    )
    try:
        result = builder.build(args.start_date, args.end_date, include_aurora=args.include_aurora)
    finally:
        builder.close()
    print(json.dumps([entry.__dict__ for entry in result.entries], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
