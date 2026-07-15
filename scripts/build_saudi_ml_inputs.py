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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Saudi Aurora and LightGBM input files from data/raw.")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--single-dir", type=Path, help="Override the ERA5 single-level directory, e.g. ./era5_single_levels_2024.")
    parser.add_argument("--pressure-dir", type=Path, help="Override the ERA5 pressure-level directory, e.g. ./era5_pressure_levels_2024.")
    parser.add_argument("--missing-pressure-dir", type=Path, help="Optional fallback directory for recovered pressure-level files.")
    parser.add_argument("--aurora-out", type=Path, default=ROOT / "data" / "processed" / "aurora_inputs")
    parser.add_argument("--indicator-nc-out", type=Path, default=ROOT / "data" / "processed" / "lightgbm_indicators_nc")
    parser.add_argument("--indicator-parquet-out", type=Path, default=ROOT / "data" / "processed" / "lightgbm_tables")
    parser.add_argument("--start-date", type=parse_date, required=True)
    parser.add_argument("--end-date", type=parse_date, required=True)
    parser.add_argument("--aurora-cadence-hours", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    builder = RawInputBuilder(
        raw_root=args.raw_root,
        aurora_out=args.aurora_out,
        indicator_nc_out=args.indicator_nc_out,
        indicator_parquet_out=args.indicator_parquet_out,
        aurora_cadence_hours=args.aurora_cadence_hours,
        single_dir=args.single_dir,
        pressure_dir=args.pressure_dir,
        missing_pressure_dir=args.missing_pressure_dir,
    )
    try:
        result = builder.build(args.start_date, args.end_date)
    finally:
        builder.close()
    print(json.dumps([entry.__dict__ for entry in result.entries], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
