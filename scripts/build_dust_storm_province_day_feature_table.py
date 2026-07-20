#!/usr/bin/env python3
"""Build a province-day dust-storm feature table from daily indicator NetCDF files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.data.dust_storm_province_features import (
    build_dust_storm_province_day_feature_table,
    summarize_dust_storm_feature_coverage,
)
from mazu_saudi.data.io import read_netcdf_dataset


FORMATS = ("csv", "json", "parquet")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a province-day dust-storm feature table from daily indicator NetCDF files.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "processed" / "lightgbm_indicators_nc",
        help="Directory containing daily indicator NetCDF files.",
    )
    parser.add_argument(
        "--glob",
        default="saudi_indicators_*.nc",
        help="Glob used to discover daily indicator files.",
    )
    parser.add_argument(
        "--boundary-path",
        type=Path,
        default=ROOT / "data" / "raw" / "admin_boundaries" / "geoBoundaries-SAU-ADM1.geojson",
        help="GeoJSON ADM1 boundary file used to map grid cells into Saudi provinces.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "training" / "dust_storm_province_day_features.parquet",
        help="Output path for the aggregated province-day feature table.",
    )
    parser.add_argument("--format", choices=FORMATS, help="Output format. Defaults to the output file suffix.")
    parser.add_argument("--province-column", default="province_name", help="Province column to create on the output table.")
    parser.add_argument(
        "--coordinate-precision",
        type=int,
        default=4,
        help="Decimal precision used when matching lat/lon grid cells to province lookup rows.",
    )
    return parser.parse_args(argv)


def _infer_format(path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    suffix = path.suffix.lower().lstrip(".")
    if suffix in FORMATS:
        return suffix
    raise ValueError(f"Could not infer table format from path: {path}")


def _is_readable_dust_file(path: Path) -> bool:
    try:
        dataset = read_netcdf_dataset(path)
    except Exception:
        return False
    try:
        return bool(getattr(dataset, "data_vars", None))
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()


def _write_table(table, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        table.to_csv(path, index=False)
        return
    if fmt == "json":
        path.write_text(table.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
        return
    try:
        table.to_parquet(path, index=False)
    except Exception as exc:
        raise RuntimeError("Parquet export requires a pandas parquet engine such as pyarrow or fastparquet") from exc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_format = _infer_format(args.output, args.format)
    feature_paths = sorted(path for path in args.input.glob(args.glob) if _is_readable_dust_file(path))
    if not feature_paths:
        raise FileNotFoundError(f"No indicator NetCDF files matched {args.input}/{args.glob}")

    table = build_dust_storm_province_day_feature_table(
        feature_paths,
        boundary_path=args.boundary_path,
        province_column=args.province_column,
        coordinate_precision=args.coordinate_precision,
    )
    _write_table(table, args.output, output_format)

    summary = {
        "rows": int(len(table)),
        "province_rows": int(table[args.province_column].astype(str).str.strip().ne("").sum()) if args.province_column in table.columns else 0,
        "region_rows": int(table["region_id"].astype(str).str.strip().ne("").sum()) if "region_id" in table.columns else 0,
        "feature_files": int(len(feature_paths)),
        "dust_feature_coverage": summarize_dust_storm_feature_coverage(table),
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
