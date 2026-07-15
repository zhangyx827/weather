#!/usr/bin/env python3
"""Flatten daily Saudi indicator NetCDF files into hazard-specific training tables."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.data import read_netcdf_dataset
from mazu_saudi.risk.layer4_features import feature_frame_from_dataset


HAZARD_TYPES = ("extreme_heat", "dry_heat_agriculture", "flash_flood")
FORMATS = ("csv", "json", "parquet")
DEFAULT_PATTERN = "saudi_indicators_*.nc"
DATE_PATTERN = re.compile(r"(\d{8})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hazard-specific Layer-4 training tables from indicator NetCDF files.")
    parser.add_argument("--input", type=Path, required=True, help="Indicator NetCDF file or directory.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed" / "layer4_training_tables")
    parser.add_argument("--hazard-type", choices=HAZARD_TYPES, action="append", help="Hazard type to export. Repeat for multiple hazards. Defaults to all.")
    parser.add_argument("--glob", default=DEFAULT_PATTERN, help="Glob pattern used when --input is a directory.")
    parser.add_argument("--format", choices=FORMATS, default="csv", help="Output table format. Defaults to csv.")
    return parser.parse_args(argv)


def _discover_input_files(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    return sorted(candidate for candidate in path.glob(pattern) if candidate.is_file())


def _coerce_timestamp(dataset: Any, source_path: Path) -> str:
    if hasattr(dataset, "coords") and "time" in dataset.coords and dataset.coords["time"].size:
        value = dataset.coords["time"].values[0]
        return str(np.datetime_as_string(np.asarray(value, dtype="datetime64[ns]"), unit="D"))
    match = DATE_PATTERN.search(source_path.stem)
    if match is None:
        raise ValueError(f"Could not infer date from dataset time coordinate or file name: {source_path}")
    token = match.group(1)
    return f"{token[0:4]}-{token[4:6]}-{token[6:8]}"


def _degradation_metadata_json(dataset: Any) -> str:
    payload = dataset.attrs.get("degradation_metadata", {})
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _build_table_for_file(source_path: Path, hazard_type: str):
    dataset = read_netcdf_dataset(source_path)
    frame = feature_frame_from_dataset(dataset, hazard_type=hazard_type)
    frame.insert(0, "date", _coerce_timestamp(dataset, source_path))
    frame.insert(1, "hazard_type", hazard_type)
    frame["source_file"] = source_path.name
    frame["source_status"] = dataset.attrs.get("source_status", "normal")
    frame["degradation_metadata"] = _degradation_metadata_json(dataset)
    return frame


def _write_table(table: Any, path: Path, fmt: str) -> None:
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
    if pd is None:
        raise RuntimeError("pandas is required to build Layer-4 training tables")
    input_files = _discover_input_files(args.input, args.glob)
    if not input_files:
        raise FileNotFoundError(f"No indicator NetCDF files matched under {args.input} with pattern {args.glob!r}")

    hazard_types = args.hazard_type or list(HAZARD_TYPES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    for hazard_type in hazard_types:
        tables = [_build_table_for_file(path, hazard_type) for path in input_files]
        table = tables[0] if len(tables) == 1 else pd.concat(tables, ignore_index=True)
        output_path = args.output_dir / f"{hazard_type}_training.{args.format}"
        _write_table(table, output_path, args.format)
        summary.append(
            {
                "hazard_type": hazard_type,
                "files": len(input_files),
                "rows": int(len(table)),
                "format": args.format,
                "output": str(output_path),
            }
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
