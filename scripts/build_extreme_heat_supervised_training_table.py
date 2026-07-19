#!/usr/bin/env python3
"""Build a compact supervised extreme-heat training table from daily indicators."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.data import build_extreme_heat_supervised_training_dataset


FORMATS = ("csv", "json", "parquet")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a supervised extreme-heat Layer-4 training table from daily indicator files and verified events."
    )
    parser.add_argument("--input", type=Path, required=True, help="Directory containing daily Saudi indicator NetCDF files.")
    parser.add_argument("--labels", type=Path, required=True, help="Verified extreme-heat event CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "training" / "extreme_heat_supervised_verified_chain.parquet",
        help="Output path for the merged supervised training table.",
    )
    parser.add_argument("--format", choices=FORMATS, help="Output format. Defaults to the output file suffix.")
    parser.add_argument("--glob", default="saudi_indicators_*.nc", help="Glob used to discover daily indicator files.")
    parser.add_argument(
        "--negative-sample-size",
        type=int,
        default=None,
        help="Number of negative samples to keep. In region-day mode this counts region-day rows; in single-point mode it counts dates.",
    )
    parser.add_argument("--point-variable", default="heat_index_c", help="Indicator variable used to select the hottest grid cell per day.")
    parser.add_argument(
        "--sample-unit",
        choices=("single_point_day", "region_day"),
        default="region_day",
        help="Training sample unit. `region_day` aggregates multiple grid cells inside each Saudi ADM1 region.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of hottest grid cells pooled per region-day sample.")
    parser.add_argument(
        "--region-boundary-path",
        type=Path,
        default=ROOT / "data" / "raw" / "admin_boundaries" / "geoBoundaries-SAU-ADM1.geojson",
        help="GeoJSON admin-1 boundary file used to map grid cells into Saudi regions for region-day supervision.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for negative-date sampling.")
    return parser.parse_args(argv)


def _infer_format(path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    suffix = path.suffix.lower().lstrip(".")
    if suffix in FORMATS:
        return suffix
    raise ValueError(f"Could not infer table format from path: {path}")


def _read_table(path: Path):
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "csv":
        import pandas as pd

        return pd.read_csv(path)
    if suffix == "json":
        import pandas as pd

        return pd.read_json(path)
    import pandas as pd

    return pd.read_parquet(path)


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

    feature_paths = sorted(args.input.glob(args.glob))
    if not feature_paths:
        raise FileNotFoundError(f"No indicator files matched {args.input}/{args.glob}")

    labels = _read_table(args.labels)
    merged = build_extreme_heat_supervised_training_dataset(
        feature_paths,
        labels,
        point_variable=args.point_variable,
        negative_sample_size=args.negative_sample_size,
        seed=args.seed,
        sample_unit=args.sample_unit,
        top_k=args.top_k,
        region_boundary_path=args.region_boundary_path,
    )
    _write_table(merged, args.output, output_format)

    summary = {
        "sample_unit": args.sample_unit,
        "rows": int(len(merged)),
        "positive_rows": int((merged["label"] > 0.5).sum()) if "label" in merged.columns else 0,
        "negative_rows": int((merged["label"] <= 0.5).sum()) if "label" in merged.columns else 0,
        "region_rows": int(merged["region_id"].astype(str).str.strip().ne("").sum()) if "region_id" in merged.columns else 0,
        "matched_event_days": int(merged["matched_event_ids"].astype(str).str.len().gt(0).sum()) if "matched_event_ids" in merged.columns else 0,
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
