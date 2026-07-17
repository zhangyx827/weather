#!/usr/bin/env python3
"""Aggregate daily regional dry-heat features and join explicit agriculture labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.data import build_dry_heat_agriculture_supervised_training_dataset


FORMATS = ("csv", "json", "parquet")
SAMPLE_UNITS = ("region-year", "region-season")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a supervised dry-heat agriculture training table from daily regional features and explicit labels."
    )
    parser.add_argument("--features", type=Path, required=True, help="Daily regional feature table.")
    parser.add_argument("--labels", type=Path, required=True, help="Explicit agriculture outcome label table.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "training" / "dry_heat_agriculture_supervised_training.csv",
        help="Output path for the merged supervised training table.",
    )
    parser.add_argument("--format", choices=FORMATS, help="Output format. Defaults to the output file suffix.")
    parser.add_argument("--sample-unit", choices=SAMPLE_UNITS, default="region-year", help="Supervision unit used for aggregation and joining.")
    parser.add_argument("--region-column", default="region_id", help="Regional key shared by features and labels.")
    parser.add_argument("--keep-unmatched", action="store_true", help="Keep aggregated feature rows with no matched explicit label.")
    return parser.parse_args(argv)


def _infer_format(path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    suffix = path.suffix.lower().lstrip(".")
    if suffix in FORMATS:
        return suffix
    raise ValueError(f"Could not infer table format from path: {path}")


def _read_table(path: Path):
    fmt = _infer_format(path, None)
    if fmt == "csv":
        import pandas as pd

        return pd.read_csv(path)
    if fmt == "json":
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
    features = _read_table(args.features)
    labels = _read_table(args.labels)
    merged = build_dry_heat_agriculture_supervised_training_dataset(
        features,
        labels,
        sample_unit=args.sample_unit,
        region_column=args.region_column,
        drop_unmatched=not args.keep_unmatched,
    )
    _write_table(merged, args.output, output_format)

    summary = {
        "rows": int(len(merged)),
        "sample_unit": args.sample_unit,
        "region_column": args.region_column,
        "labeled_rows": int(merged["is_labeled"].sum()) if "is_labeled" in merged.columns else 0,
        "matched_regions": int(merged[args.region_column].nunique()) if args.region_column in merged.columns else 0,
        "target_columns": [name for name in ("yield_anomaly", "yield_value", "label") if name in merged.columns],
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
