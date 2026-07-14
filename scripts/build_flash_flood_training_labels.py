#!/usr/bin/env python3
"""Build flash-flood training labels from a sample table and event table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.config import FlashFloodLabelMappingConfig
from mazu_saudi.data import build_flash_flood_training_labels, expand_flash_flood_events_to_daily_table


FORMATS = ("csv", "json", "parquet")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conservative flash-flood training labels for grid-day or province-day samples.")
    parser.add_argument("--samples", type=Path, required=True, help="Sample table with at least a date column.")
    parser.add_argument(
        "--events-daily",
        type=Path,
        help="Optional daily flash-flood event table. Defaults to built-in handoff seed events.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "labels" / "flash_flood_training_labels.parquet",
        help="Output path for the labeled sample table.",
    )
    parser.add_argument("--format", choices=FORMATS, help="Output format. Defaults to the output file suffix.")
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
    samples = _read_table(args.samples)
    events_daily = _read_table(args.events_daily) if args.events_daily else expand_flash_flood_events_to_daily_table()
    labeled = build_flash_flood_training_labels(samples, event_daily_table=events_daily, config=FlashFloodLabelMappingConfig())
    _write_table(labeled, args.output, output_format)

    summary = {
        "rows": int(len(labeled)),
        "positive_rows": int((labeled["label_status"] == "positive").sum()),
        "negative_rows": int((labeled["label_status"] == "negative").sum()),
        "uncertain_rows": int((labeled["label_status"] == "uncertain").sum()),
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
