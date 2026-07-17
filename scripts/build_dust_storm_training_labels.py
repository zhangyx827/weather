#!/usr/bin/env python3
"""Build dust-storm training labels from a sample table and verified event facts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.config import DustStormLabelMappingConfig
from mazu_saudi.data import (
    build_dust_storm_training_labels,
    expand_dust_storm_events_to_daily_table,
    standardize_dust_storm_event_records,
)


FORMATS = ("csv", "json", "parquet")
DEFAULT_RAW_VERIFIED_INPUT = ROOT / "data" / "raw" / "dust_storm_verified" / "user_leads_2025_dust_events.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conservative dust-storm training labels for region-day or province-day samples.")
    parser.add_argument("--samples", type=Path, required=True, help="Sample table with a date column and a region/province identifier.")
    parser.add_argument("--events-daily", type=Path, help="Optional daily dust-storm event table in csv/json/parquet format.")
    parser.add_argument(
        "--verified-input",
        type=Path,
        default=DEFAULT_RAW_VERIFIED_INPUT,
        help="Verified raw dust-storm event file used when --events-daily is omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "labels" / "dust_storm_training_labels.parquet",
        help="Output path for the labeled sample table.",
    )
    parser.add_argument("--format", choices=FORMATS, help="Output format. Defaults to the output file suffix.")
    parser.add_argument(
        "--no-event-day-negatives",
        action="store_true",
        help="Disable negative labels for non-matching regions on days that still contain resolved dust events.",
    )
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


def _default_events_daily_table(verified_input: Path):
    records = _read_table(verified_input)
    events = standardize_dust_storm_event_records(records, source_name="user_session_handoff")
    return expand_dust_storm_events_to_daily_table(events)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_format = _infer_format(args.output, args.format)
    samples = _read_table(args.samples)
    events_daily = _read_table(args.events_daily) if args.events_daily else _default_events_daily_table(args.verified_input)
    config = DustStormLabelMappingConfig(emit_event_day_negatives=not args.no_event_day_negatives)
    labeled = build_dust_storm_training_labels(samples, events_daily, config=config)
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
