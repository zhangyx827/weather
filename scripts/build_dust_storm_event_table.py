#!/usr/bin/env python3
"""Build normalized dust-storm event tables from explicit verified event facts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.data.dust_storm_event_sources import (
    dust_storm_event_table,
    expand_dust_storm_events_to_daily_table,
    standardize_dust_storm_event_records,
)


FORMATS = ("csv", "json", "parquet")
DEFAULT_INPUT = ROOT / "data" / "raw" / "verified_dust_storm.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "real_dust_storm_chain" / "dust_storm_events_2025_verified.csv"
DEFAULT_DAILY_OUTPUT = ROOT / "data" / "processed" / "real_dust_storm_chain" / "dust_storm_events_2025_verified_daily.csv"
DEFAULT_SUMMARY_OUTPUT = ROOT / "data" / "processed" / "real_dust_storm_chain" / "dust_storm_events_2025_verified_summary.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a normalized dust-storm event table from explicit verified dust-storm facts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Dust event source table in csv/json/parquet format.")
    parser.add_argument("--input-format", choices=FORMATS, help="Optional input format override.")
    parser.add_argument("--source-name", default="user_session_handoff", help="Default source name used when input rows omit it.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output path for the normalized event table.")
    parser.add_argument("--format", choices=FORMATS, default="csv", help="Output format for the normalized event table.")
    parser.add_argument("--daily-output", type=Path, default=DEFAULT_DAILY_OUTPUT, help="Output path for the inclusive daily event table.")
    parser.add_argument("--daily-format", choices=FORMATS, default="csv", help="Output format for the daily event table.")
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT, help="Optional JSON path for ingestion summary metadata.")
    return parser.parse_args(argv)


def _infer_format(path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    suffix = path.suffix.lower().lstrip(".")
    if suffix in FORMATS:
        return suffix
    raise ValueError(f"Could not infer table format from path: {path}")


def _read_table(path: Path, fmt: str):
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


def _non_empty_count(table, column: str) -> int:
    if column not in table.columns:
        return 0
    return int(table[column].fillna("").astype(str).str.strip().ne("").sum())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        raise FileNotFoundError(f"Dust event table does not exist: {args.input}")

    input_format = _infer_format(args.input, args.input_format)
    records = _read_table(args.input, input_format)
    events = standardize_dust_storm_event_records(records, source_name=args.source_name)
    event_table = dust_storm_event_table(events)
    daily_table = expand_dust_storm_events_to_daily_table(events)

    _write_table(event_table, args.output, args.format)
    _write_table(daily_table, args.daily_output, args.daily_format)

    summary = {
        "input": str(args.input),
        "rows": int(len(event_table)),
        "verified_rows": int(len(event_table[event_table["validation_status"].astype(str) == "verified"])),
        "daily_rows": int(len(daily_table)),
        "validation_status_counts": {
            str(key): int(value)
            for key, value in event_table["validation_status"].value_counts(dropna=False).to_dict().items()
        },
        "source_name_counts": {
            str(key): int(value)
            for key, value in event_table["source_name"].value_counts(dropna=False).to_dict().items()
        },
        "provenance_field_coverage": {
            "source_name_non_empty": _non_empty_count(event_table, "source_name"),
            "source_url_non_empty": _non_empty_count(event_table, "source_url"),
            "source_record_id_non_empty": _non_empty_count(event_table, "source_record_id"),
            "validation_status_non_empty": _non_empty_count(event_table, "validation_status"),
        },
        "output": str(args.output),
        "daily_output": str(args.daily_output),
    }
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["summary_output"] = str(args.summary_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
