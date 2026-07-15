#!/usr/bin/env python3
"""Build combined flash-flood event tables from seed and verified sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.data import (
    expand_flash_flood_events_to_daily_table,
    flash_flood_event_table,
    merge_flash_flood_event_sources,
    seed_flash_flood_events,
    standardize_flash_flood_event_records,
)


FORMATS = ("csv", "json", "parquet")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a combined flash-flood event table from handoff seed events and one verified external source."
    )
    parser.add_argument("--verified-input", type=Path, required=True, help="Verified event table path in csv/json/parquet format.")
    parser.add_argument("--verified-format", choices=FORMATS, help="Optional input format override for the verified event table.")
    parser.add_argument(
        "--source-name",
        default="verified_source",
        help="Source name written into standardized verified event provenance when rows do not provide one.",
    )
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="Export only standardized verified rows without merging built-in seed events.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "labels" / "flash_flood_events_verified_combined.csv",
        help="Output path for the normalized combined event table.",
    )
    parser.add_argument("--format", choices=FORMATS, default="csv", help="Output format for the normalized combined event table.")
    parser.add_argument(
        "--daily-output",
        type=Path,
        default=ROOT / "data" / "processed" / "labels" / "flash_flood_events_verified_combined_daily.csv",
        help="Output path for the inclusive daily expansion table.",
    )
    parser.add_argument("--daily-format", choices=FORMATS, default="csv", help="Output format for the daily expansion table.")
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional JSON path for ingestion summary metadata and provenance coverage counts.",
    )
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
    values = table[column]
    return int(values.fillna("").astype(str).str.strip().ne("").sum())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    verified_format = _infer_format(args.verified_input, args.verified_format)
    verified_records = _read_table(args.verified_input, verified_format)
    verified_events = standardize_flash_flood_event_records(verified_records, source_name=args.source_name)
    events = verified_events if args.verified_only else merge_flash_flood_event_sources(
        seed_events=seed_flash_flood_events(),
        verified_events=verified_events,
    )

    event_table = flash_flood_event_table(events)
    daily_table = expand_flash_flood_events_to_daily_table(events)
    _write_table(event_table, args.output, args.format)
    _write_table(daily_table, args.daily_output, args.daily_format)

    validation_status_counts = {
        str(key): int(value)
        for key, value in event_table["validation_status"].value_counts(dropna=False).to_dict().items()
    }
    source_name_counts = {
        str(key): int(value)
        for key, value in event_table["source_name"].value_counts(dropna=False).to_dict().items()
    }
    summary = {
        "verified_input": str(args.verified_input),
        "verified_rows": int(len(verified_events)),
        "seed_rows_included": 0 if args.verified_only else int(len(seed_flash_flood_events())),
        "combined_rows": int(len(event_table)),
        "daily_rows": int(len(daily_table)),
        "validation_status_counts": validation_status_counts,
        "source_name_counts": source_name_counts,
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
