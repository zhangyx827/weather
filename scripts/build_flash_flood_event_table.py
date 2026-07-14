#!/usr/bin/env python3
"""Build the handoff-approved flash-flood seed event table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.data import expand_flash_flood_events_to_daily_table, flash_flood_event_table


FORMATS = ("csv", "json", "parquet")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build flash-flood seed event tables for Layer-4 label development.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "labels" / "flash_flood_events.csv",
        help="Output path for the normalized event table.",
    )
    parser.add_argument("--format", choices=FORMATS, default="csv", help="Output format for the normalized event table.")
    parser.add_argument(
        "--daily-output",
        type=Path,
        default=ROOT / "data" / "processed" / "labels" / "flash_flood_events_daily.csv",
        help="Output path for the inclusive daily expansion table.",
    )
    parser.add_argument("--daily-format", choices=FORMATS, default="csv", help="Output format for the daily expansion table.")
    return parser.parse_args(argv)


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
    event_table = flash_flood_event_table()
    daily_table = expand_flash_flood_events_to_daily_table()
    _write_table(event_table, args.output, args.format)
    _write_table(daily_table, args.daily_output, args.daily_format)

    summary = {
        "events": int(len(event_table)),
        "daily_rows": int(len(daily_table)),
        "output": str(args.output),
        "daily_output": str(args.daily_output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
