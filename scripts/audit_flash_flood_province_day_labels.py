#!/usr/bin/env python3
"""Audit flash-flood province-day labels, unresolved rows, and resolved positives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.data import audit_flash_flood_province_day_labels, expand_flash_flood_events_to_daily_table


FORMATS = ("csv", "json", "parquet")
DEFAULT_VERIFIED_DAILY_EVENTS = (
    ROOT / "data" / "processed" / "real_flash_flood_chain" / "flash_flood_events_verified_combined_daily.csv"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit flash-flood province-day label outputs and unresolved event-day rows.")
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data" / "processed" / "labels" / "flash_flood_province_day_labels_verified_chain.parquet",
        help="Province-day label table to audit.",
    )
    parser.add_argument(
        "--events-daily",
        type=Path,
        help="Optional daily verified event table. Defaults to the verified daily chain when present, otherwise the built-in handoff seed events.",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Number of ranked rows to keep in each summary list.")
    return parser.parse_args(argv)


def _infer_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in FORMATS:
        return suffix
    raise ValueError(f"Could not infer table format from path: {path}")


def _read_table(path: Path):
    fmt = _infer_format(path)
    if fmt == "csv":
        import pandas as pd

        return pd.read_csv(path)
    if fmt == "json":
        import pandas as pd

        return pd.read_json(path)
    import pandas as pd

    return pd.read_parquet(path)


def _load_default_events_daily():
    if DEFAULT_VERIFIED_DAILY_EVENTS.exists():
        return _read_table(DEFAULT_VERIFIED_DAILY_EVENTS)
    return expand_flash_flood_events_to_daily_table()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    labels = _read_table(args.labels)
    events_daily = _read_table(args.events_daily) if args.events_daily else _load_default_events_daily()
    summary = audit_flash_flood_province_day_labels(labels, event_daily_table=events_daily, top_n=args.top_n)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
