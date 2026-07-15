#!/usr/bin/env python3
"""Audit converted Saudi STCast inputs for temporal and variable completeness."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


PRESSURE_LEVELS = [1000.0, 925.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 250.0, 200.0, 150.0, 100.0, 50.0]
PRESSURE_VARIABLES = ("z", "q", "u", "v", "t")
SURFACE_VARIABLES = ("t2m", "u10", "v10", "msl")


@dataclass
class AuditResult:
    is_complete: bool
    expected_day_count: int
    actual_pressure_day_count: int
    actual_surface_day_count: int
    expected_hours: list[str]
    discovered_hours: list[str]
    missing_days: list[str]
    extra_pressure_days: list[str]
    extra_surface_days: list[str]
    missing_pressure_files: dict[str, list[str]]
    missing_surface_files: dict[str, list[str]]
    unexpected_pressure_hours: dict[str, list[str]]
    unexpected_surface_hours: dict[str, list[str]]
    pressure_hour_count_distribution: dict[str, int]
    surface_hour_count_distribution: dict[str, int]
    pressure_timestamp_range: dict[str, str | None]
    surface_timestamp_range: dict[str, str | None]
    stats_validation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_complete": self.is_complete,
            "expected_day_count": self.expected_day_count,
            "actual_pressure_day_count": self.actual_pressure_day_count,
            "actual_surface_day_count": self.actual_surface_day_count,
            "expected_hours": self.expected_hours,
            "discovered_hours": self.discovered_hours,
            "missing_days": self.missing_days,
            "extra_pressure_days": self.extra_pressure_days,
            "extra_surface_days": self.extra_surface_days,
            "missing_pressure_files": self.missing_pressure_files,
            "missing_surface_files": self.missing_surface_files,
            "unexpected_pressure_hours": self.unexpected_pressure_hours,
            "unexpected_surface_hours": self.unexpected_surface_hours,
            "pressure_hour_count_distribution": self.pressure_hour_count_distribution,
            "surface_hour_count_distribution": self.surface_hour_count_distribution,
            "pressure_timestamp_range": self.pressure_timestamp_range,
            "surface_timestamp_range": self.surface_timestamp_range,
            "stats_validation": self.stats_validation,
        }

    def to_summary_dict(self, *, sample_limit: int = 3) -> dict[str, Any]:
        payload = self.to_dict()
        payload["unexpected_pressure_hours"] = _compact_day_mapping(self.unexpected_pressure_hours, sample_limit=sample_limit)
        payload["unexpected_surface_hours"] = _compact_day_mapping(self.unexpected_surface_hours, sample_limit=sample_limit)
        payload["missing_pressure_files"] = _compact_timestamp_mapping(self.missing_pressure_files, sample_limit=sample_limit)
        payload["missing_surface_files"] = _compact_timestamp_mapping(self.missing_surface_files, sample_limit=sample_limit)
        return payload


def _iter_expected_days(year: int) -> list[str]:
    current = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    days: list[str] = []
    while current < end:
        days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return days


def _expected_hours_from_cadence(cadence_hours: int) -> list[str]:
    if cadence_hours <= 0 or 24 % cadence_hours != 0:
        raise ValueError(f"cadence_hours must divide 24 cleanly; got {cadence_hours}")
    return [f"{hour:02d}:00:00" for hour in range(0, 24, cadence_hours)]


def _parse_pressure_name(name: str) -> tuple[str, str, str] | None:
    stem = Path(name).stem
    parts = stem.split("-")
    if len(parts) != 3:
        return None
    hour, var_name, level = parts
    return hour, var_name, level


def _parse_surface_name(name: str) -> tuple[str, str] | None:
    stem = Path(name).stem
    hour, sep, var_name = stem.partition("-")
    if not sep:
        return None
    return hour, var_name


def _pressure_day_summary(day_dir: Path) -> tuple[set[str], dict[str, set[str]]]:
    hours: set[str] = set()
    observed: dict[str, set[str]] = defaultdict(set)
    for path in day_dir.glob("*.npy"):
        parsed = _parse_pressure_name(path.name)
        if parsed is None:
            continue
        hour, var_name, level = parsed
        hours.add(hour)
        observed[hour].add(f"{var_name}-{level}")
    return hours, observed


def _surface_day_summary(day_dir: Path) -> tuple[set[str], dict[str, set[str]]]:
    hours: set[str] = set()
    observed: dict[str, set[str]] = defaultdict(set)
    for path in day_dir.glob("*.npy"):
        parsed = _parse_surface_name(path.name)
        if parsed is None:
            continue
        hour, var_name = parsed
        hours.add(hour)
        observed[hour].add(var_name)
    return hours, observed


def _timestamp_range(timestamps: list[str]) -> dict[str, str | None]:
    if not timestamps:
        return {"start": None, "end": None}
    return {"start": timestamps[0], "end": timestamps[-1]}


def _compact_day_mapping(values: dict[str, list[str]], *, sample_limit: int) -> dict[str, Any]:
    if not values:
        return {}
    items = sorted(values.items())
    sample = [{"day": day, "hours": hours[:sample_limit], "hour_count": len(hours)} for day, hours in items[:sample_limit]]
    distinct_hours = sorted({hour for hours in values.values() for hour in hours})
    return {
        "day_count": len(values),
        "sample_days": sample,
        "distinct_hours": distinct_hours,
    }


def _compact_timestamp_mapping(values: dict[str, list[str]], *, sample_limit: int) -> dict[str, Any]:
    if not values:
        return {}
    items = sorted(values.items())
    sample = [{"timestamp": stamp, "missing": missing[:sample_limit], "missing_count": len(missing)} for stamp, missing in items[:sample_limit]]
    return {
        "timestamp_count": len(values),
        "sample_timestamps": sample,
    }


def _load_stats_validation(
    stats_dir: Path | None,
    stats_start: datetime | None,
    stats_end: datetime | None,
    stats_step_hours: int | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"checked": False}
    if stats_dir is None:
        return result

    mean_std_path = stats_dir / "mean_std.json"
    single_path = stats_dir / "mean_std_single.json"
    if not mean_std_path.exists() or not single_path.exists():
        result.update({"checked": True, "valid": False, "reason": "missing_stats_files"})
        return result

    result["checked"] = True
    payload = json.loads(mean_std_path.read_text(encoding="utf-8"))
    single_payload = json.loads(single_path.read_text(encoding="utf-8"))
    result["mean_std_count"] = payload.get("count")
    result["mean_std_single_count"] = single_payload.get("count")
    result["current_date"] = payload.get("current_date")

    if stats_start is None or stats_end is None or stats_step_hours is None:
        result["valid"] = None
        result["reason"] = "window_not_provided"
        return result

    expected_count = 0
    current = stats_start
    while current <= stats_end:
        expected_count += 1
        current += timedelta(hours=stats_step_hours)
    result["expected_count"] = expected_count
    result["valid"] = payload.get("count") == expected_count and single_payload.get("count") == expected_count
    if result["valid"] is False:
        result["reason"] = "count_mismatch"
    return result


def audit_stcast_dataset(
    root_dir: Path,
    *,
    year: int,
    cadence_hours: int,
    stats_dir: Path | None = None,
    stats_start: datetime | None = None,
    stats_end: datetime | None = None,
    stats_step_hours: int | None = None,
) -> AuditResult:
    pressure_root = root_dir / str(year)
    surface_root = root_dir / "single" / str(year)
    expected_days = _iter_expected_days(year)
    expected_hours = _expected_hours_from_cadence(cadence_hours)

    actual_pressure_days = sorted(path.name for path in pressure_root.iterdir() if path.is_dir()) if pressure_root.exists() else []
    actual_surface_days = sorted(path.name for path in surface_root.iterdir() if path.is_dir()) if surface_root.exists() else []
    expected_day_set = set(expected_days)
    pressure_day_set = set(actual_pressure_days)
    surface_day_set = set(actual_surface_days)

    missing_days = sorted(expected_day_set - (pressure_day_set & surface_day_set))
    extra_pressure_days = sorted(pressure_day_set - expected_day_set)
    extra_surface_days = sorted(surface_day_set - expected_day_set)

    pressure_missing: dict[str, list[str]] = {}
    surface_missing: dict[str, list[str]] = {}
    unexpected_pressure_hours: dict[str, list[str]] = {}
    unexpected_surface_hours: dict[str, list[str]] = {}
    pressure_hours_seen: set[str] = set()
    surface_hours_seen: set[str] = set()
    pressure_hour_counts: Counter[int] = Counter()
    surface_hour_counts: Counter[int] = Counter()
    pressure_timestamps_present: list[str] = []
    surface_timestamps_present: list[str] = []

    expected_pressure_tokens = {f"{var_name}-{level}" for var_name in PRESSURE_VARIABLES for level in PRESSURE_LEVELS}
    expected_surface_tokens = set(SURFACE_VARIABLES)

    for day in expected_days:
        day_pressure_dir = pressure_root / day
        day_surface_dir = surface_root / day
        if not day_pressure_dir.exists() or not day_surface_dir.exists():
            continue

        pressure_hours, pressure_observed = _pressure_day_summary(day_pressure_dir)
        surface_hours, surface_observed = _surface_day_summary(day_surface_dir)
        pressure_hours_seen.update(pressure_hours)
        surface_hours_seen.update(surface_hours)
        pressure_hour_counts[len(pressure_hours)] += 1
        surface_hour_counts[len(surface_hours)] += 1
        for hour in sorted(pressure_hours):
            pressure_timestamps_present.append(f"{day}T{hour}")
        for hour in sorted(surface_hours):
            surface_timestamps_present.append(f"{day}T{hour}")

        extra_pressure_hours = sorted(pressure_hours - set(expected_hours))
        extra_surface_hours = sorted(surface_hours - set(expected_hours))
        if extra_pressure_hours:
            unexpected_pressure_hours[day] = extra_pressure_hours
        if extra_surface_hours:
            unexpected_surface_hours[day] = extra_surface_hours

        for hour in expected_hours:
            pressure_missing_tokens = sorted(expected_pressure_tokens - pressure_observed.get(hour, set()))
            if pressure_missing_tokens:
                pressure_missing[f"{day}T{hour}"] = pressure_missing_tokens
            surface_missing_tokens = sorted(expected_surface_tokens - surface_observed.get(hour, set()))
            if surface_missing_tokens:
                surface_missing[f"{day}T{hour}"] = surface_missing_tokens

    stats_validation = _load_stats_validation(stats_dir, stats_start, stats_end, stats_step_hours)

    is_complete = (
        not missing_days
        and not extra_pressure_days
        and not extra_surface_days
        and not pressure_missing
        and not surface_missing
        and not unexpected_pressure_hours
        and not unexpected_surface_hours
    )
    if stats_validation.get("checked") and stats_validation.get("valid") is False:
        is_complete = False

    return AuditResult(
        is_complete=is_complete,
        expected_day_count=len(expected_days),
        actual_pressure_day_count=len(actual_pressure_days),
        actual_surface_day_count=len(actual_surface_days),
        expected_hours=expected_hours,
        discovered_hours=sorted(pressure_hours_seen | surface_hours_seen),
        missing_days=missing_days,
        extra_pressure_days=extra_pressure_days,
        extra_surface_days=extra_surface_days,
        missing_pressure_files=pressure_missing,
        missing_surface_files=surface_missing,
        unexpected_pressure_hours=unexpected_pressure_hours,
        unexpected_surface_hours=unexpected_surface_hours,
        pressure_hour_count_distribution={str(key): value for key, value in sorted(pressure_hour_counts.items())},
        surface_hour_count_distribution={str(key): value for key, value in sorted(surface_hour_counts.items())},
        pressure_timestamp_range=_timestamp_range(sorted(pressure_timestamps_present)),
        surface_timestamp_range=_timestamp_range(sorted(surface_timestamps_present)),
        stats_validation=stats_validation,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a converted Saudi STCast dataset.")
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--cadence-hours", type=int, default=6, help="Expected timestamp cadence in hours.")
    parser.add_argument("--stats-dir", type=Path, help="Optional stats directory containing mean_std JSON files.")
    parser.add_argument("--stats-start", type=str, help="Optional stats window start, e.g. 2024-01-01T00:00:00.")
    parser.add_argument("--stats-end", type=str, help="Optional stats window end, e.g. 2024-10-31T18:00:00.")
    parser.add_argument("--stats-step-hours", type=int, help="Optional stats cadence in hours for count validation.")
    parser.add_argument("--output-json", type=Path, help="Optional path to write the full audit JSON payload.")
    parser.add_argument("--compact", action="store_true", help="Print a compact summary instead of the full per-day anomaly payload.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stats_start = datetime.fromisoformat(args.stats_start) if args.stats_start else None
    stats_end = datetime.fromisoformat(args.stats_end) if args.stats_end else None
    result = audit_stcast_dataset(
        args.root_dir,
        year=args.year,
        cadence_hours=args.cadence_hours,
        stats_dir=args.stats_dir,
        stats_start=stats_start,
        stats_end=stats_end,
        stats_step_hours=args.stats_step_hours,
    )
    full_payload = result.to_dict()
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(full_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = result.to_summary_dict() if args.compact else full_payload
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.is_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
