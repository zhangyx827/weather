#!/usr/bin/env python3
"""Build STCast normalization statistics from one or more converted Saudi datasets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


PRESSURE_LEVELS = [1000.0, 925.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 250.0, 200.0, 150.0, 100.0, 50.0]
PRESSURE_VARIABLES = ("z", "q", "u", "v", "t")
SURFACE_VARIABLES = ("t2m", "u10", "v10", "msl")


def _parse_source_spec(value: str) -> dict[str, object]:
    parts = [part.strip() for part in value.split(",", 2)]
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "source spec must be 'root_dir,train_start,train_end' with ISO timestamps, "
            f"got: {value!r}"
        )
    return {
        "root_dir": Path(parts[0]),
        "train_start": datetime.fromisoformat(parts[1]),
        "train_end": datetime.fromisoformat(parts[2]),
    }


def _iter_timestamps(start: datetime, end: datetime, step_hours: int) -> list[datetime]:
    values: list[datetime] = []
    current = start
    while current <= end:
        values.append(current)
        current += timedelta(hours=step_hours)
    return values


def _mean_std(values: list[np.ndarray]) -> tuple[list[float], float]:
    stacked = np.concatenate([v.reshape(v.shape[0], -1) for v in values], axis=1)
    per_channel = stacked.mean(axis=1)
    overall = float(stacked.mean())
    return per_channel.astype(float).tolist(), overall


def _std(values: list[np.ndarray]) -> tuple[list[float], float]:
    stacked = np.concatenate([v.reshape(v.shape[0], -1) for v in values], axis=1)
    per_channel = stacked.std(axis=1)
    overall = float(stacked.std())
    return per_channel.astype(float).tolist(), overall


def _load_pressure_sample(root: Path, stamp: datetime, var_name: str) -> np.ndarray:
    day_dir = root / f"{stamp:%Y}" / f"{stamp:%Y-%m-%d}"
    fields = []
    for level in PRESSURE_LEVELS:
        path = day_dir / f"{stamp:%H}:00:00-{var_name}-{level}.npy"
        fields.append(np.load(path).astype(np.float64))
    return np.stack(fields, axis=0)


def _load_surface_sample(root: Path, stamp: datetime) -> np.ndarray:
    day_dir = root / "single" / f"{stamp:%Y}" / f"{stamp:%Y-%m-%d}"
    return np.stack([np.load(day_dir / f"{stamp:%H}:00:00-{name}.npy").astype(np.float64) for name in SURFACE_VARIABLES], axis=0)


def _collect_samples(root: Path, start: datetime, end: datetime, step_hours: int) -> tuple[dict[str, list[np.ndarray]], list[np.ndarray], int]:
    timestamps = _iter_timestamps(start, end, step_hours)
    if not timestamps:
        raise ValueError("No timestamps selected for statistics generation")
    pressure_stats: dict[str, list[np.ndarray]] = {name: [] for name in PRESSURE_VARIABLES}
    surface_stats: list[np.ndarray] = []
    count = 0

    for stamp in timestamps:
        for name in PRESSURE_VARIABLES:
            pressure_stats[name].append(_load_pressure_sample(root, stamp, name))
        surface_stats.append(_load_surface_sample(root, stamp))
        count += 1

    return pressure_stats, surface_stats, count


def build_stats(sources: list[dict[str, object]], stats_dir: Path, step_hours: int) -> None:
    if not sources:
        raise ValueError("At least one source must be provided")

    pressure_stats: dict[str, list[np.ndarray]] = {name: [] for name in PRESSURE_VARIABLES}
    surface_stats: list[np.ndarray] = []
    count = 0
    latest_end: datetime | None = None

    for source in sources:
        root = Path(source["root_dir"])
        start = source["train_start"]
        end = source["train_end"]
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            raise TypeError("train_start and train_end must be datetime instances")
        source_pressure, source_surface, source_count = _collect_samples(root, start, end, step_hours)
        for name in PRESSURE_VARIABLES:
            pressure_stats[name].extend(source_pressure[name])
        surface_stats.extend(source_surface)
        count += source_count
        latest_end = end if latest_end is None or end > latest_end else latest_end

    mean_payload: dict[str, object] = {}
    std_payload: dict[str, object] = {}
    for name in PRESSURE_VARIABLES:
        mean_values, mean_overall = _mean_std(pressure_stats[name])
        std_values, std_overall = _std(pressure_stats[name])
        mean_payload[name] = mean_values
        mean_payload[f"{name}_overall"] = mean_overall
        std_payload[name] = std_values
        std_payload[f"{name}_overall"] = std_overall

    surface_mean, _ = _mean_std(surface_stats)
    surface_std, _ = _std(surface_stats)
    single_mean = {name: float(surface_mean[idx]) for idx, name in enumerate(SURFACE_VARIABLES)}
    single_std = {name: float(surface_std[idx]) for idx, name in enumerate(SURFACE_VARIABLES)}

    stats_dir.mkdir(parents=True, exist_ok=True)
    with (stats_dir / "mean_std.json").open("w", encoding="utf-8") as fh:
        current_date = f"{latest_end:%Y/%Y-%m-%dT%H:00:00.nc}" if latest_end is not None else None
        json.dump({"mean": mean_payload, "std": std_payload, "count": count, "current_date": current_date}, fh)
    with (stats_dir / "mean_std_single.json").open("w", encoding="utf-8") as fh:
        json.dump({"mean": single_mean, "std": single_std, "count": count}, fh)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=Path, help="Legacy single-source root dir.")
    parser.add_argument("--stats-dir", required=True, type=Path)
    parser.add_argument("--train-start", type=str, help="Legacy single-source start, YYYY-mm-ddTHH:MM")
    parser.add_argument("--train-end", type=str, help="Legacy single-source end, YYYY-mm-ddTHH:MM")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Repeatable source spec: root_dir,train_start,train_end. Example: data/processed/stcast_saudi_2022_6h,2022-01-01T00:00,2022-12-31T18:00",
    )
    parser.add_argument("--step-hours", default=6, type=int)
    args = parser.parse_args()

    sources: list[dict[str, object]]
    if args.source:
        sources = [_parse_source_spec(value) for value in args.source]
    else:
        if args.root_dir is None or args.train_start is None or args.train_end is None:
            parser.error("either repeat --source or provide --root-dir, --train-start, and --train-end")
        sources = [
            {
                "root_dir": args.root_dir,
                "train_start": datetime.fromisoformat(args.train_start),
                "train_end": datetime.fromisoformat(args.train_end),
            }
        ]

    build_stats(sources, args.stats_dir, args.step_hours)
    print(f"Wrote statistics to {args.stats_dir}")


if __name__ == "__main__":
    main()
