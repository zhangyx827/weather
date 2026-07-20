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
WARNING_LIMIT = 20
_warning_count = 0
_skipped_file_count = 0
_replaced_value_count = 0


def _sanitize_array(array: np.ndarray, label: str) -> np.ndarray | None:
    global _warning_count, _skipped_file_count, _replaced_value_count
    finite = np.isfinite(array)
    if finite.all():
        return array
    bad_count = int((~finite).sum())
    if not finite.any():
        _skipped_file_count += 1
        if _warning_count < WARNING_LIMIT:
            print(f"WARNING: skipping file with all non-finite values: {label}")
            _warning_count += 1
        return None
    _replaced_value_count += bad_count
    if _warning_count < WARNING_LIMIT:
        print(f"WARNING: replacing {bad_count} non-finite values with NaN in {label}")
        _warning_count += 1
    cleaned = array.copy()
    cleaned[~finite] = np.nan
    return cleaned


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


def _new_running_stats(num_channels: int) -> dict[str, np.ndarray | float]:
    return {
        "sum": np.zeros(num_channels, dtype=np.float64),
        "sumsq": np.zeros(num_channels, dtype=np.float64),
        "count": np.zeros(num_channels, dtype=np.int64),
        "total_sum": 0.0,
        "total_sumsq": 0.0,
        "total_count": 0,
    }


def _update_running_stats(stats: dict[str, np.ndarray | float], sample: np.ndarray) -> None:
    finite = np.isfinite(sample)
    if not finite.any():
        return
    safe = np.where(finite, sample, 0.0)
    per_channel = safe.reshape(sample.shape[0], -1)
    per_channel_finite = finite.reshape(sample.shape[0], -1)
    stats["sum"] += per_channel.sum(axis=1)
    stats["sumsq"] += np.square(per_channel).sum(axis=1)
    stats["count"] += per_channel_finite.sum(axis=1)
    stats["total_sum"] += float(safe[finite].sum())
    stats["total_sumsq"] += float(np.square(safe[finite]).sum())
    stats["total_count"] += int(finite.sum())


def _finalize_running_stats(stats: dict[str, np.ndarray | float]) -> tuple[list[float], float, list[float], float]:
    count = np.asarray(stats["count"], dtype=np.float64)
    if np.any(count == 0):
        raise ValueError("No finite values available for one or more channels")
    mean = np.asarray(stats["sum"], dtype=np.float64) / count
    variance = np.asarray(stats["sumsq"], dtype=np.float64) / count - np.square(mean)
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)
    total_count = float(stats["total_count"])
    if total_count <= 0:
        raise ValueError("No finite values available for aggregate stats")
    total_mean = float(stats["total_sum"]) / total_count
    total_variance = float(stats["total_sumsq"]) / total_count - total_mean * total_mean
    total_std = float(np.sqrt(max(total_variance, 0.0)))
    return mean.tolist(), total_mean, std.tolist(), total_std


def _load_pressure_sample(root: Path, stamp: datetime, var_name: str) -> np.ndarray:
    day_dir = root / f"{stamp:%Y}" / f"{stamp:%Y-%m-%d}"
    fields = []
    for level in PRESSURE_LEVELS:
        path = day_dir / f"{stamp:%H}:00:00-{var_name}-{level}.npy"
        field = np.load(path).astype(np.float64)
        field = _sanitize_array(field, str(path))
        if field is None:
            return None
        fields.append(field)
    return np.stack(fields, axis=0)


def _load_surface_sample(root: Path, stamp: datetime) -> np.ndarray:
    day_dir = root / "single" / f"{stamp:%Y}" / f"{stamp:%Y-%m-%d}"
    fields = []
    for name in SURFACE_VARIABLES:
        path = day_dir / f"{stamp:%H}:00:00-{name}.npy"
        field = np.load(path).astype(np.float64)
        field = _sanitize_array(field, str(path))
        if field is None:
            return None
        fields.append(field)
    return np.stack(fields, axis=0)


def _collect_samples(root: Path, start: datetime, end: datetime, step_hours: int) -> tuple[dict[str, dict[str, np.ndarray | float]], dict[str, np.ndarray | float], int]:
    timestamps = _iter_timestamps(start, end, step_hours)
    if not timestamps:
        raise ValueError("No timestamps selected for statistics generation")
    pressure_stats: dict[str, dict[str, np.ndarray | float]] = {name: _new_running_stats(len(PRESSURE_LEVELS)) for name in PRESSURE_VARIABLES}
    surface_stats = _new_running_stats(len(SURFACE_VARIABLES))
    count = 0

    for stamp in timestamps:
        for name in PRESSURE_VARIABLES:
            sample = _load_pressure_sample(root, stamp, name)
            if sample is None:
                continue
            _update_running_stats(pressure_stats[name], sample)
        surface_sample = _load_surface_sample(root, stamp)
        if surface_sample is not None:
            _update_running_stats(surface_stats, surface_sample)
        count += 1

    return pressure_stats, surface_stats, count


def build_stats(sources: list[dict[str, object]], stats_dir: Path, step_hours: int) -> None:
    if not sources:
        raise ValueError("At least one source must be provided")

    pressure_stats: dict[str, dict[str, np.ndarray | float]] = {name: _new_running_stats(len(PRESSURE_LEVELS)) for name in PRESSURE_VARIABLES}
    surface_stats = _new_running_stats(len(SURFACE_VARIABLES))
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
            pressure_stats[name]["sum"] += np.asarray(source_pressure[name]["sum"], dtype=np.float64)
            pressure_stats[name]["sumsq"] += np.asarray(source_pressure[name]["sumsq"], dtype=np.float64)
            pressure_stats[name]["count"] += np.asarray(source_pressure[name]["count"], dtype=np.int64)
            pressure_stats[name]["total_sum"] += float(source_pressure[name]["total_sum"])
            pressure_stats[name]["total_sumsq"] += float(source_pressure[name]["total_sumsq"])
            pressure_stats[name]["total_count"] += int(source_pressure[name]["total_count"])
        surface_stats["sum"] += np.asarray(source_surface["sum"], dtype=np.float64)
        surface_stats["sumsq"] += np.asarray(source_surface["sumsq"], dtype=np.float64)
        surface_stats["count"] += np.asarray(source_surface["count"], dtype=np.int64)
        surface_stats["total_sum"] += float(source_surface["total_sum"])
        surface_stats["total_sumsq"] += float(source_surface["total_sumsq"])
        surface_stats["total_count"] += int(source_surface["total_count"])
        count += source_count
        latest_end = end if latest_end is None or end > latest_end else latest_end

    mean_payload: dict[str, object] = {}
    std_payload: dict[str, object] = {}
    for name in PRESSURE_VARIABLES:
        mean_values, mean_overall, std_values, std_overall = _finalize_running_stats(pressure_stats[name])
        if not np.all(np.isfinite(mean_values)) or not np.all(np.isfinite(std_values)):
            raise ValueError(f"Non-finite aggregate stats found for {name}")
        if not np.isfinite(mean_overall) or not np.isfinite(std_overall):
            raise ValueError(f"Non-finite aggregate stats found for {name}")
        mean_payload[name] = mean_values
        mean_payload[f"{name}_overall"] = mean_overall
        std_payload[name] = std_values
        std_payload[f"{name}_overall"] = std_overall

    surface_mean, surface_mean_overall, surface_std, surface_std_overall = _finalize_running_stats(surface_stats)
    if not np.all(np.isfinite(surface_mean)) or not np.all(np.isfinite(surface_std)):
        raise ValueError("Non-finite aggregate stats found for surface")
    single_mean = {name: float(surface_mean[idx]) for idx, name in enumerate(SURFACE_VARIABLES)}
    single_std = {name: float(surface_std[idx]) for idx, name in enumerate(SURFACE_VARIABLES)}

    stats_dir.mkdir(parents=True, exist_ok=True)
    with (stats_dir / "mean_std.json").open("w", encoding="utf-8") as fh:
        current_date = f"{latest_end:%Y/%Y-%m-%dT%H:00:00.nc}" if latest_end is not None else None
        json.dump({"mean": mean_payload, "std": std_payload, "count": count, "current_date": current_date}, fh)
    with (stats_dir / "mean_std_single.json").open("w", encoding="utf-8") as fh:
        json.dump({"mean": single_mean, "std": single_std, "count": count}, fh)
    print(
        "WARNING SUMMARY: "
        f"skipped_files={_skipped_file_count}, "
        f"replaced_values={_replaced_value_count}, "
        f"surface_mean={surface_mean_overall:.6f}, "
        f"surface_std={surface_std_overall:.6f}"
    )


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
