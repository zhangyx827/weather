#!/usr/bin/env python3
"""Audit flash-flood province lookup coverage and unmatched coordinate hotspots."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.data import audit_flash_flood_province_lookup


FORMATS = ("csv", "json", "parquet")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit flash-flood province lookup coverage and unmatched coordinate clusters.")
    parser.add_argument(
        "--lookup",
        type=Path,
        default=ROOT / "data" / "processed" / "labels" / "flash_flood_province_lookup.parquet",
        help="Latitude/longitude to province lookup table built by the province lookup script.",
    )
    parser.add_argument("--features", type=Path, help="Optional feature table used to weight unmatched coordinates by row count.")
    parser.add_argument("--boundaries", type=Path, help="Optional admin boundary file used to classify unmatched coordinates.")
    parser.add_argument("--boundary-format", choices=("auto", "geojson", "csv", "json", "parquet"), default="auto")
    parser.add_argument("--geometry-format", choices=("geojson", "wkt"), default="geojson")
    parser.add_argument("--geometry-column", default="geometry", help="Geometry column for tabular boundary files.")
    parser.add_argument("--boundary-province-column", default="province_name", help="Province column in the boundary source.")
    parser.add_argument("--coordinate-precision", type=int, default=4, help="Decimal precision used for coordinate joins.")
    parser.add_argument("--bin-size-degrees", type=float, default=1.0, help="Coarse lat/lon bin size for hotspot summaries.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of hotspot rows to keep in each ranked output.")
    parser.add_argument(
        "--runtime-stats",
        action="store_true",
        help="Emit audit runtime counters such as zero-candidate rows and candidate-count statistics.",
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="Run the audit under cProfile and include the top function timings in the JSON output.",
    )
    parser.add_argument(
        "--cprofile-sort",
        choices=("cumulative", "tottime", "calls"),
        default="cumulative",
        help="Sort key used when summarizing cProfile output.",
    )
    parser.add_argument("--cprofile-top", type=int, default=20, help="Number of cProfile rows to keep in the JSON output.")
    return parser.parse_args(argv)


def _infer_format(path: Path, explicit: str | None) -> str:
    if explicit and explicit != "auto":
        return explicit
    suffix = path.suffix.lower().lstrip(".")
    if suffix in FORMATS or suffix == "geojson":
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


def _read_feature_coordinate_counts(path: Path, *, coordinate_precision: int):
    fmt = _infer_format(path, None)
    if fmt != "parquet":
        return _read_table(path)
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("Parquet feature streaming requires pyarrow") from exc

    import pandas as pd

    parquet_file = pq.ParquetFile(path)
    counts: dict[tuple[float, float], int] = {}
    for record_batch in parquet_file.iter_batches(columns=["latitude", "longitude"]):
        batch = record_batch.to_pandas()
        batch["latitude"] = pd.to_numeric(batch["latitude"], errors="coerce").round(coordinate_precision)
        batch["longitude"] = pd.to_numeric(batch["longitude"], errors="coerce").round(coordinate_precision)
        batch = batch.dropna(subset=["latitude", "longitude"])
        if batch.empty:
            continue
        grouped = batch.groupby(["latitude", "longitude"], dropna=False).size()
        for (latitude, longitude), value in grouped.items():
            key = (float(latitude), float(longitude))
            counts[key] = counts.get(key, 0) + int(value)
    return pd.DataFrame(
        [
            {"latitude": latitude, "longitude": longitude, "feature_row_count": feature_row_count}
            for (latitude, longitude), feature_row_count in sorted(counts.items())
        ]
    )


def _read_boundary_table(path: Path, *, boundary_format: str):
    resolved = _infer_format(path, boundary_format)
    if resolved == "geojson":
        import pandas as pd

        payload = json.loads(path.read_text(encoding="utf-8"))
        rows: list[dict[str, object]] = []
        for index, feature in enumerate(payload.get("features") or []):
            row = dict(feature.get("properties") or {})
            row.setdefault("boundary_id", feature.get("id", index))
            row["geometry"] = feature.get("geometry")
            rows.append(row)
        return pd.DataFrame.from_records(rows)
    return _read_table(path)


def _resolve_boundary_column(boundaries, requested: str, candidates: tuple[str, ...], *, label: str) -> str:
    if requested in boundaries.columns:
        return requested
    for candidate in candidates:
        if candidate in boundaries.columns:
            return candidate
    raise KeyError(f"Boundary table is missing {label} column '{requested}' and no fallback candidates were found")


def _run_audit(args, lookup, features, boundaries, boundary_province_column: str):
    return audit_flash_flood_province_lookup(
        lookup,
        feature_table=features,
        boundary_table=boundaries,
        boundary_province_column=boundary_province_column,
        geometry_column=args.geometry_column,
        geometry_format=args.geometry_format,
        coordinate_precision=args.coordinate_precision,
        bin_size_degrees=args.bin_size_degrees,
        top_n=args.top_n,
        include_runtime_stats=args.runtime_stats,
    )


def _profile_summary(stats: pstats.Stats, *, top_n: int, sort_key: str) -> list[dict[str, object]]:
    if sort_key == "tottime":
        metric_index = 2
    elif sort_key == "calls":
        metric_index = 1
    else:
        metric_index = 3
    rows: list[dict[str, object]] = []
    for function_key, stat in sorted(stats.stats.items(), key=lambda item: item[1][metric_index], reverse=True)[:top_n]:
        filename, line_number, function_name = function_key
        primitive_calls, total_calls, total_time, cumulative_time, _ = stat
        rows.append(
            {
                "function": function_name,
                "file": filename,
                "line": int(line_number),
                "primitive_calls": int(primitive_calls),
                "total_calls": int(total_calls),
                "total_time_seconds": float(total_time),
                "cumulative_time_seconds": float(cumulative_time),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    lookup = _read_table(args.lookup)
    features = _read_feature_coordinate_counts(args.features, coordinate_precision=args.coordinate_precision) if args.features else None
    boundaries = _read_boundary_table(args.boundaries, boundary_format=args.boundary_format) if args.boundaries else None
    boundary_province_column = args.boundary_province_column
    if boundaries is not None:
        boundary_province_column = _resolve_boundary_column(
            boundaries,
            args.boundary_province_column,
            ("shapeName", "NAME_1", "admin1_name", "province"),
            label="province",
        )
    if args.cprofile:
        profiler = cProfile.Profile()
        profiler.enable()
        summary = _run_audit(args, lookup, features, boundaries, boundary_province_column)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats(args.cprofile_sort)
        summary["cprofile"] = {
            "sort": args.cprofile_sort,
            "top_n": int(args.cprofile_top),
            "top_functions": _profile_summary(stats, top_n=args.cprofile_top, sort_key=args.cprofile_sort),
        }
    else:
        summary = _run_audit(args, lookup, features, boundaries, boundary_province_column)
    summary["boundary_province_column"] = boundary_province_column if boundaries is not None else None
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
