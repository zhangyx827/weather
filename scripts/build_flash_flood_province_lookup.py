#!/usr/bin/env python3
"""Build a reusable flash-flood latitude/longitude to province lookup table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.data import build_flash_flood_province_lookup


FORMATS = ("csv", "json", "parquet")
BOUNDARY_TABULAR_FORMATS = ("csv", "json", "parquet")
BOUNDARY_GEOJSON_FORMATS = ("geojson", "json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a flash-flood latitude/longitude to province lookup table.")
    parser.add_argument("--features", type=Path, required=True, help="Feature table containing latitude/longitude columns.")
    parser.add_argument("--boundaries", type=Path, required=True, help="Province boundary file, usually GeoJSON.")
    parser.add_argument("--boundary-format", choices=("auto", "geojson", "csv", "json", "parquet"), default="auto")
    parser.add_argument("--geometry-format", choices=("geojson", "wkt"), default="geojson")
    parser.add_argument("--geometry-column", default="geometry", help="Geometry column for tabular boundary files.")
    parser.add_argument("--province-column", default="province_name", help="Province column to emit in the lookup table.")
    parser.add_argument("--boundary-province-column", default="province_name", help="Province column in the boundary source.")
    parser.add_argument("--boundary-id-column", default="boundary_id", help="Optional identifier column in the boundary source.")
    parser.add_argument("--coordinate-precision", type=int, default=4, help="Decimal precision used for lookup coordinates.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "labels" / "flash_flood_province_lookup.parquet",
        help="Output path for the generated lookup table.",
    )
    parser.add_argument("--format", choices=FORMATS, help="Output format. Defaults to the output file suffix.")
    return parser.parse_args(argv)


def _infer_format(path: Path, explicit: str | None, allowed: tuple[str, ...]) -> str:
    if explicit and explicit != "auto":
        return explicit
    suffix = path.suffix.lower().lstrip(".")
    if suffix in allowed:
        return suffix
    raise ValueError(f"Could not infer format from path: {path}")


def _read_table(path: Path):
    fmt = _infer_format(path, None, FORMATS)
    if fmt == "csv":
        import pandas as pd

        return pd.read_csv(path)
    if fmt == "json":
        import pandas as pd

        return pd.read_json(path)
    import pandas as pd

    return pd.read_parquet(path)


def _read_feature_coordinates_streaming(path: Path, *, coordinate_precision: int):
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("Parquet coordinate streaming requires pyarrow") from exc

    parquet_file = pq.ParquetFile(path)
    total_rows = 0
    coordinate_batches = []
    for record_batch in parquet_file.iter_batches(columns=["latitude", "longitude"]):
        batch = record_batch.to_pandas()
        total_rows += int(len(batch))
        coordinate_batches.append(batch)
    if not coordinate_batches:
        raise ValueError(f"No rows were read from parquet input: {path}")

    import pandas as pd

    coordinates = pd.concat(coordinate_batches, ignore_index=True)
    coordinates["latitude"] = pd.to_numeric(coordinates["latitude"], errors="coerce").round(coordinate_precision)
    coordinates["longitude"] = pd.to_numeric(coordinates["longitude"], errors="coerce").round(coordinate_precision)
    coordinates = coordinates.dropna(subset=["latitude", "longitude"]).drop_duplicates().reset_index(drop=True)
    return coordinates, total_rows


def _read_boundary_table(
    path: Path,
    *,
    boundary_format: str,
    boundary_province_column: str,
    boundary_id_column: str,
):
    resolved = _infer_format(path, boundary_format, BOUNDARY_TABULAR_FORMATS + ("geojson",))
    if resolved == "geojson":
        import pandas as pd

        payload = json.loads(path.read_text(encoding="utf-8"))
        if _normalize_text(payload.get("type")) != "featurecollection":
            raise ValueError("GeoJSON boundary file must be a FeatureCollection")
        rows: list[dict[str, object]] = []
        for index, feature in enumerate(payload.get("features") or []):
            properties = feature.get("properties") or {}
            row = dict(properties)
            row.setdefault(boundary_id_column, feature.get("id", index))
            row["geometry"] = feature.get("geometry")
            rows.append(row)
        return pd.DataFrame.from_records(rows)
    if resolved == "csv":
        import pandas as pd

        return pd.read_csv(path)
    if resolved == "json":
        import pandas as pd

        return pd.read_json(path)
    import pandas as pd

    return pd.read_parquet(path)


def _normalize_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def _resolve_boundary_column(boundaries, requested: str, candidates: tuple[str, ...], *, label: str) -> str:
    if requested == "boundary_id":
        for candidate in candidates:
            if candidate in boundaries.columns:
                return candidate
    if requested in boundaries.columns:
        return requested
    for candidate in candidates:
        if candidate in boundaries.columns:
            return candidate
    raise KeyError(f"Boundary table is missing {label} column '{requested}' and no fallback candidates were found")


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


def _load_feature_coordinates(path: Path, *, coordinate_precision: int):
    feature_format = _infer_format(path, None, FORMATS)
    if feature_format == "parquet":
        return _read_feature_coordinates_streaming(path, coordinate_precision=coordinate_precision)
    table = _read_table(path)
    total_rows = int(len(table))
    coordinates = table.loc[:, ["latitude", "longitude"]].copy()
    import pandas as pd

    coordinates["latitude"] = pd.to_numeric(coordinates["latitude"], errors="coerce").round(coordinate_precision)
    coordinates["longitude"] = pd.to_numeric(coordinates["longitude"], errors="coerce").round(coordinate_precision)
    coordinates = coordinates.dropna(subset=["latitude", "longitude"]).drop_duplicates().reset_index(drop=True)
    return coordinates, total_rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_format = _infer_format(args.output, args.format, FORMATS)
    features, input_rows = _load_feature_coordinates(args.features, coordinate_precision=args.coordinate_precision)
    boundaries = _read_boundary_table(
        args.boundaries,
        boundary_format=args.boundary_format,
        boundary_province_column=args.boundary_province_column,
        boundary_id_column=args.boundary_id_column,
    )
    boundary_province_column = _resolve_boundary_column(
        boundaries,
        args.boundary_province_column,
        ("shapeName", "NAME_1", "admin1_name", "province"),
        label="province",
    )
    boundary_id_column = _resolve_boundary_column(
        boundaries,
        args.boundary_id_column,
        ("shapeID", "id", "ID_1"),
        label="boundary id",
    )
    lookup = build_flash_flood_province_lookup(
        features,
        boundaries,
        province_column=args.province_column,
        boundary_province_column=boundary_province_column,
        boundary_id_column=boundary_id_column,
        geometry_column=args.geometry_column,
        geometry_format=args.geometry_format,
        coordinate_precision=args.coordinate_precision,
    )
    _write_table(lookup, args.output, output_format)

    summary = {
        "input_rows": int(input_rows),
        "unique_coordinate_rows": int(len(lookup)),
        "matched_coordinate_rows": int((lookup["match_status"] == "matched").sum()),
        "unmatched_coordinate_rows": int((lookup["match_status"] == "unmatched").sum()),
        "province_column": args.province_column,
        "boundary_province_column": boundary_province_column,
        "boundary_id_column": boundary_id_column,
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
