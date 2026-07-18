#!/usr/bin/env python3
"""Build a province-day flash-flood feature table from Layer-4 grid-day features."""

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
    aggregate_flash_flood_features_to_province_day,
    enrich_flash_flood_features_with_province,
    province_day_numeric_feature_columns,
)


FORMATS = ("csv", "json", "parquet")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a province-day flash-flood feature table from Layer-4 grid-day features.")
    parser.add_argument("--features", type=Path, required=True, help="Flash-flood feature table, typically flash_flood_training.parquet.")
    parser.add_argument(
        "--province-lookup",
        type=Path,
        help="Optional latitude/longitude to province lookup table used when the feature table lacks a province column.",
    )
    parser.add_argument("--province-column", default="province_name", help="Province column to create/use on the feature table.")
    parser.add_argument(
        "--lookup-province-column",
        default="province_name",
        help="Province column name in the lookup table when --province-lookup is provided.",
    )
    parser.add_argument(
        "--coordinate-precision",
        type=int,
        default=4,
        help="Decimal precision used when matching latitude/longitude rows to the province lookup table.",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=250_000,
        help="Batch size for parquet streaming builds. Ignored for non-parquet runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "training" / "flash_flood_province_day_features.parquet",
        help="Output path for the province-day feature table.",
    )
    parser.add_argument("--format", choices=FORMATS, help="Output format. Defaults to the output file suffix.")
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


def _build_province_day_streaming(
    features_path: Path,
    *,
    province_lookup_path: Path | None,
    province_column: str,
    lookup_province_column: str,
    coordinate_precision: int,
    batch_rows: int,
):
    try:
        import pandas as pd
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("Parquet province-day streaming requires pyarrow and pandas") from exc

    if batch_rows <= 0:
        raise ValueError("--batch-rows must be positive")

    lookup = _read_table(province_lookup_path) if province_lookup_path is not None else None
    parquet_file = pq.ParquetFile(features_path)
    aggregate_map: dict[tuple[str, str], dict[str, object]] = {}
    input_rows = 0
    province_ready_rows = 0
    saw_numeric_feature = False

    for record_batch in parquet_file.iter_batches(batch_size=batch_rows):
        batch = record_batch.to_pandas()
        input_rows += int(len(batch))

        if province_column not in batch.columns:
            if lookup is None:
                raise KeyError(
                    f"Feature table is missing province column '{province_column}'. "
                    "Provide --province-lookup to attach province names before aggregation."
                )
            batch = enrich_flash_flood_features_with_province(
                batch,
                lookup,
                coordinate_precision=coordinate_precision,
                province_column=province_column,
                lookup_province_column=lookup_province_column,
            )

        if "hazard_type" in batch.columns:
            batch = batch[batch["hazard_type"].astype(str).str.lower() == "flash_flood"].copy()
        if batch.empty:
            continue

        batch["date"] = pd.to_datetime(batch["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        if batch["date"].isna().any():
            raise ValueError("feature_table contains invalid 'date' values")
        batch[province_column] = batch[province_column].fillna("").astype(str).str.strip().str.lower()
        province_ready_rows += int(batch[province_column].ne("").sum())
        batch = batch[batch[province_column].ne("")].copy()
        if batch.empty:
            continue

        numeric_columns = province_day_numeric_feature_columns(batch)
        if not numeric_columns:
            continue
        saw_numeric_feature = True

        grouped = batch.groupby(["date", province_column], dropna=False)
        sums = grouped[numeric_columns].sum().reset_index()
        counts = grouped.size().reset_index(name="grid_cell_count")
        degraded = None
        if "source_status" in batch.columns:
            degraded = (
                batch.assign(_is_degraded=batch["source_status"].astype(str).str.lower().eq("degraded").astype(int))
                .groupby(["date", province_column], dropna=False)["_is_degraded"]
                .sum()
                .reset_index(name="degraded_grid_cell_count")
            )

        merged = sums.merge(counts, on=["date", province_column], how="left", validate="1:1")
        if degraded is not None:
            merged = merged.merge(degraded, on=["date", province_column], how="left", validate="1:1")

        for record in merged.to_dict(orient="records"):
            key = (str(record["date"]), str(record[province_column]))
            state = aggregate_map.setdefault(
                key,
                {
                    "grid_cell_count": 0,
                    "degraded_grid_cell_count": 0,
                    "feature_sums": {},
                },
            )
            state["grid_cell_count"] = int(state["grid_cell_count"]) + int(record["grid_cell_count"])
            if "degraded_grid_cell_count" in record:
                state["degraded_grid_cell_count"] = int(state["degraded_grid_cell_count"]) + int(record["degraded_grid_cell_count"])
            for column in numeric_columns:
                current = float(state["feature_sums"].get(column, 0.0))
                state["feature_sums"][column] = current + float(record[column])

    if input_rows == 0:
        raise ValueError(f"No rows were read from parquet input: {features_path}")
    if not saw_numeric_feature:
        raise ValueError("feature_table has no numeric feature columns available for province-day aggregation")
    if not aggregate_map:
        raise ValueError(f"feature_table has no rows with a usable {province_column}")

    records: list[dict[str, object]] = []
    for (date_value, province_value), state in sorted(aggregate_map.items()):
        count = int(state["grid_cell_count"])
        row = {
            "date": date_value,
            "hazard_type": "flash_flood",
            province_column: province_value,
            "grid_cell_count": count,
        }
        degraded_count = int(state["degraded_grid_cell_count"])
        if degraded_count:
            row["degraded_grid_cell_count"] = degraded_count
        for column, total in sorted(state["feature_sums"].items()):
            row[column] = float(total) / float(count)
        records.append(row)

    province_day = pd.DataFrame.from_records(records)
    first_columns = ["date", "hazard_type", province_column, "grid_cell_count"]
    if "degraded_grid_cell_count" in province_day.columns:
        first_columns.append("degraded_grid_cell_count")
    ordered_columns = first_columns + [column for column in province_day.columns if column not in first_columns]
    province_day = province_day.loc[:, ordered_columns].reset_index(drop=True)
    return province_day, input_rows, province_ready_rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_format = _infer_format(args.output, args.format)
    if _infer_format(args.features, None) == "parquet":
        province_day, original_rows, province_ready_rows = _build_province_day_streaming(
            args.features,
            province_lookup_path=args.province_lookup,
            province_column=args.province_column,
            lookup_province_column=args.lookup_province_column,
            coordinate_precision=args.coordinate_precision,
            batch_rows=args.batch_rows,
        )
    else:
        features = _read_table(args.features)
        original_rows = int(len(features))
        if args.province_column not in features.columns:
            if args.province_lookup is None:
                raise KeyError(
                    f"Feature table is missing province column '{args.province_column}'. "
                    "Provide --province-lookup to attach province names before aggregation."
                )
            lookup = _read_table(args.province_lookup)
            features = enrich_flash_flood_features_with_province(
                features,
                lookup,
                coordinate_precision=args.coordinate_precision,
                province_column=args.province_column,
                lookup_province_column=args.lookup_province_column,
            )
        province_ready_rows = int(features[args.province_column].fillna("").astype(str).str.strip().ne("").sum())
        province_day = aggregate_flash_flood_features_to_province_day(features, province_column=args.province_column)
    _write_table(province_day, args.output, output_format)

    summary = {
        "input_rows": original_rows,
        "province_ready_rows": province_ready_rows,
        "province_missing_rows": original_rows - province_ready_rows,
        "province_day_rows": int(len(province_day)),
        "province_column": args.province_column,
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
