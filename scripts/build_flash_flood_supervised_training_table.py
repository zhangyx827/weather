#!/usr/bin/env python3
"""Join flash-flood Layer-4 feature rows with conservative event-derived labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.config import FlashFloodLabelMappingConfig
from mazu_saudi.data import build_flash_flood_supervised_training_dataset
from mazu_saudi.data.flash_flood_audit import summarize_flash_flood_supervision_quality


FORMATS = ("csv", "json", "parquet")
_PROVINCE_COLUMNS = ("province_name", "admin1_name", "region_name", "location_name")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a supervised flash-flood Layer-4 training table by joining features with labels.")
    parser.add_argument("--features", type=Path, required=True, help="Feature table, typically flash_flood_training.parquet.")
    parser.add_argument("--labels", type=Path, required=True, help="Flash-flood label table produced by build_flash_flood_training_labels.py.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "training" / "flash_flood_province_day_supervised_verified_chain.parquet",
        help="Output path for the merged supervised training table.",
    )
    parser.add_argument("--format", choices=FORMATS, help="Output format. Defaults to the output file suffix.")
    parser.add_argument("--keep-uncertain", action="store_true", help="Keep uncertain or unlabeled rows in the merged output.")
    parser.add_argument("--coordinate-precision", type=int, default=4, help="Decimal precision used for latitude/longitude join keys.")
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=250_000,
        help="Batch size for parquet passthrough builds. Ignored for non-parquet fallback joins.",
    )
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


def _empty_summary(output: Path) -> dict[str, object]:
    return {
        "rows": 0,
        "positive_rows": 0,
        "negative_rows": 0,
        "uncertain_rows": 0,
        "labeled_rows": 0,
        "rows_with_matched_event_ids": 0,
        "geometry_positive_rows": 0,
        "outside_event_footprint_negative_rows": 0,
        "label_source_mode_counts": {},
        "output": str(output),
    }


def _summarize_merged_table(merged, output: Path) -> dict[str, object]:
    label_source_mode_counts = {}
    if "label_source_mode" in merged.columns:
        label_source_mode_counts = {
            str(key): int(value)
            for key, value in merged["label_source_mode"].astype(str).value_counts(dropna=False).to_dict().items()
        }

    summary = _empty_summary(output)
    summary.update(
        {
            "rows": int(len(merged)),
            "positive_rows": int((merged["label_status"] == "positive").sum()) if "label_status" in merged.columns else 0,
            "negative_rows": int((merged["label_status"] == "negative").sum()) if "label_status" in merged.columns else 0,
            "uncertain_rows": int((merged["label_status"] == "uncertain").sum()) if "label_status" in merged.columns else 0,
            "labeled_rows": int(merged["is_labeled"].sum()) if "is_labeled" in merged.columns else 0,
            "rows_with_matched_event_ids": int(merged["matched_event_ids"].fillna("").astype(str).str.strip().ne("").sum())
            if "matched_event_ids" in merged.columns
            else 0,
            "geometry_positive_rows": int((merged["label_source_mode"] == "geometry_wkt").sum()) if "label_source_mode" in merged.columns else 0,
            "outside_event_footprint_negative_rows": int((merged["label_source_mode"] == "outside_event_footprint").sum())
            if "label_source_mode" in merged.columns
            else 0,
            "label_source_mode_counts": label_source_mode_counts,
        }
    )
    return summary


def _finalize_summary(summary: dict[str, object]) -> dict[str, object]:
    summary["supervision_quality"] = summarize_flash_flood_supervision_quality(
        total_rows=int(summary["rows"]),
        positive_rows=int(summary["positive_rows"]),
        negative_rows=int(summary["negative_rows"]),
        uncertain_rows=int(summary["uncertain_rows"]),
        rows_with_matched_event_ids=int(summary["rows_with_matched_event_ids"]),
        geometry_positive_rows=int(summary["geometry_positive_rows"]),
        outside_event_footprint_negative_rows=int(summary["outside_event_footprint_negative_rows"]),
    )
    return summary


def _write_verified_parquet(table, path: Path) -> None:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("Parquet export requires pyarrow for verified parquet writes") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        table.to_parquet(temp_path, index=False)
        pq.ParquetFile(temp_path).metadata
        temp_path.replace(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _label_table_supports_passthrough(feature_path: Path, label_path: Path) -> bool:
    try:
        import pyarrow.parquet as pq
    except Exception:
        return False

    feature_columns = set(pq.ParquetFile(feature_path).schema_arrow.names)
    label_columns = set(pq.ParquetFile(label_path).schema_arrow.names)
    required_label_columns = {"label", "label_status", "label_source_mode", "matched_event_ids", "label_provenance"}
    return feature_columns.issubset(label_columns) and required_label_columns.issubset(label_columns)


def _prepare_passthrough_batch(table, *, drop_uncertain: bool, coordinate_precision: int):
    import pandas as pd

    batch = table.copy()
    if "hazard_type" in batch.columns:
        batch = batch[batch["hazard_type"].astype(str).str.lower() == "flash_flood"].copy()
    if drop_uncertain:
        batch = batch[batch["label_status"].isin(("positive", "negative"))].copy()
    batch["is_labeled"] = batch["label"].notna()

    if {"latitude", "longitude"}.issubset(batch.columns):
        batch["training_join_mode"] = "grid_day"
        batch["training_join_key"] = (
            pd.to_datetime(batch["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            + "|"
            + pd.to_numeric(batch["latitude"], errors="coerce").round(coordinate_precision).astype(str)
            + "|"
            + pd.to_numeric(batch["longitude"], errors="coerce").round(coordinate_precision).astype(str)
        )
    else:
        province_column = next((column for column in _PROVINCE_COLUMNS if column in batch.columns), None)
        if province_column is None:
            batch["training_join_mode"] = "label_table_passthrough"
            batch["training_join_key"] = pd.to_datetime(batch["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            batch["training_join_mode"] = f"province_day:{province_column}"
            batch["training_join_key"] = (
                pd.to_datetime(batch["date"], errors="coerce").dt.strftime("%Y-%m-%d") + "|" + batch[province_column].astype(str)
            )
    return batch


def _build_passthrough_supervised_table(label_path: Path, output: Path, *, drop_uncertain: bool, coordinate_precision: int, batch_rows: int):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("Parquet passthrough requires pyarrow") from exc

    if batch_rows <= 0:
        raise ValueError("--batch-rows must be positive")

    label_file = pq.ParquetFile(label_path)
    temp_output = output.with_name(f"{output.name}.tmp")
    writer = None
    summary = _empty_summary(output)
    label_source_mode_counts: Counter[str] = Counter()

    try:
        for record_batch in label_file.iter_batches(batch_size=batch_rows):
            merged = _prepare_passthrough_batch(
                record_batch.to_pandas(),
                drop_uncertain=drop_uncertain,
                coordinate_precision=coordinate_precision,
            )
            if merged.empty:
                continue
            arrow_table = pa.Table.from_pandas(merged, preserve_index=False)
            if writer is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(temp_output, arrow_table.schema)
            else:
                arrow_table = arrow_table.cast(writer.schema, safe=False)
            writer.write_table(arrow_table)

            batch_summary = _summarize_merged_table(merged, output)
            summary["rows"] = int(summary["rows"]) + int(batch_summary["rows"])
            summary["positive_rows"] = int(summary["positive_rows"]) + int(batch_summary["positive_rows"])
            summary["negative_rows"] = int(summary["negative_rows"]) + int(batch_summary["negative_rows"])
            summary["uncertain_rows"] = int(summary["uncertain_rows"]) + int(batch_summary["uncertain_rows"])
            summary["labeled_rows"] = int(summary["labeled_rows"]) + int(batch_summary["labeled_rows"])
            summary["rows_with_matched_event_ids"] = int(summary["rows_with_matched_event_ids"]) + int(
                batch_summary["rows_with_matched_event_ids"]
            )
            summary["geometry_positive_rows"] = int(summary["geometry_positive_rows"]) + int(batch_summary["geometry_positive_rows"])
            summary["outside_event_footprint_negative_rows"] = int(summary["outside_event_footprint_negative_rows"]) + int(
                batch_summary["outside_event_footprint_negative_rows"]
            )
            label_source_mode_counts.update({str(key): int(value) for key, value in batch_summary["label_source_mode_counts"].items()})

        if writer is None:
            raise ValueError(f"No flash_flood rows were written from label parquet: {label_path}")
        writer.close()
        writer = None
        pq.ParquetFile(temp_output).metadata
        temp_output.replace(output)
    except Exception:
        if writer is not None:
            writer.close()
        if temp_output.exists():
            temp_output.unlink()
        raise

    summary["label_source_mode_counts"] = dict(sorted(label_source_mode_counts.items()))
    return _finalize_summary(summary)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_format = _infer_format(args.output, args.format)
    if (
        _infer_format(args.features, None) == "parquet"
        and _infer_format(args.labels, None) == "parquet"
        and output_format == "parquet"
        and _label_table_supports_passthrough(args.features, args.labels)
    ):
        summary = _build_passthrough_supervised_table(
            args.labels,
            args.output,
            drop_uncertain=not args.keep_uncertain,
            coordinate_precision=args.coordinate_precision,
            batch_rows=args.batch_rows,
        )
    else:
        features = _read_table(args.features)
        labels = _read_table(args.labels)
        merged = build_flash_flood_supervised_training_dataset(
            features,
            labels,
            config=FlashFloodLabelMappingConfig(),
            drop_uncertain=not args.keep_uncertain,
            coordinate_precision=args.coordinate_precision,
        )
        if output_format == "parquet":
            _write_verified_parquet(merged, args.output)
        else:
            _write_table(merged, args.output, output_format)
        summary = _finalize_summary(_summarize_merged_table(merged, args.output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
