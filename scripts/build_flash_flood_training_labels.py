#!/usr/bin/env python3
"""Build flash-flood training labels from a sample table and event table."""

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
from mazu_saudi.data import build_flash_flood_training_labels, expand_flash_flood_events_to_daily_table
from mazu_saudi.data.flash_flood_audit import (
    count_flash_flood_boundary_grounded_positive_rows,
    count_flash_flood_explicit_geometry_positive_rows,
    count_flash_flood_geometry_backed_positive_rows,
    summarize_flash_flood_geometry_backed_positive_rows,
    summarize_flash_flood_supervision_quality,
)


FORMATS = ("csv", "json", "parquet")
DEFAULT_VERIFIED_DAILY_EVENTS = (
    ROOT / "data" / "processed" / "real_flash_flood_chain" / "flash_flood_events_verified_combined_daily.csv"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conservative flash-flood training labels for grid-day or province-day samples.")
    parser.add_argument("--samples", type=Path, required=True, help="Sample table with at least a date column.")
    parser.add_argument(
        "--events-daily",
        type=Path,
        help="Optional daily flash-flood event table. Defaults to the verified daily chain when present, otherwise the built-in handoff seed events.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "labels" / "flash_flood_province_day_labels_verified_chain.parquet",
        help="Output path for the labeled sample table.",
    )
    parser.add_argument("--format", choices=FORMATS, help="Output format. Defaults to the output file suffix.")
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=250_000,
        help="Batch size for parquet-to-parquet streaming builds. Ignored for non-parquet runs.",
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
        "event_day_negative_rows": 0,
        "event_day_unresolved_rows": 0,
        "rows_with_matched_event_ids": 0,
        "geometry_positive_rows": 0,
        "boundary_grounded_positive_rows": 0,
        "explicit_geometry_positive_rows": 0,
        "geometry_positive_source_counts": {},
        "outside_event_footprint_negative_rows": 0,
        "label_source_mode_counts": {},
        "output": str(output),
    }


def _summarize_labeled_table(labeled, output: Path) -> dict[str, object]:
    label_source_mode_counts = {
        str(key): int(value)
        for key, value in labeled["label_source_mode"].astype(str).value_counts(dropna=False).to_dict().items()
    }
    geometry_positive_rows = count_flash_flood_geometry_backed_positive_rows(labeled)
    boundary_grounded_positive_rows = count_flash_flood_boundary_grounded_positive_rows(labeled)
    explicit_geometry_positive_rows = count_flash_flood_explicit_geometry_positive_rows(labeled)
    geometry_positive_source_counts = summarize_flash_flood_geometry_backed_positive_rows(labeled)
    summary = _empty_summary(output)
    summary.update(
        {
            "rows": int(len(labeled)),
            "positive_rows": int((labeled["label_status"] == "positive").sum()),
            "negative_rows": int((labeled["label_status"] == "negative").sum()),
            "uncertain_rows": int((labeled["label_status"] == "uncertain").sum()),
            "event_day_negative_rows": int((labeled["label_source_mode"] == "outside_event_footprint").sum()),
            "event_day_unresolved_rows": int((labeled["label_source_mode"] == "event_day_unresolved").sum()),
            "rows_with_matched_event_ids": int(labeled["matched_event_ids"].fillna("").astype(str).str.strip().ne("").sum()),
            "geometry_positive_rows": int(geometry_positive_rows),
            "boundary_grounded_positive_rows": int(boundary_grounded_positive_rows),
            "explicit_geometry_positive_rows": int(explicit_geometry_positive_rows),
            "geometry_positive_source_counts": geometry_positive_source_counts,
            "outside_event_footprint_negative_rows": int((labeled["label_source_mode"] == "outside_event_footprint").sum()),
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
        geometry_positive_source_counts=dict(summary["geometry_positive_source_counts"]),
        boundary_grounded_positive_rows=int(summary["boundary_grounded_positive_rows"]),
        explicit_geometry_positive_rows=int(summary["explicit_geometry_positive_rows"]),
        outside_event_footprint_negative_rows=int(summary["outside_event_footprint_negative_rows"]),
        event_day_negative_rows=int(summary["event_day_negative_rows"]),
        event_day_unresolved_rows=int(summary["event_day_unresolved_rows"]),
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


def _summary_path(output: Path) -> Path:
    return output.with_suffix(".summary.json")


def _write_summary(summary: dict[str, object], output: Path) -> None:
    _summary_path(output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_default_events_daily():
    if DEFAULT_VERIFIED_DAILY_EVENTS.exists():
        return _read_table(DEFAULT_VERIFIED_DAILY_EVENTS)
    return expand_flash_flood_events_to_daily_table()


def _build_labels_streaming(samples_path: Path, events_daily, output: Path, *, batch_rows: int):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("Parquet streaming requires pyarrow") from exc

    if batch_rows <= 0:
        raise ValueError("--batch-rows must be positive")

    parquet_file = pq.ParquetFile(samples_path)
    temp_output = output.with_name(f"{output.name}.tmp")
    writer = None
    summary = _empty_summary(output)
    label_source_mode_counts: Counter[str] = Counter()
    geometry_positive_source_counts: Counter[str] = Counter()

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_rows):
            samples = batch.to_pandas()
            labeled = build_flash_flood_training_labels(
                samples,
                event_daily_table=events_daily,
                config=FlashFloodLabelMappingConfig.from_env(),
            )
            arrow_table = pa.Table.from_pandas(labeled, preserve_index=False)
            if writer is None:
                output.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(temp_output, arrow_table.schema)
            else:
                arrow_table = arrow_table.cast(writer.schema, safe=False)
            writer.write_table(arrow_table)

            batch_summary = _summarize_labeled_table(labeled, output)
            summary["rows"] = int(summary["rows"]) + int(batch_summary["rows"])
            summary["positive_rows"] = int(summary["positive_rows"]) + int(batch_summary["positive_rows"])
            summary["negative_rows"] = int(summary["negative_rows"]) + int(batch_summary["negative_rows"])
            summary["uncertain_rows"] = int(summary["uncertain_rows"]) + int(batch_summary["uncertain_rows"])
            summary["event_day_negative_rows"] = int(summary["event_day_negative_rows"]) + int(
                batch_summary["event_day_negative_rows"]
            )
            summary["event_day_unresolved_rows"] = int(summary["event_day_unresolved_rows"]) + int(
                batch_summary["event_day_unresolved_rows"]
            )
            summary["rows_with_matched_event_ids"] = int(summary["rows_with_matched_event_ids"]) + int(
                batch_summary["rows_with_matched_event_ids"]
            )
            summary["geometry_positive_rows"] = int(summary["geometry_positive_rows"]) + int(batch_summary["geometry_positive_rows"])
            summary["boundary_grounded_positive_rows"] = int(summary["boundary_grounded_positive_rows"]) + int(
                batch_summary["boundary_grounded_positive_rows"]
            )
            summary["explicit_geometry_positive_rows"] = int(summary["explicit_geometry_positive_rows"]) + int(
                batch_summary["explicit_geometry_positive_rows"]
            )
            summary["outside_event_footprint_negative_rows"] = int(summary["outside_event_footprint_negative_rows"]) + int(
                batch_summary["outside_event_footprint_negative_rows"]
            )
            label_source_mode_counts.update({str(key): int(value) for key, value in batch_summary["label_source_mode_counts"].items()})
            geometry_positive_source_counts.update(
                {str(key): int(value) for key, value in batch_summary["geometry_positive_source_counts"].items()}
            )

        if writer is None:
            raise ValueError(f"No rows were read from parquet input: {samples_path}")
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
    summary["geometry_positive_source_counts"] = dict(sorted(geometry_positive_source_counts.items()))
    return _finalize_summary(summary)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_format = _infer_format(args.output, args.format)
    events_daily = _read_table(args.events_daily) if args.events_daily else _load_default_events_daily()
    if _infer_format(args.samples, None) == "parquet" and output_format == "parquet":
        summary = _build_labels_streaming(args.samples, events_daily, args.output, batch_rows=args.batch_rows)
    else:
        samples = _read_table(args.samples)
        labeled = build_flash_flood_training_labels(samples, event_daily_table=events_daily, config=FlashFloodLabelMappingConfig.from_env())
        if output_format == "parquet":
            _write_verified_parquet(labeled, args.output)
        else:
            _write_table(labeled, args.output, output_format)
        summary = _finalize_summary(_summarize_labeled_table(labeled, args.output))
    _write_summary(summary, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
