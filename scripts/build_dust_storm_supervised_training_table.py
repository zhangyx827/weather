#!/usr/bin/env python3
"""Join dust-storm Layer-4 feature rows with event-derived labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.config import DustStormLabelMappingConfig
from mazu_saudi.data import build_dust_storm_supervised_training_dataset


FORMATS = ("csv", "json", "parquet")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a supervised dust-storm Layer-4 training table by joining features with labels.")
    parser.add_argument("--features", type=Path, required=True, help="Feature table with region-day or province-day samples.")
    parser.add_argument("--labels", type=Path, required=True, help="Dust-storm label table produced by build_dust_storm_training_labels.py.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "training" / "dust_storm_supervised_training.parquet",
        help="Output path for the merged supervised training table.",
    )
    parser.add_argument("--format", choices=FORMATS, help="Output format. Defaults to the output file suffix.")
    parser.add_argument("--keep-uncertain", action="store_true", help="Keep uncertain or unlabeled rows in the merged output.")
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_format = _infer_format(args.output, args.format)
    features = _read_table(args.features)
    labels = _read_table(args.labels)
    merged = build_dust_storm_supervised_training_dataset(
        features,
        labels,
        config=DustStormLabelMappingConfig(),
        drop_uncertain=not args.keep_uncertain,
    )
    _write_table(merged, args.output, output_format)

    summary = {
        "rows": int(len(merged)),
        "positive_rows": int((merged["label_status"] == "positive").sum()) if "label_status" in merged.columns else 0,
        "negative_rows": int((merged["label_status"] == "negative").sum()) if "label_status" in merged.columns else 0,
        "uncertain_rows": int((merged["label_status"] == "uncertain").sum()) if "label_status" in merged.columns else 0,
        "labeled_rows": int(merged["is_labeled"].sum()) if "is_labeled" in merged.columns else 0,
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
