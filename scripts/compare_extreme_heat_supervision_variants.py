#!/usr/bin/env python3
"""Compare extreme-heat supervision variants before model changes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.data.extreme_heat_training_dataset import _date_token_from_path, _positive_date_lookup
from mazu_saudi.data import build_extreme_heat_supervised_training_dataset
from mazu_saudi.risk.ml import LightGBMAdapter


TRAIN_SCRIPT_PATH = ROOT / "examples" / "train_layer4_lightgbm.py"


def _load_training_module():
    spec = importlib.util.spec_from_file_location("train_layer4_lightgbm", TRAIN_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training script: {TRAIN_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_table(path: Path):
    suffix = path.suffix.lower().lstrip(".")
    import pandas as pd

    if suffix == "csv":
        return pd.read_csv(path)
    if suffix == "json":
        return pd.read_json(path)
    return pd.read_parquet(path)


def _evenly_sample(values: list[str], limit: int | None) -> list[str]:
    if limit is None or limit >= len(values):
        return list(values)
    if limit <= 0:
        return []
    if limit == 1:
        return [values[len(values) // 2]]
    indices = np.linspace(0, len(values) - 1, num=limit)
    selected: list[str] = []
    seen: set[str] = set()
    for index in indices:
        value = values[int(round(float(index)))]
        if value in seen:
            continue
        selected.append(value)
        seen.add(value)
    if len(selected) < limit:
        for value in values:
            if value in seen:
                continue
            selected.append(value)
            seen.add(value)
            if len(selected) >= limit:
                break
    return selected[:limit]


def _variant_name(point_variable: str, negative_sample_size: int | None) -> str:
    suffix = "balanced" if negative_sample_size is None else f"neg{negative_sample_size}"
    return f"{point_variable}_{suffix}"


def _variant_name_region_day(point_variable: str, top_k: int, negative_sample_size: int | None) -> str:
    suffix = "balanced" if negative_sample_size is None else f"neg{negative_sample_size}"
    return f"region_day_{point_variable}_top{top_k}_{suffix}"


def _parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("At least one integer value is required")
    return values


def _parse_text_list(text: str) -> list[str]:
    values = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    if not values:
        raise ValueError("At least one value is required")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare extreme-heat supervision variants using the existing Layer-4 trainer. Omit the date limits for full coverage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Directory containing daily Saudi indicator NetCDF files.")
    parser.add_argument("--labels", type=Path, required=True, help="Verified extreme-heat event CSV.")
    parser.add_argument(
        "--glob",
        default="saudi_indicators_*.nc",
        help="Glob used to discover daily indicator files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "training" / "extreme_heat_supervision_comparison_full.json",
        help="JSON file where the comparison summary is written.",
    )
    parser.add_argument(
        "--positive-date-limit",
        type=int,
        default=None,
        help="Maximum number of positive dates to keep in the comparison pass. Omit for full coverage.",
    )
    parser.add_argument(
        "--negative-date-limit",
        type=int,
        default=None,
        help="Maximum number of negative dates to keep in the comparison pass. Omit for full coverage.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed forwarded to the LightGBM trainer.")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="Validation fraction forwarded to the trainer.")
    parser.add_argument("--num-boost-round", type=int, default=120, help="Maximum boosting rounds forwarded to the trainer.")
    parser.add_argument("--early-stopping-rounds", type=int, default=20, help="Early-stopping rounds forwarded to the trainer.")
    parser.add_argument(
        "--sample-unit",
        choices=("single_point_day", "region_day"),
        default="region_day",
        help="Extreme-heat supervision unit to compare.",
    )
    parser.add_argument(
        "--point-variables",
        default="heat_index_c,tmax_c,t2m_c",
        help="Comma-separated point variables to sweep.",
    )
    parser.add_argument(
        "--top-k-values",
        default="1,3",
        help="Comma-separated region-day pooling sizes to sweep when `--sample-unit region_day` is used.",
    )
    parser.add_argument(
        "--region-boundary-path",
        type=Path,
        default=ROOT / "data" / "raw" / "admin_boundaries" / "geoBoundaries-SAU-ADM1.geojson",
        help="GeoJSON admin-1 boundary file used for region-day mapping.",
    )
    return parser.parse_args(argv)


def _build_variant_table(
    *,
    training_module,
    feature_paths: list[Path],
    labels,
    point_variable: str,
    negative_sample_size: int,
    sample_unit: str,
    top_k: int,
    region_boundary_path: Path,
    seed: int,
):
    return build_extreme_heat_supervised_training_dataset(
        feature_paths,
        labels,
        point_variable=point_variable,
        negative_sample_size=negative_sample_size,
        sample_unit=sample_unit,
        top_k=top_k,
        region_boundary_path=region_boundary_path,
        seed=seed,
    )


def _train_variant(
    *,
    training_module,
    table,
    seed: int,
    validation_fraction: float,
    num_boost_round: int,
    early_stopping_rounds: int,
):
    payload = training_module.build_training_payload_from_frame(table, "extreme_heat")
    training_summary = LightGBMAdapter().train(
        payload,
        validation_fraction=validation_fraction,
        seed=seed,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )
    target_summary = training_module.summarize_frame_training_targets(table, "extreme_heat")
    return training_summary, target_summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    training_module = _load_training_module()
    labels = _read_table(args.labels)

    feature_paths = sorted(args.input.glob(args.glob))
    if not feature_paths:
        raise FileNotFoundError(f"No indicator files matched {args.input}/{args.glob}")

    feature_map = {_date_token_from_path(path): path for path in feature_paths}
    positive_lookup = _positive_date_lookup(labels)
    positive_dates = _evenly_sample(sorted(date for date in positive_lookup if date in feature_map), args.positive_date_limit)
    negative_dates = _evenly_sample(sorted(date for date in feature_map if date not in positive_lookup), args.negative_date_limit)
    if not positive_dates:
        raise ValueError("No positive extreme_heat dates matched the available indicator files")
    if not negative_dates:
        raise ValueError("No negative extreme_heat dates matched the available indicator files")

    balanced_negative_count = min(len(negative_dates), len(positive_dates))
    wider_negative_count = len(negative_dates)
    selected_dates = positive_dates + negative_dates
    selected_feature_paths = [feature_map[date] for date in selected_dates]

    point_variables = _parse_text_list(args.point_variables)
    top_k_values = _parse_int_list(args.top_k_values)
    variant_specs: list[dict[str, object]] = []
    if args.sample_unit == "single_point_day":
        for point_variable in point_variables:
            variant_specs.append(
                {
                    "name": _variant_name(point_variable, balanced_negative_count),
                    "point_variable": point_variable,
                    "negative_sample_size": balanced_negative_count,
                    "top_k": 1,
                }
            )
            variant_specs.append(
                {
                    "name": _variant_name(point_variable, wider_negative_count),
                    "point_variable": point_variable,
                    "negative_sample_size": wider_negative_count,
                    "top_k": 1,
                }
            )
    else:
        for point_variable in point_variables:
            for top_k in top_k_values:
                variant_specs.append(
                    {
                        "name": _variant_name_region_day(point_variable, top_k, balanced_negative_count),
                        "point_variable": point_variable,
                        "negative_sample_size": balanced_negative_count,
                        "top_k": top_k,
                    }
                )
                variant_specs.append(
                    {
                        "name": _variant_name_region_day(point_variable, top_k, wider_negative_count),
                        "point_variable": point_variable,
                        "negative_sample_size": wider_negative_count,
                        "top_k": top_k,
                    }
                )

    variants: list[dict[str, object]] = []
    for spec in variant_specs:
        table = _build_variant_table(
            training_module=training_module,
            feature_paths=selected_feature_paths,
            labels=labels,
            point_variable=str(spec["point_variable"]),
            negative_sample_size=int(spec["negative_sample_size"]),
            sample_unit=args.sample_unit,
            top_k=int(spec["top_k"]),
            region_boundary_path=args.region_boundary_path,
            seed=args.seed,
        )
        training_summary, target_summary = _train_variant(
            training_module=training_module,
            table=table,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        variant_summary = {
            "name": spec["name"],
            "point_variable": spec["point_variable"],
            "negative_sample_size": int(spec["negative_sample_size"]),
            "top_k": int(spec["top_k"]),
            "rows": int(len(table)),
            "positive_rows": int((table["label"] > 0.5).sum()),
            "negative_rows": int((table["label"] <= 0.5).sum()),
            "validation_metric": training_summary["validation_metric"],
            "best_iteration": training_summary["best_iteration"],
            "split_strategy": training_summary["split_strategy"],
            "target_summary": target_summary,
            "training_summary": training_summary,
        }
        variants.append(variant_summary)

    ranked = sorted(
        variants,
        key=lambda item: float("inf") if item["validation_metric"] is None else float(item["validation_metric"]),
    )
    comparison = {
        "selection": {
            "positive_date_limit": args.positive_date_limit,
            "negative_date_limit": args.negative_date_limit,
            "selected_positive_dates": len(positive_dates),
            "selected_negative_dates": len(negative_dates),
            "balanced_negative_count": balanced_negative_count,
            "wider_negative_count": wider_negative_count,
            "sample_unit": args.sample_unit,
            "point_variables": point_variables,
            "top_k_values": top_k_values,
        },
        "variants": variants,
        "ranked_variants": [
            {
                "name": item["name"],
                "point_variable": item["point_variable"],
                "negative_sample_size": item["negative_sample_size"],
                "top_k": item["top_k"],
                "validation_metric": item["validation_metric"],
                "best_iteration": item["best_iteration"],
            }
            for item in ranked
        ],
        "best_variant": ranked[0]["name"] if ranked else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
