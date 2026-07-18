"""Run a reproducible flash-flood supervised-training demo."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.data import (
    build_flash_flood_supervised_training_dataset,
    build_flash_flood_training_labels,
    seed_flash_flood_events,
)
from mazu_saudi.risk.layer4_features import feature_names_for_hazard
from mazu_saudi.risk.ml import LightGBMAdapter

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("pandas is required for the flash-flood supervised-training demo") from exc


DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "demo_flash_flood_supervised"
TRAINING_SCRIPT = ROOT / "examples" / "train_layer4_lightgbm.py"


def _load_training_module():
    spec = importlib.util.spec_from_file_location("train_layer4_lightgbm", TRAINING_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load training module from {TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:
        return False


def build_demo_feature_table(rows_per_bucket: int = 8, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic flash-flood feature table spanning seed event days."""

    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for event in seed_flash_flood_events():
        for _ in range(rows_per_bucket):
            records.append(
                {
                    "date": event.start_date.isoformat(),
                    "hazard_type": "flash_flood",
                    "latitude": float(event.latitude) + rng.uniform(-0.06, 0.06),
                    "longitude": float(event.longitude) + rng.uniform(-0.06, 0.06),
                    "daily_precip_total": float(rng.uniform(35.0, 95.0)),
                    "daily_convective_precip": float(rng.uniform(12.0, 50.0)),
                    "daily_large_scale_precip": float(rng.uniform(8.0, 45.0)),
                    "cape": float(rng.uniform(800.0, 3200.0)),
                    "pwat": float(rng.uniform(20.0, 60.0)),
                    "ivt": float(rng.uniform(120.0, 480.0)),
                    "wind850_speed": float(rng.uniform(6.0, 24.0)),
                    "wind_shear_850_200": float(rng.uniform(10.0, 55.0)),
                    "flash_flood_risk": int(rng.integers(1, 4)),
                    "daily_precip_anomaly": float(rng.uniform(5.0, 35.0)),
                }
            )
        non_event_date = (event.start_date + timedelta(days=1)).isoformat()
        for _ in range(rows_per_bucket):
            records.append(
                {
                    "date": non_event_date,
                    "hazard_type": "flash_flood",
                    "latitude": float(event.latitude) + rng.uniform(-0.15, 0.15),
                    "longitude": float(event.longitude) + rng.uniform(-0.15, 0.15),
                    "daily_precip_total": float(rng.uniform(0.0, 8.0)),
                    "daily_convective_precip": float(rng.uniform(0.0, 4.0)),
                    "daily_large_scale_precip": float(rng.uniform(0.0, 4.0)),
                    "cape": float(rng.uniform(0.0, 900.0)),
                    "pwat": float(rng.uniform(5.0, 20.0)),
                    "ivt": float(rng.uniform(10.0, 150.0)),
                    "wind850_speed": float(rng.uniform(0.0, 12.0)),
                    "wind_shear_850_200": float(rng.uniform(0.0, 25.0)),
                    "flash_flood_risk": int(rng.integers(0, 2)),
                    "daily_precip_anomaly": float(rng.uniform(-10.0, 5.0)),
                }
            )
    return pd.DataFrame.from_records(records)


def run_demo(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    rows_per_bucket: int = 8,
    seed: int = 42,
    train_model: bool = True,
) -> dict[str, object]:
    """Build demo labels and supervised dataset, then optionally train a model."""

    output_dir.mkdir(parents=True, exist_ok=True)

    features = build_demo_feature_table(rows_per_bucket=rows_per_bucket, seed=seed)
    labels = build_flash_flood_training_labels(features)
    supervised = build_flash_flood_supervised_training_dataset(features, labels)

    feature_path = output_dir / "flash_flood_demo_features.csv"
    label_path = output_dir / "flash_flood_demo_labels.csv"
    supervised_path = output_dir / "flash_flood_demo_supervised.parquet"
    features.to_csv(feature_path, index=False)
    labels.to_csv(label_path, index=False)

    summary: dict[str, object] = {
        "feature_rows": int(len(features)),
        "label_rows": int(len(labels)),
        "supervised_rows": int(len(supervised)),
        "positive_rows": int((supervised["label_status"] == "positive").sum()),
        "negative_rows": int((supervised["label_status"] == "negative").sum()),
        "feature_path": str(feature_path),
        "label_path": str(label_path),
    }

    if _parquet_available():
        supervised.to_parquet(supervised_path, index=False)
        summary["supervised_path"] = str(supervised_path)

    if not train_model:
        summary["training_skipped_reason"] = "train_model_disabled"
        return summary
    if not _lightgbm_available():
        summary["training_skipped_reason"] = "lightgbm_unavailable"
        return summary

    train_module = _load_training_module()
    target_summary = train_module.summarize_frame_training_targets(supervised, "flash_flood")
    training_payload = train_module.build_training_payload_from_frame(supervised, "flash_flood")
    features_matrix = training_payload["features"]
    target = training_payload["labels"]
    adapter = LightGBMAdapter()
    training_summary = adapter.train(
        training_payload,
        validation_fraction=0.1,
        seed=seed,
        num_boost_round=250,
        early_stopping_rounds=20,
    )

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "flash_flood.txt"
    adapter.save_model(model_path)
    train_summary = {
        "source": str(supervised_path) if "supervised_path" in summary else str(feature_path),
        "source_format": "indicator-parquet" if "supervised_path" in summary else "demo-dataframe",
        "hazard_type": "flash_flood",
        "samples": int(features_matrix.shape[0]),
        "feature_names": list(feature_names_for_hazard("flash_flood")),
        "model": {
            "path": str(model_path),
            "backend": training_summary["backend"],
            "objective": training_summary["objective"],
            "metric": training_summary["metric"],
            "validation_metric": training_summary["validation_metric"],
            "best_iteration": training_summary["best_iteration"],
            "split_strategy": training_summary["split_strategy"],
        },
        "training_target": target_summary,
    }
    if "split_group_audit" in training_payload:
        split_group_audit = dict(training_payload["split_group_audit"])
        train_summary["model"].update(split_group_audit)
        train_summary["training_target"]["training_split_group_audit"] = split_group_audit
    train_summary_path = model_dir / "train_summary.json"
    train_summary_path.write_text(json.dumps(train_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["model_path"] = str(model_path)
    summary["train_summary_path"] = str(train_summary_path)
    summary["training_target_source"] = target_summary["target_source"]
    return summary


def main() -> None:
    result = run_demo()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
