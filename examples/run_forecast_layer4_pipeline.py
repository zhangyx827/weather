"""Run the forecast-layer to Layer-4 risk pipeline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.data.io import write_netcdf_dataset
from mazu_saudi.forecast import AIFSBenchmarkProvider, AuroraForecastProvider, GenCastForecastProvider, MockForecastProvider
from mazu_saudi.risk import LightGBMLayer4Model


def _parse_leads(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_issue_time(raw: str) -> datetime:
    issue_time = datetime.fromisoformat(raw)
    if issue_time.tzinfo is None:
        issue_time = issue_time.replace(tzinfo=timezone.utc)
    return issue_time


def _provider(name: str):
    normalized = name.strip().lower()
    if normalized == "mock":
        return MockForecastProvider()
    if normalized == "gencast":
        return GenCastForecastProvider()
    if normalized == "aifs":
        return AIFSBenchmarkProvider()
    return AuroraForecastProvider()


def _summary_stats(field: Any) -> dict[str, float]:
    arr = np.asarray(field, dtype=np.float32)
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }


def _metadata_snapshot(ds: Any) -> dict[str, Any]:
    keys = (
        "primary_provider",
        "provider_role",
        "provider_status",
        "source_status",
        "degradation_metadata_json",
        "ensemble_member_count",
        "benchmark_comparison_json",
    )
    return {key: ds.attrs[key] for key in keys if key in ds.attrs}


def build_payload(
    forecast_ds: Any,
    risk_ds: Any,
    *,
    provider_name: str,
    issue_time: datetime,
    lead_hours: list[int],
    gencast_metadata: dict[str, Any] | None,
    aifs_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "provider": provider_name,
        "issue_time": issue_time.isoformat(),
        "lead_hours": lead_hours,
        "grid": {
            "dimensions": {name: int(size) for name, size in forecast_ds.sizes.items()},
            "latitude": [float(value) for value in np.asarray(forecast_ds["latitude"].values).tolist()],
            "longitude": [float(value) for value in np.asarray(forecast_ds["longitude"].values).tolist()],
        },
        "forecast_metadata": _metadata_snapshot(forecast_ds),
        "auxiliary_metadata": {
            "gencast": gencast_metadata or {"status": "not_requested"},
            "aifs": aifs_metadata or {"status": "not_requested"},
        },
        "variables": {
            "forecast": sorted(forecast_ds.data_vars),
            "risk": sorted(risk_ds.data_vars),
        },
        "summary": {
            "ExtremeHeat_Risk_Prob": _summary_stats(risk_ds["ExtremeHeat_Risk_Prob"]),
            "DryHeatStress_Risk_Prob": _summary_stats(risk_ds["DryHeatStress_Risk_Prob"]),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Aurora forecast fields into Layer-4 risk grids.")
    parser.add_argument("--provider", default="aurora", choices=["aurora", "mock"], help="Primary deterministic provider.")
    parser.add_argument("--issue-time", default="2026-07-11T00:00:00+00:00", help="Issue time in ISO8601 format.")
    parser.add_argument("--lead-hours", default="0", help="Comma-separated lead hours.")
    parser.add_argument("--output-netcdf", default=str(ROOT / "data" / "output" / "forecast_layer4_risk.nc"))
    parser.add_argument("--output-json", default=str(ROOT / "data" / "output" / "forecast_layer4_summary.json"))
    parser.add_argument("--include-gencast-metadata", action="store_true", help="Attach GenCast ensemble metadata snapshot.")
    parser.add_argument("--include-aifs-benchmark", action="store_true", help="Attach AIFS benchmark metadata snapshot.")
    parser.add_argument("--extreme-heat-model-path", help="Override the Extreme Heat LightGBM model path.")
    parser.add_argument("--dry-heat-model-path", help="Override the Dry Heat Stress LightGBM model path.")
    args = parser.parse_args(argv)

    issue_time = _parse_issue_time(args.issue_time)
    lead_hours = _parse_leads(args.lead_hours)
    provider = _provider(args.provider)

    forecast_ds = provider.forecast_dataset(issue_time, lead_hours)
    risk_model = LightGBMLayer4Model(
        extreme_heat_model_path=args.extreme_heat_model_path,
        dry_heat_model_path=args.dry_heat_model_path,
    )
    risk_ds = risk_model.predict_fields(forecast_ds)

    gencast_metadata = None
    if args.include_gencast_metadata:
        gencast_metadata = _metadata_snapshot(GenCastForecastProvider().forecast_dataset(issue_time, lead_hours))
        gencast_metadata["status"] = "available"

    aifs_metadata = None
    if args.include_aifs_benchmark:
        aifs_metadata = _metadata_snapshot(AIFSBenchmarkProvider().forecast_dataset(issue_time, lead_hours))
        aifs_metadata["status"] = "available"

    output_netcdf = Path(args.output_netcdf)
    output_json = Path(args.output_json)
    output_netcdf.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    merged = risk_ds.assign_attrs(
        {
            **risk_ds.attrs,
            "forecast_metadata_json": json.dumps(_metadata_snapshot(forecast_ds), ensure_ascii=False, sort_keys=True),
        }
    )
    write_netcdf_dataset(output_netcdf, merged)

    payload = build_payload(
        forecast_ds,
        risk_ds,
        provider_name=args.provider,
        issue_time=issue_time,
        lead_hours=lead_hours,
        gencast_metadata=gencast_metadata,
        aifs_metadata=aifs_metadata,
    )
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"forecast provider: {args.provider}")
    print(f"risk netcdf: {output_netcdf}")
    print(f"summary json: {output_json}")
    print(f"ExtremeHeat mean prob: {payload['summary']['ExtremeHeat_Risk_Prob']['mean']:.3f}")
    print(f"DryHeatStress mean prob: {payload['summary']['DryHeatStress_Risk_Prob']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
