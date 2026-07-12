from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np

from mazu_saudi.forecast import AIFSBenchmarkProvider, GenCastForecastProvider, MockForecastProvider
from mazu_saudi.risk import LightGBMLayer4Model

try:
    import xarray as xr
except Exception:  # pragma: no cover - optional dependency
    xr = None


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "examples" / "run_forecast_layer4_pipeline.py"


class FakeBooster:
    def __init__(self, bias: float) -> None:
        self.bias = bias

    def predict(self, features):
        array = np.asarray(features, dtype=np.float32)
        temp_component = np.clip((array[:, 0] - 35.0) / 15.0, 0.0, 1.0)
        return np.clip(temp_component + self.bias, 0.0, 1.0)


def _load_example_module():
    spec = importlib.util.spec_from_file_location("run_forecast_layer4_pipeline", EXAMPLE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(xr is None, "xarray is required for forecast Layer-4 tests")
class ForecastLayer4Tests(unittest.TestCase):
    def test_mock_provider_forecast_dataset_contains_layer4_fields(self):
        provider = MockForecastProvider()
        ds = provider.forecast_dataset(datetime(2026, 7, 11, tzinfo=timezone.utc), 0)

        for name in ("temp_c", "rh_percent", "wind_speed_mps", "heat_index_c", "vpd_kpa", "relative_humidity_percent"):
            self.assertIn(name, ds.data_vars)
        self.assertEqual(ds.attrs["primary_provider"], "mock")
        self.assertEqual(ds.attrs["source_status"], "degraded")

    def test_auxiliary_provider_metadata_survives_dataset_export(self):
        issue_time = datetime(2026, 7, 11, tzinfo=timezone.utc)
        gencast = GenCastForecastProvider().forecast_dataset(issue_time, 0)
        aifs = AIFSBenchmarkProvider().forecast_dataset(issue_time, 0)

        self.assertEqual(gencast.attrs["primary_provider"], "gencast")
        self.assertEqual(gencast.attrs["ensemble_member_count"], 4)
        self.assertEqual(aifs.attrs["primary_provider"], "aifs")
        self.assertIn("benchmark_comparison_json", aifs.attrs)

    def test_layer4_grid_inference_outputs_probability_and_level_fields(self):
        ds = xr.Dataset(
            data_vars={
                "temp_c": (("time", "latitude", "longitude"), np.array([[[41.0, 43.0], [39.0, 45.0]]], dtype=np.float32)),
                "vpd_kpa": (("time", "latitude", "longitude"), np.array([[[2.0, 3.0], [1.5, 4.0]]], dtype=np.float32)),
                "heat_index_c": (("time", "latitude", "longitude"), np.array([[[42.0, 45.0], [40.0, 48.0]]], dtype=np.float32)),
                "wind_speed_mps": (("time", "latitude", "longitude"), np.array([[[5.0, 6.0], [4.0, 7.0]]], dtype=np.float32)),
                "relative_humidity_percent": (("time", "latitude", "longitude"), np.array([[[35.0, 30.0], [40.0, 28.0]]], dtype=np.float32)),
            },
            coords={
                "time": np.array(["2026-07-11T00:00:00"], dtype="datetime64[ns]"),
                "latitude": np.array([24.7, 25.7], dtype=np.float32),
                "longitude": np.array([46.7, 47.7], dtype=np.float32),
            },
        )

        model = LightGBMLayer4Model(
            extreme_heat_model=FakeBooster(0.1),
            dry_heat_model=FakeBooster(0.0),
        )
        risk_ds = model.predict_fields(ds)

        self.assertEqual(
            set(risk_ds.data_vars),
            {
                "ExtremeHeat_Risk_Prob",
                "ExtremeHeat_Risk_Level",
                "DryHeatStress_Risk_Prob",
                "DryHeatStress_Risk_Level",
            },
        )
        self.assertEqual(risk_ds["ExtremeHeat_Risk_Prob"].shape, (2, 2))
        self.assertTrue(np.all((risk_ds["ExtremeHeat_Risk_Prob"].values >= 0.0) & (risk_ds["ExtremeHeat_Risk_Prob"].values <= 1.0)))
        self.assertTrue(np.all(np.isin(risk_ds["ExtremeHeat_Risk_Level"].values, [0, 1, 2, 3])))
        self.assertEqual(risk_ds.attrs["model_family"], "lightgbm")

    def test_forecast_layer4_example_smoke(self):
        module = _load_example_module()

        class FakeLayer4Model:
            def __init__(self, *args, **kwargs):
                pass

            def predict_fields(self, dataset):
                base = np.asarray(dataset["temp_c"].values, dtype=np.float32)
                return xr.Dataset(
                    data_vars={
                        "ExtremeHeat_Risk_Prob": (dataset["temp_c"].dims, np.clip((base - 35.0) / 15.0, 0.0, 1.0)),
                        "ExtremeHeat_Risk_Level": (dataset["temp_c"].dims, np.zeros_like(base, dtype=np.int8)),
                        "DryHeatStress_Risk_Prob": (dataset["temp_c"].dims, np.clip((base - 36.0) / 14.0, 0.0, 1.0)),
                        "DryHeatStress_Risk_Level": (dataset["temp_c"].dims, np.ones_like(base, dtype=np.int8)),
                    },
                    coords={name: dataset.coords[name] for name in dataset.coords},
                    attrs={"model_family": "lightgbm", "feature_source_contract_version": "layer4_v1"},
                )

        with tempfile.TemporaryDirectory() as tmp:
            output_netcdf = Path(tmp) / "risk.nc"
            output_json = Path(tmp) / "summary.json"
            with mock.patch.object(module, "LightGBMLayer4Model", FakeLayer4Model):
                result = module.main(
                    [
                        "--provider",
                        "mock",
                        "--lead-hours",
                        "0,6",
                        "--output-netcdf",
                        str(output_netcdf),
                        "--output-json",
                        str(output_json),
                        "--include-gencast-metadata",
                        "--include-aifs-benchmark",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertTrue(output_netcdf.exists())
            self.assertTrue(output_json.exists())
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["provider"], "mock")
            self.assertEqual(payload["auxiliary_metadata"]["gencast"]["status"], "available")
            self.assertEqual(payload["auxiliary_metadata"]["aifs"]["status"], "available")
            self.assertIn("ExtremeHeat_Risk_Prob", payload["summary"])


if __name__ == "__main__":
    unittest.main()
