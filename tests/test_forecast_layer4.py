from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np

from mazu_saudi.forecast import AIFSBenchmarkProvider, GenCastForecastProvider, MockForecastProvider
from mazu_saudi.risk import LightGBMLayer4Model
from mazu_saudi.risk import model_paths

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


def _write_fake_era5_mswep_tree(root: Path, *, year: int = 2025) -> tuple[Path, Path, Path]:
    era5_dir = root / "era5_single_levels"
    precip_dir = root / "precip"
    pressure_dir = root / "era5_pressure_levels"
    era5_dir.mkdir(parents=True, exist_ok=True)
    precip_dir.mkdir(parents=True, exist_ok=True)
    pressure_dir.mkdir(parents=True, exist_ok=True)

    times = np.array([f"{year}-01-01T00:00:00"], dtype="datetime64[ns]")
    lats = np.array([24.7, 25.7], dtype=np.float32)
    lons = np.array([46.7, 47.7], dtype=np.float32)
    levels = np.array([1000, 925, 850, 700, 500, 300, 200], dtype=np.int32)
    shape_3d = (len(times), len(lats), len(lons))
    shape_4d = (len(times), len(levels), len(lats), len(lons))

    instant = xr.Dataset(
        {
            "t2m": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 303.15, dtype=np.float32), {"units": "K"}),
            "d2m": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 293.15, dtype=np.float32), {"units": "K"}),
            "u10": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 6.0, dtype=np.float32), {"units": "m s**-1"}),
            "v10": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 2.0, dtype=np.float32), {"units": "m s**-1"}),
            "sp": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 101325.0, dtype=np.float32), {"units": "Pa"}),
            "cape": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 900.0, dtype=np.float32), {"units": "J kg**-1"}),
        },
        coords={"valid_time": times, "latitude": lats, "longitude": lons},
    )
    accum = xr.Dataset(
        {
            "tp": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 0.012, dtype=np.float32), {"units": "m"}),
            "cp": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 0.004, dtype=np.float32), {"units": "m"}),
        },
        coords={"valid_time": times, "latitude": lats, "longitude": lons},
    )
    instant_path = root / "instant.nc"
    accum_path = root / "accum.nc"
    instant.to_netcdf(instant_path)
    accum.to_netcdf(accum_path)
    max_ds = xr.Dataset(
        {
            "mx2t": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 309.15, dtype=np.float32), {"units": "K"}),
            "mn2t": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 298.15, dtype=np.float32), {"units": "K"}),
        },
        coords={"valid_time": times, "latitude": lats, "longitude": lons},
    )
    max_path = root / "max.nc"
    max_ds.to_netcdf(max_path)
    with zipfile.ZipFile(era5_dir / f"era5_single_levels_{year}_01.nc", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(instant_path, "data_stream-oper_stepType-instant.nc")
        archive.write(accum_path, "data_stream-oper_stepType-accum.nc")
        archive.write(max_path, "data_stream-oper_stepType-max.nc")

    precip = xr.Dataset(
        {
            "precipitation": (("time", "lat", "lon"), np.full((1, len(lats), len(lons)), 12.0, dtype=np.float32), {"units": "mm/day"}),
        },
        coords={"time": np.array([f"{year}-01-01"], dtype="datetime64[ns]"), "lat": lats, "lon": lons},
    )
    precip.to_netcdf(precip_dir / f"{year}001.nc")

    pressure_payloads = {
        "specific_humidity": ("q", 0.01, "kg kg**-1"),
        "u_component_of_wind": ("u", 8.0, "m s**-1"),
        "v_component_of_wind": ("v", 3.0, "m s**-1"),
    }
    for suffix, (var_name, value, units) in pressure_payloads.items():
        ds = xr.Dataset(
            {
                var_name: (
                    ("valid_time", "pressure_level", "latitude", "longitude"),
                    np.full(shape_4d, value, dtype=np.float32),
                    {"units": units},
                )
            },
            coords={"valid_time": times, "pressure_level": levels, "latitude": lats, "longitude": lons},
        )
        ds.to_netcdf(pressure_dir / f"era5_pl_{year}_01_{suffix}.nc")

    return era5_dir, precip_dir, pressure_dir


def _write_fake_heat_climatology_tree(root: Path) -> Path:
    climatology_dir = root / "heat_climatology"
    climatology_dir.mkdir(parents=True, exist_ok=True)
    times = np.array(["2025-07-11T00:00:00"], dtype="datetime64[ns]")
    lats = np.array([24.7, 25.7], dtype=np.float32)
    lons = np.array([46.7, 47.7], dtype=np.float32)
    for index, (temp_c, tmax_c) in enumerate(((28.0, 33.0), (29.0, 34.0)), start=1):
        dataset = xr.Dataset(
            {
                "t2m_c": (("time", "latitude", "longitude"), np.full((1, len(lats), len(lons)), temp_c, dtype=np.float32)),
                "tmax_c": (("time", "latitude", "longitude"), np.full((1, len(lats), len(lons)), tmax_c, dtype=np.float32)),
            },
            coords={"time": times, "latitude": lats, "longitude": lons},
        )
        dataset.to_netcdf(climatology_dir / f"saudi_indicators_2024010{index}.nc")
    return climatology_dir


@unittest.skipIf(xr is None, "xarray is required for forecast Layer-4 tests")
class ForecastLayer4Tests(unittest.TestCase):
    def test_mock_provider_forecast_dataset_contains_layer4_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            heat_climatology_dir = _write_fake_heat_climatology_tree(Path(tmp))
            provider = MockForecastProvider(heat_climatology_dir=heat_climatology_dir)
            ds = provider.forecast_dataset(datetime(2026, 7, 11, tzinfo=timezone.utc), 0)

        for name in (
            "temp_c",
            "tmax_c",
            "tmin_c",
            "rh_percent",
            "wind_speed_mps",
            "heat_index_c",
            "vpd_kpa",
            "relative_humidity_percent",
            "t2m_anomaly_c",
            "tmax_anomaly_c",
            "heatwave_day_flag",
            "heatwave_duration_days",
            "daily_precip_total",
            "daily_convective_precip",
            "daily_large_scale_precip",
            "cape",
            "pwat",
            "ivt",
            "wind850_speed",
            "wind_shear_850_200",
            "flash_flood_risk",
        ):
            self.assertIn(name, ds.data_vars)
        self.assertEqual(ds.attrs["primary_provider"], "mock")
        self.assertEqual(ds.attrs["source_status"], "degraded")
        degradation_metadata = json.loads(ds.attrs["degradation_metadata_json"])
        self.assertIn("runtime_fallbacks", degradation_metadata)
        self.assertEqual(degradation_metadata["runtime_fallbacks"][0]["fallback_status"], "heuristic_proxy_fallback")
        source_metadata = json.loads(ds.attrs["source_metadata_json"])
        flood_metadata = source_metadata["grounding_gap"]["flash_flood_features"]
        self.assertEqual(flood_metadata["feature_status"]["pwat"]["status"], "direct")
        self.assertEqual(flood_metadata["feature_status"]["ivt"]["status"], "direct")
        self.assertEqual(flood_metadata["feature_status"]["wind850_speed"]["status"], "direct")
        self.assertEqual(flood_metadata["feature_status"]["wind_shear_850_200"]["status"], "direct")
        self.assertEqual(flood_metadata["feature_status"]["daily_convective_precip"]["status"], "proxy")
        self.assertEqual(flood_metadata["feature_status"]["daily_large_scale_precip"]["status"], "proxy")
        self.assertIn("daily_convective_precip", flood_metadata["proxy_features"])
        self.assertIn("daily_large_scale_precip", flood_metadata["proxy_features"])
        self.assertEqual(flood_metadata["precipitation_partition"]["status"], "heuristic_proxy_fallback")
        self.assertEqual(flood_metadata["precipitation_partition"]["fallback_reason"], "direct_precip_partition_unavailable")
        self.assertEqual(flood_metadata["precipitation_partition"]["source_pair"], ["mock", "cape_proxy"])
        heat_metadata = source_metadata["grounding_gap"]["heat_features"]
        self.assertEqual(heat_metadata["status"], "comparison_available")
        self.assertEqual(heat_metadata["feature_status"]["heatwave_day_flag"]["status"], "derived_context_only")
        self.assertEqual(heat_metadata["feature_status"]["heatwave_duration_days"]["status"], "derived_context_only")
        self.assertEqual(heat_metadata["feature_status"]["t2m_anomaly_c"]["status"], "derived")
        self.assertEqual(heat_metadata["feature_status"]["tmax_anomaly_c"]["status"], "derived")
        self.assertEqual(heat_metadata["feature_status"]["t2m_anomaly_c"]["comparison_source_id"], "historical_indicator_archive")
        self.assertEqual(heat_metadata["feature_status"]["tmax_anomaly_c"]["comparison_source_id"], "historical_indicator_archive")
        self.assertEqual(heat_metadata["feature_status"]["t2m_anomaly_c"]["method"], "day_of_year_archive_mean")
        self.assertEqual(heat_metadata["feature_status"]["tmax_anomaly_c"]["method"], "day_of_year_archive_mean")
        self.assertEqual(heat_metadata["source_pair"], ["mock", "historical_indicator_archive"])
        self.assertEqual(heat_metadata["heatwave_context"]["status"], "single_day_context_only")
        self.assertEqual(heat_metadata["heatwave_context"]["thresholds"]["tmax_c_ge"], 45.0)
        self.assertEqual(heat_metadata["feature_status"]["t2m_anomaly_c"]["comparison_schedule"][0]["day_of_year"], 192)
        self.assertEqual(heat_metadata["feature_status"]["tmax_anomaly_c"]["comparison_schedule"][0]["day_of_year"], 192)
        t2m_anomaly = np.asarray(ds["t2m_anomaly_c"].values, dtype=np.float32)
        tmax_anomaly = np.asarray(ds["tmax_anomaly_c"].values, dtype=np.float32)
        self.assertEqual(int(np.isfinite(t2m_anomaly).sum()), 3)
        self.assertEqual(int(np.isfinite(tmax_anomaly).sum()), 3)
        self.assertTrue(np.allclose(np.sort(t2m_anomaly[np.isfinite(t2m_anomaly)]), np.array([9.5, 12.5, 16.5], dtype=np.float32)))
        self.assertTrue(np.allclose(np.sort(tmax_anomaly[np.isfinite(tmax_anomaly)]), np.array([6.5, 9.5, 13.5], dtype=np.float32)))
        heatwave_flag = np.asarray(ds["heatwave_day_flag"].values)
        heatwave_duration = np.asarray(ds["heatwave_duration_days"].values)
        self.assertEqual(int(np.isfinite(heatwave_flag).sum()), 3)
        self.assertEqual(int(np.isfinite(heatwave_duration).sum()), 3)
        self.assertTrue(np.all(heatwave_flag[np.isfinite(heatwave_flag)] == 1.0))
        self.assertTrue(np.all(heatwave_duration[np.isfinite(heatwave_duration)] == 1.0))

    def test_auxiliary_provider_metadata_survives_dataset_export(self):
        issue_time = datetime(2026, 7, 11, tzinfo=timezone.utc)
        gencast = GenCastForecastProvider().forecast_dataset(issue_time, 0)
        aifs = AIFSBenchmarkProvider().forecast_dataset(issue_time, 0)

        self.assertEqual(gencast.attrs["primary_provider"], "gencast")
        self.assertEqual(gencast.attrs["ensemble_member_count"], 4)
        self.assertEqual(aifs.attrs["primary_provider"], "aifs")
        self.assertIn("benchmark_comparison_json", aifs.attrs)

    def test_era5_mswep_forecast_dataset_marks_pressure_fields_direct(self):
        from mazu_saudi.forecast import ERA5MSWEPForecastProvider

        with tempfile.TemporaryDirectory() as tmp:
            era5_dir, precip_dir, pressure_dir = _write_fake_era5_mswep_tree(Path(tmp))
            heat_climatology_dir = _write_fake_heat_climatology_tree(Path(tmp))
            provider = ERA5MSWEPForecastProvider(
                era5_dir=era5_dir,
                precip_dir=precip_dir,
                pressure_dir=pressure_dir,
                heat_climatology_dir=heat_climatology_dir,
            )
            try:
                ds = provider.forecast_dataset(datetime(2025, 1, 1, tzinfo=timezone.utc), 0)
            finally:
                provider.close()

        for name in (
            "tmax_c",
            "tmin_c",
            "t2m_anomaly_c",
            "tmax_anomaly_c",
            "heatwave_day_flag",
            "heatwave_duration_days",
            "pwat",
            "ivt",
            "wind850_speed",
            "wind_shear_850_200",
            "daily_convective_precip",
            "daily_large_scale_precip",
            "flash_flood_risk",
        ):
            self.assertIn(name, ds.data_vars)
        for name in (
            "tmax_c",
            "tmin_c",
            "heatwave_day_flag",
            "heatwave_duration_days",
            "pwat",
            "ivt",
            "wind850_speed",
            "wind_shear_850_200",
            "daily_convective_precip",
            "daily_large_scale_precip",
            "flash_flood_risk",
        ):
            self.assertTrue(np.isfinite(np.asarray(ds[name].values)).any(), name)
        source_metadata = json.loads(ds.attrs["source_metadata_json"])
        flood_metadata = source_metadata["grounding_gap"]["flash_flood_features"]
        self.assertEqual(ds.attrs["source_status"], "normal")
        self.assertEqual(json.loads(ds.attrs["degradation_metadata_json"]), {})
        self.assertEqual(flood_metadata["feature_status"]["pwat"]["status"], "direct")
        self.assertEqual(flood_metadata["feature_status"]["ivt"]["status"], "direct")
        self.assertEqual(flood_metadata["feature_status"]["wind850_speed"]["status"], "direct")
        self.assertEqual(flood_metadata["feature_status"]["wind_shear_850_200"]["status"], "direct")
        self.assertEqual(flood_metadata["feature_status"]["daily_convective_precip"]["status"], "direct")
        self.assertEqual(flood_metadata["feature_status"]["daily_large_scale_precip"]["status"], "derived")
        self.assertNotIn("daily_convective_precip", flood_metadata["proxy_features"])
        self.assertNotIn("daily_large_scale_precip", flood_metadata["proxy_features"])
        self.assertEqual(flood_metadata["precipitation_partition"]["status"], "same_source_residual")
        self.assertEqual(flood_metadata["precipitation_partition"]["source_pair"], ["era5_mswep", "era5_mswep"])
        heat_metadata = source_metadata["grounding_gap"]["heat_features"]
        self.assertEqual(heat_metadata["status"], "comparison_available")
        self.assertEqual(heat_metadata["feature_status"]["heatwave_day_flag"]["status"], "derived_context_only")
        self.assertEqual(heat_metadata["feature_status"]["heatwave_duration_days"]["status"], "derived_context_only")
        self.assertEqual(heat_metadata["feature_status"]["t2m_anomaly_c"]["status"], "derived")
        self.assertEqual(heat_metadata["feature_status"]["tmax_anomaly_c"]["status"], "derived")
        self.assertEqual(
            heat_metadata["feature_status"]["t2m_anomaly_c"]["method"],
            "historical_indicator_archive_mean_fallback",
        )
        self.assertEqual(
            heat_metadata["feature_status"]["tmax_anomaly_c"]["method"],
            "historical_indicator_archive_mean_fallback",
        )
        self.assertEqual(heat_metadata["source_pair"], ["era5_mswep", "historical_indicator_archive"])
        self.assertEqual(heat_metadata["heatwave_context"]["status"], "single_day_context_only")
        self.assertAlmostEqual(float(np.asarray(ds["daily_convective_precip"].values)[0, 0, 0]), 4.0, places=4)
        self.assertAlmostEqual(float(np.asarray(ds["daily_large_scale_precip"].values)[0, 0, 0]), 8.0, places=4)
        self.assertAlmostEqual(float(np.asarray(ds["tmax_c"].values)[0, 0, 0]), 36.0, places=4)
        self.assertAlmostEqual(float(np.asarray(ds["tmin_c"].values)[0, 0, 0]), 25.0, places=4)
        self.assertTrue(np.isfinite(np.asarray(ds["t2m_anomaly_c"].values)).all())
        self.assertTrue(np.isfinite(np.asarray(ds["tmax_anomaly_c"].values)).all())
        self.assertAlmostEqual(float(np.asarray(ds["t2m_anomaly_c"].values)[0, 0, 0]), 1.5, places=4)
        self.assertAlmostEqual(float(np.asarray(ds["tmax_anomaly_c"].values)[0, 0, 0]), 2.5, places=4)
        self.assertAlmostEqual(float(np.asarray(ds["heatwave_day_flag"].values)[0, 0, 0]), 0.0, places=4)
        self.assertAlmostEqual(float(np.asarray(ds["heatwave_duration_days"].values)[0, 0, 0]), 0.0, places=4)

        model = LightGBMLayer4Model(flash_flood_model=FakeBooster(0.05))
        flood_features, flood_shape = model._aligned_feature_matrix(ds, "flash_flood", model.flash_flood_model)
        flash_flood_prob = model._predict_probability(model.flash_flood_model, flood_features, flood_shape)
        self.assertEqual(flash_flood_prob.shape, (2, 2))
        self.assertTrue(np.all((flash_flood_prob >= 0.0) & (flash_flood_prob <= 1.0)))

    def test_layer4_grid_inference_outputs_probability_and_level_fields(self):
        ds = xr.Dataset(
            data_vars={
                "temp_c": (("time", "latitude", "longitude"), np.array([[[41.0, 43.0], [39.0, 45.0]]], dtype=np.float32)),
                "tmax_c": (("time", "latitude", "longitude"), np.array([[[43.0, 45.0], [41.0, 47.0]]], dtype=np.float32)),
                "tmin_c": (("time", "latitude", "longitude"), np.array([[[31.0, 32.0], [30.0, 33.0]]], dtype=np.float32)),
                "vpd_kpa": (("time", "latitude", "longitude"), np.array([[[2.0, 3.0], [1.5, 4.0]]], dtype=np.float32)),
                "heat_index_c": (("time", "latitude", "longitude"), np.array([[[42.0, 45.0], [40.0, 48.0]]], dtype=np.float32)),
                "wind_speed_mps": (("time", "latitude", "longitude"), np.array([[[5.0, 6.0], [4.0, 7.0]]], dtype=np.float32)),
                "relative_humidity_percent": (("time", "latitude", "longitude"), np.array([[[35.0, 30.0], [40.0, 28.0]]], dtype=np.float32)),
                "daily_precip_total": (("time", "latitude", "longitude"), np.array([[[8.0, 12.0], [5.0, 15.0]]], dtype=np.float32)),
                "daily_convective_precip": (("time", "latitude", "longitude"), np.array([[[2.0, 5.0], [1.0, 6.0]]], dtype=np.float32)),
                "daily_large_scale_precip": (("time", "latitude", "longitude"), np.array([[[3.0, 4.0], [2.0, 5.0]]], dtype=np.float32)),
                "cape": (("time", "latitude", "longitude"), np.array([[[500.0, 900.0], [300.0, 1200.0]]], dtype=np.float32)),
                "pwat": (("time", "latitude", "longitude"), np.array([[[22.0, 28.0], [20.0, 31.0]]], dtype=np.float32)),
                "ivt": (("time", "latitude", "longitude"), np.array([[[80.0, 120.0], [75.0, 150.0]]], dtype=np.float32)),
                "wind850_speed": (("time", "latitude", "longitude"), np.array([[[7.0, 9.0], [6.0, 11.0]]], dtype=np.float32)),
                "wind_shear_850_200": (("time", "latitude", "longitude"), np.array([[[18.0, 24.0], [16.0, 28.0]]], dtype=np.float32)),
                "flash_flood_risk": (("time", "latitude", "longitude"), np.array([[[1.0, 2.0], [1.0, 3.0]]], dtype=np.float32)),
                "daily_precip_anomaly": (("time", "latitude", "longitude"), np.array([[[1.0, 4.0], [0.5, 6.0]]], dtype=np.float32)),
                "t2m_anomaly_c": (("time", "latitude", "longitude"), np.array([[[0.5, 1.0], [0.2, 1.5]]], dtype=np.float32)),
                "tmax_anomaly_c": (("time", "latitude", "longitude"), np.array([[[1.0, 1.4], [0.5, 1.8]]], dtype=np.float32)),
                "heatwave_day_flag": (("time", "latitude", "longitude"), np.array([[[0, 1], [0, 1]]], dtype=np.int16)),
                "heatwave_duration_days": (("time", "latitude", "longitude"), np.array([[[0, 2], [0, 3]]], dtype=np.int16)),
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
            flash_flood_model=FakeBooster(0.05),
        )
        risk_ds = model.predict_fields(ds)

        self.assertEqual(
            set(risk_ds.data_vars),
            {
                "ExtremeHeat_Risk_Prob",
                "ExtremeHeat_Risk_Level",
                "DryHeatStress_Risk_Prob",
                "DryHeatStress_Risk_Level",
                "FlashFlood_Risk_Prob",
                "FlashFlood_Risk_Level",
            },
        )
        self.assertEqual(risk_ds["ExtremeHeat_Risk_Prob"].shape, (2, 2))
        self.assertEqual(risk_ds["FlashFlood_Risk_Prob"].shape, (2, 2))
        self.assertTrue(np.all((risk_ds["ExtremeHeat_Risk_Prob"].values >= 0.0) & (risk_ds["ExtremeHeat_Risk_Prob"].values <= 1.0)))
        self.assertTrue(np.all(np.isin(risk_ds["ExtremeHeat_Risk_Level"].values, [0, 1, 2, 3])))
        self.assertTrue(np.all((risk_ds["FlashFlood_Risk_Prob"].values >= 0.0) & (risk_ds["FlashFlood_Risk_Prob"].values <= 1.0)))
        self.assertEqual(risk_ds.attrs["model_family"], "lightgbm")

    def test_layer4_model_resolves_flash_flood_verified_chain_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            verified_model = repo_root / "data" / "processed" / "models" / "flash_flood_province_day_verified_chain_baseline_quick" / "flash_flood.txt"
            verified_model.parent.mkdir(parents=True, exist_ok=True)
            verified_model.write_text("stub", encoding="utf-8")
            with mock.patch.object(model_paths, "REPO_ROOT", repo_root), mock.patch.object(
                model_paths,
                "DEFAULT_LAYER4_MODEL_DIR",
                repo_root / "models" / "layer4",
            ), mock.patch.object(LightGBMLayer4Model, "_load_optional_booster", return_value="flash-booster"):
                model = LightGBMLayer4Model(
                    extreme_heat_model=FakeBooster(0.1),
                    dry_heat_model=FakeBooster(0.0),
                )
        self.assertEqual(model.flash_flood_model_path, verified_model)
        self.assertEqual(model.flash_flood_model, "flash-booster")

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
                        "FlashFlood_Risk_Prob": (dataset["temp_c"].dims, np.clip(base / 100.0, 0.0, 1.0)),
                        "FlashFlood_Risk_Level": (dataset["temp_c"].dims, np.full_like(base, 2, dtype=np.int8)),
                    },
                    coords={name: dataset.coords[name] for name in dataset.coords},
                    attrs={"model_family": "lightgbm", "feature_source_contract_version": "layer4_v2"},
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
            forecast_metadata = json.loads(payload["forecast_metadata"]["source_metadata_json"])
            self.assertEqual(
                forecast_metadata["grounding_gap"]["flash_flood_features"]["precipitation_partition"]["status"],
                "heuristic_proxy_fallback",
            )
            self.assertIn("ExtremeHeat_Risk_Prob", payload["summary"])
            self.assertIn("FlashFlood_Risk_Prob", payload["summary"])


if __name__ == "__main__":
    unittest.main()
