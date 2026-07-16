from __future__ import annotations

import json
import tempfile
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from mazu_saudi.indicators.build_inputs import PRESSURE_LEVELS, RawInputBuilder


def _write_dataset(path: Path, ds: xr.Dataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _write_fake_srtm_tile(path: Path) -> None:
    array = np.full((1201, 1201), 100, dtype=np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("N16E034.hgt", array.astype(">i2").tobytes())


def _build_fake_raw_tree(
    root: Path,
    *,
    year: int = 2025,
    include_oisst: bool = True,
    include_jpl_mur: bool = False,
    oisst_value_c: float = 25.0,
    jpl_value_c: float = 27.0,
    supplemental_pressure_levels: dict[str, tuple[int, ...]] | None = None,
    single_layout: str = "zip",
) -> Path:
    raw_root = root / "data" / "raw"
    single_dir = raw_root / f"era5_single_levels_{year}"
    pressure_dir = raw_root / f"era5_pressure_levels_{year}"
    missing_pressure_dir = raw_root / f"era5_pressure_levels_{year}_missing"
    precip_dir = raw_root / "precip"
    dust_dir = raw_root / "dust"
    sst_dir = raw_root / "sst"
    nis_dir = root / "data" / "output" / "nis"

    times = np.array(
        [
            f"{year}-01-01T00:00:00",
            f"{year}-01-01T06:00:00",
            f"{year}-01-01T12:00:00",
            f"{year}-01-01T18:00:00",
            f"{year}-01-02T00:00:00",
        ],
        dtype="datetime64[ns]",
    )
    lats = np.array([32.0, 16.0], dtype=np.float32)
    lons = np.array([34.0, 56.0], dtype=np.float32)
    levels = np.array(PRESSURE_LEVELS, dtype=np.int32)
    shape_3d = (len(times), len(lats), len(lons))
    shape_4d = (len(times), len(levels), len(lats), len(lons))

    instant = xr.Dataset(
        {
            "t2m": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 300.0, dtype=np.float32), {"units": "K"}),
            "d2m": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 290.0, dtype=np.float32), {"units": "K"}),
            "u10": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 5.0, dtype=np.float32), {"units": "m s**-1"}),
            "v10": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 1.0, dtype=np.float32), {"units": "m s**-1"}),
            "sp": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 101325.0, dtype=np.float32), {"units": "Pa"}),
            "cape": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 1200.0, dtype=np.float32), {"units": "J kg**-1"}),
            "cin": (("valid_time", "latitude", "longitude"), np.full(shape_3d, -50.0, dtype=np.float32), {"units": "J kg**-1"}),
            "tcc": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 0.3, dtype=np.float32), {"units": "(0 - 1)"}),
            "lcc": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 0.1, dtype=np.float32), {"units": "(0 - 1)"}),
            "mcc": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 0.2, dtype=np.float32), {"units": "(0 - 1)"}),
            "hcc": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 0.4, dtype=np.float32), {"units": "(0 - 1)"}),
            "z": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 100.0, dtype=np.float32), {"units": "m**2 s**-2"}),
        },
        coords={"valid_time": times, "latitude": lats, "longitude": lons},
    )
    instant["t2m"].values[0, 0, 0] = 9.96921e36
    instant["d2m"].values[0, 0, 0] = 9.96921e36
    accum_steps = np.array([0.0, 0.001, 0.002, 0.003, 0.004], dtype=np.float32)[:, None, None]
    accum = xr.Dataset(
        {
            "tp": (("valid_time", "latitude", "longitude"), np.broadcast_to(accum_steps, shape_3d).copy(), {"units": "m"}),
            "cp": (("valid_time", "latitude", "longitude"), np.broadcast_to(accum_steps * 0.5, shape_3d).copy(), {"units": "m"}),
            "ssrd": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 100.0, dtype=np.float32), {"units": "J m-2"}),
            "strd": (("valid_time", "latitude", "longitude"), np.full(shape_3d, -40.0, dtype=np.float32), {"units": "J m-2"}),
            "sshf": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 50.0, dtype=np.float32)),
            "slhf": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 100.0, dtype=np.float32)),
        },
        coords={"valid_time": times, "latitude": lats, "longitude": lons},
    )
    max_ds = xr.Dataset(
        {
            "mx2t": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 305.0, dtype=np.float32), {"units": "K"}),
            "mn2t": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 295.0, dtype=np.float32), {"units": "K"}),
        },
        coords={"valid_time": times, "latitude": lats, "longitude": lons},
    )

    tmp_instant = root / "instant.nc"
    tmp_accum = root / "accum.nc"
    tmp_max = root / "max.nc"
    instant.to_netcdf(tmp_instant)
    accum.to_netcdf(tmp_accum)
    max_ds.to_netcdf(tmp_max)
    primary_path = single_dir / f"era5_single_levels_{year}_01.nc"
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    if single_layout == "zip":
        with zipfile.ZipFile(primary_path, "w") as archive:
            archive.write(tmp_instant, "data_stream-oper_stepType-instant.nc")
            archive.write(tmp_accum, "data_stream-oper_stepType-accum.nc")
            archive.write(tmp_max, "data_stream-oper_stepType-max.nc")
    elif single_layout == "direct_plus_supplement":
        primary_direct = xr.merge(
            [
                instant[["t2m", "u10", "v10", "z"]],
                xr.Dataset(
                    {
                        "msl": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 101000.0, dtype=np.float32), {"units": "Pa"}),
                        "lsm": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 1.0, dtype=np.float32)),
                        "slt": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 2.0, dtype=np.float32)),
                    },
                    coords={"valid_time": times, "latitude": lats, "longitude": lons},
                ),
            ],
            compat="override",
        )
        _write_dataset(primary_path, primary_direct)
        supplement_path = single_dir / f"era5_single_levels_{year}_01_supplement.nc"
        direct_missing = instant[["d2m", "sp", "cape", "cin", "tcc", "lcc", "mcc", "hcc"]]
        tmp_direct_missing = root / "instant_missing.nc"
        direct_missing.to_netcdf(tmp_direct_missing)
        with zipfile.ZipFile(supplement_path, "w") as archive:
            archive.write(tmp_direct_missing, "data_stream-oper_stepType-instant.nc")
            archive.write(tmp_accum, "data_stream-oper_stepType-accum.nc")
            archive.write(tmp_max, "data_stream-oper_stepType-max.nc")
    else:
        raise ValueError(f"unsupported single_layout: {single_layout}")

    aurora_single = xr.Dataset(
        {
            "lsm": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 1.0, dtype=np.float32)),
            "msl": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 101000.0, dtype=np.float32), {"units": "Pa"}),
            "slt": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 2.0, dtype=np.float32)),
        },
        coords={"valid_time": times, "latitude": lats, "longitude": lons},
    )
    _write_dataset(single_dir / f"era5_single_levels_{year}_01_aurora.nc", aurora_single)
    _write_dataset(single_dir / f"era5_single_levels_{year}_01_static.nc", aurora_single[["lsm", "slt"]].isel(valid_time=slice(0, 1)))

    pressure_payloads = {
        "specific_humidity": ("q", 0.01, "kg kg**-1"),
        "u_component_of_wind": ("u", 8.0, "m s**-1"),
        "v_component_of_wind": ("v", 3.0, "m s**-1"),
        "geopotential": ("z", 58000.0, "m**2 s**-2"),
        "temperature": ("t", 280.0, "K"),
        "vertical_velocity": ("w", -0.2, "Pa s**-1"),
        "divergence": ("d", -1e-5, "s**-1"),
        "relative_vorticity": ("r", 60.0, "%"),
    }
    supplemental_pressure_levels = supplemental_pressure_levels or {}
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
        supplemental_levels = supplemental_pressure_levels.get(suffix, ())
        if supplemental_levels:
            primary_levels = [level for level in levels.tolist() if level not in supplemental_levels]
            _write_dataset(pressure_dir / f"era5_pl_{year}_01_{suffix}.nc", ds.sel(pressure_level=primary_levels))
            _write_dataset(
                missing_pressure_dir / f"era5_pl_{year}_01_{suffix}_missing.nc",
                ds.sel(pressure_level=list(supplemental_levels)),
            )
            continue
        _write_dataset(pressure_dir / f"era5_pl_{year}_01_{suffix}.nc", ds)

    gpm = xr.Dataset(
        {"precipitation": (("time", "lat", "lon"), np.full((1, len(lats), len(lons)), 12.0, dtype=np.float32), {"units": "mm/day"})},
        coords={"time": np.array([f"{year}-01-01"], dtype="datetime64[ns]"), "lat": lats[::-1], "lon": lons},
    )
    _write_dataset(precip_dir / f"GPM_3IMERGDF_{year}0101.nc4", gpm)

    dust = xr.Dataset(
        {
            "DUEXTTAU": (("time", "lat", "lon"), np.full((2, len(lats), len(lons)), 0.4, dtype=np.float32), {"units": "1"}),
            "DUCMASS": (("time", "lat", "lon"), np.full((2, len(lats), len(lons)), 0.2, dtype=np.float32), {"units": "kg m-2"}),
            "DUSMASS": (("time", "lat", "lon"), np.full((2, len(lats), len(lons)), 0.001, dtype=np.float32), {"units": "kg m-3"}),
        },
        coords={"time": np.array([f"{year}-01-01T00:00:00", f"{year}-01-01T12:00:00"], dtype="datetime64[ns]"), "lat": lats, "lon": lons},
    )
    _write_dataset(dust_dir / f"MERRA2_{year}0101.nc4", dust)

    if include_oisst:
        sst = xr.Dataset(
            {"sst": (("time", "lat", "lon"), np.full((1, len(lats), len(lons)), oisst_value_c, dtype=np.float32), {"units": "degC"})},
            coords={"time": np.array([f"{year}-01-01"], dtype="datetime64[ns]"), "lat": lats, "lon": lons},
        )
        _write_dataset(sst_dir / f"oisst.day.mean.{year}.nc", sst)
    if include_jpl_mur:
        jpl = xr.Dataset(
            {
                "analysed_sst": (
                    ("time", "lat", "lon"),
                    np.full((1, len(lats), len(lons)), jpl_value_c, dtype=np.float32),
                    {"units": "degC"},
                )
            },
            coords={"time": np.array([f"{year}-01-01"], dtype="datetime64[ns]"), "lat": lats, "lon": lons},
        )
        _write_dataset(sst_dir / "jplMURSST41_72e4_84a7_e7ac.nc", jpl)

    chirps = xr.Dataset(
        {"precip": (("time", "latitude", "longitude"), np.full((1, len(lats), len(lons)), 31.0, dtype=np.float32), {"units": "mm/month"})},
        coords={"time": np.array([f"{year}-01-01"], dtype="datetime64[ns]"), "latitude": lats, "longitude": lons},
    )
    _write_dataset(precip_dir / f"chirps-v3.0.{year}.monthly.nc", chirps)

    elevation = xr.Dataset(
        {"elevation_m": (("latitude", "longitude"), np.array([[100.0, 110.0], [90.0, 95.0]], dtype=np.float32))},
        coords={"latitude": lats[::-1], "longitude": lons},
    )
    _write_dataset(nis_dir / "nis_elevation_grid.nc", elevation)

    return raw_root


class TestBuildInputs:
    def test_build_daily_indicators(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_fake_raw_tree(Path(tmp))
            builder = RawInputBuilder(
                raw_root=raw_root,
                aurora_out=Path(tmp) / "aurora",
                indicator_nc_out=Path(tmp) / "nc",
                indicator_parquet_out=Path(tmp) / "pq",
            )
            try:
                ds = builder.build_daily_indicators(date(2025, 1, 1))
            finally:
                builder.close()

        assert "t2m_c" in ds.data_vars
        assert "ivt" in ds.data_vars
        assert "flash_flood_risk" in ds.data_vars
        assert "monthly_chirps_precip_total" in ds.data_vars
        assert "apparent_temp_c" in ds.data_vars
        assert "absolute_vorticity850" in ds.data_vars
        assert ds["flash_flood_risk"].dims == ("time", "latitude", "longitude")
        assert int(ds.sizes["time"]) == 1
        assert "missing_indicator_groups" in ds.attrs
        assert np.isfinite(ds["vpd_kpa"].values).any()
        assert float(ds["pwat"].min()) >= 0.0
        assert float(ds["daily_precip_total"].min()) >= 0.0
        assert not np.isinf(ds["convective_precip_ratio"].values).any()
        assert not np.isinf(ds["gpm_era5_precip_ratio"].values).any()
        source_metadata = json.loads(ds.attrs["source_metadata_json"])
        assert source_metadata["resolved_sources"]["sst"]["resolved_source"] == "oisst"
        assert source_metadata["validation_status"]["sst"] == "primary_only"
        assert source_metadata["resolved_sources"]["dust"]["resolved_source"] == "merra2_dust"
        assert source_metadata["validation_status"]["dust"] == "primary_only"
        assert source_metadata["resolved_sources"]["precip_daily"]["resolved_source"] == "gpm_imerg_daily"
        assert source_metadata["validation_status"]["precip_daily"] == "compared"
        assert source_metadata["resolved_sources"]["precip_monthly"]["resolved_source"] == "chirps_monthly"
        assert source_metadata["validation_status"]["precip_monthly"] == "primary_only"

    def test_build_aurora_input_uses_namespaced_variables(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_fake_raw_tree(Path(tmp))
            builder = RawInputBuilder(
                raw_root=raw_root,
                aurora_out=Path(tmp) / "aurora",
                indicator_nc_out=Path(tmp) / "nc",
                indicator_parquet_out=Path(tmp) / "pq",
            )
            try:
                ds = builder.build_aurora_input(datetime(2025, 1, 1, 6, tzinfo=timezone.utc))
            finally:
                builder.close()

        assert "surf_2t" in ds.data_vars
        assert "static_z" in ds.data_vars
        assert "atmos_z" in ds.data_vars
        assert "z" not in ds.data_vars
        assert ds["surf_2t"].dims == ("time", "lat", "lon")
        assert ds["atmos_q"].dims == ("time", "level", "lat", "lon")
        assert list(ds["level"].values) == list(PRESSURE_LEVELS)

    def test_build_daily_indicators_uses_primary_sst_and_compares_secondary(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_fake_raw_tree(
                Path(tmp),
                include_oisst=True,
                include_jpl_mur=True,
                oisst_value_c=25.0,
                jpl_value_c=27.0,
            )
            builder = RawInputBuilder(
                raw_root=raw_root,
                aurora_out=Path(tmp) / "aurora",
                indicator_nc_out=Path(tmp) / "nc",
                indicator_parquet_out=Path(tmp) / "pq",
            )
            try:
                ds = builder.build_daily_indicators(date(2025, 1, 1))
            finally:
                builder.close()

        assert float(ds["sst_celsius"].isel(time=0, latitude=0, longitude=0)) == 25.0
        source_metadata = json.loads(ds.attrs["source_metadata_json"])
        sst_meta = source_metadata["resolved_sources"]["sst"]
        assert sst_meta["resolved_source"] == "oisst"
        assert sst_meta["secondary_sources"] == ["jpl_mur"]
        assert sst_meta["comparison_summary"][0]["dataset_id"] == "jpl_mur"
        assert sst_meta["comparison_summary"][0]["mean_abs_diff"] == 2.0
        assert source_metadata["validation_status"]["sst"] == "compared"
        precip_meta = source_metadata["resolved_sources"]["precip_daily"]
        assert precip_meta["comparison_summary"][0]["against_source"] == "era5"
        assert np.isclose(precip_meta["comparison_summary"][0]["mean_abs_diff"], 9.0)

    def test_build_daily_indicators_falls_back_to_secondary_sst(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_fake_raw_tree(
                Path(tmp),
                include_oisst=False,
                include_jpl_mur=True,
                jpl_value_c=27.0,
            )
            builder = RawInputBuilder(
                raw_root=raw_root,
                aurora_out=Path(tmp) / "aurora",
                indicator_nc_out=Path(tmp) / "nc",
                indicator_parquet_out=Path(tmp) / "pq",
            )
            try:
                ds = builder.build_daily_indicators(date(2025, 1, 1))
            finally:
                builder.close()

        assert float(ds["sst_celsius"].isel(time=0, latitude=0, longitude=0)) == 27.0
        source_metadata = json.loads(ds.attrs["source_metadata_json"])
        sst_meta = source_metadata["resolved_sources"]["sst"]
        assert sst_meta["resolved_source"] == "jpl_mur"
        assert sst_meta["fallback_chain"][0]["status"] == "missing"
        assert sst_meta["fallback_chain"][1]["status"] == "available"

    def test_build_manifest_includes_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_root = _build_fake_raw_tree(root, include_oisst=True, include_jpl_mur=True)
            builder = RawInputBuilder(
                raw_root=raw_root,
                aurora_out=root / "aurora",
                indicator_nc_out=root / "nc",
                indicator_parquet_out=root / "pq",
            )
            try:
                builder.build(date(2025, 1, 1), date(2025, 1, 1))
            finally:
                builder.close()

            manifest = json.loads((root / "build_manifest.json").read_text(encoding="utf-8"))

        indicator_entry = next(entry for entry in manifest if entry["kind"] == "indicator_nc")
        assert indicator_entry["metadata"]["resolved_sources"]["sst"]["resolved_source"] == "oisst"
        assert indicator_entry["metadata"]["resolved_sources"]["dust"]["resolved_source"] == "merra2_dust"
        assert indicator_entry["metadata"]["resolved_sources"]["precip_daily"]["resolved_source"] == "gpm_imerg_daily"

    def test_build_falls_back_to_csv_when_parquet_engine_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_root = _build_fake_raw_tree(root, include_oisst=True, include_jpl_mur=True)
            builder = RawInputBuilder(
                raw_root=raw_root,
                aurora_out=root / "aurora",
                indicator_nc_out=root / "nc",
                indicator_parquet_out=root / "pq",
            )
            original_to_parquet = getattr(builder._daily_to_frame(builder.build_daily_indicators(date(2025, 1, 1)), date(2025, 1, 1)).__class__, "to_parquet")

            def _raise_missing_engine(self, path, index=False):
                raise ImportError("missing parquet engine")

            try:
                import pandas as pd

                pd.DataFrame.to_parquet = _raise_missing_engine
                result = builder.build(date(2025, 1, 1), date(2025, 1, 1), include_aurora=False)
            finally:
                import pandas as pd

                pd.DataFrame.to_parquet = original_to_parquet
                builder.close()

            table_entry = next(entry for entry in result.entries if entry.kind == "indicator_table")
            csv_path = root / "pq" / "saudi_indicator_samples_20250101.csv"
            csv_exists = csv_path.exists()

        assert table_entry.status == "ok"
        assert table_entry.detail == str(csv_path)
        assert csv_exists

    def test_build_daily_indicators_supports_explicit_2024_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_fake_raw_tree(Path(tmp), year=2024, single_layout="direct_plus_supplement")
            builder = RawInputBuilder(
                raw_root=raw_root,
                aurora_out=Path(tmp) / "aurora",
                indicator_nc_out=Path(tmp) / "nc",
                indicator_parquet_out=Path(tmp) / "pq",
                single_dir=raw_root / "era5_single_levels_2024",
                pressure_dir=raw_root / "era5_pressure_levels_2024",
            )
            try:
                ds = builder.build_daily_indicators(date(2024, 1, 1))
            finally:
                builder.close()

        assert "flash_flood_risk" in ds.data_vars
        assert str(ds["time"].values[0]).startswith("2024-01-01")

    def test_build_daily_indicators_rebuilds_missing_nis_netcdf_from_raw_tiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_root = _build_fake_raw_tree(root, year=2024, single_layout="direct_plus_supplement")
            nis_path = root / "data" / "output" / "nis" / "nis_elevation_grid.nc"
            nis_path.unlink()
            _write_fake_srtm_tile(raw_root / "nis" / "N16E034.SRTMGL1.hgt.zip")
            builder = RawInputBuilder(
                raw_root=raw_root,
                aurora_out=root / "aurora",
                indicator_nc_out=root / "nc",
                indicator_parquet_out=root / "pq",
                single_dir=raw_root / "era5_single_levels_2024",
                pressure_dir=raw_root / "era5_pressure_levels_2024",
            )
            try:
                ds = builder.build_daily_indicators(date(2024, 1, 1))
            finally:
                builder.close()
            nis_exists = nis_path.exists()

        assert "flash_flood_risk" in ds.data_vars
        assert nis_exists

    def test_build_daily_indicators_supports_year_specific_directory_mappings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_root = _build_fake_raw_tree(root, year=2022)
            _build_fake_raw_tree(root, year=2024, single_layout="direct_plus_supplement")

            single_2022 = raw_root / "era5_single_levels_2022"
            pressure_2022 = raw_root / "era5_pressure_levels_2022"
            single_2022_6h = raw_root / "era5_single_levels_2022_6h"
            pressure_2022_6h = raw_root / "era5_pressure_levels_2022_6h"
            for supplement in single_2022.glob("*_supplement.nc"):
                supplement.rename(supplement.with_name(supplement.name.replace("_supplement.nc", "_supplement_backfill.nc")))
            single_2022.rename(single_2022_6h)
            pressure_2022.rename(pressure_2022_6h)

            builder = RawInputBuilder(
                raw_root=raw_root,
                aurora_out=root / "aurora",
                indicator_nc_out=root / "nc",
                indicator_parquet_out=root / "pq",
                single_dirs_by_year={
                    2022: single_2022_6h,
                    2024: raw_root / "era5_single_levels_2024",
                },
                pressure_dirs_by_year={
                    2022: pressure_2022_6h,
                    2024: raw_root / "era5_pressure_levels_2024",
                },
            )
            try:
                ds_2022 = builder.build_daily_indicators(date(2022, 1, 1))
                ds_2024 = builder.build_daily_indicators(date(2024, 1, 1))
            finally:
                builder.close()

        assert str(ds_2022["time"].values[0]).startswith("2022-01-01")
        assert str(ds_2024["time"].values[0]).startswith("2024-01-01")
        assert "flash_flood_risk" in ds_2022.data_vars
        assert "flash_flood_risk" in ds_2024.data_vars

    def test_build_daily_indicators_merges_supplemental_pressure_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_fake_raw_tree(
                Path(tmp),
                supplemental_pressure_levels={"temperature": (300, 200)},
            )
            builder = RawInputBuilder(
                raw_root=raw_root,
                aurora_out=Path(tmp) / "aurora",
                indicator_nc_out=Path(tmp) / "nc",
                indicator_parquet_out=Path(tmp) / "pq",
                missing_pressure_dir=raw_root / "era5_pressure_levels_2025_missing",
            )
            try:
                ds = builder.build_daily_indicators(date(2025, 1, 1))
            finally:
                builder.close()

            assert "pwat" in ds.data_vars
            aurora = RawInputBuilder(
                raw_root=raw_root,
                aurora_out=Path(tmp) / "aurora",
                indicator_nc_out=Path(tmp) / "nc",
                indicator_parquet_out=Path(tmp) / "pq",
                missing_pressure_dir=raw_root / "era5_pressure_levels_2025_missing",
            )
            try:
                aurora_ds = aurora.build_aurora_input(datetime(2025, 1, 1, 6, tzinfo=timezone.utc))
            finally:
                aurora.close()

            assert list(aurora_ds["level"].values) == list(PRESSURE_LEVELS)
