from __future__ import annotations

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


def _build_fake_raw_tree(root: Path) -> Path:
    raw_root = root / "data" / "raw"
    single_dir = raw_root / "era5_single_levels_2025"
    pressure_dir = raw_root / "era5_pressure_levels_2025"
    precip_dir = raw_root / "precip"
    dust_dir = raw_root / "dust"
    sst_dir = raw_root / "sst"
    nis_dir = root / "data" / "output" / "nis"

    times = np.array(
        [
            "2025-01-01T00:00:00",
            "2025-01-01T06:00:00",
            "2025-01-01T12:00:00",
            "2025-01-01T18:00:00",
            "2025-01-02T00:00:00",
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
    zip_path = single_dir / "era5_single_levels_2025_01.nc"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(tmp_instant, "data_stream-oper_stepType-instant.nc")
        archive.write(tmp_accum, "data_stream-oper_stepType-accum.nc")
        archive.write(tmp_max, "data_stream-oper_stepType-max.nc")

    aurora_single = xr.Dataset(
        {
            "lsm": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 1.0, dtype=np.float32)),
            "msl": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 101000.0, dtype=np.float32), {"units": "Pa"}),
            "slt": (("valid_time", "latitude", "longitude"), np.full(shape_3d, 2.0, dtype=np.float32)),
        },
        coords={"valid_time": times, "latitude": lats, "longitude": lons},
    )
    _write_dataset(single_dir / "era5_single_levels_2025_01_aurora.nc", aurora_single)
    _write_dataset(single_dir / "era5_single_levels_2025_01_static.nc", aurora_single[["lsm", "slt"]].isel(valid_time=slice(0, 1)))

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
        _write_dataset(pressure_dir / f"era5_pl_2025_01_{suffix}.nc", ds)

    gpm = xr.Dataset(
        {"precipitation": (("time", "lat", "lon"), np.full((1, len(lats), len(lons)), 12.0, dtype=np.float32), {"units": "mm/day"})},
        coords={"time": np.array(["2025-01-01"], dtype="datetime64[ns]"), "lat": lats[::-1], "lon": lons},
    )
    _write_dataset(precip_dir / "GPM_3IMERGDF_20250101.nc4", gpm)

    dust = xr.Dataset(
        {
            "DUEXTTAU": (("time", "lat", "lon"), np.full((2, len(lats), len(lons)), 0.4, dtype=np.float32), {"units": "1"}),
            "DUCMASS": (("time", "lat", "lon"), np.full((2, len(lats), len(lons)), 0.2, dtype=np.float32), {"units": "kg m-2"}),
            "DUSMASS": (("time", "lat", "lon"), np.full((2, len(lats), len(lons)), 0.001, dtype=np.float32), {"units": "kg m-3"}),
        },
        coords={"time": np.array(["2025-01-01T00:00:00", "2025-01-01T12:00:00"], dtype="datetime64[ns]"), "lat": lats, "lon": lons},
    )
    _write_dataset(dust_dir / "MERRA2_20250101.nc4", dust)

    sst = xr.Dataset(
        {"sst": (("time", "lat", "lon"), np.full((1, len(lats), len(lons)), 25.0, dtype=np.float32), {"units": "degC"})},
        coords={"time": np.array(["2025-01-01"], dtype="datetime64[ns]"), "lat": lats, "lon": lons},
    )
    _write_dataset(sst_dir / "oisst.day.mean.2025.nc", sst)

    chirps = xr.Dataset(
        {"precip": (("time", "latitude", "longitude"), np.full((1, len(lats), len(lons)), 31.0, dtype=np.float32), {"units": "mm/month"})},
        coords={"time": np.array(["2025-01-01"], dtype="datetime64[ns]"), "latitude": lats, "longitude": lons},
    )
    _write_dataset(precip_dir / "chirps-v3.0.2025.monthly.nc", chirps)

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
        assert not np.isinf(ds["convective_precip_ratio"].values).any()
        assert not np.isinf(ds["gpm_era5_precip_ratio"].values).any()

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
