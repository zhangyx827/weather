from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "download_era5_saudi.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("download_era5_saudi", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pressure_variable_mapping_uses_relative_vorticity_filename():
    module = _load_module()

    assert ("vorticity", "relative_vorticity") in module.PRESSURE_VARIABLES
    target = module.pressure_target_path(Path("data/raw/era5_pressure_levels_2025"), "2025", "01", "relative_vorticity")
    assert target.as_posix().endswith("data/raw/era5_pressure_levels_2025/era5_pl_2025_01_relative_vorticity.nc")


def test_daily_times_uses_daily_resolution():
    module = _load_module()

    assert module._daily_times() == ["00:00"]


def test_iter_months_expands_selected_years_by_month():
    module = _load_module()

    months = module._iter_months(["2024"], None)
    assert months[0] == ("2024", "01")
    assert months[-1] == ("2024", "12")
    assert len(months) == 12


def test_iter_months_deduplicates_explicit_dates_within_same_month():
    module = _load_module()

    months = module._iter_months(["2024"], ["2024-11-01", "2024-11-25", "2024-12-03"])
    assert months == [("2024", "11"), ("2024", "12")]


def test_requests_use_unarchived_netcdf_format():
    module = _load_module()

    single_request = module._single_level_request("2024", "11", ["01", "02"], ["2m_temperature"])
    pressure_request = module._pressure_level_request("2024", "11", ["01", "02"], "temperature")

    assert single_request["download_format"] == "unarchived"
    assert single_request["data_format"] == "netcdf"
    assert single_request["format"] == "netcdf"
    assert pressure_request["download_format"] == "unarchived"
    assert pressure_request["data_format"] == "netcdf"
    assert pressure_request["format"] == "netcdf"


def test_rewrite_zip_as_netcdf_merges_nc_members(tmp_path: Path):
    module = _load_module()
    zip_named_nc = tmp_path / "era5_single_levels_2022_01_supplement_backfill.nc"
    member_a = tmp_path / "member_a.nc"
    member_b = tmp_path / "member_b.nc"

    xr.Dataset(data_vars={"t2m": (("time",), [300.0])}).to_netcdf(member_a)
    xr.Dataset(data_vars={"tp": (("time",), [0.1])}).to_netcdf(member_b)

    with zipfile.ZipFile(zip_named_nc, "w") as archive:
        archive.write(member_a, arcname="instant.nc")
        archive.write(member_b, arcname="accum.nc")

    module._rewrite_zip_as_netcdf(zip_named_nc)

    assert not zipfile.is_zipfile(zip_named_nc)
    ds = xr.open_dataset(zip_named_nc)
    try:
        assert "t2m" in ds.data_vars
        assert "tp" in ds.data_vars
    finally:
        ds.close()


def test_missing_single_level_variables_only_returns_missing_fields(tmp_path: Path):
    module = _load_module()
    year_dir = tmp_path / "era5_single_levels_2022_6h"
    year_dir.mkdir()

    xr.Dataset(
        data_vars={
            "t2m": (("time",), [300.0]),
            "u10": (("time",), [5.0]),
            "v10": (("time",), [2.0]),
            "msl": (("time",), [101000.0]),
        }
    ).to_netcdf(year_dir / "era5_single_levels_2022_11.nc")

    missing = module._missing_single_level_variables(tmp_path, "2022", "11")

    assert "2m_temperature" not in missing
    assert "10m_u_component_of_wind" not in missing
    assert "surface_pressure" in missing
    assert "2m_dewpoint_temperature" in missing
    assert "total_precipitation" in missing


def test_missing_pressure_variables_accepts_legacy_vorticity_filename(tmp_path: Path):
    module = _load_module()
    year_dir = tmp_path / "era5_pressure_levels_2024"
    year_dir.mkdir()

    for suffix in (
        "u_component_of_wind",
        "v_component_of_wind",
        "geopotential",
        "temperature",
        "specific_humidity",
        "vertical_velocity",
        "divergence",
        "vorticity",
    ):
        (year_dir / f"era5_pl_2024_11_{suffix}.nc").touch()

    missing = module._missing_pressure_variables(tmp_path, "2024", "11")
    missing_suffixes = {output_suffix for _, output_suffix in missing}

    assert "relative_vorticity" not in missing_suffixes


def test_pressure_target_uses_resolved_year_directory(tmp_path: Path):
    module = _load_module()
    year_dir = tmp_path / "era5_pressure_levels_2022_6h"
    year_dir.mkdir()

    resolved = module._resolve_year_dir(tmp_path, "era5_pressure_levels", "2022")
    target = module.pressure_target_path(resolved, "2022", "01", "vertical_velocity")

    assert target == year_dir / "era5_pl_2022_01_vertical_velocity.nc"
