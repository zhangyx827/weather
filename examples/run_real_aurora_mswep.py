"""Run the official Microsoft Aurora model with ERA5 input and MSWEP context.

This script is intentionally strict: Aurora inputs must come from real ERA5
fields required by the official ``microsoft-aurora`` package. MSWEP is merged
only as an observed/background precipitation field for downstream validation
and risk processing; it is not fed into Aurora.
"""

from __future__ import annotations

import argparse
import calendar
import dataclasses
import sys
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
import xarray as xr

from aurora import AuroraPretrained, AuroraSmallPretrained, Batch, Metadata, rollout

from mazu_saudi.data import write_netcdf_dataset


SAUDI_BBOX = (16.0, 34.0, 32.0, 56.0)
PRESSURE_LEVELS = (925, 850, 700, 500, 300, 200)
HISTORY_HOURS = 6

SURF_VAR_ALIASES = {
    "2t": ("t2m", "2t"),
    "10u": ("u10", "10u"),
    "10v": ("v10", "10v"),
    "msl": ("msl",),
}
STATIC_VAR_ALIASES = {
    "lsm": ("lsm",),
    "z": ("z",),
    "slt": ("slt",),
}
ATMOS_VAR_ALIASES = {
    "z": ("z",),
    "u": ("u",),
    "v": ("v",),
    "t": ("t",),
    "q": ("q",),
}

CDS_SINGLE_NAMES = {
    "msl": "mean_sea_level_pressure",
    "lsm": "land_sea_mask",
    "slt": "soil_type",
}
CDS_PRESSURE_NAMES = {
    "z": "geopotential",
    "t": "temperature",
    "q": "specific_humidity",
}


def parse_time(value: str) -> datetime:
    """Parse an ISO timestamp as UTC."""

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def history_times(issue_time: datetime) -> tuple[datetime, datetime]:
    """Return the two Aurora history timestamps."""

    return (issue_time - timedelta(hours=HISTORY_HOURS), issue_time)


def month_key(value: datetime) -> tuple[int, int]:
    return value.year, value.month


def normalise_coords(ds: xr.Dataset) -> xr.Dataset:
    """Normalise common CDS coordinate names to Aurora-friendly names."""

    rename = {}
    if "valid_time" in ds.coords:
        rename["valid_time"] = "time"
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if "pressure_level" in ds.coords:
        rename["pressure_level"] = "level"
    ds = ds.rename(rename)
    if "lon" in ds.coords and float(ds["lon"].min()) < 0:
        ds = ds.assign_coords(lon=(ds["lon"] % 360)).sortby("lon")
    return ds


def open_dataset_maybe_zip(path: Path) -> xr.Dataset:
    """Open a NetCDF file or a CDS zip response with NetCDF members."""

    if zipfile.is_zipfile(path):
        loaded = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    if member.endswith("/"):
                        continue
                    extracted = Path(tmpdir) / Path(member).name
                    extracted.write_bytes(archive.read(member))
                    with xr.open_dataset(extracted) as ds:
                        loaded.append(normalise_coords(ds).load())
        return xr.merge(loaded, compat="override")

    with xr.open_dataset(path) as ds:
        return normalise_coords(ds).load()


def merge_datasets(paths: list[Path]) -> xr.Dataset:
    """Merge all readable datasets."""

    datasets = [open_dataset_maybe_zip(path) for path in paths if path.exists()]
    if not datasets:
        return xr.Dataset()
    return xr.merge(datasets, compat="override", join="outer")


def single_level_paths(root: Path, year: int, month: int) -> list[Path]:
    month_str = f"{month:02d}"
    candidates = [
        root / f"era5_single_levels_{year}_{month_str}.nc",
        root / f"era5_single_levels_{year}_{month_str}_aurora.nc",
    ]
    candidates.extend(sorted(root.glob(f"*{year}_{month_str}*single*.nc")))
    candidates.extend(sorted(root.glob(f"*{year}_{month_str}*aurora*.nc")))
    return sorted(set(candidates))


def pressure_level_path(root: Path, year: int, month: int, cds_name: str) -> Path:
    return root / f"era5_pl_{year}_{month:02d}_{cds_name}.nc"


def find_data_var(ds: xr.Dataset, aliases: tuple[str, ...]) -> str | None:
    for name in aliases:
        if name in ds.data_vars:
            return name
    return None


@dataclasses.dataclass
class MissingInputs:
    single_vars: set[str]
    static_vars: set[str]
    atmos_vars: set[str]

    def any(self) -> bool:
        return bool(self.single_vars or self.static_vars or self.atmos_vars)

    def lines(self) -> list[str]:
        lines = []
        if self.single_vars:
            lines.append(f"single-level: {', '.join(sorted(self.single_vars))}")
        if self.static_vars:
            lines.append(f"static: {', '.join(sorted(self.static_vars))}")
        if self.atmos_vars:
            lines.append(f"pressure-level: {', '.join(sorted(self.atmos_vars))}")
        return lines


def inspect_inputs(single_ds: xr.Dataset, pressure_ds: xr.Dataset) -> MissingInputs:
    """Return all missing official Aurora input variables."""

    missing_single = {
        target
        for target, aliases in SURF_VAR_ALIASES.items()
        if find_data_var(single_ds, aliases) is None
    }
    missing_static = {
        target
        for target, aliases in STATIC_VAR_ALIASES.items()
        if find_data_var(single_ds, aliases) is None
    }
    missing_atmos = {
        target
        for target, aliases in ATMOS_VAR_ALIASES.items()
        if find_data_var(pressure_ds, aliases) is None
    }
    return MissingInputs(missing_single, missing_static, missing_atmos)


def download_missing_inputs(
    single_dir: Path,
    pressure_dir: Path,
    times: tuple[datetime, datetime],
    missing: MissingInputs,
) -> None:
    """Download missing ERA5 inputs with CDS API."""

    import cdsapi

    client = cdsapi.Client()
    grouped_times: dict[tuple[int, int], list[datetime]] = defaultdict(list)
    for value in times:
        grouped_times[month_key(value)].append(value)

    for (year, month), month_times in grouped_times.items():
        if missing.single_vars or missing.static_vars:
            variables = [
                CDS_SINGLE_NAMES[name]
                for name in sorted(missing.single_vars | missing.static_vars)
                if name in CDS_SINGLE_NAMES
            ]
            if variables:
                target = single_dir / f"era5_single_levels_{year}_{month:02d}_aurora.nc"
                _cds_retrieve_single(client, target, year, month, month_times, variables)

        for var in sorted(missing.atmos_vars):
            cds_name = CDS_PRESSURE_NAMES.get(var)
            if cds_name is None:
                continue
            target = pressure_level_path(pressure_dir, year, month, cds_name)
            _cds_retrieve_pressure(client, target, year, month, month_times, cds_name)


def _cds_days_and_hours(times: list[datetime]) -> tuple[list[str], list[str]]:
    days = sorted({f"{value.day:02d}" for value in times})
    hours = sorted({f"{value.hour:02d}:00" for value in times})
    return days, hours


def _cds_retrieve_single(
    client: Any,
    target: Path,
    year: int,
    month: int,
    times: list[datetime],
    variables: list[str],
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    days, hours = _cds_days_and_hours(times)
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": variables,
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": hours,
            "area": [SAUDI_BBOX[2] + 1, SAUDI_BBOX[1], SAUDI_BBOX[0], SAUDI_BBOX[3] + 1],
        },
        str(target),
    )


def _cds_retrieve_pressure(
    client: Any,
    target: Path,
    year: int,
    month: int,
    times: list[datetime],
    variable: str,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    days, hours = _cds_days_and_hours(times)
    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "data_format": "netcdf",
            "variable": [variable],
            "pressure_level": [str(level) for level in PRESSURE_LEVELS],
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": hours,
            "area": [SAUDI_BBOX[2] + 1, SAUDI_BBOX[1], SAUDI_BBOX[0], SAUDI_BBOX[3] + 1],
        },
        str(target),
    )


def load_single_inputs(single_dir: Path, times: tuple[datetime, datetime]) -> xr.Dataset:
    paths: list[Path] = []
    for year, month in sorted({month_key(value) for value in times}):
        paths.extend(single_level_paths(single_dir, year, month))
    return merge_datasets(paths)


def load_pressure_inputs(pressure_dir: Path, times: tuple[datetime, datetime]) -> xr.Dataset:
    paths: list[Path] = []
    for year, month in sorted({month_key(value) for value in times}):
        for cds_name in (
            "u_component_of_wind",
            "v_component_of_wind",
            "geopotential",
            "temperature",
            "specific_humidity",
        ):
            paths.append(pressure_level_path(pressure_dir, year, month, cds_name))
    return merge_datasets(paths)


def select_history(da: xr.DataArray, times: tuple[datetime, datetime]) -> xr.DataArray:
    requested = np.array([np.datetime64(value.replace(tzinfo=None)) for value in times])
    selected = da.sel(time=requested)
    return selected


def align_domain(single_ds: xr.Dataset, pressure_ds: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """Crop to a patch-compatible Saudi regional grid."""

    single_ds = single_ds.sortby("lat", ascending=False).sortby("lon")
    pressure_ds = pressure_ds.sortby("lat", ascending=False).sortby("lon")
    single_ds = single_ds.sel(lat=slice(SAUDI_BBOX[2], SAUDI_BBOX[0]), lon=slice(SAUDI_BBOX[1], SAUDI_BBOX[3]))
    pressure_ds = pressure_ds.sel(lat=single_ds["lat"], lon=single_ds["lon"], method="nearest")

    h = (single_ds.sizes["lat"] // 4) * 4
    w = (single_ds.sizes["lon"] // 4) * 4
    if h < 4 or w < 4:
        raise RuntimeError("Aurora input domain is too small after crop")
    single_ds = single_ds.isel(lat=slice(0, h), lon=slice(0, w))
    pressure_ds = pressure_ds.sel(lat=single_ds["lat"], lon=single_ds["lon"], method="nearest")
    return single_ds, pressure_ds


def build_batch(single_ds: xr.Dataset, pressure_ds: xr.Dataset, times: tuple[datetime, datetime]) -> Batch:
    """Build the official Aurora Batch from ERA5 fields."""

    single_ds, pressure_ds = align_domain(single_ds, pressure_ds)
    lat = torch.tensor(single_ds["lat"].values, dtype=torch.float32)
    lon = torch.tensor(single_ds["lon"].values, dtype=torch.float32)

    surf_vars = {
        target: torch.tensor(
            select_history(single_ds[find_data_var(single_ds, aliases)], times).values[None],
            dtype=torch.float32,
        )
        for target, aliases in SURF_VAR_ALIASES.items()
    }
    static_vars = {
        target: torch.tensor(
            single_ds[find_data_var(single_ds, aliases)].isel(time=-1).values
            if "time" in single_ds[find_data_var(single_ds, aliases)].dims
            else single_ds[find_data_var(single_ds, aliases)].values,
            dtype=torch.float32,
        )
        for target, aliases in STATIC_VAR_ALIASES.items()
    }
    atmos_vars = {}
    for target, aliases in ATMOS_VAR_ALIASES.items():
        name = find_data_var(pressure_ds, aliases)
        da = select_history(pressure_ds[name], times).sel(level=list(PRESSURE_LEVELS))
        atmos_vars[target] = torch.tensor(da.values[None], dtype=torch.float32)

    return Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=Metadata(
            lat=lat,
            lon=lon,
            time=(times[-1].replace(tzinfo=None),),
            atmos_levels=PRESSURE_LEVELS,
        ),
    )


def build_batch_from_prebuilt(ds: xr.Dataset, issue_time: datetime) -> Batch:
    """Build the official Aurora Batch from a prebuilt direct-input NetCDF file."""

    ds = normalise_coords(ds)
    required = (
        "surf_2t",
        "surf_10u",
        "surf_10v",
        "surf_msl",
        "static_lsm",
        "static_z",
        "static_slt",
        "atmos_z",
        "atmos_u",
        "atmos_v",
        "atmos_t",
        "atmos_q",
    )
    missing = [name for name in required if name not in ds.data_vars]
    if missing:
        raise ValueError(f"Prebuilt Aurora input is missing variables: {', '.join(missing)}")

    lat = torch.tensor(ds["lat"].values, dtype=torch.float32)
    lon = torch.tensor(ds["lon"].values, dtype=torch.float32)
    surf_vars = {
        "2t": torch.tensor(ds["surf_2t"].values[None], dtype=torch.float32),
        "10u": torch.tensor(ds["surf_10u"].values[None], dtype=torch.float32),
        "10v": torch.tensor(ds["surf_10v"].values[None], dtype=torch.float32),
        "msl": torch.tensor(ds["surf_msl"].values[None], dtype=torch.float32),
    }
    static_vars = {
        "lsm": torch.tensor(ds["static_lsm"].values, dtype=torch.float32),
        "z": torch.tensor(ds["static_z"].values, dtype=torch.float32),
        "slt": torch.tensor(ds["static_slt"].values, dtype=torch.float32),
    }
    atmos_vars = {
        "z": torch.tensor(ds["atmos_z"].values[None], dtype=torch.float32),
        "u": torch.tensor(ds["atmos_u"].values[None], dtype=torch.float32),
        "v": torch.tensor(ds["atmos_v"].values[None], dtype=torch.float32),
        "t": torch.tensor(ds["atmos_t"].values[None], dtype=torch.float32),
        "q": torch.tensor(ds["atmos_q"].values[None], dtype=torch.float32),
    }
    return Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=Metadata(
            lat=lat,
            lon=lon,
            time=(issue_time.replace(tzinfo=None),),
            atmos_levels=PRESSURE_LEVELS,
        ),
    )


def batch_to_dataset(batch: Batch) -> xr.Dataset:
    """Convert an Aurora prediction batch to an xarray Dataset."""

    lat = batch.metadata.lat.detach().cpu().numpy()
    lon = batch.metadata.lon.detach().cpu().numpy()
    valid_time = np.datetime64(batch.metadata.time[0])
    data_vars = {}
    for name, tensor in batch.surf_vars.items():
        data_vars[f"aurora_surf_{name}"] = (
            ("time", "lat", "lon"),
            tensor.detach().cpu().numpy()[0],
        )
    for name, tensor in batch.atmos_vars.items():
        data_vars[f"aurora_atmos_{name}"] = (
            ("time", "level", "lat", "lon"),
            tensor.detach().cpu().numpy()[0],
        )
    return xr.Dataset(
        data_vars,
        coords={"time": [valid_time], "level": list(batch.metadata.atmos_levels), "lat": lat, "lon": lon},
        attrs={
            "provider": "microsoft-aurora",
            "rollout_step": batch.metadata.rollout_step,
            "note": "MSWEP, if present, is downstream precipitation context and not an Aurora input.",
        },
    )


def add_mswep_context(ds: xr.Dataset, precip_dir: Path, valid_time: datetime) -> xr.Dataset:
    """Merge same-day MSWEP precipitation onto the Aurora output grid."""

    doy = valid_time.timetuple().tm_yday
    path = precip_dir / f"{valid_time.year}{doy:03d}.nc"
    if not path.exists():
        ds.attrs["mswep_status"] = f"missing: {path}"
        return ds

    with xr.open_dataset(path) as precip_ds:
        precip = normalise_coords(precip_ds)["precipitation"].load()
    if "time" in precip.coords:
        precip = precip.sel(time=np.datetime64(valid_time.date()), method="nearest")
    precip = precip.interp(lat=ds["lat"], lon=ds["lon"], method="nearest")
    ds["mswep_precip_24h_mm"] = (("time", "lat", "lon"), precip.values[None].astype("float32"))
    ds["mswep_precip_24h_mm"].attrs["role"] = "observed/background precipitation for downstream validation; not Aurora input"
    ds.attrs["mswep_status"] = f"merged: {path}"
    return ds


def run_aurora(batch: Batch, model_size: str, steps: int, device: str) -> list[Batch]:
    """Load official Aurora weights and run rollout."""

    model = AuroraSmallPretrained() if model_size == "small" else AuroraPretrained()
    model.load_checkpoint()
    model.eval()
    model.to(device)
    with torch.inference_mode():
        return [prediction.to("cpu") for prediction in rollout(model, batch.to(device), steps=steps)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run real Microsoft Aurora with ERA5 input and MSWEP context.")
    parser.add_argument("--issue-time", default="2025-01-01T06:00:00+00:00", help="Aurora issue time. Needs issue and issue-6h ERA5 fields.")
    parser.add_argument("--steps", type=int, default=1, help="Number of 6-hour Aurora rollout steps.")
    parser.add_argument("--model-size", choices=["small", "full"], default="small", help="Official Aurora checkpoint size.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.add_argument("--single-dir", default="data/raw/era5_single_levels_2025")
    parser.add_argument("--pressure-dir", default="data/raw/era5_pressure_levels_2025")
    parser.add_argument("--precip-dir", default="data/raw/precip")
    parser.add_argument("--output-netcdf", default="data/output/real_aurora_mswep_forecast.nc")
    parser.add_argument("--aurora-input-file", help="Optional prebuilt Aurora direct-input NetCDF from scripts/build_saudi_ml_inputs.py.")
    parser.add_argument("--download-missing", action="store_true", help="Download missing ERA5 variables using CDS API.")
    parser.add_argument("--check-only", action="store_true", help="Only check required inputs; do not run Aurora.")
    args = parser.parse_args(argv)

    issue_time = parse_time(args.issue_time)
    times = history_times(issue_time)
    single_dir = Path(args.single_dir)
    pressure_dir = Path(args.pressure_dir)

    batch: Batch
    if args.aurora_input_file:
        with xr.open_dataset(args.aurora_input_file) as prebuilt:
            batch = build_batch_from_prebuilt(prebuilt.load(), issue_time)
        print(f"Loaded prebuilt Aurora input: {args.aurora_input_file}")
    else:
        single_ds = load_single_inputs(single_dir, times)
        pressure_ds = load_pressure_inputs(pressure_dir, times)
        missing = inspect_inputs(single_ds, pressure_ds)

        if missing.any():
            print("Missing required real ERA5 inputs for official Aurora:")
            for line in missing.lines():
                print(f"  - {line}")
            if not args.download_missing:
                print("Run again with --download-missing to fetch these ERA5 variables via CDS.")
                return 2
            download_missing_inputs(single_dir, pressure_dir, times, missing)
            single_ds = load_single_inputs(single_dir, times)
            pressure_ds = load_pressure_inputs(pressure_dir, times)
            missing = inspect_inputs(single_ds, pressure_ds)
            if missing.any():
                print("Inputs are still missing after download:")
                for line in missing.lines():
                    print(f"  - {line}")
                return 2

        print("All official Aurora ERA5 inputs are present.")
        batch = build_batch(single_ds, pressure_ds, times)

    if args.check_only:
        return 0

    predictions = run_aurora(batch, args.model_size, args.steps, args.device)
    datasets = []
    for prediction in predictions:
        forecast = batch_to_dataset(prediction)
        valid_time = issue_time + timedelta(hours=6 * prediction.metadata.rollout_step)
        forecast = forecast.assign_coords(time=[np.datetime64(valid_time.replace(tzinfo=None))])
        forecast = add_mswep_context(forecast, Path(args.precip_dir), valid_time)
        datasets.append(forecast)

    output = xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]
    output.attrs.update(
        {
            "issue_time": issue_time.isoformat(),
            "model_size": args.model_size,
            "aurora_checkpoint_source": "microsoft/aurora",
            "history_times": [value.isoformat() for value in times],
        }
    )
    output_path = Path(args.output_netcdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_netcdf_dataset(output_path, output)

    print(f"Saved real Aurora + MSWEP output: {output_path}")
    print(f"dims: {dict(output.sizes)}")
    print(f"variables: {', '.join(sorted(output.data_vars))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
