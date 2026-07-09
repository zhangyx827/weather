# Data Access Probe

The real-data probe only works on the CDS-backed ERA5 download at `data/raw/era5_saudi_20250616.nc`.
It does not generate substitute NetCDF data. If the download or read step fails, the script exits with an error.

## What to check

1. Confirm that `~/.cdsapirc` exists and the CDS credentials are active.
2. Verify access to the Copernicus Climate Data Store.
3. Re-run `python examples/real_data_probe.py` after fixing credentials or network.
