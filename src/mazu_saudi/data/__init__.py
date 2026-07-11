"""Data access helpers and sample resource hooks."""

from .io import (
    FEATURE_UNITS,
    SAUDI_BBOX,
    STANDARD_GRID_RESOLUTION_DEG,
    check_missing_values,
    compute_daily_precipitation_statistics,
    crop_to_bbox,
    crop_to_saudi,
    derive_xarray_physical_indicators,
    generate_standard_grid,
    read_json_features,
    read_netcdf_dataset,
    read_zarr_dataset,
    validate_time_dimension,
    validate_units,
    write_json_features,
    write_netcdf_dataset,
    write_zarr_dataset,
)
from .srtm import (
    SRTMElevationIndex,
    discover_srtm_tiles,
    enrich_features_with_elevation,
)

__all__ = [
    "FEATURE_UNITS",
    "SAUDI_BBOX",
    "STANDARD_GRID_RESOLUTION_DEG",
    "check_missing_values",
    "compute_daily_precipitation_statistics",
    "crop_to_bbox",
    "crop_to_saudi",
    "derive_xarray_physical_indicators",
    "generate_standard_grid",
    "read_json_features",
    "read_netcdf_dataset",
    "read_zarr_dataset",
    "validate_time_dimension",
    "validate_units",
    "write_json_features",
    "write_netcdf_dataset",
    "write_zarr_dataset",
    "SRTMElevationIndex",
    "discover_srtm_tiles",
    "enrich_features_with_elevation",
]
