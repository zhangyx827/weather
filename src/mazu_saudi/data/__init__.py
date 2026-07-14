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
from .indicator_features import (
    highest_indicator_point_from_dataset,
    indicator_point_from_dataset,
    indicator_point_from_netcdf,
    read_indicator_dataset,
)
from .flash_flood_labels import (
    FlashFloodEvent,
    expand_flash_flood_events_to_daily_records,
    expand_flash_flood_events_to_daily_table,
    flash_flood_event_records,
    flash_flood_event_table,
    seed_flash_flood_events,
)
from .flash_flood_event_sources import (
    flash_flood_event_table_from_sources,
    merge_flash_flood_event_sources,
    standardize_flash_flood_event_records,
)
from .flash_flood_mapping import build_flash_flood_training_labels
from .flash_flood_training_dataset import build_flash_flood_supervised_training_dataset
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
    "highest_indicator_point_from_dataset",
    "indicator_point_from_dataset",
    "indicator_point_from_netcdf",
    "read_indicator_dataset",
    "FlashFloodEvent",
    "expand_flash_flood_events_to_daily_records",
    "expand_flash_flood_events_to_daily_table",
    "flash_flood_event_records",
    "flash_flood_event_table",
    "flash_flood_event_table_from_sources",
    "merge_flash_flood_event_sources",
    "seed_flash_flood_events",
    "standardize_flash_flood_event_records",
    "build_flash_flood_training_labels",
    "build_flash_flood_supervised_training_dataset",
    "SRTMElevationIndex",
    "discover_srtm_tiles",
    "enrich_features_with_elevation",
]
