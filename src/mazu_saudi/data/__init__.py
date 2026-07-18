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
from .agriculture_supervised_training import (
    aggregate_dry_heat_agriculture_features,
    build_dry_heat_agriculture_supervised_training_dataset,
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
from .dust_storm_event_sources import (
    DustStormEvent,
    dust_storm_event_table,
    dust_storm_event_records,
    expand_dust_storm_events_to_daily_records,
    expand_dust_storm_events_to_daily_table,
    standardize_dust_storm_event_records,
)
from .dust_storm_mapping import build_dust_storm_training_labels
from .dust_storm_training_dataset import build_dust_storm_supervised_training_dataset
from .flash_flood_mapping import build_flash_flood_training_labels
from .flash_flood_province_features import (
    aggregate_flash_flood_features_to_province_day,
    enrich_flash_flood_features_with_province,
    province_day_numeric_feature_columns,
)
from .flash_flood_province_audit import audit_flash_flood_province_lookup
from .flash_flood_audit import (
    count_flash_flood_boundary_grounded_positive_rows,
    count_flash_flood_explicit_geometry_positive_rows,
    count_flash_flood_geometry_backed_positive_rows,
    summarize_flash_flood_geometry_backed_positive_rows,
)
from .flash_flood_label_audit import audit_flash_flood_province_day_labels
from .flash_flood_province_lookup import build_flash_flood_province_lookup
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
    "aggregate_dry_heat_agriculture_features",
    "build_dry_heat_agriculture_supervised_training_dataset",
    "FlashFloodEvent",
    "expand_flash_flood_events_to_daily_records",
    "expand_flash_flood_events_to_daily_table",
    "flash_flood_event_records",
    "flash_flood_event_table",
    "flash_flood_event_table_from_sources",
    "merge_flash_flood_event_sources",
    "seed_flash_flood_events",
    "standardize_flash_flood_event_records",
    "DustStormEvent",
    "dust_storm_event_table",
    "dust_storm_event_records",
    "expand_dust_storm_events_to_daily_records",
    "expand_dust_storm_events_to_daily_table",
    "standardize_dust_storm_event_records",
    "build_dust_storm_training_labels",
    "build_dust_storm_supervised_training_dataset",
    "build_flash_flood_training_labels",
    "enrich_flash_flood_features_with_province",
    "aggregate_flash_flood_features_to_province_day",
    "province_day_numeric_feature_columns",
    "audit_flash_flood_province_lookup",
    "audit_flash_flood_province_day_labels",
    "count_flash_flood_boundary_grounded_positive_rows",
    "count_flash_flood_explicit_geometry_positive_rows",
    "count_flash_flood_geometry_backed_positive_rows",
    "summarize_flash_flood_geometry_backed_positive_rows",
    "build_flash_flood_province_lookup",
    "build_flash_flood_supervised_training_dataset",
    "SRTMElevationIndex",
    "discover_srtm_tiles",
    "enrich_features_with_elevation",
]
