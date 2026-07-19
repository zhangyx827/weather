"""Build province-day dust-storm feature tables from daily NetCDF inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

try:
    import xarray as xr
except Exception:  # pragma: no cover - optional dependency
    xr = None

from mazu_saudi.config import DustStormLabelMappingConfig
from mazu_saudi.data.flash_flood_province_features import enrich_flash_flood_features_with_province, province_day_numeric_feature_columns
from mazu_saudi.data.flash_flood_province_lookup import build_flash_flood_province_lookup
from mazu_saudi.data.io import read_netcdf_dataset


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGION_BOUNDARY_PATH = ROOT / "data" / "raw" / "admin_boundaries" / "geoBoundaries-SAU-ADM1.geojson"


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for dust-storm province feature assembly")


def _require_xarray() -> None:
    if xr is None:
        raise RuntimeError("xarray is required for dust-storm province feature assembly")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip().lower()


def _normalize_location_token(value: Any) -> str:
    token = _normalize_text(value)
    token = token.strip("\"'`")
    token = token.replace("-", " ").replace("_", " ")
    token = " ".join(token.split())
    return token


def _date_from_path(path: Path) -> str:
    token = path.stem.split("_")[-1]
    parsed = pd.to_datetime(token, format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Could not infer date from dust input file name: {path}")
    return parsed.strftime("%Y-%m-%d")


def _is_readable_netcdf(path: Path) -> bool:
    try:
        dataset = read_netcdf_dataset(path)
    except Exception:
        return False
    try:
        return bool(getattr(dataset, "data_vars", None))
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()


def _source_metadata_json(dataset: Any) -> str:
    payload = getattr(dataset, "attrs", {}).get("source_metadata_json", {})
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _degradation_metadata_json(dataset: Any) -> str:
    payload = getattr(dataset, "attrs", {}).get("degradation_metadata", {})
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _source_status(dataset: Any) -> str:
    attrs = getattr(dataset, "attrs", {})
    metadata = attrs.get("source_metadata_json")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}
    if isinstance(metadata, dict) and _normalize_text(metadata.get("source_status")):
        return _normalize_text(metadata.get("source_status"))
    degradation = attrs.get("degradation_metadata", {})
    if isinstance(degradation, str):
        try:
            degradation = json.loads(degradation)
        except Exception:
            degradation = {}
    if isinstance(degradation, dict) and degradation:
        return "degraded"
    return "normal"


def _load_boundary_table(path: Path) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for feature in payload.get("features", []):
        properties = dict(feature.get("properties", {}))
        properties["geometry"] = feature.get("geometry")
        rows.append(properties)
    if not rows:
        raise ValueError(f"boundary file contains no features: {path}")
    return pd.DataFrame(rows)


def _daily_grid_frame(path: Path) -> Any:
    _require_xarray()
    dataset = read_netcdf_dataset(path)
    try:
        normalized = dataset
        rename: dict[str, str] = {}
        if "lat" in normalized.dims and "lon" in normalized.dims and "latitude" not in normalized.dims and "longitude" not in normalized.dims:
            rename["lat"] = "longitude"
            rename["lon"] = "latitude"
        elif "lat" in normalized.dims and "latitude" not in normalized.dims:
            rename["lat"] = "latitude"
        elif "lon" in normalized.dims and "longitude" not in normalized.dims:
            rename["lon"] = "longitude"
        if rename:
            normalized = normalized.rename(rename)
        if "latitude" not in normalized.coords or "longitude" not in normalized.coords:
            raise ValueError(f"dust input file lacks latitude/longitude coordinates: {path}")
        if "time" in normalized.dims and normalized.sizes.get("time", 0):
            normalized = normalized.max(dim="time", skipna=True)

        frame = normalized.to_dataframe().reset_index()
        if "time" in frame.columns:
            frame = frame.drop(columns=["time"])
        frame["date"] = _date_from_path(path)
        frame["hazard_type"] = "dust_storm"
        frame["source_file"] = path.name
        frame["source"] = str(path)
        frame["source_status"] = _source_status(dataset)
        frame["source_metadata"] = _source_metadata_json(dataset)
        frame["degradation_metadata"] = _degradation_metadata_json(dataset)
        frame["grounding_gap"] = json.dumps({}, ensure_ascii=False, sort_keys=True)
        frame["grid_id"] = frame.apply(
            lambda row: f"saudi_{float(row['latitude']):.4f}_{float(row['longitude']):.4f}",
            axis=1,
        )
        return frame.reset_index(drop=True)
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()


def _canonical_region_id(value: Any) -> str:
    config = DustStormLabelMappingConfig()
    token = _normalize_location_token(value)
    mapped = config.location_aliases.get(token)
    if not mapped:
        region_ids = config.location_to_region_ids.get(token)
        mapped = region_ids[0] if region_ids else token
    mapped = mapped.replace(" ", "_")
    return "" if mapped in {"", "unknown"} else mapped


def build_dust_storm_province_day_feature_table(
    feature_paths: list[Path] | tuple[Path, ...] | Any,
    *,
    boundary_path: Path = DEFAULT_REGION_BOUNDARY_PATH,
    province_column: str = "province_name",
    coordinate_precision: int = 4,
):
    """Aggregate dust raw-grid files into province-day samples."""

    _require_pandas()
    paths = [Path(path) for path in feature_paths]
    dated_paths: list[Path] = []
    skipped_paths = 0
    for path in paths:
        try:
            _date_from_path(path)
            if not _is_readable_netcdf(path):
                raise ValueError(f"dust input file is not a readable NetCDF dataset: {path}")
        except ValueError:
            skipped_paths += 1
            continue
        except Exception:
            skipped_paths += 1
            continue
        dated_paths.append(path)
    if not dated_paths:
        raise FileNotFoundError("No dust-storm NetCDF files were provided")

    boundary_table = _load_boundary_table(boundary_path)

    reference_frame = _daily_grid_frame(dated_paths[0])
    province_lookup = build_flash_flood_province_lookup(
        reference_frame,
        boundary_table,
        province_column=province_column,
        boundary_province_column="shapeName",
        boundary_id_column="shapeID",
        geometry_column="geometry",
        geometry_format="geojson",
    )

    records: list[dict[str, Any]] = []
    for path in dated_paths:
        frame = _daily_grid_frame(path)
        enriched = enrich_flash_flood_features_with_province(
            frame,
            province_lookup,
            coordinate_precision=coordinate_precision,
            province_column=province_column,
            lookup_province_column=province_column,
        )
        enriched = enriched[enriched[province_column].astype(str).str.strip().ne("")].copy()
        if enriched.empty:
            continue
        enriched[province_column] = enriched[province_column].astype(str).str.strip().str.lower()
        enriched["region_id"] = enriched[province_column].map(_canonical_region_id)
        numeric_columns = province_day_numeric_feature_columns(enriched)
        if not numeric_columns:
            continue
        grouped = enriched.groupby(["date", province_column, "region_id"], dropna=False)
        aggregated = grouped[numeric_columns].mean().reset_index()
        counts = grouped.size().reset_index(name="grid_cell_count")
        aggregated = aggregated.merge(counts, on=["date", province_column, "region_id"], how="left", validate="1:1")
        if "source_status" in enriched.columns:
            degraded = (
                enriched.assign(_is_degraded=enriched["source_status"].astype(str).str.lower().eq("degraded").astype(int))
                .groupby(["date", province_column, "region_id"], dropna=False)["_is_degraded"]
                .sum()
                .reset_index(name="degraded_grid_cell_count")
            )
            aggregated = aggregated.merge(degraded, on=["date", province_column, "region_id"], how="left", validate="1:1")
        aggregated["hazard_type"] = "dust_storm"
        aggregated["sample_unit"] = "region_day"
        records.extend(aggregated.to_dict(orient="records"))

    if not records:
        raise ValueError("No dust-storm province-day rows could be built from the provided inputs")

    province_day = pd.DataFrame.from_records(records)
    province_day.attrs["skipped_paths"] = skipped_paths
    first_columns = ["date", "hazard_type", "sample_unit", province_column, "region_id", "grid_cell_count"]
    if "degraded_grid_cell_count" in province_day.columns:
        first_columns.append("degraded_grid_cell_count")
    ordered_columns = first_columns + [column for column in province_day.columns if column not in first_columns]
    return province_day.loc[:, ordered_columns].reset_index(drop=True)
