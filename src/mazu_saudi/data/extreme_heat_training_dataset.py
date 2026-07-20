"""Build extreme-heat supervised training tables from daily indicator files."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

from mazu_saudi.config import ExtremeHeatLabelMappingConfig, FlashFloodLabelMappingConfig
from mazu_saudi.data.flash_flood_province_lookup import build_flash_flood_province_lookup
from mazu_saudi.data.indicator_features import highest_indicator_point_from_dataset
from mazu_saudi.data.io import read_netcdf_dataset
from mazu_saudi.risk.layer4_features import all_feature_names_for_hazard, prepare_feature_frame


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGION_BOUNDARY_PATH = ROOT / "data" / "raw" / "admin_boundaries" / "geoBoundaries-SAU-ADM1.geojson"
EXTREME_HEAT_SAMPLE_UNITS = ("single_point_day", "region_day")

_EXTREME_HEAT_FEATURE_SOURCE_COLUMNS = {
    "temp_c": ("temp_c", "t2m_c"),
    "tmax_c": ("tmax_c",),
    "tmin_c": ("tmin_c",),
    "heat_index_c": ("heat_index_c",),
    "vpd_kpa": ("vpd_kpa",),
    "wind_speed_mps": ("wind_speed_mps", "wind10_speed"),
    "relative_humidity_percent": ("relative_humidity_percent", "rh2m", "rh_percent"),
    "sst_celsius": ("sst_celsius",),
    "t2m_anomaly_c": ("t2m_anomaly_c",),
    "tmax_anomaly_c": ("tmax_anomaly_c",),
    "heatwave_day_flag": ("heatwave_day_flag",),
    "heatwave_duration_days": ("heatwave_duration_days",),
}


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for extreme-heat supervised training dataset assembly")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip().lower()


def _normalize_date(series: Any):
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        raise ValueError("label_table contains invalid date values")
    return parsed.dt.strftime("%Y-%m-%d")


def _normalize_hazard_type(series: Any):
    return series.astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False).str.replace("-", "_", regex=False)


def _canonicalize_location_text(value: Any) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _date_token_from_path(path: Path) -> str:
    stem = path.stem
    token = stem.split("_")[-1]
    parsed = pd.to_datetime(token, format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Could not infer date from indicator file name: {path}")
    return parsed.strftime("%Y-%m-%d")


def _degradation_metadata_json(dataset: Any) -> str:
    payload = getattr(dataset, "attrs", {}).get("degradation_metadata", {})
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _source_metadata_json(dataset: Any) -> str:
    payload = getattr(dataset, "attrs", {}).get("source_metadata_json", {})
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _source_status(dataset: Any) -> str:
    metadata = getattr(dataset, "attrs", {}).get("source_metadata_json")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}
    if isinstance(metadata, dict):
        status = _normalize_text(metadata.get("source_status"))
        if status:
            return status
    degradation = getattr(dataset, "attrs", {}).get("degradation_metadata", {})
    if isinstance(degradation, str):
        try:
            degradation = json.loads(degradation)
        except Exception:
            degradation = {}
    if isinstance(degradation, dict) and degradation:
        return "degraded"
    return "normal"


def _positive_date_lookup(label_table: Any) -> dict[str, list[str]]:
    _require_pandas()
    if not isinstance(label_table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for label_table, got {type(label_table)!r}")
    required = {"start_date", "end_date"}
    missing = sorted(required - set(label_table.columns))
    if missing:
        raise KeyError(f"label_table is missing required columns: {missing}")

    working = label_table.copy()
    if "hazard_type" in working.columns:
        hazard_mask = _normalize_hazard_type(working["hazard_type"]) == "extreme_heat"
        working = working[hazard_mask].copy()
    if working.empty:
        raise ValueError("label_table has no extreme_heat rows to join")

    working["start_date"] = _normalize_date(working["start_date"])
    working["end_date"] = _normalize_date(working["end_date"])
    lookup: dict[str, list[str]] = {}
    for _, row in working.iterrows():
        date_range = pd.date_range(row["start_date"], row["end_date"], freq="D").strftime("%Y-%m-%d")
        event_id = str(row.get("event_id", "")).strip()
        record_id = str(row.get("record_id", "")).strip()
        for date_value in date_range:
            payload = lookup.setdefault(str(date_value), [])
            if event_id and event_id not in payload:
                payload.append(event_id)
            elif record_id and record_id not in payload:
                payload.append(record_id)
    return lookup


def _load_daily_row(path: Path, *, point_variable: str) -> dict[str, Any]:
    dataset = read_netcdf_dataset(path)
    try:
        point = highest_indicator_point_from_dataset(dataset, point_variable, source=str(path))
        row: dict[str, Any] = {
            "date": _date_token_from_path(path),
            "hazard_type": "extreme_heat",
            "latitude": float(point.grid.lat),
            "longitude": float(point.grid.lon),
            "grid_id": point.grid.id,
            "valid_time": point.valid_time.isoformat(),
            "source_file": path.name,
            "source_status": point.source_status,
            "source": point.source or str(path),
            "source_metadata": json.dumps(point.source_metadata, ensure_ascii=False, sort_keys=True),
            "grounding_gap": json.dumps(point.grounding_gap, ensure_ascii=False, sort_keys=True),
            "degradation_metadata": _degradation_metadata_json(dataset),
        }
        row.update(point.values)
        return row
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()


def _normalize_dataset_for_frame(dataset: Any) -> Any:
    rename: dict[str, str] = {}
    if "lat" in getattr(dataset, "dims", {}) and "latitude" not in dataset.dims:
        rename["lat"] = "latitude"
    if "lon" in getattr(dataset, "dims", {}) and "longitude" not in dataset.dims:
        rename["lon"] = "longitude"
    if rename:
        dataset = dataset.rename(rename)
    if "latitude" not in dataset.coords or "longitude" not in dataset.coords:
        raise ValueError("indicator dataset must expose latitude and longitude coordinates")
    return dataset


def _dataset_frame(path: Path) -> Any:
    dataset = read_netcdf_dataset(path)
    try:
        normalized = _normalize_dataset_for_frame(dataset)
        reduced_vars: dict[str, Any] = {}
        for name, data_array in normalized.data_vars.items():
            selected = data_array
            for dim in list(getattr(selected, "dims", ())):
                if dim in {"latitude", "longitude"}:
                    continue
                selected = selected.isel({dim: 0}, drop=True)
            if not set(getattr(selected, "dims", ())).issubset({"latitude", "longitude"}):
                continue
            reduced_vars[name] = selected
        if not reduced_vars:
            raise ValueError(f"indicator dataset contains no grid-like data variables: {path}")
        frame = normalized[list(reduced_vars)].to_dataframe().reset_index()
        if "time" in frame.columns:
            frame = frame.drop(columns=["time"])
        frame["date"] = _date_token_from_path(path)
        frame["hazard_type"] = "extreme_heat"
        frame["source_file"] = path.name
        frame["source"] = str(path)
        frame["source_status"] = _source_status(normalized)
        frame["source_metadata"] = _source_metadata_json(normalized)
        frame["degradation_metadata"] = _degradation_metadata_json(normalized)
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


def _canonical_region_id(value: Any) -> str:
    base_aliases = FlashFloodLabelMappingConfig().location_to_province
    token = _normalize_text(value).replace("_", " ").replace("-", " ")
    token = re.sub(r"\([^)]*\)", " ", token)
    token = " ".join(token.split())
    mapped = base_aliases.get(token, token)
    mapped = mapped.replace(" ", "_")
    return "" if mapped in {"", "unknown"} else mapped


def _coordinate_join_key(latitude: Any, longitude: Any, *, precision: int = 4) -> str:
    try:
        lat_value = round(float(latitude), precision)
        lon_value = round(float(longitude), precision)
    except Exception:
        return ""
    return f"{lat_value:.{precision}f}|{lon_value:.{precision}f}"


def _location_candidates(value: Any) -> list[str]:
    text = _normalize_text(value)
    if not text:
        return []
    if text in {"multiple cities", "saudi arabia (nationwide)", "saudi arabia"}:
        return [text]
    working = text.replace("/", ",")
    working = re.sub(r"\band\b", ",", working)
    working = working.replace(" - ", ",")
    pieces = [piece.strip() for piece in working.split(",") if piece.strip()]
    candidates: list[str] = [text]
    for piece in pieces:
        candidates.append(piece)
        candidates.append(re.sub(r"\([^)]*\)", " ", piece).strip())
        if "-" in piece and piece not in {"al-ahsa", "al-kharj", "al-majmaah"}:
            candidates.extend(sub.strip() for sub in piece.split("-") if sub.strip())
    return [candidate for candidate in candidates if candidate]


def _resolved_region_ids(location_name: Any) -> list[str]:
    canonical_text = _canonicalize_location_text(location_name)
    if not canonical_text:
        return []

    direct_match = ExtremeHeatLabelMappingConfig().location_to_region_ids.get(canonical_text)
    if direct_match is not None:
        return list(direct_match)

    base_aliases = FlashFloodLabelMappingConfig().location_to_province
    alias_patterns: list[tuple[str, str]] = []
    seen_aliases: set[str] = set()
    for alias, province in base_aliases.items():
        canonical_alias = _canonicalize_location_text(alias)
        if not canonical_alias or canonical_alias in seen_aliases:
            continue
        seen_aliases.add(canonical_alias)
        alias_patterns.append((canonical_alias, _canonical_region_id(province)))
    for alias, region_ids in ExtremeHeatLabelMappingConfig().location_to_region_ids.items():
        if len(region_ids) != 1:
            continue
        if alias in seen_aliases:
            continue
        seen_aliases.add(alias)
        alias_patterns.append((alias, region_ids[0]))

    alias_patterns.sort(key=lambda item: len(item[0]), reverse=True)
    padded_value = f" {canonical_text} "
    regions: list[str] = []
    seen_regions: set[str] = set()
    for alias, region_id in alias_patterns:
        if f" {alias} " not in padded_value:
            continue
        if region_id in seen_regions:
            continue
        seen_regions.add(region_id)
        regions.append(region_id)
    return regions


def _positive_region_lookup(label_table: Any) -> dict[str, dict[str, Any]]:
    _require_pandas()
    if not isinstance(label_table, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame for label_table, got {type(label_table)!r}")
    required = {"start_date", "end_date"}
    missing = sorted(required - set(label_table.columns))
    if missing:
        raise KeyError(f"label_table is missing required columns: {missing}")

    working = label_table.copy()
    if "hazard_type" in working.columns:
        working = working[_normalize_hazard_type(working["hazard_type"]) == "extreme_heat"].copy()
    if working.empty:
        raise ValueError("label_table has no extreme_heat rows to join")

    working["start_date"] = _normalize_date(working["start_date"])
    working["end_date"] = _normalize_date(working["end_date"])
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in working.iterrows():
        date_range = pd.date_range(row["start_date"], row["end_date"], freq="D").strftime("%Y-%m-%d")
        event_id = str(row.get("event_id", "")).strip() or str(row.get("record_id", "")).strip() or "unknown_event"
        region_ids = _resolved_region_ids(row.get("location_name"))
        nationwide = "saudi_arabia" in region_ids
        scoped_region_ids = [region_id for region_id in region_ids if region_id != "saudi_arabia"]
        for date_value in date_range:
            payload = lookup.setdefault(
                str(date_value),
                {
                    "all_regions": False,
                    "event_ids": [],
                    "region_event_ids": {},
                    "unresolved_event_ids": [],
                },
            )
            if event_id not in payload["event_ids"]:
                payload["event_ids"].append(event_id)
            if nationwide:
                payload["all_regions"] = True
            if scoped_region_ids:
                for region_id in scoped_region_ids:
                    region_payload = payload["region_event_ids"].setdefault(region_id, [])
                    if event_id not in region_payload:
                        region_payload.append(event_id)
            if not nationwide and not scoped_region_ids and event_id not in payload["unresolved_event_ids"]:
                payload["unresolved_event_ids"].append(event_id)
    return lookup


def _aggregate_region_day_group(group, *, point_variable: str, top_k: int) -> dict[str, Any]:
    selector = pd.to_numeric(group[point_variable], errors="coerce") if point_variable in group.columns else pd.Series(np.nan, index=group.index)
    ranked = group.assign(_selector=selector).sort_values("_selector", ascending=False, na_position="last")
    valid_ranked = ranked[ranked["_selector"].notna()].copy()
    pooled = valid_ranked.head(max(1, top_k)) if not valid_ranked.empty else ranked.head(max(1, top_k))

    row: dict[str, Any] = {
        "date": str(group["date"].iloc[0]),
        "hazard_type": "extreme_heat",
        "region_id": str(group["region_id"].iloc[0]),
        "sample_unit": "region_day",
        "aggregation_mode": f"topk_mean:{point_variable}",
        "grid_cell_count": int(len(group)),
        "pooled_grid_cell_count": int(len(pooled)),
        "degraded_grid_cell_count": int(group["source_status"].astype(str).str.lower().eq("degraded").sum())
        if "source_status" in group.columns
        else 0,
        "source_status": "degraded" if "source_status" in group.columns and group["source_status"].astype(str).str.lower().eq("degraded").any() else "normal",
        "source_file": str(group["source_file"].iloc[0]) if "source_file" in group.columns else "",
    }
    if "grid_id" in pooled.columns:
        row["representative_grid_ids"] = ",".join(sorted({str(value) for value in pooled["grid_id"].astype(str).tolist() if str(value).strip()}))
    for feature_name in all_feature_names_for_hazard("extreme_heat"):
        source_column = next((name for name in _EXTREME_HEAT_FEATURE_SOURCE_COLUMNS.get(feature_name, (feature_name,)) if name in group.columns), None)
        if source_column is None:
            continue
        pooled_series = pd.to_numeric(pooled[source_column], errors="coerce").dropna()
        full_series = pd.to_numeric(group[source_column], errors="coerce").dropna()
        if not pooled_series.empty:
            if feature_name == "heatwave_duration_days":
                row[feature_name] = float(pooled_series.max())
            else:
                row[feature_name] = float(pooled_series.mean())
        if full_series.empty:
            continue
        row[f"{feature_name}_max"] = float(full_series.max())
        row[f"{feature_name}_p90"] = float(full_series.quantile(0.9))
        if feature_name == "relative_humidity_percent":
            row[f"{feature_name}_min"] = float(full_series.min())
        if feature_name == "heatwave_day_flag":
            row["heatwave_grid_fraction"] = float(full_series.mean())
        if feature_name == "heatwave_duration_days":
            row["heatwave_duration_days_max"] = float(full_series.max())
    return row


def _assign_region_labels(aggregated, *, label_lookup: dict[str, dict[str, Any]]) -> Any:
    rows: list[dict[str, Any]] = []
    for record in aggregated.to_dict(orient="records"):
        date_value = str(record["date"])
        region_id = str(record["region_id"])
        payload = label_lookup.get(date_value)
        matched_event_ids: list[str] = []
        if payload is None:
            label_status = "negative"
            label = 0.0
            label_source_mode = "no_event_day"
        elif payload.get("all_regions"):
            matched_event_ids = sorted(payload.get("event_ids", []))
            label_status = "positive"
            label = 1.0
            label_source_mode = "nationwide_event_day"
        elif region_id in payload.get("region_event_ids", {}):
            matched_event_ids = sorted(payload["region_event_ids"].get(region_id, []))
            label_status = "positive"
            label = 1.0
            label_source_mode = "region_day_overlap"
        elif payload.get("region_event_ids"):
            label_status = "negative"
            label = 0.0
            label_source_mode = "outside_event_region"
        else:
            label_status = "uncertain"
            label = np.nan
            label_source_mode = "event_day_unresolved"
        record["label"] = label
        record["label_status"] = label_status
        record["label_source_mode"] = label_source_mode
        record["matched_event_ids"] = ",".join(matched_event_ids)
        record["label_provenance"] = json.dumps(
            {
                "date": date_value,
                "sample_region_id": region_id,
                "matched_event_ids": matched_event_ids,
                "day_event_ids": [] if payload is None else sorted(payload.get("event_ids", [])),
                "day_region_ids": [] if payload is None else sorted(payload.get("region_event_ids", {}).keys()),
                "all_regions": False if payload is None else bool(payload.get("all_regions")),
                "unresolved_event_ids": [] if payload is None else sorted(payload.get("unresolved_event_ids", [])),
                "label_source": "verified_extreme_heat.csv",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _sample_negative_region_rows(table, *, negative_sample_size: int | None, seed: int):
    positives = table[table["label_status"] == "positive"].copy()
    negatives = table[table["label_status"] == "negative"].copy()
    if negative_sample_size is None:
        negative_sample_size = len(positives)
    if negative_sample_size < 0:
        raise ValueError("negative_sample_size must be non-negative")
    if negatives.empty or negative_sample_size >= len(negatives):
        return pd.concat([positives, negatives], ignore_index=True)

    informative = negatives[negatives["label_source_mode"] == "outside_event_region"].copy()
    background = negatives[negatives["label_source_mode"] != "outside_event_region"].copy()
    if len(informative) >= negative_sample_size:
        sampled = informative.sample(n=negative_sample_size, random_state=seed, replace=False)
    else:
        remaining = negative_sample_size - len(informative)
        sampled = informative
        if remaining > 0 and not background.empty:
            sampled = pd.concat(
                [sampled, background.sample(n=min(remaining, len(background)), random_state=seed, replace=False)],
                ignore_index=True,
            )
    return pd.concat([positives, sampled], ignore_index=True)


def _build_single_point_day_dataset(
    feature_paths: Iterable[Path],
    label_table: Any,
    *,
    point_variable: str,
    negative_sample_size: int | None,
    seed: int,
):
    positive_lookup = _positive_date_lookup(label_table)
    feature_map = {_date_token_from_path(Path(path)): Path(path) for path in feature_paths}
    if not feature_map:
        raise FileNotFoundError("No daily indicator files were provided")

    positive_dates = sorted(date for date in positive_lookup if date in feature_map)
    if not positive_dates:
        raise ValueError("No positive extreme_heat dates matched the available indicator files")

    negative_pool = sorted(date for date in feature_map if date not in positive_lookup)
    if negative_sample_size is None:
        negative_sample_size = len(positive_dates)
    if negative_sample_size < 0:
        raise ValueError("negative_sample_size must be non-negative")
    if negative_pool:
        rng = np.random.default_rng(seed)
        selected_negative_dates = sorted(rng.choice(negative_pool, size=min(negative_sample_size, len(negative_pool)), replace=False).tolist())
    else:
        selected_negative_dates = []

    selected_dates = positive_dates + selected_negative_dates
    rows: list[dict[str, Any]] = []
    for date_value in selected_dates:
        path = feature_map[date_value]
        row = _load_daily_row(path, point_variable=point_variable)
        matched_event_ids = positive_lookup.get(date_value, [])
        is_positive = bool(matched_event_ids)
        row["label"] = 1.0 if is_positive else 0.0
        row["label_status"] = "positive" if is_positive else "negative"
        row["label_source_mode"] = "date_range_overlap" if is_positive else "no_event_day"
        row["matched_event_ids"] = ",".join(sorted(matched_event_ids))
        row["label_provenance"] = json.dumps(
            {
                "date": date_value,
                "matched_event_ids": sorted(matched_event_ids),
                "matched_event_count": len(matched_event_ids),
                "source_file": path.name,
                "label_source": "verified_extreme_heat.csv",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    prepared = prepare_feature_frame(frame, hazard_type="extreme_heat")
    frame = frame.loc[prepared.index].reset_index(drop=True)
    frame.loc[:, prepared.columns] = prepared.reset_index(drop=True)
    frame["hazard_type"] = "extreme_heat"
    frame["sample_unit"] = "single-point-day"
    return frame.reset_index(drop=True)


def _build_region_day_dataset(
    feature_paths: Iterable[Path],
    label_table: Any,
    *,
    point_variable: str,
    negative_sample_size: int | None,
    seed: int,
    top_k: int,
    region_boundary_path: Path,
    precomputed_table: Any | None = None,
    return_un_sampled: bool = False,
):
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if precomputed_table is None:
        feature_list = [Path(path) for path in feature_paths]
        if not feature_list:
            raise FileNotFoundError("No daily indicator files were provided")

        boundary_table = _load_boundary_table(region_boundary_path)
        reference_frame = _dataset_frame(feature_list[0])
        province_lookup = build_flash_flood_province_lookup(
            reference_frame,
            boundary_table,
            province_column="province_name",
            boundary_province_column="shapeName",
            boundary_id_column="shapeID",
            geometry_column="geometry",
            geometry_format="geojson",
        )
        coordinate_lookup = province_lookup.loc[:, ["latitude", "longitude", "province_name"]].drop_duplicates(
            subset=["latitude", "longitude"],
            keep="last",
        )
        coordinate_lookup["province_name"] = coordinate_lookup["province_name"].map(_normalize_text)
        coordinate_lookup["region_id"] = coordinate_lookup["province_name"].map(_canonical_region_id)
        coordinate_lookup = coordinate_lookup.loc[
            coordinate_lookup["region_id"].astype(str).str.strip().ne(""),
            ["latitude", "longitude", "province_name", "region_id"],
        ].reset_index(drop=True)

        aggregated_rows: list[dict[str, Any]] = []
        for index, path in enumerate(feature_list):
            frame = reference_frame if index == 0 else _dataset_frame(path)
            enriched = frame.reset_index(drop=True).merge(
                coordinate_lookup,
                on=["latitude", "longitude"],
                how="left",
                validate="m:1",
            )
            enriched = enriched[enriched["region_id"].astype(str).str.strip().ne("")].copy()
            if enriched.empty:
                continue

            for _, group in enriched.groupby(["date", "region_id"], dropna=False, sort=True):
                aggregated_rows.append(_aggregate_region_day_group(group, point_variable=point_variable, top_k=top_k))

        if not aggregated_rows:
            raise ValueError("No daily indicator grid cells could be mapped into a Saudi admin-1 region")
        labeled = _assign_region_labels(
            pd.DataFrame(aggregated_rows),
            label_lookup=_positive_region_lookup(label_table),
        )
    else:
        labeled = precomputed_table.copy()

    if return_un_sampled:
        return labeled.reset_index(drop=True)

    selected = _sample_negative_region_rows(labeled, negative_sample_size=negative_sample_size, seed=seed)
    prepared = prepare_feature_frame(selected, hazard_type="extreme_heat")
    selected = selected.loc[prepared.index].reset_index(drop=True)
    selected.loc[:, prepared.columns] = prepared.reset_index(drop=True)
    selected["hazard_type"] = "extreme_heat"
    selected["training_join_key"] = selected[["date", "region_id"]].astype(str).agg("|".join, axis=1)
    selected["is_labeled"] = True
    return selected.reset_index(drop=True)


def build_extreme_heat_supervised_training_dataset(
    feature_paths: Iterable[Path],
    label_table: Any,
    *,
    point_variable: str = "heat_index_c",
    negative_sample_size: int | None = None,
    seed: int = 42,
    sample_unit: str = "single_point_day",
    top_k: int = 3,
    region_boundary_path: Path | None = None,
    precomputed_region_day_table: Any | None = None,
    return_un_sampled_region_day: bool = False,
):
    """Build an extreme-heat supervision table from daily indicator files."""

    _require_pandas()
    normalized_sample_unit = sample_unit.strip().lower()
    if normalized_sample_unit not in EXTREME_HEAT_SAMPLE_UNITS:
        raise ValueError(f"Unsupported sample_unit: {sample_unit}")
    if normalized_sample_unit == "single_point_day":
        return _build_single_point_day_dataset(
            feature_paths,
            label_table,
            point_variable=point_variable,
            negative_sample_size=negative_sample_size,
            seed=seed,
        )
    return _build_region_day_dataset(
        feature_paths,
        label_table,
        point_variable=point_variable,
        negative_sample_size=negative_sample_size,
        seed=seed,
        top_k=top_k,
        region_boundary_path=region_boundary_path or DEFAULT_REGION_BOUNDARY_PATH,
        precomputed_table=precomputed_region_day_table,
        return_un_sampled=return_un_sampled_region_day,
    )
