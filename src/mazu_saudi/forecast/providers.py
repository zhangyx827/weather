"""Forecast provider abstraction and mock implementations."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from mazu_saudi.data import crop_to_saudi as data_crop_to_saudi
from mazu_saudi.data import read_netcdf_dataset, read_zarr_dataset
from mazu_saudi.schemas import ForecastField, GridCell

SAUDI_BBOX = (16.0, 34.0, 32.0, 56.0)

VARIABLE_ALIASES = {
    "2t": "temp_c",
    "t2m": "temp_c",
    "temperature": "temp_c",
    "relative_humidity": "rh_percent",
    "rh": "rh_percent",
    "10u": "wind_u_mps",
    "10v": "wind_v_mps",
    "wind": "wind_speed_mps",
    "tp": "precip_1h_mm",
    "precipitation": "precip_1h_mm",
}


class BaseForecastProvider(ABC):
    """Unified forecast provider interface."""

    name = "base"
    dataset: Any = None

    @abstractmethod
    def fetch(self, variable: str, valid_time: datetime | None = None, bbox: tuple[float, float, float, float] = SAUDI_BBOX) -> ForecastField:
        """Fetch or create a standard forecast field."""

    def load_from_local(self, path: str | Path) -> "BaseForecastProvider":
        """Load a local JSON/NetCDF/Zarr forecast dataset into the provider."""

        path_obj = Path(path)
        if path_obj.suffix.lower() == ".json":
            self.dataset = json.loads(path_obj.read_text(encoding="utf-8"))
        elif path_obj.suffix.lower() in {".nc", ".netcdf"}:
            self.dataset = read_netcdf_dataset(path_obj)
        else:
            self.dataset = read_zarr_dataset(path_obj)
        return self

    def get_forecast(
        self,
        issue_time: datetime,
        lead_hours: int | list[int],
        bbox: tuple[float, float, float, float] = SAUDI_BBOX,
        variables: list[str] | None = None,
    ) -> dict[str, ForecastField]:
        """Return forecast fields keyed by ``variable:+lead_hour``."""

        leads = lead_hours if isinstance(lead_hours, list) else [lead_hours]
        selected = variables or ["temp_c", "rh_percent", "wind_speed_mps", "precip_1h_mm"]
        fields = {}
        for lead in leads:
            valid = issue_time + timedelta(hours=int(lead))
            for variable in selected:
                normalized = self.normalize_variables(variable)
                fields[f"{normalized}:+{int(lead)}h"] = self.fetch(normalized, valid_time=valid, bbox=bbox)
        return fields

    def normalize_variables(self, variable: str) -> str:
        """Normalize provider-specific variable names to MAZU names."""

        return VARIABLE_ALIASES.get(variable, variable)

    def crop_to_saudi(self, field: ForecastField) -> ForecastField:
        """Crop a forecast field to the Saudi operating domain."""

        return data_crop_to_saudi(field)

    def resample_to_grid(self, field: ForecastField, grid: list[GridCell] | None = None) -> ForecastField:
        """Placeholder for future conservative/bilinear grid remapping."""

        metadata = dict(field.metadata)
        metadata.update({"resample_status": "placeholder", "target_grid_size": len(grid or field.grid)})
        return ForecastField(field.provider, field.variable, field.units, field.valid_time, list(field.values), list(field.grid), metadata)

    def crop_to_bbox(self, field: ForecastField, bbox: tuple[float, float, float, float] = SAUDI_BBOX) -> ForecastField:
        """Return points inside ``(min_lat, min_lon, max_lat, max_lon)``."""

        min_lat, min_lon, max_lat, max_lon = bbox
        pairs = [(v, g) for v, g in zip(field.values, field.grid) if min_lat <= g.lat <= max_lat and min_lon <= g.lon <= max_lon]
        if not pairs:
            return ForecastField(field.provider, field.variable, field.units, field.valid_time, [], [], dict(field.metadata))
        values, grid = zip(*pairs)
        return ForecastField(field.provider, field.variable, field.units, field.valid_time, list(values), list(grid), dict(field.metadata))


class MockForecastProvider(BaseForecastProvider):
    """Deterministic mock forecast provider for demos and tests."""

    name = "mock"

    def fetch(self, variable: str, valid_time: datetime | None = None, bbox: tuple[float, float, float, float] = SAUDI_BBOX) -> ForecastField:
        valid = valid_time or datetime.now(timezone.utc)
        variable = self.normalize_variables(variable)
        grid = [
            GridCell(id="riyadh", lat=24.7136, lon=46.6753, elevation_m=612, region="Riyadh"),
            GridCell(id="jeddah", lat=21.4858, lon=39.1925, elevation_m=12, region="Makkah"),
            GridCell(id="dammam", lat=26.4207, lon=50.0888, elevation_m=10, region="Eastern Province"),
        ]
        defaults = {
            "temp_c": [45.0, 38.0, 41.0],
            "rh_percent": [18.0, 72.0, 64.0],
            "wind_speed_mps": [8.0, 10.0, 14.0],
            "precip_1h_mm": [2.0, 12.0, 5.0],
        }
        values = defaults.get(variable, [0.0 for _ in grid])
        units = {"temp_c": "degC", "rh_percent": "%", "wind_speed_mps": "m/s", "precip_1h_mm": "mm"}.get(variable, "unknown")
        field = ForecastField(
            self.name,
            variable,
            units,
            valid,
            values,
            grid,
            {
                "source": "deterministic_mock",
                "structure": {"dims": ["time", "lat", "lon"], "time": [valid.isoformat()], "lat": [g.lat for g in grid], "lon": [g.lon for g in grid]},
            },
        )
        return self.crop_to_bbox(field, bbox)


class JSONForecastProvider(BaseForecastProvider):
    """Read a local JSON forecast sample and return a standard field."""

    name = "json"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def fetch(self, variable: str, valid_time: datetime | None = None, bbox: tuple[float, float, float, float] = SAUDI_BBOX) -> ForecastField:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        valid = valid_time or datetime.fromisoformat(payload.get("valid_time", datetime.now(timezone.utc).isoformat()))
        grid = [GridCell(**item) for item in payload["grid"]]
        normalized = self.normalize_variables(variable)
        values = payload["variables"][normalized]["values"]
        units = payload["variables"][normalized].get("units", "unknown")
        metadata = {"source_path": str(self.path), "structure": payload.get("structure", {"dims": ["time", "lat", "lon"]})}
        field = ForecastField(self.name, normalized, units, valid, values, grid, metadata)
        return self.crop_to_bbox(field, bbox)


class AuroraForecastProvider(MockForecastProvider):
    """Aurora integration placeholder with the standard provider interface."""

    name = "aurora_placeholder"


class GenCastForecastProvider(MockForecastProvider):
    """GenCast integration placeholder with the standard provider interface."""

    name = "gencast_placeholder"

    def fetch(self, variable: str, valid_time: datetime | None = None, bbox: tuple[float, float, float, float] = SAUDI_BBOX) -> ForecastField:
        field = super().fetch(variable, valid_time, bbox)
        member_values = []
        for offset in (-0.15, 0.0, 0.12, 0.25):
            member_values.append([float(value) * (1.0 + offset) for value in field.values])
        metadata = dict(field.metadata)
        metadata.update({"member_count": len(member_values), "member_values": member_values, "structure": {"dims": ["member", "time", "lat", "lon"]}})
        return ForecastField(field.provider, field.variable, field.units, field.valid_time, field.values, field.grid, metadata)

    def member_count(self, field: ForecastField) -> int:
        """Return ensemble member count from metadata."""

        return int(field.metadata.get("member_count", len(field.metadata.get("member_values", [])) or 1))

    def ensemble_mean(self, field: ForecastField) -> list[float]:
        """Compute pointwise ensemble mean."""

        members = field.metadata.get("member_values") or [field.values]
        return [sum(values) / len(values) for values in zip(*members)]

    def ensemble_spread(self, field: ForecastField) -> list[float]:
        """Compute pointwise ensemble standard deviation."""

        means = self.ensemble_mean(field)
        members = field.metadata.get("member_values") or [field.values]
        spread = []
        for index, mean in enumerate(means):
            variance = sum((member[index] - mean) ** 2 for member in members) / len(members)
            spread.append(variance ** 0.5)
        return spread

    def exceedance_probability(self, field: ForecastField, threshold: float, variable: str | None = None) -> list[float]:
        """Compute pointwise probability that ensemble members exceed threshold."""

        if variable is not None and self.normalize_variables(variable) != field.variable:
            raise ValueError(f"field variable {field.variable!r} does not match requested {variable!r}")
        members = field.metadata.get("member_values") or [field.values]
        return [sum(1 for member in members if member[index] >= threshold) / len(members) for index in range(len(field.values))]


class AIFSBenchmarkProvider(MockForecastProvider):
    """AIFS benchmark integration placeholder with the standard provider interface."""

    name = "aifs_benchmark_placeholder"


def load_netcdf_or_zarr_placeholder(path: str | Path, variable: str) -> dict[str, Any]:
    """Placeholder for future xarray-backed NetCDF/Zarr loading."""

    path_obj = Path(path)
    try:
        dataset = read_netcdf_dataset(path_obj) if path_obj.suffix.lower() in {".nc", ".netcdf"} else read_zarr_dataset(path_obj)
    except Exception as exc:
        return {"path": str(path), "variable": variable, "status": "xarray_loader_unavailable", "error": str(exc)}
    return {"path": str(path), "variable": variable, "status": "loaded", "dims": dict(getattr(dataset, "sizes", {}))}
