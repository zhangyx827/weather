"""Forecast provider abstraction and mock implementations."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mazu_saudi.schemas import ForecastField, GridCell

SAUDI_BBOX = (16.0, 34.0, 32.0, 56.0)


class BaseForecastProvider(ABC):
    """Unified forecast provider interface."""

    name = "base"

    @abstractmethod
    def fetch(self, variable: str, valid_time: datetime | None = None, bbox: tuple[float, float, float, float] = SAUDI_BBOX) -> ForecastField:
        """Fetch or create a standard forecast field."""

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
        field = ForecastField(self.name, variable, units, valid, values, grid, {"source": "deterministic_mock"})
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
        values = payload["variables"][variable]["values"]
        units = payload["variables"][variable].get("units", "unknown")
        field = ForecastField(self.name, variable, units, valid, values, grid, {"source_path": str(self.path)})
        return self.crop_to_bbox(field, bbox)


class AuroraForecastProvider(MockForecastProvider):
    """Aurora integration placeholder with the standard provider interface."""

    name = "aurora_placeholder"


class GenCastForecastProvider(MockForecastProvider):
    """GenCast integration placeholder with the standard provider interface."""

    name = "gencast_placeholder"


class AIFSBenchmarkProvider(MockForecastProvider):
    """AIFS benchmark integration placeholder with the standard provider interface."""

    name = "aifs_benchmark_placeholder"


def load_netcdf_or_zarr_placeholder(path: str | Path, variable: str) -> dict[str, Any]:
    """Placeholder for future xarray-backed NetCDF/Zarr loading."""

    return {"path": str(path), "variable": variable, "status": "xarray_loader_placeholder"}
