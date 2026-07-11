"""Helpers for converting bundled SRTM elevation tiles into model-ready grids."""

from __future__ import annotations

import json
import math
import re
import zipfile
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from mazu_saudi.schemas import GridCell
from .io import SAUDI_BBOX, _inclusive_range

SRTM_NO_DATA_VALUE = -32768

_SRTM_TILE_RE = re.compile(
    r"^(?P<lat_hemi>[NS])(?P<lat_deg>\d{2})(?P<lon_hemi>[EW])(?P<lon_deg>\d{3})\.SRTMGL1\.hgt\.zip$"
)


@dataclass(frozen=True)
class SRTMTile:
    """Metadata for one HGT tile."""

    path: Path
    south_lat: int
    west_lon: int

    @property
    def key(self) -> tuple[int, int]:
        return self.south_lat, self.west_lon


def discover_srtm_tiles(tile_dir: str | Path) -> list[SRTMTile]:
    """Return parsed tile metadata for every SRTM archive in ``tile_dir``."""

    directory = Path(tile_dir)
    tiles: list[SRTMTile] = []
    for path in sorted(directory.glob("*.zip")):
        if not zipfile.is_zipfile(path):
            continue
        match = _SRTM_TILE_RE.match(path.name)
        if not match:
            continue
        south_lat = int(match.group("lat_deg"))
        if match.group("lat_hemi") == "S":
            south_lat = -south_lat
        west_lon = int(match.group("lon_deg"))
        if match.group("lon_hemi") == "W":
            west_lon = -west_lon
        tiles.append(SRTMTile(path=path, south_lat=south_lat, west_lon=west_lon))
    return tiles


@lru_cache(maxsize=32)
def _load_hgt_array(path: Path) -> np.ndarray:
    """Load one HGT member from a ZIP archive as a square int16 array."""

    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".hgt")]
        if not members:
            raise RuntimeError(f"Archive {path} does not contain an HGT member")
        raw = archive.read(members[0])

    array = np.frombuffer(raw, dtype=">i2")
    side = int(math.isqrt(array.size))
    if side * side != array.size:
        raise ValueError(f"Unexpected HGT size in {path}: got {array.size}, expected a square grid")
    return array.reshape((side, side))


class SRTMElevationIndex:
    """In-memory tile index with on-demand HGT sampling."""

    def __init__(self, tile_dir: str | Path) -> None:
        self.tile_dir = Path(tile_dir)
        self.tiles = {tile.key: tile for tile in discover_srtm_tiles(tile_dir)}
        if not self.tiles:
            raise FileNotFoundError(f"No SRTM .hgt.zip files found in {self.tile_dir}")

    def available_tiles(self) -> list[SRTMTile]:
        return list(self.tiles.values())

    def sample(self, lat: float, lon: float) -> float | None:
        """Return nearest-neighbour elevation in meters for one lat/lon point."""

        tile = self._resolve_tile(lat, lon)
        if tile is None:
            return None

        data = _load_hgt_array(tile.path)
        row, col = self._point_to_indices(lat, lon, tile, data.shape[0])
        value = int(data[row, col])
        if value == SRTM_NO_DATA_VALUE:
            return None
        return float(value)

    def build_grid(
        self,
        bbox: tuple[float, float, float, float] = SAUDI_BBOX,
        resolution_deg: float = 0.1,
        prefix: str = "saudi",
    ) -> list[GridCell]:
        """Build a standard Saudi grid with elevation populated on each cell."""

        min_lat, min_lon, max_lat, max_lon = bbox
        lats = _inclusive_range(min_lat, max_lat, resolution_deg)
        lons = _inclusive_range(min_lon, max_lon, resolution_deg)
        cells: list[GridCell] = []
        for lat in lats:
            for lon in lons:
                cells.append(
                    GridCell(
                        id=f"{prefix}_{lat:.1f}_{lon:.1f}",
                        lat=lat,
                        lon=lon,
                        elevation_m=self.sample(lat, lon),
                    )
                )
        return cells

    def build_grid_table(
        self,
        bbox: tuple[float, float, float, float] = SAUDI_BBOX,
        resolution_deg: float = 0.1,
        prefix: str = "saudi",
    ) -> dict[str, Any]:
        """Return a JSON-serializable summary plus grid records."""

        cells = self.build_grid(bbox=bbox, resolution_deg=resolution_deg, prefix=prefix)
        missing = sum(1 for cell in cells if cell.elevation_m is None)
        return {
            "source": "srtm_hgt_zip",
            "tile_dir": str(self.tile_dir),
            "bbox": {
                "min_lat": bbox[0],
                "min_lon": bbox[1],
                "max_lat": bbox[2],
                "max_lon": bbox[3],
            },
            "resolution_deg": resolution_deg,
            "cell_count": len(cells),
            "missing_count": missing,
            "cells": [cell.to_dict() for cell in cells],
        }

    def to_xarray_dataset(
        self,
        bbox: tuple[float, float, float, float] = SAUDI_BBOX,
        resolution_deg: float = 0.1,
    ) -> Any:
        """Convert the sampled grid to an xarray Dataset when xarray is installed."""

        try:
            import xarray as xr
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("xarray is required to build a NetCDF elevation dataset") from exc

        min_lat, min_lon, max_lat, max_lon = bbox
        lats = _inclusive_range(min_lat, max_lat, resolution_deg)
        lons = _inclusive_range(min_lon, max_lon, resolution_deg)
        data = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)

        for lat_index, lat in enumerate(lats):
            for lon_index, lon in enumerate(lons):
                value = self.sample(lat, lon)
                if value is not None:
                    data[lat_index, lon_index] = float(value)

        return xr.Dataset(
            {
                "elevation_m": (("latitude", "longitude"), data),
            },
            coords={
                "latitude": lats,
                "longitude": lons,
            },
            attrs={
                "source": "srtm_hgt_zip",
                "tile_dir": str(self.tile_dir),
                "bbox": json.dumps(
                    {
                        "min_lat": bbox[0],
                        "min_lon": bbox[1],
                        "max_lat": bbox[2],
                        "max_lon": bbox[3],
                    }
                ),
                "resolution_deg": resolution_deg,
            },
        )

    def _resolve_tile(self, lat: float, lon: float) -> SRTMTile | None:
        lat_floor = math.floor(lat)
        lon_floor = math.floor(lon)
        tile = self.tiles.get((lat_floor, lon_floor))
        if tile is not None:
            return tile

        # Boundary points can land on the upper edge of the Saudi bbox.
        return self.tiles.get((lat_floor - 1, lon_floor - 1)) or self.tiles.get((lat_floor - 1, lon_floor)) or self.tiles.get((lat_floor, lon_floor - 1))

    @staticmethod
    def _point_to_indices(lat: float, lon: float, tile: SRTMTile, side: int) -> tuple[int, int]:
        samples_per_degree = side - 1
        row = int(round((tile.south_lat + 1.0 - lat) * samples_per_degree))
        col = int(round((lon - tile.west_lon) * samples_per_degree))
        row = max(0, min(side - 1, row))
        col = max(0, min(side - 1, col))
        return row, col


def enrich_features_with_elevation(features: list[GridCell], index: SRTMElevationIndex) -> list[GridCell]:
    """Attach sampled elevation to a list of grid cells."""

    return [replace(cell, elevation_m=index.sample(cell.lat, cell.lon)) for cell in features]
