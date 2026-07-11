"""Tests for SRTM tile conversion helpers."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import numpy as np

from mazu_saudi.data import SRTMElevationIndex, discover_srtm_tiles


def _write_fake_tile(path: Path) -> None:
    array = np.zeros((1201, 1201), dtype=np.int16)
    array[0, 0] = 101
    array[600, 600] = 303
    array[-1, -1] = 202
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("N17E042.hgt", array.astype(">i2").tobytes())


def test_discover_srtm_tiles_parses_tile_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "N17E042.SRTMGL1.hgt.zip"
        _write_fake_tile(path)

        tiles = discover_srtm_tiles(tmp)

    assert len(tiles) == 1
    assert tiles[0].key == (17, 42)


def test_srtm_sampling_uses_correct_tile_orientation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "N17E042.SRTMGL1.hgt.zip"
        _write_fake_tile(path)

        index = SRTMElevationIndex(tmp)

        assert index.sample(18.0, 42.0) == 101.0
        assert index.sample(17.5, 42.5) == 303.0
        assert index.sample(17.0, 43.0) == 202.0


def test_srtm_grid_builder_populates_elevation_cells() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "N17E042.SRTMGL1.hgt.zip"
        _write_fake_tile(path)

        index = SRTMElevationIndex(tmp)
        cells = index.build_grid(bbox=(17.0, 42.0, 17.1, 42.1), resolution_deg=0.1)

    assert len(cells) == 4
    assert all(cell.elevation_m is not None for cell in cells)

