"""Convert bundled NIS SRTM tiles into model-ready elevation products."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.data import SRTMElevationIndex, write_netcdf_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing NIS SRTM .hgt.zip tiles.")
    parser.add_argument("output_dir", type=Path, help="Directory for converted products.")
    parser.add_argument("--min-lat", type=float, default=16.0)
    parser.add_argument("--min-lon", type=float, default=34.0)
    parser.add_argument("--max-lat", type=float, default=32.0)
    parser.add_argument("--max-lon", type=float, default=56.0)
    parser.add_argument("--resolution", type=float, default=0.1, help="Grid resolution in degrees.")
    return parser.parse_args()


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    bbox: tuple[float, float, float, float],
    resolution_deg: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_zip_count = len(list(input_dir.glob("*.zip")))
    index = SRTMElevationIndex(input_dir)
    table = index.build_grid_table(bbox=bbox, resolution_deg=resolution_deg)
    json_path = output_dir / "nis_elevation_grid.json"
    json_path.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")

    summary: dict[str, object] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "json_path": str(json_path),
        "raw_zip_count": raw_zip_count,
        "tile_count": len(index.available_tiles()),
        "skipped_count": raw_zip_count - len(index.available_tiles()),
        "cell_count": table["cell_count"],
        "missing_count": table["missing_count"],
        "bbox": table["bbox"],
        "resolution_deg": resolution_deg,
    }

    try:
        dataset = index.to_xarray_dataset(bbox=bbox, resolution_deg=resolution_deg)
    except Exception as exc:
        summary["netcdf"] = {"written": False, "reason": str(exc)}
    else:
        nc_path = output_dir / "nis_elevation_grid.nc"
        write_netcdf_dataset(nc_path, dataset)
        summary["netcdf"] = {"written": True, "path": str(nc_path)}

    return summary


def main() -> int:
    args = parse_args()
    bbox = (args.min_lat, args.min_lon, args.max_lat, args.max_lon)
    summary = convert_directory(args.input_dir, args.output_dir, bbox, args.resolution)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
