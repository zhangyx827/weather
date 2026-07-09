"""Crop a directory of global precipitation NetCDF files to the Saudi bbox."""

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

import xarray as xr  # noqa: E402

from mazu_saudi.data import crop_to_bbox  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing global NetCDF files")
    parser.add_argument("output_dir", type=Path, help="Directory to write cropped NetCDF files")
    parser.add_argument("--min-lat", type=float, default=16.0)
    parser.add_argument("--min-lon", type=float, default=34.0)
    parser.add_argument("--max-lat", type=float, default=32.0)
    parser.add_argument("--max-lon", type=float, default=56.0)
    return parser.parse_args()


def crop_directory(input_dir: Path, output_dir: Path, bbox: tuple[float, float, float, float]) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {input_dir}")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "bbox": {
            "min_lat": bbox[0],
            "min_lon": bbox[1],
            "max_lat": bbox[2],
            "max_lon": bbox[3],
        },
        "files": [],
    }

    for src_path in files:
        dst_path = output_dir / src_path.name
        with xr.open_dataset(src_path) as ds:
            cropped = crop_to_bbox(ds, bbox).load()
        cropped.to_netcdf(dst_path, engine="h5netcdf")
        summary["files"].append(
            {
                "name": src_path.name,
                "output": str(dst_path),
                "dims": {key: int(value) for key, value in cropped.sizes.items()},
            }
        )

    summary["file_count"] = len(files)
    return summary


def main() -> int:
    args = parse_args()
    bbox = (args.min_lat, args.min_lon, args.max_lat, args.max_lon)
    summary = crop_directory(args.input_dir, args.output_dir, bbox)
    manifest_path = args.output_dir / "crop_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["bbox"], ensure_ascii=False))
    print(f"cropped {summary['file_count']} files -> {args.output_dir}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
