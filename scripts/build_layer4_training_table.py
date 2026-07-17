#!/usr/bin/env python3
"""Flatten daily Saudi indicator NetCDF files into hazard-specific training tables."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mazu_saudi.data import read_netcdf_dataset
from mazu_saudi.risk.layer4_features import feature_frame_from_dataset


HAZARD_TYPES = ("extreme_heat", "dry_heat_agriculture", "flash_flood")
FORMATS = ("csv", "json", "parquet")
DEFAULT_PATTERN = "saudi_indicators_*.nc"
DATE_PATTERN = re.compile(r"(\d{8})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hazard-specific Layer-4 training tables from indicator NetCDF files.")
    parser.add_argument("--input", type=Path, required=True, help="Indicator NetCDF file or directory.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed" / "layer4_training_tables")
    parser.add_argument("--hazard-type", choices=HAZARD_TYPES, action="append", help="Hazard type to export. Repeat for multiple hazards. Defaults to all.")
    parser.add_argument("--glob", default=DEFAULT_PATTERN, help="Glob pattern used when --input is a directory.")
    parser.add_argument("--format", choices=FORMATS, default="csv", help="Output table format. Defaults to csv.")
    return parser.parse_args(argv)


def _discover_input_files(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    return sorted(candidate for candidate in path.glob(pattern) if candidate.is_file())


def _coerce_timestamp(dataset: Any, source_path: Path) -> str:
    if hasattr(dataset, "coords") and "time" in dataset.coords and dataset.coords["time"].size:
        value = dataset.coords["time"].values[0]
        return str(np.datetime_as_string(np.asarray(value, dtype="datetime64[ns]"), unit="D"))
    match = DATE_PATTERN.search(source_path.stem)
    if match is None:
        raise ValueError(f"Could not infer date from dataset time coordinate or file name: {source_path}")
    token = match.group(1)
    return f"{token[0:4]}-{token[4:6]}-{token[6:8]}"


def _degradation_metadata_json(dataset: Any) -> str:
    payload = dataset.attrs.get("degradation_metadata", {})
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _source_signature(source_path: Path) -> dict[str, Any]:
    stat = source_path.stat()
    return {
        "source_size_bytes": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        # Use microseconds for incremental matching because they fit safely in float64
        # when older parquet rows are read back with a widened numeric dtype.
        "source_mtime_us": int(stat.st_mtime_ns // 1_000),
        "source_mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _build_table_from_dataset(dataset: Any, source_path: Path, hazard_type: str, source_signature: dict[str, Any]):
    """优化抽离：接受已打开的 dataset 避免重复进行文件 I/O"""
    frame = feature_frame_from_dataset(dataset, hazard_type=hazard_type)
    frame.insert(0, "date", _coerce_timestamp(dataset, source_path))
    frame.insert(1, "hazard_type", hazard_type)
    frame["source_file"] = source_path.name
    frame["source_status"] = dataset.attrs.get("source_status", "normal")
    frame["degradation_metadata"] = _degradation_metadata_json(dataset)
    frame["source_size_bytes"] = source_signature["source_size_bytes"]
    frame["source_mtime_ns"] = source_signature["source_mtime_ns"]
    frame["source_mtime_us"] = source_signature["source_mtime_us"]
    frame["source_mtime_utc"] = source_signature["source_mtime_utc"]
    return frame


def _build_table_for_file(source_path: Path, hazard_type: str):
    """保持向后兼容的原始接口"""
    dataset = read_netcdf_dataset(source_path)
    source_signature = _source_signature(source_path)
    return _build_table_from_dataset(dataset, source_path, hazard_type, source_signature)


def _write_table(table: Any, path: Path, fmt: str) -> None:
    if fmt == "csv":
        table.to_csv(path, index=False)
        return
    if fmt == "json":
        path.write_text(table.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
        return
    try:
        table.to_parquet(path, index=False)
    except Exception as exc:
        raise RuntimeError("Parquet export requires a pandas parquet engine such as pyarrow or fastparquet") from exc


def _read_table(path: Path, fmt: str):
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "json":
        return pd.read_json(path)
    return pd.read_parquet(path)


def _incremental_merge(existing_table, new_tables):
    if existing_table is None or existing_table.empty:
        return pd.concat(new_tables, ignore_index=True) if len(new_tables) > 1 else new_tables[0]
    if not new_tables:
        return existing_table

    retained = existing_table.copy()
    
    # 优化点 3：通过集合收集所有要被替换的 source_file，利用向量化 .isin() 一次性过滤，避免循环过滤
    if "source_file" in retained.columns:
        files_to_remove = {str(table["source_file"].iloc[0]) for table in new_tables if not table.empty}
        if files_to_remove:
            retained = retained[~retained["source_file"].astype(str).isin(files_to_remove)]

    return pd.concat([retained, *new_tables], ignore_index=True)


def _existing_signature_map(table: Any) -> dict[str, tuple[int, int]]:
    if table is None or "source_file" not in table.columns or "source_size_bytes" not in table.columns:
        return {}

    df = table.dropna(subset=["source_file", "source_size_bytes"]).drop_duplicates(subset=["source_file"]).copy()
    if "source_mtime_us" in df.columns:
        mtime_us = df["source_mtime_us"]
    else:
        mtime_us = pd.Series(np.nan, index=df.index)

    if "source_mtime_ns" in df.columns:
        legacy_mtime_us = df["source_mtime_ns"] // 1_000
        mtime_us = mtime_us.where(~mtime_us.isna(), legacy_mtime_us)

    df["signature_mtime_us"] = mtime_us
    df = df.dropna(subset=["signature_mtime_us"])
    if df.empty:
        return {}

    return dict(
        zip(
            df["source_file"].astype(str),
            zip(df["source_size_bytes"].astype(int), df["signature_mtime_us"].astype(int)),
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if pd is None:
        raise RuntimeError("pandas is required to build Layer-4 training tables")
    input_files = _discover_input_files(args.input, args.glob)
    if not input_files:
        raise FileNotFoundError(f"No indicator NetCDF files matched under {args.input} with pattern {args.glob!r}")

    hazard_types = args.hazard_type or list(HAZARD_TYPES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 优化点 1：提前读取所有灾害的历史大表，并构建高性能的签名哈希映射缓存
    existing_tables = {}
    existing_signatures = {}  # 格式: {hazard_type: {source_file: (size, mtime_us)}}
    
    for hazard_type in hazard_types:
        output_path = args.output_dir / f"{hazard_type}_training.{args.format}"
        if output_path.exists():
            table = _read_table(output_path, args.format)
            existing_tables[hazard_type] = table
            existing_signatures[hazard_type] = _existing_signature_map(table)
        else:
            existing_tables[hazard_type] = None
            existing_signatures[hazard_type] = {}

    # 初始化收集容器
    new_tables_by_hazard = {h: [] for h in hazard_types}
    skipped_counts = {h: 0 for h in hazard_types}

    # 优化点 2：颠倒循环，外层循环文件，确保一个大 NetCDF 文件只做一次 I/O 读取
    for path in input_files:
        source_file = path.name
        source_signature = _source_signature(path)
        
        # 找出当前文件需要为哪些灾害类型生成特征数据（未匹配上签名的灾害）
        hazards_to_process = []
        for hazard_type in hazard_types:
            sigs = existing_signatures[hazard_type]
            if source_file in sigs:
                prior_size, prior_mtime_us = sigs[source_file]
                if prior_size == source_signature["source_size_bytes"] and prior_mtime_us == source_signature["source_mtime_us"]:
                    skipped_counts[hazard_type] += 1
                    continue
            hazards_to_process.append(hazard_type)
        
        # 只有在至少有一种灾害需要处理时，才加载 NetCDF 文件
        if hazards_to_process:
            dataset = read_netcdf_dataset(path)
            for hazard_type in hazards_to_process:
                frame = _build_table_from_dataset(dataset, path, hazard_type, source_signature)
                new_tables_by_hazard[hazard_type].append(frame)

    # 按原始要求的灾害顺序，进行增量合并写入并构建最终的 Summary 输出
    summary: list[dict[str, Any]] = []
    for hazard_type in hazard_types:
        output_path = args.output_dir / f"{hazard_type}_training.{args.format}"
        existing_table = existing_tables[hazard_type]
        new_tables = new_tables_by_hazard[hazard_type]
        
        table = _incremental_merge(existing_table, new_tables)
        _write_table(table, output_path, args.format)
        
        summary.append(
            {
                "hazard_type": hazard_type,
                "files": len(input_files),
                "skipped_files": skipped_counts[hazard_type],
                "rows": int(len(table)),
                "format": args.format,
                "output": str(output_path),
            }
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
