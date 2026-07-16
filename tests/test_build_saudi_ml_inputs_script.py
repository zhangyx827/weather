from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_saudi_ml_inputs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_saudi_ml_inputs", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_era5_source_spec_supports_optional_missing_dir():
    module = _load_module()

    year, single_dir, pressure_dir, missing_pressure_dir = module.parse_era5_source_spec(
        "2024,./era5_single_levels_2024,./era5_pressure_levels_2024,./era5_pressure_levels_2024_missing"
    )

    assert year == 2024
    assert single_dir == Path("./era5_single_levels_2024")
    assert pressure_dir == Path("./era5_pressure_levels_2024")
    assert missing_pressure_dir == Path("./era5_pressure_levels_2024_missing")


def test_build_year_directory_mappings_preserves_year_specific_sources():
    module = _load_module()

    single_dirs, pressure_dirs, missing_pressure_dirs = module.build_year_directory_mappings(
        [
            (2022, Path("./era5_single_levels_2022_6h"), Path("./era5_pressure_levels_2022_6h"), None),
            (
                2024,
                Path("./era5_single_levels_2024"),
                Path("./era5_pressure_levels_2024"),
                Path("./era5_pressure_levels_2024_missing"),
            ),
        ]
    )

    assert single_dirs[2022] == Path("./era5_single_levels_2022_6h")
    assert pressure_dirs[2024] == Path("./era5_pressure_levels_2024")
    assert missing_pressure_dirs[2022] is None
    assert missing_pressure_dirs[2024] == Path("./era5_pressure_levels_2024_missing")
