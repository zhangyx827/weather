"""Shared Layer-4 runtime model path resolution."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LAYER4_MODEL_DIR = REPO_ROOT / "models" / "layer4"

_LAYER4_ENV_KEYS = {
    "flash_flood": "MAZU_LAYER4_FLASH_FLOOD_MODEL",
    "extreme_heat": "MAZU_LAYER4_EXTREME_HEAT_MODEL",
    "dry_heat_agriculture": "MAZU_LAYER4_DRY_HEAT_MODEL",
}

_LAYER4_DEFAULT_NAMES = {
    "flash_flood": "flash_flood.txt",
    "extreme_heat": "extreme_heat.txt",
    "dry_heat_agriculture": "dry_heat_stress.txt",
}

_LAYER4_CANDIDATES = {
    "flash_flood": (
        "models/layer4/flash_flood.txt",
        "data/processed/models/flash_flood_province_day_verified_chain_baseline_quick/flash_flood.txt",
        "data/processed/models/flash_flood_province_day_clean_baseline_quick/flash_flood.txt",
    ),
    "extreme_heat": ("models/layer4/extreme_heat.txt",),
    "dry_heat_agriculture": ("models/layer4/dry_heat_stress.txt",),
}


def layer4_model_env_key(hazard_type: str) -> str:
    normalized = hazard_type.strip().lower()
    return _LAYER4_ENV_KEYS.get(normalized, f"MAZU_LAYER4_{normalized.upper()}_MODEL")


def _normalize_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def resolve_layer4_model_path(
    hazard_type: str,
    *,
    explicit: str | Path | None = None,
    env_key: str | None = None,
    default_name: str | None = None,
    allow_missing: bool = False,
) -> Path | None:
    normalized = hazard_type.strip().lower()
    resolved_env_key = env_key or layer4_model_env_key(normalized)
    configured = explicit or os.environ.get(resolved_env_key)
    if configured is not None:
        configured_path = _normalize_path(configured)
        if configured_path.exists():
            return configured_path
        raise FileNotFoundError(
            f"Configured Layer-4 model file not found for {normalized}: {configured_path}. "
            f"Check {resolved_env_key} or the explicit model path."
        )

    candidate_rel_paths = _LAYER4_CANDIDATES.get(normalized)
    if candidate_rel_paths is None:
        fallback_name = default_name or _LAYER4_DEFAULT_NAMES.get(normalized)
        candidate_rel_paths = tuple(filter(None, [f"models/layer4/{fallback_name}" if fallback_name else None]))

    for rel_path in candidate_rel_paths:
        candidate = _normalize_path(rel_path)
        if candidate.exists():
            return candidate

    if allow_missing:
        return None

    expected = default_name or _LAYER4_DEFAULT_NAMES.get(normalized) or f"{normalized}.txt"
    raise FileNotFoundError(
        f"Layer-4 LightGBM model file not found for {normalized}. "
        f"Checked default runtime locations and expected {DEFAULT_LAYER4_MODEL_DIR / expected}. "
        f"Set {resolved_env_key} to a trained LightGBM booster file."
    )
