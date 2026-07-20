"""Runtime configuration for briefing, LLM, and Strands integrations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_json(name: str) -> dict[str, Any]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return {}
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


@dataclass
class BriefingProviderSettings:
    provider: str = "template"

    @classmethod
    def from_env(cls) -> "BriefingProviderSettings":
        return cls(provider=os.getenv("MAZU_BRIEFING_PROVIDER", "template").strip().lower() or "template")


@dataclass
class GroundingPolicySettings:
    precip_daily_abs_diff_mm_threshold: float = 10.0
    require_note_on_missing_grounding: bool = True

    @classmethod
    def from_env(cls) -> "GroundingPolicySettings":
        return cls(
            precip_daily_abs_diff_mm_threshold=_env_float("MAZU_GROUNDING_PRECIP_DAILY_ABS_DIFF_MM_THRESHOLD", 10.0),
            require_note_on_missing_grounding=_env_flag("MAZU_GROUNDING_REQUIRE_NOTE_ON_MISSING", True),
        )


@dataclass
class LLMSettings:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout_seconds: float = 30.0
    chat_path: str = "/v1/chat/completions"
    system_prompt: str = (
        "You generate concise but specific multi-industry Chinese weather warning briefings. "
        "Always return valid JSON that follows the requested schema exactly."
    )
    temperature: float = 0.2
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "LLMSettings":
        return cls(
            base_url=os.getenv("MAZU_LLM_BASE_URL", "").strip(),
            api_key=os.getenv("MAZU_LLM_API_KEY", "").strip(),
            model=os.getenv("MAZU_LLM_MODEL", "").strip(),
            timeout_seconds=_env_float("MAZU_LLM_TIMEOUT_SECONDS", 30.0),
            chat_path=os.getenv("MAZU_LLM_CHAT_PATH", "/v1/chat/completions").strip() or "/v1/chat/completions",
            system_prompt=os.getenv(
                "MAZU_LLM_SYSTEM_PROMPT",
                "You generate concise but specific multi-industry Chinese weather warning briefings. "
                "Always return valid JSON that follows the requested schema exactly.",
            ).strip(),
            temperature=_env_float("MAZU_LLM_TEMPERATURE", 0.2),
            extra_body=_env_json("MAZU_LLM_EXTRA_BODY_JSON"),
        )


@dataclass
class StrandsSettings:
    enabled: bool = False
    model_id: str = "amazon.nova-lite-v1:0"
    region: str = "us-east-1"
    provider: str = "bedrock"
    agent_name: str = "MAZU Saudi Warning Agent"
    timeout_seconds: float = 30.0
    system_prompt: str = (
        "You are a strict workflow orchestrator. Execute the standard operating procedure exactly once. "
        "Call the validation tool, then the warning pipeline tool, then the response assembly tool. "
        "Do not retry, branch, reflect, explain, or modify output formats."
    )
    extra_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "StrandsSettings":
        return cls(
            enabled=_env_flag("MAZU_STRANDS_ENABLED", False),
            model_id=os.getenv("MAZU_STRANDS_MODEL_ID", "amazon.nova-lite-v1:0").strip() or "amazon.nova-lite-v1:0",
            region=os.getenv("MAZU_STRANDS_REGION", "us-east-1").strip() or "us-east-1",
            provider=os.getenv("MAZU_STRANDS_PROVIDER", "bedrock").strip().lower() or "bedrock",
            agent_name=os.getenv("MAZU_STRANDS_AGENT_NAME", "MAZU Saudi Warning Agent").strip() or "MAZU Saudi Warning Agent",
            timeout_seconds=_env_float("MAZU_STRANDS_TIMEOUT_SECONDS", 30.0),
            system_prompt=os.getenv(
                "MAZU_STRANDS_SYSTEM_PROMPT",
                "You are a strict workflow orchestrator. Execute the standard operating procedure exactly once. "
                "Call the validation tool, then the warning pipeline tool, then the response assembly tool. "
                "Do not retry, branch, reflect, explain, or modify output formats.",
            ).strip(),
            extra_config=_env_json("MAZU_STRANDS_EXTRA_CONFIG_JSON"),
        )
