"""Risk-level mapping utilities."""

from dataclasses import dataclass
from typing import Any

from mazu_saudi.schemas import RiskLevel


@dataclass(frozen=True)
class RiskThresholdConfig:
    """Configurable inclusive lower-bound risk thresholds."""

    low: tuple[float, float] = (0.0, 0.25)
    medium: tuple[float, float] = (0.25, 0.5)
    high: tuple[float, float] = (0.5, 0.75)
    extreme: tuple[float, float] = (0.75, 1.0)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "RiskThresholdConfig":
        if not payload:
            return cls()
        values = {}
        for level in ("low", "medium", "high", "extreme"):
            item = payload.get(level)
            if isinstance(item, dict):
                values[level] = (float(item.get("min", 0.0)), float(item.get("max", 1.0)))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                values[level] = (float(item[0]), float(item[1]))
        return cls(**values)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "low": [self.low[0], self.low[1]],
            "medium": [self.medium[0], self.medium[1]],
            "high": [self.high[0], self.high[1]],
            "extreme": [self.extreme[0], self.extreme[1]],
        }


DEFAULT_RISK_THRESHOLDS = RiskThresholdConfig()


def probability_to_level(probability: float, thresholds: RiskThresholdConfig | None = None) -> RiskLevel:
    """Map 0-1 model output to the four-level risk scale."""

    p = max(0.0, min(1.0, float(probability)))
    config = thresholds or DEFAULT_RISK_THRESHOLDS
    if p >= config.extreme[0]:
        return RiskLevel.EXTREME
    if p >= config.high[0]:
        return RiskLevel.HIGH
    if p >= config.medium[0]:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW
