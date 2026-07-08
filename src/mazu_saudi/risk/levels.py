"""Risk-level mapping utilities."""

from mazu_saudi.schemas import RiskLevel


def probability_to_level(probability: float) -> RiskLevel:
    """Map 0-1 model output to the four-level risk scale."""

    return RiskLevel.from_probability(probability)
