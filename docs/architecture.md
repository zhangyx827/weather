# Architecture Notes

The MVP is intentionally rule-based and lightweight. Forecast model providers expose stable interfaces for later Aurora, GenCast, and AIFS integration without downloading model weights.

The system flow is:

`features -> indicators -> risk models -> knowledge graph explanations -> industry warning briefings -> API`.

The current rules are screening models. They are not operational thresholds and should be calibrated with Saudi observations, exposure data, warning policy, and CMA operational guidance before production use.
