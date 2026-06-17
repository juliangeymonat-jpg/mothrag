# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Per-aspect primitive disaggregation -- spectral decomposition of the
γ / L4b / cross-arm-agreement signals into per-claim granularity.

Background
----------

In the MOTHRAG 1 production stack, the three structural
signals (γ verifier, L4b temporal-stability, cross-arm agreement) fire
at the **whole-answer** granularity: γ_status ∈ {valid, partial, invalid}
applies to the entire chosen answer; L4b cancellation triggers on the
iter arm as a whole; cross-arm agreement is the pairwise cosine of
*entire* arm outputs.

For the SubQuestionRerouteCascade strategy (#9) Layer 2, the cascade
needs to know which **specific aspect** of the answer is the low-signal
component so it can formulate a targeted sub-question over that aspect
only. Example: given "Paris is the capital of France and has 2.1M
people", the cascade may want to verify the population claim alone
while trusting the capital-of claim.

This subpackage provides:

- :mod:`mothrag.core.spectral.aspects` -- aspect extraction over an
  answer string. Naive Capitalised-NP regex by default; spaCy
  dep-parse when ``mothrag[active-learning]`` is installed.
- :mod:`mothrag.core.spectral.disaggregation` -- per-aspect score
  helpers for γ, L4b, agreement. At v0.5.0 alpha each helper broadcasts
  the whole-answer signal across aspects (defensive default); the
  full per-aspect verifier surface ports in v0.5.1 when the granular γ
  pipeline is wired.

The contract for each disaggregator is::

    fn(answer: str, ..., aspects: list[str] | None = None) -> dict[str, float]

with values clamped to [0, 1] and missing aspects filled with the
appropriate neutral default (gamma=1.0, faith=1.0, agreement=0.0).
"""

from __future__ import annotations

from mothrag.core.spectral.aspects import (
    extract_aspects,
    DEFAULT_MAX_ASPECTS,
)
from mothrag.core.spectral.disaggregation import (
    gamma_per_aspect,
    l4b_per_aspect,
    agreement_per_aspect,
)

__all__ = [
    "extract_aspects",
    "DEFAULT_MAX_ASPECTS",
    "gamma_per_aspect",
    "l4b_per_aspect",
    "agreement_per_aspect",
]
