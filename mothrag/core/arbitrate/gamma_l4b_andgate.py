# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""gamma + L4b AND-gate composition primitive.

Generic system-telemetry rule combining two existing primitives:

  * gamma verifier output (``gamma_final_status`` in {"valid", "invalid",
    "partial", "refuse", None}). An empirical calibration on HP/2W/MQ
    iter outputs found that the gamma-as-binary-correctness-predictor
    delivers P=0.805 (HP) / P=0.818 (2W) / P=0.631 (MQ) when used alone.

  * L4b stability score in ``[0, 1]``: float arms-agreement fraction
    (1.0 = unanimous, 0.0 = all-disagree). Empirically, L4b
    cancellation fires 1.7% with 100% gamma-invalid coupling and a
    +26pp F1 lift on the cancelled cohort.

The AND-gate composition formalizes the "keep when BOTH signals
positive" idea: intersect gamma-valid with high-stability to push
effective precision above either signal alone. Whether it crosses any
specific threshold in production is a downstream operator decision from
empirical data -- this module supplies the deterministic primitive only.

Anti-leak contract:
The rule reads ONLY system telemetry (gamma status, stability score).
It does NOT inspect gold answers, F1, or any held-out signal. It does
NOT carry per-dataset arguments. Threshold default 0.5 is the symmetric
midpoint of the [0, 1] stability range -- callers can override but the
default is theory-derived, not F1-fitted.

Compatible with both legacy mothrag arbitrate paths and the
Stage 2.5 wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


__all__ = [
    "ANDGateDecision",
    "gamma_l4b_andgate_decision",
    "VALID_GAMMA_STATUSES",
]


# Valid γ status strings produced by the gamma verifier.
# ``None`` is also accepted as a "no-status" sentinel; treated as not-valid.
VALID_GAMMA_STATUSES: frozenset[str] = frozenset(
    {"valid", "invalid", "partial", "refuse"},
)


ANDGateDecision = Literal["keep", "defer"]


@dataclass(frozen=True)
class _Decision:
    """Internal: structured return of the decision + reason for telemetry."""

    decision: ANDGateDecision
    reason: str


def gamma_l4b_andgate_decision(
    gamma_status: Optional[str],
    l4b_stability_score: float,
    threshold: float = 0.5,
) -> ANDGateDecision:
    """Return ``"keep"`` iff gamma reports ``"valid"`` AND L4b stability is
    at or above ``threshold``; otherwise ``"defer"``.

    Args:
        gamma_status: gamma verifier output. Recognized values:
            ``"valid"`` (only status that allows ``"keep"``),
            ``"invalid"`` / ``"partial"`` / ``"refuse"`` / ``None`` (all
            force ``"defer"``).
        l4b_stability_score: arms-agreement fraction in ``[0, 1]``.
        threshold: minimum stability to accept ``"keep"``. Default
            ``0.5`` is the symmetric midpoint of the [0, 1] range,
            theory-derived (not F1-tuned). Caller may override.

    Returns:
        ``"keep"`` when BOTH conditions hold; ``"defer"`` otherwise.

    Raises:
        ValueError: if ``l4b_stability_score`` is not in ``[0, 1]`` or
            ``threshold`` is not in ``[0, 1]``.
    """
    if not (0.0 <= l4b_stability_score <= 1.0):
        raise ValueError(
            f"l4b_stability_score must be in [0, 1], "
            f"got {l4b_stability_score!r}"
        )
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"threshold must be in [0, 1], got {threshold!r}"
        )
    if gamma_status == "valid" and l4b_stability_score >= threshold:
        return "keep"
    return "defer"


def gamma_l4b_andgate_diagnostic(
    gamma_status: Optional[str],
    l4b_stability_score: float,
    threshold: float = 0.5,
) -> _Decision:
    """Diagnostic variant that returns ``(decision, reason)``.

    Useful for telemetry-rich downstream callers (Stage 5 arbitration
    score_breakdown, Stage 6 provenance log). The :func:`gamma_l4b_andgate_decision`
    plain variant remains the canonical entry point.
    """
    if not (0.0 <= l4b_stability_score <= 1.0):
        raise ValueError(
            f"l4b_stability_score must be in [0, 1], "
            f"got {l4b_stability_score!r}"
        )
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"threshold must be in [0, 1], got {threshold!r}"
        )
    if gamma_status != "valid":
        return _Decision(
            decision="defer",
            reason=f"gamma_not_valid({gamma_status!r})",
        )
    if l4b_stability_score < threshold:
        return _Decision(
            decision="defer",
            reason=(
                f"stability_below_threshold("
                f"{l4b_stability_score:.3f}<{threshold:.3f})"
            ),
        )
    return _Decision(
        decision="keep",
        reason=(
            f"gamma_valid_and_stable("
            f"stab={l4b_stability_score:.3f}>={threshold:.3f})"
        ),
    )
