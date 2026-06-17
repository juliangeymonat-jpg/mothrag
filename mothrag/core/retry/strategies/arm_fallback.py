# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""ArmFallbackStrategy (#2) — re-route to a sibling arm when iter abstained.

Cost: 0 LLM calls (uses already-computed arm outputs).
"""

from __future__ import annotations

from mothrag.core.retry.protocol import RetryContext


def _ok(text: str | None) -> bool:
    if not text:
        return False
    t = text.lower().strip()
    return t not in ("not in passages", "unknown", "no answer", "none", "i don't know")


class ArmFallbackStrategy:
    """Pick the highest-quality sibling arm output when the chosen arm abstained."""

    name = "arm_fallback"
    cost_estimate = 0

    def applicable(self, ctx: RetryContext) -> bool:
        if ctx.abstention_signal not in (
            "iter_abstain", "h4_refuse", "h12_refuse", "empty_answer", "cross_arm_disagree"
        ):
            return False
        # Need at least one alternate non-empty arm output.
        return any(_ok(p) for p in (ctx.v3bu_pred, ctx.dec_pred, ctx.iter_pred))

    def try_recover(self, ctx: RetryContext) -> str | None:
        # Preference order: prefer a sibling arm to whatever produced the
        # chosen answer. We use a simple "longest unique non-empty" heuristic
        # in place of a calibrated confidence score (calibration is a
        # follow-up).
        candidates = [
            (ctx.iter_pred, "iter"),
            (ctx.dec_pred, "dec"),
            (ctx.v3bu_pred, "v3bu"),
        ]
        # Drop the arm whose output equals the chosen empty / uncertain answer.
        candidates = [(p, n) for p, n in candidates if _ok(p) and p != ctx.chosen]
        if not candidates:
            return None
        candidates.sort(key=lambda pair: len(pair[0] or ""), reverse=True)
        return candidates[0][0]


__all__ = ["ArmFallbackStrategy"]
