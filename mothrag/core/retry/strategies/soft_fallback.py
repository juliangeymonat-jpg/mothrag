# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""SoftFallbackStrategy (#7) — terminal guarantee of a non-empty answer.

Cost: 0 LLM calls. Picks the best-confidence non-empty prediction across
the arms already executed by ``_query_production`` (preference: iter > dec
> v3bu, since the iter arm has had the most refinement budget). Never
returns None; callers can rely on this as the cascade terminus.
"""

from __future__ import annotations

from mothrag.core.retry.protocol import RetryContext


def _is_uncertain(text: str) -> bool:
    if not text:
        return True
    t = text.lower().strip()
    return t in (
        "not in passages", "unknown", "no answer", "none",
        "i don't know", "i do not know", "n/a",
    )


class SoftFallbackStrategy:
    """Terminal fallback: pick best non-empty arm output."""

    name = "soft_fallback"
    cost_estimate = 0

    def applicable(self, ctx: RetryContext) -> bool:  # noqa: ARG002
        # Soft fallback is *always* applicable; the orchestrator already
        # ensures it runs last.
        return True

    def try_recover(self, ctx: RetryContext) -> str | None:
        # Preference order: iter > dec > v3bu (iter is the deepest-budget arm).
        for cand in (ctx.iter_pred, ctx.dec_pred, ctx.v3bu_pred):
            if cand and not _is_uncertain(cand):
                return cand
        # Original chosen if non-empty even if 'uncertain'-looking, else
        # surface the first non-empty raw arm output to keep telemetry
        # actionable.
        if ctx.chosen:
            return ctx.chosen
        for cand in (ctx.iter_pred, ctx.dec_pred, ctx.v3bu_pred):
            if cand:
                return cand
        return None


__all__ = ["SoftFallbackStrategy"]
