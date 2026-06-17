# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""IterBudgetExtensionStrategy (#1) — re-run iter arm with a wider budget.

Cost: 0 LLM calls (uses the existing reader / vector store; the additional
iter steps are charged against the reader's normal pipeline cost, not the
orchestrator's LLM-call budget).
"""

from __future__ import annotations

import logging

from mothrag.core.retry.protocol import RetryContext

logger = logging.getLogger(__name__)


class IterBudgetExtensionStrategy:
    """Re-run the iter arm with extended ``max_iter_steps`` + ``top_k``."""

    name = "iter_extension"
    cost_estimate = 0   # additional reader calls are inside the iter arm itself

    def applicable(self, ctx: RetryContext) -> bool:
        if ctx.run_arm_iter is None:
            return False
        if ctx.abstention_signal not in ("gamma_refuse", "iter_abstain", "empty_answer"):
            return False
        # Only fire if the iter arm was actually run and abstained or returned empty.
        return (ctx.iter_pred is None) or (not ctx.iter_pred) or _looks_uncertain(ctx.iter_pred)

    def try_recover(self, ctx: RetryContext) -> str | None:
        cur_max = int(ctx.config.get("max_iter_steps", 3))
        cur_top_k = int(ctx.top_k)
        ext_max = max(cur_max * 2, cur_max + 4)
        ext_top_k = max(int(cur_top_k * 1.5), cur_top_k + 5)
        try:
            answer = ctx.run_arm_iter(
                question=ctx.question,
                passages=ctx.passages,
                q_emb=ctx.q_emb,
                top_k=ext_top_k,
                max_steps=ext_max,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("iter_extension recovery raised %s; skipping.", exc)
            return None
        if answer and not _looks_uncertain(answer):
            return answer
        return None


def _looks_uncertain(text: str) -> bool:
    if not text:
        return True
    t = text.lower().strip()
    return t in ("not in passages", "unknown", "no answer", "none", "i don't know")


__all__ = ["IterBudgetExtensionStrategy"]
