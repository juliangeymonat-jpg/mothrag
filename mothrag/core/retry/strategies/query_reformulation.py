# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""QueryReformulationStrategy (#3) — LAST-RESORT before SoftFallback.

Cost: 1 LLM call. Asks the reader to re-state the question more precisely
given the retrieved passages, then re-runs the v3bu arm on the reformulated
query. Guarded against infinite recursion by
:attr:`RetryContext.escalation_depth`.

This strategy is intentionally placed after the zero-LLM strategies in
:data:`mothrag.core.retry.orchestrator.DEFAULT_PRIORITY` so it only fires
when cheaper alternatives have all returned None.
"""

from __future__ import annotations

import logging

from mothrag.core.retry.protocol import RetryContext

logger = logging.getLogger(__name__)


_REFORMULATION_PROMPT = (
    "Re-state the following question more precisely so it can be answered "
    "from the retrieved passages. Output ONLY the rewritten question on a "
    "single line, no preamble.\n\n"
    "ORIGINAL QUESTION: {question}\n\n"
    "PASSAGE EXCERPTS (first 3, truncated):\n{passages}\n\n"
    "REWRITTEN QUESTION:"
)


class QueryReformulationStrategy:
    """Ask the reader to rewrite the question, then re-run v3bu."""

    name = "query_reformulation"
    cost_estimate = 2   # reformulation call + recovery read call

    def __init__(self, max_recursion_depth: int = 1) -> None:
        self.max_recursion_depth = max_recursion_depth

    def applicable(self, ctx: RetryContext) -> bool:
        if ctx.reader is None or ctx.run_arm_v3bu is None:
            return False
        if ctx.escalation_depth >= self.max_recursion_depth:
            return False
        if not ctx.passages:
            return False
        return ctx.abstention_signal in (
            "gamma_refuse", "iter_abstain", "h4_refuse",
            "cross_arm_disagree", "empty_answer",
        )

    def try_recover(self, ctx: RetryContext) -> str | None:
        if not ctx.spend(self.cost_estimate):
            logger.debug("query_reformulation: cascade budget exhausted; skipping.")
            return None

        passage_excerpt = "\n".join(p[:400] for p in ctx.passages[:3])
        prompt = _REFORMULATION_PROMPT.format(
            question=ctx.question, passages=passage_excerpt,
        )
        try:
            reformulated = ctx.reader.read(prompt, ctx.passages[:3]).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("query_reformulation reader call failed: %s", exc)
            return None
        # Take the first non-empty line.
        reformulated = next(
            (line.strip() for line in reformulated.splitlines() if line.strip()),
            "",
        )
        # Sanity: don't accept a "rewrite" identical to the original or empty.
        if not reformulated or reformulated.strip().lower() == ctx.question.strip().lower():
            return None
        ctx.escalation_depth += 1
        try:
            answer = ctx.run_arm_v3bu(question=reformulated, passages=ctx.passages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("query_reformulation v3bu re-run failed: %s", exc)
            return None
        if answer:
            return answer
        return None


__all__ = ["QueryReformulationStrategy"]
