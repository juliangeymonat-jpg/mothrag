# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""L4bAnchorRetryStrategy (#6) — swap L4b temporal anchor + retry iter.

Cost: 0 LLM calls.

L4b is the temporal-stability primitive in the full iter
pipeline (see :mod:`mothrag.eval.iterative_pipeline`); it cancels iterations
whose anchor cosine drift exceeds the temporal threshold. When L4b fires
and cancels the iter arm, this strategy retries the arm with the *second-
highest-cosine* anchor instead of the rank-1 anchor.

Activation timeline:

* **v0.5.0 alpha**: the strategy was instantiable but
  silently dead because :meth:`MothRAG._query_production` never populated
  ``c7_info["l4b"]`` and :meth:`MothRAG._run_arm_iter_for_retry` declared
  but discarded the ``l4b_anchor`` kwarg. The fix: the alpha pipeline now
  populates
  ``c7_info["l4b"] = {cancelled: bool, anchors: [int], alpha_substitute: True}``
  with anchors being positional indices into the shared retrieved
  passages, and ``_run_arm_iter_for_retry`` honours ``l4b_anchor`` by
  moving the matching passage to index 0 before delegating to the iter
  arm. The retry strategy therefore now executes whenever the iter arm
  returns empty and there are ≥ 2 retrieved passages.
* **v0.5.1+ (deferred)**: when the full L4b temporal cancellation logic
  lands inside :meth:`MothRAG._arm_iter`, the same context-shape contract
  is honoured — anchors will become embedding-derived chunk identifiers
  and ``alpha_substitute`` will be set to False.
"""

from __future__ import annotations

import logging

from mothrag.core.retry.protocol import RetryContext

logger = logging.getLogger(__name__)


class L4bAnchorRetryStrategy:
    """Re-run iter arm with an alternate L4b temporal anchor."""

    name = "l4b_anchor_retry"
    cost_estimate = 0

    def applicable(self, ctx: RetryContext) -> bool:
        if ctx.run_arm_iter is None:
            return False
        if ctx.abstention_signal not in ("iter_abstain", "gamma_refuse"):
            return False
        info = self._l4b_info(ctx)
        if info is None:
            return False
        # Fires only if L4b actually cancelled and we have a runner-up anchor.
        anchors = info.get("anchors") or []
        return bool(info.get("cancelled")) and len(anchors) >= 2

    def try_recover(self, ctx: RetryContext) -> str | None:
        info = self._l4b_info(ctx)
        if info is None:
            return None
        anchors = info.get("anchors") or []
        # Use second-best anchor (index 1), skipping the rank-1 anchor that
        # was already tried.
        if len(anchors) < 2:
            return None
        alt_anchor = anchors[1]
        try:
            answer = ctx.run_arm_iter(
                question=ctx.question,
                passages=ctx.passages,
                q_emb=ctx.q_emb,
                top_k=ctx.top_k,
                max_steps=ctx.config.get("max_iter_steps", 3),
                l4b_anchor=alt_anchor,
            )
        except TypeError:
            # The alpha run_arm_iter signature does not yet accept
            # ``l4b_anchor``; degrade silently.
            logger.debug("l4b_anchor_retry: run_arm_iter lacks l4b_anchor; skipping.")
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("l4b_anchor_retry runtime error: %s", exc)
            return None
        if answer:
            return answer
        return None

    @staticmethod
    def _l4b_info(ctx: RetryContext) -> dict | None:
        """Extract L4b state from c7_info if present.

        Expected shape (post-v0.5.1):
        ``c7_info = {"l4b": {"cancelled": bool, "anchors": list[str], ...}}``
        """
        if not ctx.c7_info or not isinstance(ctx.c7_info, dict):
            return None
        l4b = ctx.c7_info.get("l4b")
        if not isinstance(l4b, dict):
            return None
        return l4b


__all__ = ["L4bAnchorRetryStrategy"]
