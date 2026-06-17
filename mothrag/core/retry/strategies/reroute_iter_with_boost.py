# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""RerouteIterWithBoostStrategy -- P3 handler for cross-arm disagreement
where gamma flagged the original answer as valid but all arms abstained.

Failure-pattern mining identified a small but 100%-F1=0-precision
cohort: queries where ``h3_fires`` is True (cross-arm disagreement
detection) AND ``gamma`` reports ``valid``. On these 20 queries (0.6%
of the abstention pool but 100% true failures), the routing classifier
correctly picked an arm a priori, gamma correctly assessed the candidate
as valid, but the chosen arm's output is "Not in passages" -- a
retrieval failure inside the chosen arm, not an arm-selection failure.

The recovery lever: re-run the iter arm with a higher
``bottom_up_boost`` factor (default 1.0 -> 1.5) so the bottom-up entity
boost during retrieval surfaces passages that the standard boost missed.
The arm SAME (iter) is re-invoked; no arm-swap, no sel_v2 override.

This mirrors the production lever that already exists
inside :class:`mothrag.eval.iterative_pipeline.IterativeMothRAG` --
the strategy just wires it as a deterministic post-hoc retry rather
than a per-query upfront knob.

Empirical coverage / precision:
  - coverage:                       20 entries (0.6%)
  - F1=0 precision when fired:    100% (20/20 are true failures)
  - uncertain pred share:         100% ("Not in passages" answers)
  - recovery potential:           novel passages from re-boosted retrieval

The strategy requires:
  - ``ctx.run_arm_iter`` wired (the iter runner from the production
    shim must accept a ``bottom_up_boost`` kwarg)

When the iter runner doesn't accept ``bottom_up_boost`` (older shims),
the strategy catches the TypeError and falls back to calling the iter
runner with default boost; the recovery surface degrades to a plain
iter re-run rather than no-op.
"""

from __future__ import annotations

import logging

from mothrag.core.retry.protocol import RetryContext

logger = logging.getLogger(__name__)


DEFAULT_BOOST_FACTOR = 1.5
DEFAULT_FALLBACK_BOOST_FACTOR = 2.0


def _is_uncertain(text: str) -> bool:
    if not text:
        return True
    t = text.lower().strip()
    return t in (
        "not in passages", "unknown", "no answer", "none",
        "i don't know", "i do not know", "n/a",
    )


class RerouteIterWithBoostStrategy:
    """Re-run iter arm with elevated bottom_up_boost.

    Parameters
    ----------
    boost_factor
        First-pass bottom_up_boost override (default 1.5; 
        production uses 1.0).
    fallback_boost_factor
        Second-pass override if the first-pass re-run still returns
        uncertain (default 2.0). Set to ``None`` to disable the second
        pass.
    """

    name = "reroute_iter_with_boost"
    cost_estimate = 2  # iter re-run is ~1 LLM call per iteration; cap at 2 for safety

    def __init__(
        self,
        *,
        boost_factor: float = DEFAULT_BOOST_FACTOR,
        fallback_boost_factor: float | None = DEFAULT_FALLBACK_BOOST_FACTOR,
    ) -> None:
        self.boost_factor = float(boost_factor)
        self.fallback_boost_factor = (
            float(fallback_boost_factor) if fallback_boost_factor is not None
            else None
        )

    def applicable(self, ctx: RetryContext) -> bool:
        if ctx.run_arm_iter is None:
            return False
        if not self._h3_signal_present(ctx):
            return False
        # The pattern fires when the chosen path admitted uncertainty
        # (so a re-run with elevated boost has something to improve over).
        if ctx.chosen and not _is_uncertain(ctx.chosen):
            return False
        return True

    def try_recover(self, ctx: RetryContext) -> str | None:
        if not ctx.spend(1):
            logger.debug(
                "reroute_iter_with_boost: budget exhausted before first-pass."
            )
            return None
        first = self._run_iter_with_boost(ctx, self.boost_factor)
        if first and not _is_uncertain(first):
            return first
        if self.fallback_boost_factor is None:
            return None
        if not ctx.spend(1):
            logger.debug(
                "reroute_iter_with_boost: budget exhausted before fallback-pass."
            )
            return None
        second = self._run_iter_with_boost(ctx, self.fallback_boost_factor)
        if second and not _is_uncertain(second):
            return second
        return None

    # ------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------

    def _h3_signal_present(self, ctx: RetryContext) -> bool:
        """Detect the h3 cross-arm-disagree pattern across multiple surfaces."""
        # 1. Explicit h3 marker in c7_info dict.
        if isinstance(ctx.c7_info, dict):
            if ctx.c7_info.get("h3_fires"):
                return True
            if ctx.c7_info.get("h3") is True:
                return True
        # 2. The "h3_gamma_valid_disagree_iter" marker as
        #    arbitrate_reason on the production stack.
        if ctx.arbitrate_reason and "h3" in ctx.arbitrate_reason.lower():
            return True
        # 3. Generic cross_arm_disagree abstention_signal also triggers
        #    the boost retry -- the iter arm gets the same elevated
        #    retrieval surface either way.
        if ctx.abstention_signal == "cross_arm_disagree":
            return True
        return False

    def _run_iter_with_boost(
        self, ctx: RetryContext, boost_factor: float,
    ) -> str:
        """Invoke run_arm_iter with the elevated boost.

        Older shims may not accept ``bottom_up_boost`` -- catch TypeError
        and fall back to a plain re-run (still useful because retrieval
        may now hit cached passages or new context).
        """
        kwargs = dict(
            question=ctx.question,
            passages=list(ctx.passages),
            q_emb=ctx.q_emb,
            top_k=ctx.top_k,
            bottom_up_boost=boost_factor,
        )
        try:
            return ctx.run_arm_iter(**kwargs) or ""
        except TypeError as exc:
            # Older shim without bottom_up_boost kwarg support; fall
            # back to plain call.
            if "bottom_up_boost" not in str(exc):
                raise
            logger.debug(
                "reroute_iter_with_boost: shim lacks bottom_up_boost; "
                "falling back to plain iter re-run.",
            )
            kwargs.pop("bottom_up_boost", None)
            try:
                return ctx.run_arm_iter(**kwargs) or ""
            except Exception as exc2:  # noqa: BLE001
                logger.warning(
                    "reroute_iter_with_boost: plain iter re-run failed: %s",
                    exc2,
                )
                return ""
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "reroute_iter_with_boost: iter re-run failed: %s", exc,
            )
            return ""


__all__ = ["RerouteIterWithBoostStrategy"]
