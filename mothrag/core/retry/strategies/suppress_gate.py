# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""SuppressGateStrategy -- P1 cascade short-circuit for gamma false-positives.

Failure-pattern mining established that 46% of
``gamma_final_status == "invalid"`` queries on the
abstention pool actually had F1 == 1. The gamma verifier is over-cautious:
on the largest single abstention cohort (1071 queries, 31.8% of the pool),
the substantive chosen answer is correct and any retry-cascade compute
spent on these queries is wasted.

SuppressGate is the pre-cascade gate that recognises this cohort and
short-circuits: if ``gamma`` flagged the answer invalid AND the answer
is NOT an uncertain-template ("Not in passages", "Unknown", "I don't
know", ...), the strategy returns the chosen pred unchanged and the
cascade stops with ``recovered_by="suppress_gate"``. No downstream
strategies fire, no LLM calls are spent.

This is a pure deterministic predicate; cost_estimate == 0 (no LLM).
Place SuppressGate FIRST in the cascade priority (before any LLM-spending
strategy) so the short-circuit fires before more expensive strategies
have a chance to apply.

Empirical coverage / precision (from failure-pattern mining on a 3368-entry pool):
  - coverage:                     1071 entries (31.8%)
  - F1=0 precision when fired:    0% (all 1071 are F1=1 -> correct answers)
  - estimated cascade compute saved:  ~30% of would-be invocations

Out-of-scope (NOT a SuppressGate concern):
  - gamma_status ``refuse`` (81% F1=0, cascade should fire)
  - gamma_status ``partial`` (97% F1=0, cascade should fire)
  - gamma_status ``valid`` co-firing with h3/h4/h12 (99% F1=0, cascade should fire)
  - Uncertain-template chosen pred (e.g. "Not in passages") -- the
    cascade may still recover by re-retrieving / decomposing

The gate uses :data:`UNCERTAIN_TEMPLATES` as the canonical refuse-string
list (shared with the other cascade strategies for consistency).
"""

from __future__ import annotations

import logging

from mothrag.core.retry.protocol import RetryContext

logger = logging.getLogger(__name__)


UNCERTAIN_TEMPLATES = (
    "not in passages",
    "unknown",
    "not enough information",
    "don't know",
    "do not know",
    "no answer",
    "insufficient",
    "cannot determine",
    "no answer found",
    "no relevant information",
    "no information",
    "n/a",
    "none",
    "unable to answer",
    "i cannot answer",
    "i cannot find",
    "i'm not sure",
    "i am not sure",
    "i don't have",
    "no info",
)

# Minimum normalised length below which a pred is treated as "no substance"
# regardless of template match (catches "no", "n", "  ", etc.).
_MIN_SUBSTANCE_LEN = 3


def pred_has_substance(pred: str) -> bool:
    """Public predicate shared by SuppressGate (P1) AND the #8/#9
    precision gate (empirical finding):
    strategies that REPLACE the original pred (not augment / re-rank)
    must decline when the original pred is substantive -- empirically,
    replacing a substantive partial-correct pred with an alternative on
    gamma=invalid causes -13 to -23pp F1 regression on the 0<F1<0.3
    mid-range cohort (HP T1 -21.76pp #8 / -18.75pp #9; 2W T1 -20.09pp /
    -22.79pp; MQ T1 -13.87pp / -16.42pp).

    Returns True when the pred carries enough surface content that any
    "replace" recovery strategy should defer to the original answer.
    """
    if not pred:
        return False
    norm = pred.strip().lower()
    if len(norm) < _MIN_SUBSTANCE_LEN:
        return False
    return not any(t in norm for t in UNCERTAIN_TEMPLATES)


def _is_uncertain_template(pred: str) -> bool:
    """True iff ``pred`` matches one of the canonical uncertain templates.

    Inverse of :func:`pred_has_substance`. Kept as a separate symbol for
    callers that want the "uncertain" framing rather than the "substantive"
    framing.
    """
    return not pred_has_substance(pred)


class SuppressGateStrategy:
    """Pre-cascade short-circuit for gamma=invalid + substantive answer.

    When applicable, returns the chosen pred unchanged. The cascade
    terminates with ``recovered_by="suppress_gate"`` and the original
    answer is preserved verbatim -- no LLM calls, no re-retrieval,
    no arm re-dispatch.

    Trade-off: this strategy biases toward trusting the substantive
    answer over the gamma flag on the cohort where gamma is empirically
    miscalibrated. On the 4% of cases where gamma=invalid + substantive
    pred IS a hallucination (empirically: 39% F1=0 within gamma=invalid
    overall, but only 0% within the "substantive pred" sub-cohort),
    the gate masks a real failure. The empirical 0% within-cohort F1=0
    rate makes this an acceptable trade for the ~30% compute saving.

    Implementation note: the strategy does NOT consume any cascade
    budget (``cost_estimate=0``). ``ctx.spend(0)`` is a no-op so the
    short-circuit is "free" w.r.t. downstream budget calculations.
    """

    name = "suppress_gate"
    cost_estimate = 0

    def applicable(self, ctx: RetryContext) -> bool:
        gamma_status = self._extract_gamma_status(ctx)
        if gamma_status != "invalid":
            return False
        if not ctx.chosen:
            return False
        if _is_uncertain_template(ctx.chosen):
            return False
        return True

    def try_recover(self, ctx: RetryContext) -> str | None:
        # No LLM call; just return the chosen pred unchanged.
        logger.debug(
            "suppress_gate: gamma_fp short-circuit fired on question %r; "
            "returning chosen pred unchanged.", ctx.question[:80],
        )
        return ctx.chosen

    @staticmethod
    def _extract_gamma_status(ctx: RetryContext):
        """Best-effort: surface gamma_status from c7_info dict or abstention_signal."""
        if isinstance(ctx.c7_info, dict):
            for key in ("gamma_status", "gamma", "gamma_final_status"):
                v = ctx.c7_info.get(key)
                if v is not None:
                    return v
        if ctx.abstention_signal == "gamma_refuse":
            return "invalid"
        return None


__all__ = ["SuppressGateStrategy", "UNCERTAIN_TEMPLATES", "pred_has_substance"]
