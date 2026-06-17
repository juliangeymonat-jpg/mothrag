# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""γ-weighted answer pooling for the Iterative Ragnatela.

Given the four arms' answers + per-answer γ, partition into HIGH (facts) / MID
(uncertain) / LOW (anti-context) bands and pool a single answer:

  * prefer the FACTS pool (HIGH band); fall back to UNCERTAIN, then ANTI-CONTEXT
    only if nothing better exists,
  * within the chosen pool, group answers that agree (normalised text) and pick
    the group with the greatest total γ-weight (so consensus among confident
    arms wins),
  * the pooled γ is that winning group's mean γ — the loop's convergence signal.

LOW-band (anti-context) answers never dominate the pool; they only survive as a
last resort and otherwise feed verification sub-questions. Anti-leak: text + γ
only, no gold.
"""
from __future__ import annotations

import re

from mothrag.iterative_ragnatela.types import (
    ArmAnswer,
    GammaBand,
    PoolOutcome,
    RagnatelaConfig,
)

_WS = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """Canonical form for answer-agreement grouping (lower + collapse ws)."""
    return _WS.sub(" ", (text or "").strip().lower())


def classify_band(gamma: float, cfg: RagnatelaConfig) -> GammaBand:
    """Map a γ score to its band per the config thresholds."""
    g = 0.0 if gamma < 0.0 else 1.0 if gamma > 1.0 else gamma
    if g >= cfg.gamma_high:
        return GammaBand.HIGH
    if g < cfg.gamma_low:
        return GammaBand.LOW
    return GammaBand.MID


def pool_answers(arm_answers, cfg: RagnatelaConfig) -> PoolOutcome:
    """γ-weighted pooling over the arms' answers. Returns a :class:`PoolOutcome`."""
    bands: dict[str, GammaBand] = {}
    high: list[ArmAnswer] = []
    mid: list[ArmAnswer] = []
    low: list[ArmAnswer] = []
    for a in arm_answers:
        band = classify_band(a.clamped_gamma(), cfg)
        bands[a.arm] = band
        (high if band is GammaBand.HIGH
         else mid if band is GammaBand.MID else low).append(a)

    # Prefer facts, then uncertain, then anti-context (degenerate last resort).
    pool = high or mid or low
    if not pool:
        return PoolOutcome(answer="", pooled_gamma=0.0, bands=bands,
                           high=high, mid=mid, low=low)

    # Group answers that agree; weight each group by total γ. Empty answers
    # contribute no weight but are remembered as a fallback.
    groups: dict[str, list] = {}      # norm -> [repr_answer, total_gamma, members]
    for a in pool:
        key = normalize_answer(a.answer)
        if not key:
            continue
        g = groups.setdefault(key, [a.answer, 0.0, []])
        g[1] += a.clamped_gamma()
        g[2].append(a)

    if not groups:
        # every pooled answer was empty — surface the strongest by γ.
        best = max(pool, key=lambda a: a.clamped_gamma())
        return PoolOutcome(answer=best.answer, pooled_gamma=best.clamped_gamma(),
                           bands=bands, high=high, mid=mid, low=low)

    # Winning group: max total γ-weight; deterministic tie-break by member
    # count then normalised text (stable across runs).
    win_key = max(
        groups,
        key=lambda k: (groups[k][1], len(groups[k][2]), k),
    )
    repr_answer, total_gamma, members = groups[win_key]
    pooled_gamma = total_gamma / len(members) if members else 0.0
    return PoolOutcome(answer=repr_answer, pooled_gamma=pooled_gamma,
                       bands=bands, high=high, mid=mid, low=low)


__all__ = ["normalize_answer", "classify_band", "pool_answers"]
