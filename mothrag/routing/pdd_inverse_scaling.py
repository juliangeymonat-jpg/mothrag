# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""PDD reader-inverse-scaling routing signal.

Empirical finding: the PDD (Pool-Diversity
Dispatch / ``iter_dup_a``) F1=0-cohort lift is INVERSELY proportional to the
reader's baseline capability on hard queries — it is an *arbitration prosthesis
for weak readers*. HP T1 F1=0 cohort, n=30:

    reader hard-cohort baseline F1 → PDD Δ
      Llama-3.3-70B   0.0222 →  +18.29 pp
      Llama-3.1-8B    0.2034 →   +1.85 pp
      GPT-4o          0.2533 →   +0.67 pp

The lift tracks *how much room there is to rescue*: a deeply-broken reader
(baseline ≈ 0) has huge headroom; a capable reader has already solved most of
the cohort, so PDD adds little. This also explains the apparent Llama-8B
"anomaly" (weakest model, mid lift): its hard-cohort baseline (0.20) is HIGHER
than Llama-70B's (0.02), so less headroom → the relationship is monotone in
BASELINE, not model size.

This module turns that into a routing signal: given an estimate of the reader's
capability on the current (hard) query — a runtime confidence signal and/or a
per-reader prior — it dials the PDD **cardinality UP** for weak readers (more
``iter_dup`` arms = stronger prosthesis) and leaves it at the locked base for
capable readers. PDD is NEVER dialled below its locked base (``iter_dup_a`` stays
in the pool per the cardinality-bounded asymmetric-amplification LOCK); this only
ADDS extra dups where the evidence says they pay off.

Anti-leak: keyed on reader capability (a property of the READER / its per-query
confidence), never on the dataset. General-purpose across readers + corpora.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Capability axis ∈ [0, 1] = estimated P(reader answers this hard query right).
# Anchors = observed (hard-cohort baseline F1, PDD Δpp), bracketed by
# the limiting endpoints (capability 0 → max headroom; capability 1 → none).
_LIFT_ANCHORS: tuple[tuple[float, float], ...] = (
    (0.0, 18.29),
    (0.0222, 18.29),   # Llama-3.3-70B
    (0.2034, 1.85),    # Llama-3.1-8B
    (0.2533, 0.67),    # GPT-4o
    (1.0, 0.0),
)
_MAX_LIFT_PP = 18.29

# Per-reader priors = the hard-cohort baseline F1 (substring-matched on
# the reader model id). A capable reader has a HIGHER prior (less PDD headroom).
READER_CAPABILITY_PRIOR: dict[str, float] = {
    "gpt-4o": 0.2533,
    "gpt-4": 0.2533,
    "gpt-3.5": 0.18,
    "llama-3.1-8b": 0.2034,
    "llama-3.1-8b-instant": 0.2034,
    "llama-3-8b": 0.2034,
    "llama-3.3-70b": 0.0222,
    "llama-3.1-70b": 0.05,
    "llama-3-70b": 0.05,
    "gemini-2.5-flash": 0.12,
    "gemini-flash": 0.12,
    "gemini-2.5-pro": 0.22,
}


@dataclass
class PDDScalingConfig:
    """Knobs for the reader-inverse PDD scaling. All keyword-defaulted."""

    base_cardinality: int = 1          # the LOCKED iter_dup_a — never go below
    max_cardinality: int = 3           # cap on total dups (anti-runaway)
    min_weight: float = 1.0            # locked-base PDD arbitration weight
    max_weight: float = 2.0            # max amplification for the weakest reader
    # Add EXTRA dups only when the expected lift clears this bar (anti-waste:
    # the extra arm has compute cost; a ~marginal reader keeps the base).
    extra_dup_threshold_pp: float = 3.0
    # Prior used when neither a runtime confidence nor a known reader is given.
    # Mid-weak (0.10) → assume some prosthesis value rather than none.
    default_capability: float = 0.10


@dataclass
class PDDRoutingDecision:
    """Outcome of the PDD reader-inverse routing for one query / reader."""

    capability: float
    expected_lift_pp: float
    intensity: float        # lift normalised to [0, 1]
    pdd_weight: float       # arbitration weight for the dup arm(s)
    pdd_cardinality: int    # how many iter_dup arms to dispatch (>= base)
    add_extra_dups: bool    # True iff cardinality > base_cardinality
    capability_source: str  # "runtime" | "reader_prior:<name>" | "default"


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def expected_pdd_lift_pp(capability: float) -> float:
    """Piecewise-linear interpolation of the expected PDD Δpp at ``capability``."""
    c = _clamp01(capability)
    anchors = _LIFT_ANCHORS
    if c <= anchors[0][0]:
        return anchors[0][1]
    if c >= anchors[-1][0]:
        return anchors[-1][1]
    for (c0, l0), (c1, l1) in zip(anchors, anchors[1:]):
        if c0 <= c <= c1:
            if c1 == c0:
                return l1
            frac = (c - c0) / (c1 - c0)
            return l0 + frac * (l1 - l0)
    return anchors[-1][1]  # pragma: no cover


def pdd_intensity(capability: float) -> float:
    """Expected lift normalised to [0, 1] (1 = weakest reader, max prosthesis)."""
    return _clamp01(expected_pdd_lift_pp(capability) / _MAX_LIFT_PP)


def pdd_weight(capability: float, cfg: Optional[PDDScalingConfig] = None) -> float:
    """Continuous PDD arbitration weight ∈ [min_weight, max_weight]."""
    cfg = cfg or PDDScalingConfig()
    return cfg.min_weight + (cfg.max_weight - cfg.min_weight) * pdd_intensity(capability)


def pdd_cardinality(capability: float, cfg: Optional[PDDScalingConfig] = None) -> int:
    """Number of ``iter_dup`` arms to dispatch (>= locked base)."""
    cfg = cfg or PDDScalingConfig()
    if expected_pdd_lift_pp(capability) < cfg.extra_dup_threshold_pp:
        return cfg.base_cardinality
    headroom = max(0, cfg.max_cardinality - cfg.base_cardinality)
    extra = round(headroom * pdd_intensity(capability))
    return cfg.base_cardinality + extra


def estimate_reader_capability(reader_model: Optional[str]) -> Optional[float]:
    """Per-reader prior capability (substring match), or None if unknown."""
    if not reader_model:
        return None
    key = reader_model.strip().lower()
    # Longest matching substring wins (so 'llama-3.3-70b' beats 'llama-3').
    best: Optional[tuple[int, float]] = None
    for name, cap in READER_CAPABILITY_PRIOR.items():
        if name in key and (best is None or len(name) > best[0]):
            best = (len(name), cap)
    return best[1] if best is not None else None


class PDDRouter:
    """Reader-inverse PDD routing. Prefers a runtime confidence signal, falls
    back to a per-reader prior, then the config default."""

    def __init__(self, cfg: Optional[PDDScalingConfig] = None) -> None:
        self.cfg = cfg or PDDScalingConfig()

    def route(
        self,
        *,
        reader_confidence: Optional[float] = None,
        reader_model: Optional[str] = None,
    ) -> PDDRoutingDecision:
        cfg = self.cfg
        if reader_confidence is not None:
            capability = _clamp01(reader_confidence)
            source = "runtime"
        else:
            prior = estimate_reader_capability(reader_model)
            if prior is not None:
                capability = _clamp01(prior)
                source = f"reader_prior:{reader_model}"
            else:
                capability = _clamp01(cfg.default_capability)
                source = "default"

        lift = expected_pdd_lift_pp(capability)
        card = pdd_cardinality(capability, cfg)
        return PDDRoutingDecision(
            capability=round(capability, 4),
            expected_lift_pp=round(lift, 2),
            intensity=round(pdd_intensity(capability), 4),
            pdd_weight=round(pdd_weight(capability, cfg), 4),
            pdd_cardinality=card,
            add_extra_dups=card > cfg.base_cardinality,
            capability_source=source,
        )


def route_pdd(
    *,
    reader_confidence: Optional[float] = None,
    reader_model: Optional[str] = None,
    cfg: Optional[PDDScalingConfig] = None,
) -> PDDRoutingDecision:
    """Convenience one-shot wrapper around :class:`PDDRouter`."""
    return PDDRouter(cfg).route(
        reader_confidence=reader_confidence, reader_model=reader_model)


__all__ = [
    "PDDScalingConfig",
    "PDDRoutingDecision",
    "PDDRouter",
    "route_pdd",
    "expected_pdd_lift_pp",
    "pdd_intensity",
    "pdd_weight",
    "pdd_cardinality",
    "estimate_reader_capability",
    "READER_CAPABILITY_PRIOR",
]
