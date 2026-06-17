# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
r"""Per-arm applicability predicates for MOTHRAG pool composition.

Per the feature-audit Phase 6 verdict (correlation
feature × arm_winner on HP/2W/MQ T1, n≈1000/DS), no single feature
crossed the strong-predictive gate (|corr| ≥ 0.20 on at least 1 DS).
However, FIVE features cleared the reliability gate (P ≥ 0.85 AND
R ≥ 0.70 AND F1 ≥ 0.85) in the labeling audit (n=50):

  single_hop          F1 1.00 / P 1.00 / R 1.00  → InfoboxArm
  is_polar_comparison F1 1.00 / P 1.00 / R 1.00  → V3+bu
  two_entity          F1 0.91 / P 0.91 / R 0.91  → decompose
  chain_marker        F1 0.90 / P 1.00 / R 0.82  → iter
  comparison_marker   F1 0.89 / P 1.00 / R 0.80  → (also V3+bu, wired
                                                      in PAM-lite)

These are the SOLID features safe to use as per-arm
``applicable(question)`` triggers in a component-autonomy
architecture. The HYBRID architecture verdict
keeps PAM-orchestrator for the routing-by-winner-prediction loop,
but allows component-autonomy on ``applicable()`` (a precision-
oriented predicate: false positives matter; false negatives are
just missed opportunity).

Pool-safety integration
-----------------------

When :class:`mothrag.core.arbitrate.ArbitratorV2.arbitrate_pool` is
called, it consults each arm's ``applicable(question)`` via the
``arm_applicability`` parameter. Arms with ``applicable=False`` are
filtered from the firing subset BEFORE any signal composition; they
contribute zero weight to the arbitration (pool-safety axiom).

The helpers in this module are STANDALONE predicates so legacy
function-based arms (V3+bu / decompose / iter, executed via the
``_run_*`` runners in ``scripts/route_prospective.py``) can be gated
without a full class-based migration. The matching :class:`Arm`-Protocol
wrappers live in :mod:`mothrag.arms.legacy`.

All predicates are linguistic regex over question text. No training,
no per-dataset tuning, no test inspection.
"""

from __future__ import annotations

from mothrag.core.query_type_classifier import is_polar_comparison
from mothrag.routing.semantic_features import extract_semantic_features


def is_v3bu_applicable(question: str, features=None) -> bool:
    """V3+bu wins on polar / set-comparison questions.

    Feature: ``is_polar_comparison`` (F1=1.00 in n=50 audit).
    Empirically the ONLY positive cross-DS correlation in Phase 6
    (HP +0.085 / 2W +0.153 → V3+bu).

    NB this is a SUFFICIENT condition for V3+bu autonomy, not a
    necessary one. Legacy production routing still runs V3+bu in
    many other contexts (default semantic_rich qtype). This
    predicate exists for MOTHRAG component-autonomy where
    V3+bu acts as a self-electing arm on its signature cohort.
    """
    if not question or not question.strip():
        return False
    return is_polar_comparison(question)


def is_decompose_applicable(question: str, features=None) -> bool:
    """decompose wins on two-entity bridge questions.

    Feature: ``two_entity`` (F1=0.91 in n=50 audit). Cohort:
    questions with exactly two distinct capitalized noun phrases
    (e.g. "Who is X's spouse where Y was born?" -- two entities X
    and Y bridged via a relation).
    """
    if not question or not question.strip():
        return False
    f = features or extract_semantic_features(question)
    return f.two_entity > 0


def is_iter_applicable(question: str, features=None) -> bool:
    """iter wins on explicit chain / temporal-order questions.

    Feature: ``chain_marker`` (F1=0.90 in n=50 audit; P=1.00 R=0.82).
    Cohort: questions with explicit chain lexicon (first / then /
    later / subsequently / after that / prior to / following /
    next), indicating multi-step temporal reasoning.
    """
    if not question or not question.strip():
        return False
    f = features or extract_semantic_features(question)
    return f.chain_marker > 0


def is_infobox_arm_applicable(question: str, features=None) -> bool:
    """InfoboxArm wins on single-hop possessive entity-attribute lookups.

    Feature: ``single_hop`` (F1=1.00 in n=50 audit). Cohort:
    possessive forms like "X's Y is Z" -- structured-fact lookup
    where the entity-attribute triple IS the answer.

    NB :class:`mothrag.arms.InfoboxArm` ALSO exposes
    ``InfoboxArm.applicable(question)`` based on
    :func:`mothrag.core.retrieval.extract_question_hints` (hint
    extraction). The two predicates are related but not
    identical -- ``single_hop`` is BROADER (any "X's Y" form)
    while ``extract_question_hints`` is NARROWER (must match one
    of 7 entity-attribute patterns). For sel_v2-style routing,
    prefer ``single_hop`` (broader recall on the high-precision
    cohort); for the InfoboxArm execution gate, prefer the
    arm's own ``applicable`` (avoids running the lookup when no
    hint matches).
    """
    if not question or not question.strip():
        return False
    f = features or extract_semantic_features(question)
    return f.single_hop > 0


# Registry: arm_name -> applicability predicate.
APPLICABILITY_PREDICATES = {
    "v3bu": is_v3bu_applicable,
    "decompose": is_decompose_applicable,
    "iter": is_iter_applicable,
    "infobox_arm": is_infobox_arm_applicable,
}


def applicability_snapshot(question: str, arms_pool):
    """Compute ``{arm_name: bool}`` snapshot for the arms in ``arms_pool``.

    Used by :meth:`mothrag.core.arbitrate.ArbitratorV2.arbitrate_pool`
    via its ``arm_applicability`` parameter. Arms without a registered
    predicate default to ``True`` (assume applicable; the arm's own
    ``Arm.applicable`` or downstream filters then decide).
    """
    features = extract_semantic_features(question)
    out: dict[str, bool] = {}
    for arm in arms_pool:
        pred = APPLICABILITY_PREDICATES.get(arm)
        if pred is None:
            out[arm] = True
        else:
            out[arm] = pred(question, features)
    return out


__all__ = [
    "APPLICABILITY_PREDICATES",
    "applicability_snapshot",
    "is_decompose_applicable",
    "is_infobox_arm_applicable",
    "is_iter_applicable",
    "is_v3bu_applicable",
]
