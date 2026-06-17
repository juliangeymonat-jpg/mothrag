# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Generalized signal-level dup primitive for PDD universality testing.

Earlier empirical work showed Pool Diversity Dispatch (PDD) lifts F1=0
cohort recovery via the
:func:`mothrag.core.arbitrate.signals.pairwise_agreement` aggregator's
``N-1`` denominator effect.

This module probes the universality claim:
"does the fantoccio (dup primitive) work only on arms, or can it
amplify any consensus aggregator?"

Falsifiable structural hypothesis:

  PDD amplifies signal S via duplication iff the consensus aggregator
  AGG(S_1, ..., S_N) computes a cardinality-normalized average
  (denominator = N or N-1). It does NOT amplify when AGG uses a
  fixed-weight sum without cardinality normalization.

This module provides:

  * :func:`dup_signal_into_aggregator` -- a generalized dup primitive
    that wraps any per-voter signal mapping ``{voter_id: signal}`` and
    returns ``{voter_id: signal, dup_voter_id: signal_of_base}``. The
    dup deterministically mirrors the base voter's signal.

  * :func:`pdd_lift_predicted` -- structural classifier: returns True
    iff the named aggregator's formula contains a cardinality
    denominator (lookup table; no F1 inspection).

  * :func:`apply_cardinality_average` -- demonstration aggregator
    used in tests to show the PDD lift propagates analytically.

Anti-leak contract: ALL signal manipulation here is deterministic
mirroring of a base voter. NO gold inspection. NO F1-derived
constants. The cardinality-vs-fixed-weight distinction is a pure
mathematical property of the aggregator code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


__all__ = [
    "DupSignalResult",
    "dup_signal_into_aggregator",
    "pdd_lift_predicted",
    "apply_cardinality_average",
    "apply_fixed_weighted_sum",
    "AGGREGATOR_NORMALIZATION_TABLE",
]


# Lookup table of known consensus aggregators in mothrag (from
# code archaeology):
#
#   pairwise_agreement     : cardinality-normalized (denom = N-1).
#                            PDD predicted to AMPLIFY.
#   agreement_per_aspect   : same N-1 denom at aspect granularity.
#                            PDD predicted to AMPLIFY.
#   DeterministicArbitrator: fixed-weight sum (w_gamma * gamma +
#                            w_agree * agreement + w_faith * faith).
#                            NOT normalized over N voters.
#                            PDD predicted to NOT amplify directly.
#   gamma_l4b_andgate      : binary 2-signal AND. Not multi-voter.
#                            PDD predicted NOT APPLICABLE.
#   MultiModalRetriever    : additive prepend (infobox + dense). Not
#                            voting consensus. PDD predicted NOT
#                            APPLICABLE.
AGGREGATOR_NORMALIZATION_TABLE: dict[str, str] = {
    "pairwise_agreement":      "cardinality_normalized",
    "agreement_per_aspect":    "cardinality_normalized",
    "DeterministicArbitrator": "fixed_weighted_sum",
    "ArbitratorV2":            "fixed_weighted_sum",
    "gamma_l4b_andgate":       "boolean_and",
    "MultiModalRetriever":     "additive_prepend",
}


@dataclass(frozen=True)
class DupSignalResult:
    """Outcome of dup-signal injection.

    ``extended_signals`` is the original ``base_signals`` with the dup
    voter added (``dup_voter_id``: same value as ``base_voter_id``).
    """

    extended_signals: Mapping[str, float]
    base_voter_id: str
    dup_voter_id: str
    base_value: float


def dup_signal_into_aggregator(
    signals: Mapping[str, float],
    base_voter_id: str,
    dup_voter_id: str,
) -> DupSignalResult:
    """Inject a deterministic mirror of ``base_voter_id`` as
    ``dup_voter_id`` into the ``signals`` mapping.

    Args:
        signals: original per-voter signal mapping
            ``{voter_id: signal_value_in_[0,1]}``.
        base_voter_id: the voter whose signal is mirrored.
        dup_voter_id: the new key under which the mirror is stored.
            MUST be distinct from ``base_voter_id`` and MUST NOT
            already exist in ``signals``.

    Returns:
        :class:`DupSignalResult` with ``extended_signals`` carrying
        ``base_voter_id`` -> original value AND ``dup_voter_id`` ->
        same value.

    Raises:
        KeyError: if ``base_voter_id`` not in ``signals``.
        ValueError: if ``dup_voter_id`` collides with an existing key
            or equals ``base_voter_id``.
    """
    if base_voter_id not in signals:
        raise KeyError(
            f"base_voter_id {base_voter_id!r} not in signals "
            f"{list(signals.keys())}"
        )
    if dup_voter_id == base_voter_id:
        raise ValueError("dup_voter_id must differ from base_voter_id")
    if dup_voter_id in signals:
        raise ValueError(
            f"dup_voter_id {dup_voter_id!r} already exists in signals"
        )
    base_value = float(signals[base_voter_id])
    extended = dict(signals)
    extended[dup_voter_id] = base_value
    return DupSignalResult(
        extended_signals=extended,
        base_voter_id=base_voter_id,
        dup_voter_id=dup_voter_id,
        base_value=base_value,
    )


def apply_cardinality_average(signals: Mapping[str, float]) -> float:
    """Cardinality-normalized average aggregator.

    Returns ``sum(signals.values()) / len(signals)``. Reference
    implementation that PDD predicted to amplify under dup injection
    (numerator gains +base_value, denominator gains +1, net effect
    pulls the average toward base_value when base_value differs from
    the average of others).
    """
    if not signals:
        return 0.0
    return sum(signals.values()) / len(signals)


def apply_fixed_weighted_sum(
    signals: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    """Fixed-weight sum aggregator (no cardinality normalization).

    Returns ``sum(weights[v] * signals[v] for v in signals)``.
    Voters missing from ``weights`` default to weight 1.0. Reference
    implementation that PDD predicted to NOT amplify directly under
    dup injection (the dup voter contributes a NEW term to the sum,
    but the score scaling is dominated by the fixed weight on the
    dup voter -- when weight_dup = weight_base, the score grows
    linearly with N, not relative to other arms; when weight_dup = 0
    (default for unseen voters), the dup contributes nothing).
    """
    return sum(weights.get(v, 1.0) * signals[v] for v in signals)


def pdd_lift_predicted(aggregator_name: str) -> bool:
    """Structural classifier: is the named aggregator predicted to
    amplify under dup-signal injection?

    Returns True iff the aggregator's normalization mode is
    ``"cardinality_normalized"``. Unknown aggregator names return
    False (fail-safe to "no lift predicted").

    Anti-leak: prediction is from
    :data:`AGGREGATOR_NORMALIZATION_TABLE` lookup; no F1 inspection.
    """
    mode = AGGREGATOR_NORMALIZATION_TABLE.get(aggregator_name, "unknown")
    return mode == "cardinality_normalized"
