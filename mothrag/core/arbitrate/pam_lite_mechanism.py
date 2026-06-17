# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
r"""PAM-lite arbitrator mechanism walkthrough.

In-code analytical walkthrough of the arbitrate path that produced the
empirical Pool Diversity Dispatch (PDD) signal observed on an F1=0
cohort across HP / 2W / MQ with the dup-v3bu helper.

Scope: this module supplies the deterministic mechanism trace +
formulas + N-dependencies for each step.

----------------------------------------------------------------------
Step 1: per-arm sigmoid scoring (pre-normalization)
----------------------------------------------------------------------

Source: :func:`mothrag.core.query_type_classifier.arm_subset_pam_lite`
        -> per-arm ``_score_<arm>_p_arm(features)`` helpers.

Formula (per arm A, given features f and hop-class hop):
    raw_A(f, hop) = sigmoid(w_A . f + b_A) * mu_A(hop)

where
    sigmoid(x) = 1 / (1 + e^-x)
    w_A, b_A    = hand-derived linear coefficients (NO training, NO
                  F1 fitting). Source: feature catalogue in
                  :mod:`mothrag.routing.semantic_features`.
    mu_A(hop)   = hop multiplier (specialist 2.0 /
                  anti-specialist 0.1 / neutral 1.0); audited
                  theory-derived.

N-dependency: NONE at this step. Each arm scores independently.
PDD contribution at step 1: ZERO. Adding more arms to the pool does
NOT change any existing arm's raw score.

----------------------------------------------------------------------
Step 2: NO softmax normalization
----------------------------------------------------------------------

Source: same module returns the raw_A values directly as the ``P_arm``
mapping. No softmax, no projection onto the simplex.

Implication for PDD: a frequent hypothesis is "adding more arms flattens
softmax distribution and demotes the leader". That hypothesis does NOT
apply here -- the code path is NOT a softmax router. P_arm values are
independent per-arm scalars in roughly [0, 2] (post-multiplier).

N-dependency: NONE. PDD contribution at step 2: ZERO.

----------------------------------------------------------------------
Step 3: variable-K subset filter
----------------------------------------------------------------------

Source: ``arm_subset_pam_lite`` returns ``[arm | P_arm > threshold]``
with the always-non-empty guarantee (single argmax fallback when every
P_arm is below threshold).

Formula:
    subset(P_arm, theta) = { A : P_arm[A] > theta }
    if subset is empty: subset = { argmax_A P_arm[A] }

N-dependency: WEAK. Independent per-arm test; the threshold is fixed,
the existing arms' inclusion status is unchanged when a new arm is
added. The only effect: a new arm may itself qualify and enter the
subset.

PDD contribution at step 3: small and indirect. A new arm name in the
subset changes which arms execute downstream, but the existing arms
still execute as before.

----------------------------------------------------------------------
Step 4: arm execution
----------------------------------------------------------------------

Source: ``_run_ensemble_arbitrate`` in ``scripts/route_prospective.py``
        runs each subset arm; per the pool-safety axiom
        the dup-arm helper re-uses the BASE arm's
        execution result for dup names (no duplicate API call). So
        candidates[``v3bu_dup_a``] is a literal alias of
        candidates[``v3bu``] (same prediction text).

N-dependency at step 4: cost-linear; mechanism-neutral for the F1
signal as long as the dup re-uses the base result.

PDD contribution at step 4: ZERO at the per-arm-prediction level.

----------------------------------------------------------------------
Step 5: pairwise cross-arm agreement (KEY MECHANISM STEP)
----------------------------------------------------------------------

Source: :func:`mothrag.core.arbitrate.signals.pairwise_agreement`.

Formula (per arm i with answer a_i, against the pool of N arms):
    agree(i) = |{j != i : non_empty(a_j) and cos(embed(a_i), embed(a_j))
                          >= tau}|
               / |{j != i : non_empty(a_j)}|

with tau = 0.70 (cosine threshold, on normalized embeddings).

Critical N-dependency:
    The DENOMINATOR is N-1 (count of other non-empty arms).
    The NUMERATOR counts SEMANTIC-AGREEING others.

When the pool is extended with a dup-v3bu (whose answer is BYTE-
IDENTICAL to v3bu's), the cosine similarity cos(v3bu, v3bu_dup_a) = 1.0
exceeds tau trivially. Therefore:

    agree(v3bu) | with dup    = (1 + agree_count_other) / (N-1)
    agree(v3bu) | without dup = agree_count_other      / (N-2)

For HP/2W F1=0 cohort where v3bu's answer typically disagrees with
decompose / iter (agree_count_other = 0):

    agree(v3bu) | with dup    = 1 / 3 = 0.333
    agree(v3bu) | without dup = 0 / 2 = 0.000
    delta on v3bu agreement   = +0.333 (guaranteed lift)

For decompose / iter, whose denominator goes from N-2=2 to N-1=3 but
whose numerator (matches against v3bu and v3bu_dup_a) stays at most
the same:

    if decompose agrees with iter only:
        agree(decompose) | with dup    = 1 / 3 = 0.333
        agree(decompose) | without dup = 1 / 2 = 0.500
        delta on decompose agreement   = -0.167

The relative ranking of v3bu's score vs decompose/iter therefore tilts
toward v3bu by ~ 0.5 (delta_v3bu - delta_decompose ~= 0.333 + 0.167 = 0.5)
on the per-arm agreement axis.

Hypothesized PDD lift contribution at step 5: PRIMARY.

----------------------------------------------------------------------
Step 6: weighted score combination
----------------------------------------------------------------------

Source: :class:`mothrag.core.arbitrate.arbitrator.DeterministicArbitrator`.

Formula:
    score(arm) = P_arm[arm] * (w_gamma * gamma(arm)
                              + w_agree * agree(arm)
                              + w_faith * faith(arm))

Default weights:
    w_gamma = 1.0
    w_agree = 0.5
    w_faith = 0.3

PDD effect via step 5 propagation:
    delta_score(v3bu)      = P_v3bu * w_agree * 0.333 ~= P_v3bu * 0.167
    delta_score(decompose) = P_decompose * w_agree * (-0.167) ~= -0.083 * P_decompose

Net score swing: with reasonable P_arm values around 0.5, dup-v3bu
adds ~0.083 to v3bu's score and subtracts ~0.042 from decompose's
score, producing a relative swing of ~0.125 in favor of v3bu.

This is the mechanism magnitude that explains the empirical HP
+18.29pp F1 lift observed in evaluation: the agreement signal is the
lever, NOT the new arm's content (v3bu_dup_a is never selected as the
winning arm in the telemetry, but its presence boosts v3bu's
score via the agreement denominator).

----------------------------------------------------------------------
Step 7: argmax tie-break + fallback paths
----------------------------------------------------------------------

Source: same arbitrator. Tie-break by alphabetical arm name (NOT by
priority order in this code path -- the priority order applies only
to ``arbitrate_pam_lite``'s argmax mode).

Fallback paths:
    * empty answers dict           -> fallback "" with arbitrate_signal="fallback"
    * all answers uncertain        -> longest non-empty (last resort)
    * single-arm subset            -> bypass arbitrate, return that arm
    * all arms zero-score          -> fallback "" with arbitrate_signal="fallback"

N-dependency at step 7: in the alphabetical tie-break, dup names like
``v3bu_dup_a`` may win ties against ``v3bu`` (alphabetical order:
v3bu < v3bu_dup_a). In practice this is a 0% rate in the
telemetry because the score before the tie-break already differs.

PDD contribution at step 7: NEGLIGIBLE.

======================================================================
Summary: N-dependency by step
======================================================================

    Step                                   N-dependency      PDD contrib
    -----------------------------------    --------------    -----------
    1. per-arm sigmoid raw score           none              0
    2. softmax normalization               (none -- skipped) 0
    3. variable-K subset filter            weak              small
    4. arm execution (dup re-uses base)    cost-linear       0
    5. pairwise cross-arm agreement        STRONG            PRIMARY
    6. weighted score combination          inherits step 5   inherits
    7. argmax tie-break + fallback         negligible        ~0

======================================================================
Suggested ablation tests (to validate hypotheses)
======================================================================

A. Disable agreement signal (w_agree = 0). Hypothesis: PDD lift
   disappears entirely. Expected F1: matches 3-arm baseline.

B. Hold w_agree default + replace dup-v3bu with random-noise arm
   (whose answer is a random unrelated string). Hypothesis: PDD lift
   disappears -- random answer does NOT match v3bu via cosine, so the
   agreement numerator does not gain the +1 boost. Expected F1:
   matches 3-arm baseline.

C. Hold w_agree default + use a literal v3bu dup BUT cap the
   agreement denominator at N=3 (legacy 3-arm pool size). Hypothesis:
   the +0.333 numerator gain still lifts v3bu (mostly via numerator
   even with N-1=2 denominator) -- predict partial lift, smaller
   magnitude than the full dup effect.

D. Hold w_agree default + use a literal v3bu dup with N=5 (add a
   second dup). Hypothesis: agreement(v3bu) -> 2/4 = 0.5 (vs the
   single-dup 1/3 = 0.333). Score swing on v3bu grows further; F1 lift may grow
   sub-linearly (diminishing returns once agreement >= 0.5 saturates
   under faith / gamma signals).

Each of these is a controlled mechanism A/B that isolates a specific
step in the trace. The diagnostic instrumentation flags enable
A, B, C, D in :func:`arbitrate_pam_lite`.

----------------------------------------------------------------------
Anti-leak contract for this module
----------------------------------------------------------------------

Pure code reading + math derivation. NO eval F1 inspection used to
derive any formula or constant here. The empirical magnitudes cited
(HP +18.29pp etc.) come from a LIVE eval and are quoted
verbatim as the observation that motivated the trace; they are NOT
used as targets to fit any constant in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


__all__ = [
    "MechanismTrace",
    "trace_pam_lite_mechanism",
    "AGREEMENT_THRESHOLD_DEFAULT",
]


# Default cosine threshold used by pairwise_agreement (signals.py).
# Re-exported for trace consumers that want to replay the math at
# alternate thresholds.
AGREEMENT_THRESHOLD_DEFAULT: float = 0.70


@dataclass(frozen=True)
class MechanismTrace:
    """Structured trace of the PAM-lite arbitrate mechanism per query.

    Each field captures one step's relevant quantity for downstream
    A/B mechanism testing.
    """

    arms: tuple[str, ...]
    raw_scores: Mapping[str, float]         # step 1
    p_arm: Mapping[str, float]              # step 2 (same as raw -- no softmax)
    subset: tuple[str, ...]                 # step 3
    answers: Mapping[str, str]              # step 4
    agreement: Mapping[str, float]          # step 5
    gamma: Mapping[str, float]              # input to step 6
    faith: Mapping[str, float]              # input to step 6
    combined_score: Mapping[str, float]     # step 6 output
    winner: str                             # step 7 argmax
    winner_reason: str                      # diagnostic


def trace_pam_lite_mechanism(
    arms: tuple[str, ...],
    raw_scores: Mapping[str, float],
    answers: Mapping[str, str],
    agreement: Mapping[str, float],
    gamma: Mapping[str, float] | None = None,
    faith: Mapping[str, float] | None = None,
    *,
    threshold: float = 0.3,
    w_gamma: float = 1.0,
    w_agree: float = 0.5,
    w_faith: float = 0.3,
) -> MechanismTrace:
    """Reconstruct the 7-step mechanism trace from the per-arm inputs.

    Pure function -- given the same inputs, produces the same trace.
    Useful for A/B mechanism tests where one input
    (e.g. agreement) is monkeyed and the resulting winner is recorded.

    Args:
        arms: tuple of arm names in dispatch order (incl. dup names).
        raw_scores: per-arm raw P_arm from
            :func:`arm_subset_pam_lite` (post-multiplier).
        answers: per-arm prediction text.
        agreement: per-arm pairwise-agreement signal in ``[0, 1]``.
        gamma, faith: optional per-arm signals; default 1.0 per arm.
        threshold: subset inclusion threshold (variable-K).
        w_gamma, w_agree, w_faith: arbitration weights.

    Returns:
        :class:`MechanismTrace` with all per-step quantities populated.
    """
    g = {a: (gamma or {}).get(a, 1.0) for a in arms}
    f = {a: (faith or {}).get(a, 1.0) for a in arms}
    p_arm = {a: float(raw_scores.get(a, 0.0)) for a in arms}

    subset = tuple(a for a in arms if p_arm[a] > threshold)
    if not subset:
        # always-non-empty argmax fallback (matches arm_subset_pam_lite)
        if arms:
            subset = (max(arms, key=lambda a: p_arm.get(a, 0.0)),)

    combined: dict[str, float] = {}
    for a in arms:
        ag = float(agreement.get(a, 0.0))
        score = p_arm[a] * (w_gamma * g[a] + w_agree * ag + w_faith * f[a])
        combined[a] = score

    if combined:
        max_score = max(combined.values())
        ties = sorted(a for a, s in combined.items() if s == max_score)
        winner = ties[0]
        reason = (
            f"argmax_score={max_score:.4f}_tie_break_alphabetical"
            if len(ties) > 1 else f"argmax_score={max_score:.4f}"
        )
    else:
        winner = ""
        reason = "no_arms"

    return MechanismTrace(
        arms=tuple(arms),
        raw_scores=dict(raw_scores),
        p_arm=p_arm,
        subset=subset,
        answers=dict(answers),
        agreement=dict(agreement),
        gamma=g,
        faith=f,
        combined_score=combined,
        winner=winner,
        winner_reason=reason,
    )
