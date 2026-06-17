# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""PAM-lite-native arbitrator (Option B).

Fixes a subtle bug: arbitrate_post output was
``--router``-independent because both legacy downstream arbitrate
functions (``arbitrate_with_c7`` + ``arbitrate_excl_v3bu``) internally
call ``classify_query_v2(question)`` -- a label-based router that
discards the upstream arm subset / P_arm vector.

``arbitrate_pam_lite`` consumes the (preds, P_arm) pair directly, so
cfde114 / hop-aware PAM-lite scoring deltas finally drive the chosen
answer.

Three modes:

* ``argmax`` (default) -- pick the prediction from the arm with the
  highest ``P_arm``, with an uncertain-prediction filter and a v3bu
  fallback when every arm is uncertain (pool-safety).
* ``weighted_mix`` -- weighted-vote across distinct prediction strings
  using ``P_arm`` as weights. Ties broken by argmax-arm priority.
* ``subset`` -- threshold filter (``P_arm > threshold``) then argmax
  inside the filtered subset; empty subset falls back to global argmax.

Continuous P_arm philosophy: preserved. No training, no per-dataset
tuning. Pool-safety
axiom: arms with ``P_arm == 0`` contribute zero weight in weighted_mix
and are excluded in subset mode.
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import Literal, Mapping

from mothrag.core.selective_ensemble import is_uncertain


__all__ = [
    "arbitrate_pam_lite",
    "arbitrate_pam_lite_traced",
    "PamLiteDiagnostic",
    "TieBreakStrategy",
]


_ARM_PRIORITY = ("v3bu", "decompose", "iter")

# Seeded RNG for the "random" tie-break strategy. Reproducible by
# default; callers may rebind this module attribute for alternate
# seeds in mechanism A/B tests.
_RANDOM_TIE_RNG: _random.Random = _random.Random(0)


# Mechanism ablation surface.
# Tie-break strategy controls how arbitrate_pam_lite_traced resolves
# ties when two arms have the same numeric score:
#   "priority"     : _ARM_PRIORITY dict order (default, byte-compat
#                    with arbitrate_pam_lite legacy)
#   "lexicographic": sorted arm name ascending
#   "first"        : insertion order of `preds`
#   "random"       : seeded random pick (for the ablation CLI);
#                    reproducible via global random.Random(seed)
TieBreakStrategy = Literal["priority", "lexicographic", "first", "random"]


@dataclass(frozen=True)
class PamLiteDiagnostic:
    """Diagnostic trace of one ``arbitrate_pam_lite_traced`` call.

    Captures every mechanism-relevant quantity so A/B
    experiments can replay the math at alternate inputs. Anti-leak: no
    gold answers or F1 numbers in the trace; only system telemetry.
    """

    mode: str
    raw_p_arm: Mapping[str, float]
    eligible_arms: tuple[str, ...]
    eligible_preds: Mapping[str, str]
    subset_arms: tuple[str, ...]
    winner_arm: str
    winner_pred: str
    winner_score: float
    tie_break_strategy: str
    tie_break_fired: bool
    fallback_fired: bool
    fallback_path: str
    reason: str
    extra: Mapping[str, object] = field(default_factory=dict)


def _stable_eligible(
    preds: Mapping[str, str | None],
    p_arm: Mapping[str, float],
    *,
    iteration_order: tuple[str, ...] | None = None,
) -> list[tuple[str, str, float]]:
    """Return (arm, pred, p) tuples for arms whose prediction is not
    uncertain.

    ``iteration_order`` controls the source order of the eligibility
    scan. Default ``None`` -> _ARM_PRIORITY then any extra arms in
    insertion order (covers dup-arm names + opt-in arms like
    infobox_arm / mothgraph_arm without rewriting callers). Pass
    a tuple of arm names (e.g. ``tuple(preds.keys())``) to scan in
    insertion order -- used by the "first" tie-break strategy in
    :func:`arbitrate_pam_lite_traced`.
    """
    if iteration_order is None:
        extras = tuple(a for a in preds.keys() if a not in _ARM_PRIORITY)
        order: tuple[str, ...] = _ARM_PRIORITY + extras
    else:
        order = iteration_order
    out: list[tuple[str, str, float]] = []
    for arm in order:
        pred = preds.get(arm)
        if pred is None or pred == "":
            continue
        if is_uncertain(pred):
            continue
        p = float(p_arm.get(arm, 0.0))
        out.append((arm, pred, p))
    return out


def _argmax(eligible: list[tuple[str, str, float]]) -> tuple[str, str, float]:
    """Pick the (arm, pred, p) with the highest p. Ties broken by
    _ARM_PRIORITY (encoded by list order, since sorted() is stable).
    """
    return max(eligible, key=lambda t: t[2])


def _argmax_with_strategy(
    eligible: list[tuple[str, str, float]],
    strategy: TieBreakStrategy,
) -> tuple[tuple[str, str, float], bool]:
    """Pick winner by max-score with explicit tie-break strategy.

    Returns ``((arm, pred, p), tie_break_fired)`` where ``tie_break_fired``
    is True iff multiple eligible arms shared the max score.
    """
    if not eligible:
        return ("", "", 0.0), False
    max_score = max(t[2] for t in eligible)
    tied = [t for t in eligible if t[2] == max_score]
    if len(tied) == 1:
        return tied[0], False

    if strategy == "priority":
        priority_index = {name: i for i, name in enumerate(_ARM_PRIORITY)}
        sentinel = len(_ARM_PRIORITY)
        tied.sort(key=lambda t: priority_index.get(t[0], sentinel))
        return tied[0], True
    if strategy == "lexicographic":
        tied.sort(key=lambda t: t[0])
        return tied[0], True
    if strategy == "first":
        # eligible preserves insertion order, so the first tied entry is
        # the first-occurring arm
        return tied[0], True
    if strategy == "random":
        # Seeded random pick (mechanism ablation -- isolates the
        # contribution of priority ordering by replacing it with
        # uniform random among ties). Seed fixed at 0 so traces are
        # reproducible across runs; callers wanting different seeds
        # should monkey-patch _RANDOM_TIE_RNG before the call.
        winner = _RANDOM_TIE_RNG.choice(tied)
        return winner, True
    raise ValueError(f"unknown tie_break strategy: {strategy!r}")


def arbitrate_pam_lite(
    preds: Mapping[str, str | None],
    p_arm: Mapping[str, float],
    *,
    threshold: float = 0.3,
    mode: str = "argmax",
) -> tuple[str, str]:
    """Drive arbitration from PAM-lite continuous probabilities.

    Args:
        preds: per-arm predictions, e.g.
            ``{"v3bu": "Paris", "decompose": "Lyon", "iter": None}``.
            Arms with None / empty / uncertain (per ``is_uncertain``)
            predictions are skipped.
        p_arm: per-arm continuous probability, e.g.
            ``{"v3bu": 0.9, "decompose": 0.4, "iter": 0.1}``. Missing
            keys default to 0.0.
        threshold: subset-mode inclusion threshold. Ignored in
            ``argmax`` and ``weighted_mix`` modes.
        mode: one of ``"argmax"``, ``"weighted_mix"``, ``"subset"``.

    Returns:
        ``(chosen_pred, reason)`` where reason is prefixed
        ``"pamlite:<mode>:..."``.

    Contract:
        * ``mode`` must be valid; raises ``ValueError`` otherwise.
        * If no eligible (non-uncertain) prediction exists, falls back
          to a non-uncertain v3bu_pred if available, else the first
          non-empty pred in priority order, else ``""``.
        * ``argmax``: pick highest-P eligible arm.
        * ``weighted_mix``: bucket eligibles by normalized prediction
          string, sum P per bucket, pick highest-sum.
        * ``subset``: filter eligible by ``p > threshold``; argmax in
          filtered set. If filter empties everything, falls back to
          global argmax over eligibles.

    Pool-safety axiom: arms with ``p_arm == 0`` contribute zero weight
    in ``weighted_mix`` and are excluded in ``subset``.
    """
    if mode not in ("argmax", "weighted_mix", "subset"):
        raise ValueError(
            f"mode must be one of argmax / weighted_mix / subset; got {mode!r}",
        )

    eligible = _stable_eligible(preds, p_arm)
    if not eligible:
        # Fallback: try to return any non-empty prediction; else "".
        for arm in _ARM_PRIORITY:
            p = preds.get(arm)
            if p:  # non-empty
                return p, f"pamlite:{mode}:all_uncertain_fallback_{arm}"
        return "", f"pamlite:{mode}:no_preds"

    if mode == "argmax":
        arm, pred, p = _argmax(eligible)
        return pred, f"pamlite:argmax_{arm}_P={p:.3f}"

    if mode == "subset":
        filtered = [t for t in eligible if t[2] > threshold]
        if filtered:
            arm, pred, p = _argmax(filtered)
            return pred, f"pamlite:subset_{arm}_P={p:.3f}_thr={threshold:.2f}"
        # Filter zeroed-out -> fall back to global argmax for pool-safety.
        arm, pred, p = _argmax(eligible)
        return pred, (
            f"pamlite:subset_fallback_argmax_{arm}_P={p:.3f}_thr={threshold:.2f}"
        )

    # mode == "weighted_mix"
    # Group by normalized pred string; sum P per bucket. Track which arm
    # contributed the highest individual P to each bucket (for tie-break).
    from mothrag.core.selective_ensemble import normalize_answer

    buckets: dict[str, dict] = {}
    for arm, pred, p in eligible:
        key = normalize_answer(pred)
        b = buckets.setdefault(key, {"pred": pred, "total_p": 0.0, "top_arm": arm, "top_p": p})
        b["total_p"] += p
        if p > b["top_p"]:
            b["top_arm"] = arm
            b["top_p"] = p
    # Pick highest total_p; ties broken by _ARM_PRIORITY of top_arm.
    def _rank(item):
        key, b = item
        try:
            arm_rank = _ARM_PRIORITY.index(b["top_arm"])
        except ValueError:
            arm_rank = len(_ARM_PRIORITY)
        return (-b["total_p"], arm_rank)
    best_key, best = sorted(buckets.items(), key=_rank)[0]
    return best["pred"], (
        f"pamlite:weighted_mix_{best['top_arm']}_total_P={best['total_p']:.3f}"
    )


# ============================================================
# Diagnostic instrumentation + mechanism ablation flags
# ============================================================


def arbitrate_pam_lite_traced(
    preds: Mapping[str, str | None],
    p_arm: Mapping[str, float],
    *,
    threshold: float = 0.3,
    mode: str = "argmax",
    tie_break: TieBreakStrategy = "priority",
    disable_fallback: bool = False,
) -> tuple[str, str, PamLiteDiagnostic]:
    """Diagnostic variant of :func:`arbitrate_pam_lite`.

    Returns the same ``(chosen_pred, reason)`` as the canonical
    function plus a :class:`PamLiteDiagnostic` instance with the full
    per-step trace. Used by A/B mechanism tests.

    Args:
        preds, p_arm, threshold, mode: same as
            :func:`arbitrate_pam_lite`.
        tie_break: ``"priority"`` (default; matches legacy _ARM_PRIORITY
            ordering, byte-compat), ``"lexicographic"`` (sorted arm name
            ascending), or ``"first"`` (insertion order of ``preds``).
        disable_fallback: when True, skip the all-uncertain fallback +
            the subset-empty fallback. Returns ``("", reason)`` with
            an explicit "no_fallback" reason when those paths would
            normally fire. Used to isolate the argmax-only behavior.

    Anti-leak: no gold / F1 inputs; pure system telemetry tracing.
    """
    if mode not in ("argmax", "weighted_mix", "subset"):
        raise ValueError(
            f"mode must be one of argmax / weighted_mix / subset; got {mode!r}",
        )
    if tie_break not in ("priority", "lexicographic", "first", "random"):
        raise ValueError(
            f"tie_break must be priority / lexicographic / first / random; "
            f"got {tie_break!r}"
        )

    raw_p_arm_snapshot: dict[str, float] = {
        a: float(p_arm.get(a, 0.0)) for a in (set(preds) | set(p_arm))
    }

    iteration_order: tuple[str, ...] | None = None
    if tie_break == "first":
        iteration_order = tuple(preds.keys())
    eligible = _stable_eligible(preds, p_arm, iteration_order=iteration_order)

    eligible_arms = tuple(t[0] for t in eligible)
    eligible_preds = {t[0]: t[1] for t in eligible}

    if not eligible:
        if disable_fallback:
            diag = PamLiteDiagnostic(
                mode=mode,
                raw_p_arm=raw_p_arm_snapshot,
                eligible_arms=(),
                eligible_preds={},
                subset_arms=(),
                winner_arm="",
                winner_pred="",
                winner_score=0.0,
                tie_break_strategy=tie_break,
                tie_break_fired=False,
                fallback_fired=True,
                fallback_path="no_fallback_flag",
                reason=f"pamlite:{mode}:no_fallback_disabled",
            )
            return "", diag.reason, diag
        # Legacy fallback path
        for arm in _ARM_PRIORITY:
            p = preds.get(arm)
            if p:
                reason = f"pamlite:{mode}:all_uncertain_fallback_{arm}"
                diag = PamLiteDiagnostic(
                    mode=mode,
                    raw_p_arm=raw_p_arm_snapshot,
                    eligible_arms=(),
                    eligible_preds={},
                    subset_arms=(),
                    winner_arm=arm,
                    winner_pred=p,
                    winner_score=0.0,
                    tie_break_strategy=tie_break,
                    tie_break_fired=False,
                    fallback_fired=True,
                    fallback_path=f"all_uncertain_fallback_{arm}",
                    reason=reason,
                )
                return p, reason, diag
        reason = f"pamlite:{mode}:no_preds"
        diag = PamLiteDiagnostic(
            mode=mode,
            raw_p_arm=raw_p_arm_snapshot,
            eligible_arms=(),
            eligible_preds={},
            subset_arms=(),
            winner_arm="",
            winner_pred="",
            winner_score=0.0,
            tie_break_strategy=tie_break,
            tie_break_fired=False,
            fallback_fired=True,
            fallback_path="no_preds",
            reason=reason,
        )
        return "", reason, diag

    if mode == "argmax":
        (arm, pred, p), tie_fired = _argmax_with_strategy(eligible, tie_break)
        reason = f"pamlite:argmax_{arm}_P={p:.3f}"
        diag = PamLiteDiagnostic(
            mode=mode,
            raw_p_arm=raw_p_arm_snapshot,
            eligible_arms=eligible_arms,
            eligible_preds=eligible_preds,
            subset_arms=(),
            winner_arm=arm,
            winner_pred=pred,
            winner_score=p,
            tie_break_strategy=tie_break,
            tie_break_fired=tie_fired,
            fallback_fired=False,
            fallback_path="",
            reason=reason,
        )
        return pred, reason, diag

    if mode == "subset":
        filtered = [t for t in eligible if t[2] > threshold]
        subset_arms = tuple(t[0] for t in filtered)
        if filtered:
            (arm, pred, p), tie_fired = _argmax_with_strategy(filtered, tie_break)
            reason = f"pamlite:subset_{arm}_P={p:.3f}_thr={threshold:.2f}"
            diag = PamLiteDiagnostic(
                mode=mode,
                raw_p_arm=raw_p_arm_snapshot,
                eligible_arms=eligible_arms,
                eligible_preds=eligible_preds,
                subset_arms=subset_arms,
                winner_arm=arm,
                winner_pred=pred,
                winner_score=p,
                tie_break_strategy=tie_break,
                tie_break_fired=tie_fired,
                fallback_fired=False,
                fallback_path="",
                reason=reason,
            )
            return pred, reason, diag
        # Filter empty -> fallback to global argmax (or hard refuse)
        if disable_fallback:
            reason = (
                f"pamlite:subset:no_fallback_disabled_thr={threshold:.2f}"
            )
            diag = PamLiteDiagnostic(
                mode=mode,
                raw_p_arm=raw_p_arm_snapshot,
                eligible_arms=eligible_arms,
                eligible_preds=eligible_preds,
                subset_arms=(),
                winner_arm="",
                winner_pred="",
                winner_score=0.0,
                tie_break_strategy=tie_break,
                tie_break_fired=False,
                fallback_fired=True,
                fallback_path="subset_empty_no_fallback",
                reason=reason,
            )
            return "", reason, diag
        (arm, pred, p), tie_fired = _argmax_with_strategy(eligible, tie_break)
        reason = (
            f"pamlite:subset_fallback_argmax_{arm}_P={p:.3f}_thr={threshold:.2f}"
        )
        diag = PamLiteDiagnostic(
            mode=mode,
            raw_p_arm=raw_p_arm_snapshot,
            eligible_arms=eligible_arms,
            eligible_preds=eligible_preds,
            subset_arms=(),
            winner_arm=arm,
            winner_pred=pred,
            winner_score=p,
            tie_break_strategy=tie_break,
            tie_break_fired=tie_fired,
            fallback_fired=True,
            fallback_path="subset_empty_global_argmax",
            reason=reason,
        )
        return pred, reason, diag

    # mode == "weighted_mix"
    from mothrag.core.selective_ensemble import normalize_answer

    buckets: dict[str, dict] = {}
    for arm, pred, p in eligible:
        key = normalize_answer(pred)
        b = buckets.setdefault(
            key, {"pred": pred, "total_p": 0.0, "top_arm": arm, "top_p": p},
        )
        b["total_p"] += p
        if p > b["top_p"]:
            b["top_arm"] = arm
            b["top_p"] = p

    if tie_break == "priority":
        priority_index = {name: i for i, name in enumerate(_ARM_PRIORITY)}
        sentinel = len(_ARM_PRIORITY)

        def _rank(item):
            _key, b = item
            return (-b["total_p"], priority_index.get(b["top_arm"], sentinel))
    elif tie_break == "lexicographic":
        def _rank(item):
            _key, b = item
            return (-b["total_p"], b["top_arm"])
    else:  # "first"
        order_index = {a: i for i, a in enumerate(preds.keys())}

        def _rank(item):
            _key, b = item
            return (-b["total_p"], order_index.get(b["top_arm"], len(order_index)))

    sorted_buckets = sorted(buckets.items(), key=_rank)
    _best_key, best = sorted_buckets[0]
    # tie detection: multiple buckets sharing the top total_p
    top_total = best["total_p"]
    tied_buckets = [b for _k, b in buckets.items() if b["total_p"] == top_total]
    tie_fired = len(tied_buckets) > 1
    reason = (
        f"pamlite:weighted_mix_{best['top_arm']}_total_P={best['total_p']:.3f}"
    )
    diag = PamLiteDiagnostic(
        mode=mode,
        raw_p_arm=raw_p_arm_snapshot,
        eligible_arms=eligible_arms,
        eligible_preds=eligible_preds,
        subset_arms=(),
        winner_arm=best["top_arm"],
        winner_pred=best["pred"],
        winner_score=best["total_p"],
        tie_break_strategy=tie_break,
        tie_break_fired=tie_fired,
        fallback_fired=False,
        fallback_path="",
        reason=reason,
        extra={"bucket_count": len(buckets)},
    )
    return best["pred"], reason, diag
