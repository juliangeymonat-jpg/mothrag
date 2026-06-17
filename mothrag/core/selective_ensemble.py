# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Selective ensemble of V3+bottom-up (best EM) + decompose (chain-of-facts complement).

Strategy:
  Base = V3+bu prediction (best standalone EM)
  Override with decompose IF:
    - V3+bu == "Not in passages" / empty AND decompose has answer
    - decompose has F1 overlap > threshold with V3+bu but is SHORTER and contains
      a known entity-like span (rough heuristic)
    - question matches a chain-of-facts pattern (co-author, predecessor, ...)

This does NOT add LLM cost — it only re-arbitrates between two existing predictions.
"""

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """SQuAD/HotpotQA canonical normalization (Yang et al. 2018)."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred = normalize_answer(prediction).split()
    gt = normalize_answer(ground_truth).split()
    if not pred or not gt:
        return float(pred == gt)
    common = Counter(pred) & Counter(gt)
    num = sum(common.values())
    if num == 0:
        return 0.0
    p = num / len(pred)
    r = num / len(gt)
    return 2 * p * r / (p + r)


def em_score(p: str, g: str) -> float:
    return float(normalize_answer(p) == normalize_answer(g))


# P24 bug-pattern: unify abstain markers across iter,
# arbitrator, and decompose. The legacy set missed "I don't know" /
# "I do not know" / "Cannot answer" → arbitrator treated them as legitimate
# answers instead of abstains. Gated under env var to preserve
# composite-then-bisect verdict.
# Markers below are pre-normalized (lowercase, no punctuation, no articles)
# so they match normalize_answer() output directly.
_LEGACY_UNCERTAIN = frozenset({"not in passages", "unknown", "no answer", "none"})
_EXTENDED_UNCERTAIN = frozenset({
    "not in passages",
    "i dont know",      # I don't know  → punctuation stripped
    "i do not know",
    "unknown",
    "no answer",
    "none",
    "cannot answer",
    "no idea",
    "not sure",
})


def is_uncertain(pred: str) -> bool:
    import os as _os  # local import — keeps top-of-file imports stable
    markers = (_EXTENDED_UNCERTAIN
               if _os.environ.get("MOTHRAG_BUG_PATTERN_WAVE_A") == "1"
               else _LEGACY_UNCERTAIN)
    n = normalize_answer(pred)
    return (not n) or n in markers


# Chain-of-facts pattern keywords: questions where gold answer is a SECONDARY fact
# (not the immediate named entity in the question). Decompose handles these better.
CHAIN_PATTERNS = [
    r"\bco-?commentator\b",
    r"\bco-?author\b",
    r"\bco-?host\b",
    r"\bco-?star\b",
    r"\bformer\b",
    r"\bpredecessor\b",
    r"\bsuccessor\b",
    r"\bsucceeded by\b",
    r"\bpreceded by\b",
    r"\boriginally (named|called)\b",
    r"\bnow located\b",
    r"\bcreator of\b",
    r"\bcreated the\b.+\bthat\b",
]
CHAIN_RE = re.compile("|".join(CHAIN_PATTERNS), re.IGNORECASE)


def is_chain_pattern(question: str) -> bool:
    return bool(CHAIN_RE.search(question))


def selective_arbitrate(v3bu_pred: str, dec_pred: str,
                        question: str = "") -> tuple[str, str]:
    """Decision rule. Returns ``(final_pred, reason)``.

    Rule order:
      1. V3+bu uncertain -> use decompose if not also uncertain.
      2. Both agree (normalized) -> V3+bu.
      3. Chain-of-facts pattern in question -> decompose wins (must come BEFORE
         overlap-longer because chain answers can have token overlap with a wrong
         answer; chain-pattern is the stronger signal).
      4. F1 overlap >= 0.5 -> prefer LONGER (matches HotpotQA verbose gold pattern).
      5. Low overlap, no chain pattern -> V3+bu wins (stronger standalone).
    """
    if is_uncertain(v3bu_pred):
        if not is_uncertain(dec_pred):
            return dec_pred, "v3bu-uncertain-use-decompose"
        return v3bu_pred, "both-uncertain"

    if is_uncertain(dec_pred):
        return v3bu_pred, "dec-uncertain-use-v3bu"

    if normalize_answer(v3bu_pred) == normalize_answer(dec_pred):
        return v3bu_pred, "agree"

    if is_chain_pattern(question):
        return dec_pred, "chain-pattern-use-decompose"

    overlap = f1_score(v3bu_pred, dec_pred)
    if overlap >= 0.5:
        winner = v3bu_pred if len(v3bu_pred) >= len(dec_pred) else dec_pred
        label = "overlap-v3bu-longer" if winner == v3bu_pred else "overlap-dec-longer"
        return winner, label

    return v3bu_pred, "disagree-v3bu-wins"


def route_by_query_type_v2(v3bu_pred: str, dec_pred: str, question: str,
                            iter_pred: str | None = None) -> tuple[str, str]:
    """sel_v2 3-class router with optional iterative-arm prediction.

    Behavior::

        chain_deep question     -> use iter_pred if provided, else dec_pred
                                   (signals caller that iterative arm SHOULD have run)
        bridge_entity question  -> force decompose arm (returns dec_pred)
        semantic_rich question  -> selective_arbitrate (sel_v1 fallback)

    Returns ``(final_pred, reason)`` with prefix ``router_v2:`` for short-circuit
    or ``sel_v1:`` for arbitration fallback. The chain_deep branch with
    ``iter_pred=None`` returns ``dec_pred`` and tags the reason
    ``router_v2:chain-deep-no-iter`` so the caller knows the iterative arm
    was missing — useful for ablation when iterative is disabled.
    """
    from mothrag.core.query_type_classifier import classify_query_v2

    qtype = classify_query_v2(question)
    if qtype == "chain_deep":
        if iter_pred is not None and not is_uncertain(iter_pred):
            return iter_pred, "router_v2:chain-deep-use-iter"
        if not is_uncertain(dec_pred):
            return dec_pred, "router_v2:chain-deep-no-iter-fallback-decompose"
        return v3bu_pred, "router_v2:chain-deep-fallback-v3bu"
    if qtype == "bridge_entity":
        if not is_uncertain(dec_pred):
            return dec_pred, "router_v2:bridge-force-decompose"
        return v3bu_pred, "router_v2:bridge-decompose-uncertain-fallback-v3bu"
    pred, label = selective_arbitrate(v3bu_pred, dec_pred, question)
    return pred, f"sel_v1:{label}"


def apply_c7(chosen: str, candidates: list[str | None], *,
              use_c7: bool = False,
              c7_trigger: str = "gated",
              gamma_status: str | None = None,
              embedder=None,
              query_embed=None) -> dict | None:
    """Apply Aurora L6 C7 phase-cancellation post-hoc on a chosen vs candidates set.

    Decoupled from arbitration choice so it can be called after ANY arbitrate
    function (``selective_arbitrate``, ``route_by_query_type_v2``,
    ``arbitrate_excl_v3bu``). The unchosen distinct candidates become
    ``rejected_chains`` for the Method-D auto-phase filter.

    Returns the c7_info dict (with ``chosen_kept`` flag + diagnostics) or
    ``None`` when C7 is disabled, gated-and-not-triggered, or no rejected
    chains remain after dedup.
    """
    if not use_c7 or embedder is None:
        return None
    if c7_trigger == "gated" and gamma_status not in ("partial", "invalid"):
        return None

    chosen_norm = normalize_answer(chosen)
    seen: set[str] = set()
    rejected: list[str] = []
    for c in candidates:
        if c is None:
            continue
        c_norm = normalize_answer(c)
        if not c_norm or c_norm == chosen_norm or c_norm in seen:
            continue
        seen.add(c_norm)
        rejected.append(c)
    if not rejected:
        return None

    from mothrag.aurora import c7_aurora_rejected_chains
    try:
        return c7_aurora_rejected_chains(chosen, rejected, embedder,
                                          query_embed=query_embed)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:160], "chosen_kept": True}


_ARM_PRIORITY_PLUMB = ("v3bu", "decompose", "iter")


def _select_via_subset_p_arm(
    preds: dict[str, str | None],
    subset: list[str] | set[str] | tuple[str, ...],
    p_arm: dict[str, float] | None,
) -> tuple[str, str]:
    """Option A: pick a prediction from the (subset, p_arm)
    pair supplied upstream.

    Restricts the candidate space to arms in ``subset``. When ``p_arm``
    is provided, picks argmax-by-p_arm with uncertain-pred filter and
    stable priority tie-break. When ``p_arm`` is None, picks first
    non-uncertain candidate in _ARM_PRIORITY_PLUMB order.

    Pool-safety: if no in-subset arm has a non-uncertain prediction,
    falls back to the first non-uncertain pred over ALL arms; else the
    first non-empty pred; else "".
    """
    subset_set = set(subset)
    in_subset: list[tuple[str, str, float]] = []
    for arm in _ARM_PRIORITY_PLUMB:
        if arm not in subset_set:
            continue
        pred = preds.get(arm)
        if pred is None or pred == "":
            continue
        if is_uncertain(pred):
            continue
        p = float((p_arm or {}).get(arm, 0.0))
        in_subset.append((arm, pred, p))

    if in_subset:
        if p_arm is not None:
            arm, pred, p = max(in_subset, key=lambda t: t[2])
            return pred, f"plumb:argmax_{arm}_P={p:.3f}"
        arm, pred, _ = in_subset[0]
        return pred, f"plumb:first_in_subset_{arm}"

    # Pool-safety fallback: take first non-uncertain pred across ALL arms.
    for arm in _ARM_PRIORITY_PLUMB:
        pred = preds.get(arm)
        if pred and not is_uncertain(pred):
            return pred, f"plumb:fallback_outside_subset_{arm}"
    for arm in _ARM_PRIORITY_PLUMB:
        pred = preds.get(arm)
        if pred:
            return pred, f"plumb:fallback_uncertain_{arm}"
    return "", "plumb:no_preds"


def arbitrate_with_c7(v3bu_pred: str, dec_pred: str, question: str,
                       *,
                       iter_pred: str | None = None,
                       use_c7: bool = False,
                       c7_trigger: str = "gated",
                       gamma_status: str | None = None,
                       embedder=None,
                       query_embed=None,
                       use_router_v2: bool = False,
                       subset: list[str] | set[str] | tuple[str, ...] | None = None,
                       p_arm: dict[str, float] | None = None,
                       bridge_pred: str | None = None,
                       bridge_primary_qtypes: tuple[str, ...] = (
                           "bridge_entity", "chain_deep"),
                       ) -> tuple[str, str, dict | None]:
    """ENS arbitrate + Aurora L6 C7 phase-cancellation in the right place.

    Architectural placement (per Aurora handoff spec): C7 operates POST-HOC on
    the ensemble-arbitrate output, where multiple chain candidates exist
    (V3+bu / decompose / iterative). The chosen answer is fed alongside the
    *unchosen* chains as ``rejected_chains`` to the Method-D auto-phase filter.

    C7 produces a meta-confidence flag (``chosen_kept`` bool) — it does NOT
    rewrite the chosen answer. Downstream callers use the flag for Soft EM
    conditional metrics (kept = high-confidence, cancelled = low-confidence).

    Args:
        v3bu_pred / dec_pred: arm predictions (sel_v1 inputs).
        iter_pred: optional iterative-arm prediction (sel_v2 input).
        use_c7: master switch.
        c7_trigger: ``"gated"`` (only when ``gamma_status`` in
            ``{"partial", "invalid"}``) or ``"blanket"`` (always).
        gamma_status: γ verifier overall_status, required for gated trigger.
        embedder: callable ``(list[str]) -> ndarray (K, D)`` for C7 embedding.
        query_embed: optional pre-computed query embedding (Aurora-spec
            faithful axis); else C7 uses centroid bisection.
        use_router_v2: route via :func:`route_by_query_type_v2` (3-arm) instead
            of :func:`selective_arbitrate` (2-arm).
        subset: Option A. When provided, skip internal
            classify_query_v2 routing and pick from ``subset`` instead.
        p_arm: Option A. When provided AND subset provided,
            argmax-by-p_arm within subset; else first-in-subset by
            priority.

    Returns:
        ``(chosen_pred, reason, c7_info)`` where ``c7_info`` is ``None`` when
        C7 is disabled / not triggered, else a dict with ``chosen_kept`` plus
        Method-D diagnostics.
    """
    if subset is not None:
        # Option A: bypass label-based downstream router; use
        # upstream-provided subset (+ optional p_arm vector).
        chosen, reason = _select_via_subset_p_arm(
            preds={"v3bu": v3bu_pred, "decompose": dec_pred, "iter": iter_pred},
            subset=subset,
            p_arm=p_arm,
        )
    elif use_router_v2:
        chosen, reason = route_by_query_type_v2(v3bu_pred, dec_pred, question, iter_pred)
    else:
        chosen, reason = selective_arbitrate(v3bu_pred, dec_pred, question)

    # Opt-in bridge arm. When a substantive bridge_pred is
    # supplied, it competes as a 4th ensemble candidate: on the multi-hop
    # cohorts it is purpose-built for (bridge_entity / chain_deep) it becomes
    # the chosen answer; on other qtypes it stays a C7 candidate only. With
    # bridge_pred=None (default) behaviour is byte-identical to the 3-arm path.
    if bridge_pred is not None and not is_uncertain(bridge_pred):
        from mothrag.core.query_type_classifier import classify_query_v2
        if classify_query_v2(question) in bridge_primary_qtypes:
            chosen, reason = bridge_pred, "bridge_arm:multi-hop-primary"

    candidates = [v3bu_pred, dec_pred] + ([iter_pred] if iter_pred is not None else [])
    if bridge_pred is not None:
        candidates.append(bridge_pred)
    c7_info = apply_c7(chosen, candidates, use_c7=use_c7,
                        c7_trigger=c7_trigger, gamma_status=gamma_status,
                        embedder=embedder, query_embed=query_embed)
    return chosen, reason, c7_info


def arbitrate_excl_v3bu(dec_pred: str, iter_pred: str | None,
                         question: str,
                         v3bu_fallback: str | None = None,
                         *,
                         subset: list[str] | set[str] | tuple[str, ...] | None = None,
                         p_arm: dict[str, float] | None = None,
                         ) -> tuple[str, str]:
    """2-arm arbitration for queries where V3+bu was excluded by arm_subset.

    chain_deep  → iter (primary), decompose fallback, v3bu_fallback last resort
    bridge_entity → decompose (primary), iter fallback, v3bu_fallback last resort
    semantic_rich → iter as primary (stronger on complex 2Wiki queries),
                    selective_arbitrate(iter, decompose) to resolve disagreement.

    v3bu_fallback: when provided (post-hoc mode where V3+bu file already ran),
        used as last resort when both decompose and iter are uncertain. This
        prevents regression vs the baseline when no iter arm is available yet.

    subset / p_arm: Option A plumb-through. When ``subset`` is
        provided, bypass internal ``classify_query_v2`` routing entirely
        and pick from arms in ``subset`` (v3bu is fed as ``v3bu_fallback``,
        included in subset choice only if "v3bu" is in subset). When
        ``p_arm`` is also provided, argmax-by-P within subset.
    """
    from mothrag.core.query_type_classifier import classify_query_v2

    if subset is not None:
        # Option A: upstream-provided subset bypasses label
        # routing. v3bu_fallback is the "v3bu" candidate for subset
        # filtering; when "v3bu" not in subset, only dec / iter are
        # considered.
        chosen, reason = _select_via_subset_p_arm(
            preds={
                "v3bu": v3bu_fallback,
                "decompose": dec_pred,
                "iter": iter_pred,
            },
            subset=subset,
            p_arm=p_arm,
        )
        return chosen, f"excl_v3bu:{reason}"

    def _last_resort(label_suffix: str) -> tuple[str, str]:
        if v3bu_fallback and not is_uncertain(v3bu_fallback):
            return v3bu_fallback, f"excl_v3bu:{label_suffix}-v3bu-fallback"
        return dec_pred, f"excl_v3bu:{label_suffix}-all-uncertain"

    qtype = classify_query_v2(question)

    if qtype == "chain_deep":
        if iter_pred and not is_uncertain(iter_pred):
            return iter_pred, "excl_v3bu:chain-deep-iter"
        if not is_uncertain(dec_pred):
            return dec_pred, "excl_v3bu:chain-deep-fallback-decompose"
        return _last_resort("chain-deep")

    if qtype == "bridge_entity":
        if not is_uncertain(dec_pred):
            return dec_pred, "excl_v3bu:bridge-decompose"
        if iter_pred and not is_uncertain(iter_pred):
            return iter_pred, "excl_v3bu:bridge-fallback-iter"
        return _last_resort("bridge")

    # semantic_rich: iter as primary (stronger on complex queries); decompose as backup
    if iter_pred and not is_uncertain(iter_pred):
        pred, label = selective_arbitrate(iter_pred, dec_pred, question)
        return pred, f"excl_v3bu:semantic-rich-iter-primary:{label}"
    if not is_uncertain(dec_pred):
        return dec_pred, "excl_v3bu:semantic-rich-no-iter-use-decompose"
    return _last_resort("semantic-rich")


def route_by_query_type(v3bu_pred: str, dec_pred: str, question: str) -> tuple[str, str]:
    """Query-type-classifier router: short-circuits the ensemble.

    Behavior::

        bridge_entity question  -> force decompose arm  (returns dec_pred)
        semantic_rich question  -> fall back to selective_arbitrate (sel_v1)

    The classifier is a deterministic linguistic prior (see
    :mod:`mothrag.core.query_type_classifier`). It does NOT add LLM cost; it only
    re-arbitrates between two predictions already produced by the two arms.

    Returns ``(final_pred, reason)`` where ``reason`` is prefixed with
    ``router:`` when the classifier short-circuited, or ``sel_v1:`` when it
    deferred to the standard arbitration rule.
    """
    from mothrag.core.query_type_classifier import classify_query

    qtype = classify_query(question)
    if qtype == "bridge_entity":
        if not is_uncertain(dec_pred):
            return dec_pred, "router:bridge-force-decompose"
        # Decompose abstained — fall through to V3+bu rather than emit empty.
        return v3bu_pred, "router:bridge-decompose-uncertain-fallback-v3bu"
    pred, label = selective_arbitrate(v3bu_pred, dec_pred, question)
    return pred, f"sel_v1:{label}"
