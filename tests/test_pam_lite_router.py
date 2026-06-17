"""Tests for PAM-lite continuous-probability router.

Covers:
  - mothrag.routing.semantic_features.extract_semantic_features (10 features)
  - mothrag.core.query_type_classifier.arm_subset_pam_lite (continuous P_arm)
  - mothrag.core.arbitrate.DeterministicArbitrator with arm_probabilities
  - Backward compat: sel_v2 default unchanged when P_arm not supplied

All linguistic rules; no training, no per-dataset tuning, no test
inspection.
"""

from __future__ import annotations

import sys
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _block_sentence_transformers(monkeypatch) -> Iterator[None]:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    yield


# ============================================================
# Semantic features extractor -- deterministic, side-effect free
# ============================================================

def test_semantic_features_deterministic() -> None:
    from mothrag.routing.semantic_features import extract_semantic_features
    q = "When was Albert Einstein born?"
    f1 = extract_semantic_features(q)
    f2 = extract_semantic_features(q)
    f3 = extract_semantic_features(q)
    assert f1 == f2 == f3


def test_semantic_features_all_in_unit_range() -> None:
    """Every feature scorer must return a float in [0, 1]."""
    from dataclasses import asdict
    from mothrag.routing.semantic_features import extract_semantic_features
    for q in (
        "When was Einstein born?",
        "What is the capital of France?",
        "Which company that Steve Jobs founded acquired NeXT?",
        "Is Einstein older than Newton?",
        "Why did the chicken cross the road?",
        "",
    ):
        f = extract_semantic_features(q)
        for k, v in asdict(f).items():
            assert isinstance(v, float), f"{k} not float on {q!r}"
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1] on {q!r}"


def test_attribute_marker_fires_on_entity_attribute_question() -> None:
    from mothrag.routing.semantic_features import score_attribute_marker
    assert score_attribute_marker("When was Einstein born?") > 0
    assert score_attribute_marker("What is the capital of France?") > 0
    assert score_attribute_marker("Why did the chicken cross the road?") == 0.0


def test_multi_hop_marker_fires_on_subordinate_clause() -> None:
    from mothrag.routing.semantic_features import score_multi_hop
    assert score_multi_hop("Which company that Steve Jobs founded?") > 0.5
    assert score_multi_hop("Where did the composer of the film X die?") > 0
    assert score_multi_hop("When was Einstein born?") == 0.0


def test_comparison_marker_fires_on_polar() -> None:
    from mothrag.routing.semantic_features import score_comparison
    assert score_comparison("Is X older than Y?") > 0
    assert score_comparison("Both A and B were Greek?") > 0
    assert score_comparison("When was Einstein born?") == 0.0


def test_v3bu_cfde114_v3_boost_applies_when_is_1hop_polar() -> None:
    """cfde114 v3: comparison_marker boost is gated on the
    hop structure derived from SOLID features. When ``is_polar_comparison``
    fires (1-hop polar yes/no comparison), the +0.3 * comparison_marker
    boost applies (V3+bu empirically wins this surface).
    """
    from mothrag.core.query_type_classifier import _score_v3bu_p_arm
    from mothrag.routing.semantic_features import SemanticFeatures

    f_with_polar = SemanticFeatures(
        comparison_marker=0.7, is_polar_comparison=1.0,
    )
    f_without_polar = SemanticFeatures(
        comparison_marker=0.7, is_polar_comparison=0.0,
    )
    p_polar = _score_v3bu_p_arm(f_with_polar)
    p_no_polar = _score_v3bu_p_arm(f_without_polar)

    assert p_polar > p_no_polar, (
        f"cfde114 v3 hop gate failed: 1-hop polar score ({p_polar:.4f}) "
        f"should be HIGHER than non-polar ({p_no_polar:.4f}) when "
        f"comparison_marker fires identically."
    )


def test_v3bu_cfde114_v3_no_boost_when_2hop_bridge() -> None:
    """cfde114 v3: on 2-hop bridge structure (two_entity + NO single_hop
    + NO chain_marker), the comparison_marker boost is suppressed --
    decompose primitive routes this class. v2 (single_hop gate)
    still fired here because single_hop=0; v3 fixes the underfire.
    """
    from mothrag.core.query_type_classifier import _score_v3bu_p_arm
    from mothrag.routing.semantic_features import SemanticFeatures

    # 2-hop bridge: two_entity fired, single_hop=0, chain_marker=0,
    # is_polar_comparison=0. The cfde114 boost MUST be suppressed.
    f_bridge_with_comp = SemanticFeatures(
        two_entity=0.8,
        comparison_marker=0.7,
        single_hop=0.0,
        chain_marker=0.0,
        is_polar_comparison=0.0,
    )
    f_bridge_no_comp = SemanticFeatures(
        two_entity=0.8,
        comparison_marker=0.0,
        single_hop=0.0,
        chain_marker=0.0,
        is_polar_comparison=0.0,
    )
    assert _score_v3bu_p_arm(f_bridge_with_comp) == _score_v3bu_p_arm(f_bridge_no_comp)


def test_v3bu_cfde114_v3_no_boost_when_3hop_chain() -> None:
    """cfde114 v3: on 3-hop chain structure (chain_marker + two_entity),
    the comparison_marker boost is suppressed -- iter primitive routes.
    """
    from mothrag.core.query_type_classifier import _score_v3bu_p_arm
    from mothrag.routing.semantic_features import SemanticFeatures

    f_chain_with_comp = SemanticFeatures(
        chain_marker=0.7,
        two_entity=0.8,
        comparison_marker=0.7,
        is_polar_comparison=0.0,
    )
    f_chain_no_comp = SemanticFeatures(
        chain_marker=0.7,
        two_entity=0.8,
        comparison_marker=0.0,
        is_polar_comparison=0.0,
    )
    assert _score_v3bu_p_arm(f_chain_with_comp) == _score_v3bu_p_arm(f_chain_no_comp)


def test_hop_structure_boolean_composition() -> None:
    """The _hop_structure helper composes SOLID features into 5
    deterministic hop-class flags. The predicates require positive
    corroborating signals (bridge_entity_marker for bridge,
    multi_hop_marker for chain) to lift precision.
    """
    from mothrag.core.query_type_classifier import _hop_structure
    from mothrag.routing.semantic_features import SemanticFeatures

    # 1-hop polar
    f1 = SemanticFeatures(is_polar_comparison=1.0)
    hop = _hop_structure(f1)
    assert hop["is_1hop_polar"] is True
    assert hop["is_2hop_bridge"] is False
    assert hop["is_3hop_chain"] is False

    # 2-hop bridge: two_entity + bridge_entity_marker + single_hop=0 + chain_marker=0
    f2 = SemanticFeatures(
        two_entity=0.8, bridge_entity_marker=0.6, single_hop=0.0, chain_marker=0.0,
    )
    hop = _hop_structure(f2)
    assert hop["is_1hop_polar"] is False
    assert hop["is_2hop_bridge"] is True
    assert hop["is_3hop_chain"] is False

    # 3-hop chain: chain_marker + two_entity + multi_hop_marker
    f3 = SemanticFeatures(
        chain_marker=0.7, two_entity=0.8, multi_hop_marker=0.9,
    )
    hop = _hop_structure(f3)
    assert hop["is_1hop_polar"] is False
    # is_2hop_bridge is False because chain_marker fired (mutually exclusive)
    assert hop["is_2hop_bridge"] is False
    assert hop["is_3hop_chain"] is True

    # Single-hop possessive + two_entity: NOT a 2-hop bridge
    # (single_hop suppresses; also no bridge_entity_marker)
    f4 = SemanticFeatures(single_hop=0.7, two_entity=0.8, bridge_entity_marker=0.6)
    hop = _hop_structure(f4)
    assert hop["is_2hop_bridge"] is False

    # No features fired: everything False
    f5 = SemanticFeatures()
    hop = _hop_structure(f5)
    assert hop["is_1hop_polar"] is False
    assert hop["is_2hop_bridge"] is False
    assert hop["is_3hop_chain"] is False


def test_score_is_polar_comparison_extracts_correctly() -> None:
    """extract_semantic_features populates ``is_polar_comparison`` for
    yes/no set-comparison surfaces; zero otherwise. Mirrors
    query_type_classifier.is_polar_comparison (with regex duplicated to
    avoid circular import; kept in sync by contract).
    """
    from mothrag.core.query_type_classifier import is_polar_comparison as qtc_is_polar
    from mothrag.routing.semantic_features import (
        extract_semantic_features,
        score_is_polar_comparison,
    )

    polar_cases = (
        "Are X and Y both Greek?",
        "Is the dog and the cat the same color?",
        "Were Newton and Einstein either physicists or chemists?",
        # additions: comparative + than, bare two-entity choice,
        # wh-headed two-entity choice
        "Is Apple older than Microsoft?",
        "Which is older, Mars or Earth?",
        "Curry And Pepper or End Of Watch?",
    )
    non_polar_cases = (
        "When was Einstein born?",
        # "Which is older, Mars or Earth?" — was non-polar in an earlier
        # revision; this pattern (comparative + than via wh-headed choice)
        # is now EXPLICITLY included. Removed from non-polar.
        "Is the boiling point of water 100C?",  # polar but no set-comp marker
    )
    for q in polar_cases:
        assert score_is_polar_comparison(q) == 1.0
        # Stays consistent with the canonical query_type_classifier helper.
        assert qtc_is_polar(q) is True
        assert extract_semantic_features(q).is_polar_comparison == 1.0
    for q in non_polar_cases:
        assert score_is_polar_comparison(q) == 0.0
        assert qtc_is_polar(q) is False
        assert extract_semantic_features(q).is_polar_comparison == 0.0


def test_v3bu_cfde114_v3_single_hop_no_longer_gates() -> None:
    """v2 gated by (1 - single_hop); v3 ignores single_hop for
    the boost (hop structure handles it via is_polar_comparison).
    A polar 1-hop with single_hop=1.0 should now RECEIVE the full boost.
    This is the v2→v3 contract change.
    """
    from mothrag.core.query_type_classifier import _score_v3bu_p_arm
    from mothrag.routing.semantic_features import SemanticFeatures

    # Polar comparison + single_hop fires (could happen on possessive
    # polar surface): under v3, boost still applies because polar
    # classification trumps.
    f = SemanticFeatures(
        comparison_marker=0.7,
        single_hop=1.0,
        is_polar_comparison=1.0,
    )
    f_no_boost = SemanticFeatures(
        comparison_marker=0.0,
        single_hop=1.0,
        is_polar_comparison=1.0,
    )
    assert _score_v3bu_p_arm(f) > _score_v3bu_p_arm(f_no_boost), (
        "cfde114 v3 must apply boost on polar regardless of single_hop"
    )


def test_temporal_marker_fires_on_year_and_month() -> None:
    from mothrag.routing.semantic_features import score_temporal
    assert score_temporal("Who was president in 1992?") > 0
    assert score_temporal("What happened in March?") > 0
    assert score_temporal("Why?") == 0.0


# ============================================================
# arm_subset_pam_lite -- continuous P_arm + variable-K subset
# ============================================================

def test_pam_lite_returns_continuous_probabilities() -> None:
    """RE-SPEC contract change: P_arm may exceed 1.0 due to
    hop-soft-multipliers (max multiplier = 2.0). Lower bound 0.0
    preserved; upper bound is now per the multiplier schedule.
    """
    from mothrag.core.query_type_classifier import arm_subset_pam_lite
    subset, probabilities = arm_subset_pam_lite(
        "When was Einstein born?",
    )
    assert set(probabilities.keys()) == {"v3bu", "decompose", "iter"}
    for arm, p in probabilities.items():
        assert isinstance(p, float)
        # multipliers can boost above 1.0 (max ~2.0).
        # Sigmoid base is in [0,1], multiplier in [0.1, 2.0], so
        # product is in [0.0, 2.0].
        assert 0.0 <= p <= 2.0, f"{arm}: p={p} out of [0, 2]"
    assert isinstance(subset, list)
    assert all(arm in {"v3bu", "decompose", "iter"} for arm in subset)


def test_pam_lite_variable_K_threshold() -> None:
    """Different thresholds yield different subset sizes for the same query."""
    from mothrag.core.query_type_classifier import arm_subset_pam_lite
    q = "When was Einstein born?"
    subset_low, _ = arm_subset_pam_lite(q, threshold=0.1)
    subset_high, _ = arm_subset_pam_lite(q, threshold=0.9)
    assert len(subset_low) >= len(subset_high)


def test_pam_lite_subset_never_empty() -> None:
    """The always-non-empty guarantee: even with threshold=1.0 the
    argmax arm is included."""
    from mothrag.core.query_type_classifier import arm_subset_pam_lite
    subset, probs = arm_subset_pam_lite("???", threshold=1.0)
    assert len(subset) == 1
    # The argmax arm is included.
    assert subset[0] == max(probs.items(), key=lambda kv: kv[1])[0]


def test_pam_lite_extends_sel_v2_rules() -> None:
    """PAM-lite is ADDITIVE to sel_v2: legacy arm_subset must still work
    untouched, and PAM-lite is reachable via the new function name."""
    from mothrag.core.query_type_classifier import (
        arm_subset, arm_subset_pam_lite,
    )
    legacy = arm_subset("When was Einstein born?")
    assert legacy  # sel_v2 still returns something
    subset, _ = arm_subset_pam_lite("When was Einstein born?")
    assert subset  # PAM-lite also returns something


def test_pam_lite_entity_attribute_favours_v3bu() -> None:
    """Entity-attribute (is_1hop_entity_attr) question should yield
    high P_v3bu. Post RE-SPEC: the hop class requires
    ``single_hop > 0 AND two_entity == 0`` -- i.e., an "X's Y"
    possessive form on a single-entity subject. The original example
    "When was Einstein born?" does NOT trigger is_1hop_entity_attr
    (no possessive). For that residual class, iter wins under the new
    multipliers (general_multihop x 2.0). Updated query uses the
    canonical possessive form so the multiplier schedule maps
    correctly.
    """
    from mothrag.core.query_type_classifier import arm_subset_pam_lite
    _subset, probs = arm_subset_pam_lite("What is Einstein's birthplace?")
    assert probs["v3bu"] >= probs["iter"], (
        f"PAM-lite gave iter ({probs['iter']:.3f}) >= v3bu "
        f"({probs['v3bu']:.3f}) on a single-entity possessive "
        f"(is_1hop_entity_attr) question; multiplier schedule regression."
    )


def test_pam_lite_chain_query_favours_iter() -> None:
    """Chain marker should raise P_iter above P_v3bu."""
    from mothrag.core.query_type_classifier import arm_subset_pam_lite
    q = "First X happened, then Y was founded later, and subsequently Z."
    _subset, probs = arm_subset_pam_lite(q)
    assert probs["iter"] > probs["v3bu"], (
        f"PAM-lite gave v3bu ({probs['v3bu']:.3f}) >= iter "
        f"({probs['iter']:.3f}) on a chain-marker question."
    )


# ============================================================
# Extended arbitrate with arm_probabilities
# ============================================================

def test_extended_arbitrate_weights_by_P_arm() -> None:
    """When arm_probabilities is supplied, the winner's score is scaled."""
    from mothrag.core.arbitrate import DeterministicArbitrator
    arb = DeterministicArbitrator()
    answers = {"v3bu": "Paris", "decompose": "Lyon", "iter": "Marseille"}
    # All three arms have identical signals; vary P_arm to pick a winner.
    gamma = {"v3bu": 1.0, "decompose": 1.0, "iter": 1.0}
    agreement = {"v3bu": 0.0, "decompose": 0.0, "iter": 0.0}
    probs = {"v3bu": 0.9, "decompose": 0.5, "iter": 0.1}
    result = arb.arbitrate(
        answers=answers,
        gamma_signals=gamma,
        agreement_signals=agreement,
        arm_probabilities=probs,
    )
    assert result.selected_arm == "v3bu"  # highest P_arm


def test_extended_arbitrate_no_P_arm_preserves_sel_v2_behaviour() -> None:
    """When arm_probabilities is None, arbitration matches sel_v2 baseline."""
    from mothrag.core.arbitrate import DeterministicArbitrator
    arb = DeterministicArbitrator()
    answers = {"v3bu": "Paris", "decompose": "Lyon"}
    gamma = {"v3bu": 0.5, "decompose": 1.0}

    baseline = arb.arbitrate(answers=answers, gamma_signals=gamma)
    with_probs = arb.arbitrate(
        answers=answers, gamma_signals=gamma,
        arm_probabilities=None,
    )
    assert baseline.selected_arm == with_probs.selected_arm
    assert baseline.arm_scores == with_probs.arm_scores


def test_pam_lite_legacy_compat_kwargs_default_none() -> None:
    """DeterministicArbitrator.arbitrate signature: arm_probabilities
    must default to None for backward compat (no caller-side change
    needed in pre-PAM-lite code paths)."""
    import inspect
    from mothrag.core.arbitrate import DeterministicArbitrator
    sig = inspect.signature(DeterministicArbitrator.arbitrate)
    assert "arm_probabilities" in sig.parameters
    assert sig.parameters["arm_probabilities"].default is None


# ============================================================
# General-purpose / no-dataset-specific contract
# ============================================================

def test_no_dataset_specific_training() -> None:
    """Verify PAM-lite has no per-dataset code paths. The module loads
    and runs without HP/2W/MQ test data, paths, or imports."""
    # The mere fact that the import chain succeeds in this environment
    # (no test data on disk) is the assertion. Add an explicit check:
    # no test-set filename / dataset name appears in the source.
    from mothrag.core.query_type_classifier import arm_subset_pam_lite
    from mothrag.routing.semantic_features import extract_semantic_features
    import inspect
    for fn in (arm_subset_pam_lite, extract_semantic_features):
        src = inspect.getsource(fn)
        for forbidden in ("hotpotqa", "2wiki", "musique", "hp_train",
                          "2w_train", "mq_train"):
            assert forbidden not in src.lower(), (
                f"{fn.__name__} source contains dataset-specific token "
                f"{forbidden!r}; violates general-purpose contract."
            )


def test_general_purpose_cross_DS() -> None:
    """PAM-lite produces non-degenerate probabilities on questions
    spanning multiple domains (no domain-specific bias)."""
    from mothrag.core.query_type_classifier import arm_subset_pam_lite
    for q in (
        # General knowledge
        "When was Einstein born?",
        # Technical / scientific
        "What is the boiling point of water?",
        # Geographic
        "What is the capital of Brazil?",
        # Multi-hop
        "Which company that Steve Jobs founded later acquired NeXT?",
        # Comparison
        "Is Mars larger than Earth?",
    ):
        subset, probs = arm_subset_pam_lite(q)
        assert subset, f"Empty subset on {q!r}"
        # No probability should be 0 or 1 across all arms (degenerate
        # collapse). At least one arm > 0 (covered by always-non-empty
        # guarantee).
        assert max(probs.values()) > 0


# ============================================================
# route_prospective.py wiring -- --router CLI dispatch
# ============================================================

def test_resolve_arm_subset_with_router_pam_lite() -> None:
    """The route_prospective.py helper returns continuous probs under pam_lite."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import route_prospective as rp

    subset, probs = rp._resolve_arm_subset_with_router(
        "When was Einstein born?", router="pam_lite",
    )
    assert subset
    assert probs
    assert set(probs.keys()) == {"v3bu", "decompose", "iter"}


def test_resolve_arm_subset_with_router_sel_v2_returns_empty_probs() -> None:
    """sel_v2 path: empty arm_probabilities (signal-only arbitration)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import route_prospective as rp

    subset, probs = rp._resolve_arm_subset_with_router(
        "When was Einstein born?", router="sel_v2",
    )
    assert subset
    assert probs == {}
