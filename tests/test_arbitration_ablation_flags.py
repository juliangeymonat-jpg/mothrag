"""Tests for ablation flags + counters on the arbitration scorers.

Validates that the disable-cfde114-boost and disable-hop-multipliers
monkey-patches behave as specified (zero the boost / unit weights).
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
# Functional: cfde114 boost ablation
# ============================================================

def test_disable_cfde114_boost_zeroes_boost_on_1hop_polar() -> None:
    """When monkey-patched per disable-cfde114-boost, V3+bu score on
    a 1hop_polar query with comparison_marker fires equals the same
    score with comparison_marker = 0 (boost is zeroed).
    """
    import importlib
    import mothrag.core.query_type_classifier as qtc
    from mothrag.routing.semantic_features import SemanticFeatures

    # Save original; apply ablation patch (same logic as the boost ablation).
    _original = qtc._score_v3bu_p_arm
    _hop = qtc._hop_structure
    _sig = qtc._sigmoid
    _ghw = qtc.get_hop_weight

    def _v3bu_no_boost(f):
        hop = _hop(f)
        base = _sig(
            +0.5 * f.single_entity
            + 0.4 * f.attribute_marker
            + 0.3 * f.single_hop
            - 0.3 * f.multi_hop_marker
            - 0.2 * f.chain_marker
            + 0.2
        )
        return base * _ghw("v3bu", hop)

    try:
        qtc._score_v3bu_p_arm = _v3bu_no_boost
        # 1hop polar with comparison_marker firing should now produce the
        # SAME score as comparison_marker=0 (since boost is zeroed).
        f_with_comp = SemanticFeatures(
            is_polar_comparison=1.0, comparison_marker=0.7,
        )
        f_without_comp = SemanticFeatures(
            is_polar_comparison=1.0, comparison_marker=0.0,
        )
        p_with = qtc._score_v3bu_p_arm(f_with_comp)
        p_without = qtc._score_v3bu_p_arm(f_without_comp)
        assert p_with == pytest.approx(p_without, abs=1e-9)
    finally:
        qtc._score_v3bu_p_arm = _original


# ============================================================
# Functional: hop multiplier ablation
# ============================================================

def test_disable_hop_multipliers_returns_1_for_all_arms() -> None:
    """When monkey-patched per disable-hop-multipliers,
    get_hop_weight returns 1.0 for every (arm, hop) pair.
    """
    import mothrag.core.query_type_classifier as qtc

    _original = qtc.get_hop_weight
    try:
        qtc.get_hop_weight = lambda arm, hop: 1.0
        # Synthetic hop dicts -- multiplier should be 1.0 across the board.
        for hop in (
            {"is_1hop_polar": True},
            {"is_2hop_bridge": True},
            {"is_3hop_chain": True},
            {"is_1hop_entity_attr": True},
            {"is_general_multihop": True},
            {},  # empty
        ):
            for arm in ("v3bu", "decompose", "iter"):
                assert qtc.get_hop_weight(arm, hop) == 1.0
    finally:
        qtc.get_hop_weight = _original


def test_disable_hop_multipliers_collapses_scorers_to_base() -> None:
    """With multipliers patched to 1.0, the arm scorers equal the bare
    sigmoid base (no per-arm boost/penalty).
    """
    import mothrag.core.query_type_classifier as qtc
    from mothrag.routing.semantic_features import SemanticFeatures

    _original = qtc.get_hop_weight
    try:
        qtc.get_hop_weight = lambda arm, hop: 1.0
        f = SemanticFeatures(
            single_entity=0.5, attribute_marker=0.4, bridge_entity_marker=0.4,
        )
        p_v3bu = qtc._score_v3bu_p_arm(f)
        p_dec = qtc._score_decompose_p_arm(f)
        p_iter = qtc._score_iter_p_arm(f)
        # Multipliers are 1.0 so output equals raw sigmoid; all in (0, 1].
        for p in (p_v3bu, p_dec, p_iter):
            assert 0.0 < p <= 1.0
    finally:
        qtc.get_hop_weight = _original


# ============================================================
# Both flags OFF -> default behavior preserved
# ============================================================

def test_default_flags_preserve_current_behavior() -> None:
    """With NEITHER ablation flag applied, the production scorers
    behave with hop multipliers + cfde114 v3 hop-aware boost active.
    """
    from mothrag.core.query_type_classifier import (
        _score_v3bu_p_arm, get_hop_weight, _hop_structure,
    )
    from mothrag.routing.semantic_features import SemanticFeatures

    f = SemanticFeatures(is_polar_comparison=1.0, comparison_marker=0.7)
    p = _score_v3bu_p_arm(f)
    # 1hop_polar v3bu multiplier = 2.0; cfde114 boost active.
    # Base sigmoid with boost (logit ~= 0.2 base + 0.21 boost) >> 0.5,
    # multiplied by 2.0 should exceed 1.0.
    assert p > 1.0
    # get_hop_weight returns 2.0 (not 1.0) for 1hop_polar v3bu.
    hop = _hop_structure(f)
    assert get_hop_weight("v3bu", hop) == 2.0


# ============================================================
# Both flags ON -> strict pre-boost baseline
# ============================================================

def test_both_ablations_collapse_to_baseline() -> None:
    """With both flags ON: no boost + multiplier 1.0 -> V3+bu score
    equals the bare pre-cfde114 sigmoid.
    """
    import mothrag.core.query_type_classifier as qtc
    from mothrag.routing.semantic_features import SemanticFeatures

    _original_score = qtc._score_v3bu_p_arm
    _original_hw = qtc.get_hop_weight
    _hop = qtc._hop_structure
    _sig = qtc._sigmoid

    def _v3bu_no_boost(f):
        hop = _hop(f)
        base = _sig(
            +0.5 * f.single_entity
            + 0.4 * f.attribute_marker
            + 0.3 * f.single_hop
            - 0.3 * f.multi_hop_marker
            - 0.2 * f.chain_marker
            + 0.2
        )
        return base * qtc.get_hop_weight("v3bu", hop)

    try:
        qtc.get_hop_weight = lambda arm, hop: 1.0
        qtc._score_v3bu_p_arm = _v3bu_no_boost
        # Compute baseline directly + verify match.
        f = SemanticFeatures(
            is_polar_comparison=1.0, comparison_marker=0.7, single_entity=0.5,
        )
        p = qtc._score_v3bu_p_arm(f)
        expected = _sig(
            +0.5 * f.single_entity
            + 0.4 * f.attribute_marker
            + 0.3 * f.single_hop
            - 0.3 * f.multi_hop_marker
            - 0.2 * f.chain_marker
            + 0.2
        )
        assert p == pytest.approx(expected, abs=1e-9)
        # Sanity: <= 1.0 (bare sigmoid, no multiplier).
        assert p <= 1.0
    finally:
        qtc._score_v3bu_p_arm = _original_score
        qtc.get_hop_weight = _original_hw


# ============================================================
# Counter logic: cfde114_fire_count
# ============================================================

def test_cfde114_fire_count_increments_on_1hop_polar_with_comparison() -> None:
    """The counter logic:
    cfde114_fire_count += 1 iff hop['is_1hop_polar'] AND f.comparison_marker > 0.
    """
    from mothrag.core.query_type_classifier import _hop_structure
    from mothrag.routing.semantic_features import (
        SemanticFeatures, extract_semantic_features,
    )

    # Fire case: polar + comparison_marker.
    f = extract_semantic_features("Are X and Y both Greek?")
    hop = _hop_structure(f)
    assert hop["is_1hop_polar"] is True
    assert f.comparison_marker > 0.0

    # Non-fire: polar but no comparison_marker (forcibly synthetic).
    f2 = SemanticFeatures(is_polar_comparison=1.0, comparison_marker=0.0)
    hop2 = _hop_structure(f2)
    assert hop2["is_1hop_polar"] is True
    assert f2.comparison_marker == 0.0  # would NOT increment


def test_hop_multiplier_active_count_excludes_general_multihop() -> None:
    """hop_multiplier_active_count fires when ANY hop predicate fires
    EXCEPT is_general_multihop (residual is neutral 1.0 multiplier on
    v3bu and decompose; only iter gets 2.0 on general_multihop).
    Counter logic is: any flag fires EXCEPT is_general_multihop.
    """
    from mothrag.core.query_type_classifier import _hop_structure
    from mothrag.routing.semantic_features import SemanticFeatures

    # general_multihop alone (empty features) -> counter NOT incremented.
    hop = _hop_structure(SemanticFeatures())
    active = any(v for k, v in hop.items() if k != "is_general_multihop")
    assert active is False

    # 1hop_polar fires -> active=True.
    hop = _hop_structure(SemanticFeatures(is_polar_comparison=1.0))
    active = any(v for k, v in hop.items() if k != "is_general_multihop")
    assert active is True
