"""Tests for hop-structured soft multipliers on PAM-lite arm scorers.

Verifies that the boolean composition of SOLID features (validated by an
audit, F1>=0.85) yields the expected hop-class flags AND that
``get_hop_weight`` applies the correct per-(arm, hop) multiplier so the
final ``_score_*_p_arm`` outputs shape per the spec.

Framing: "Hop-structured probability shaping" — continuous P_arm
preserved, soft multipliers gated by validated SOLID features.
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
# Hop predicates (extended)
# ============================================================

def test_hop_structure_includes_new_predicates() -> None:
    from mothrag.core.query_type_classifier import _hop_structure
    from mothrag.routing.semantic_features import SemanticFeatures

    hop = _hop_structure(SemanticFeatures())
    assert "is_1hop_polar"       in hop
    assert "is_2hop_bridge"      in hop
    assert "is_3hop_chain"       in hop
    assert "is_1hop_entity_attr" in hop
    assert "is_general_multihop" in hop


def test_is_1hop_entity_attr_fires_on_single_entity_with_attribute() -> None:
    """Lexicon revision: entity-attr now gated by single_entity
    (a SOLID feature, F1>=0.85) AND attribute_marker AND no
    multi-hop signals."""
    from mothrag.core.query_type_classifier import _hop_structure
    from mothrag.routing.semantic_features import SemanticFeatures

    # single_entity + attribute_marker + no two_entity / multi_hop / chain
    f = SemanticFeatures(single_entity=0.8, attribute_marker=0.5)
    hop = _hop_structure(f)
    assert hop["is_1hop_entity_attr"] is True


def test_is_1hop_entity_attr_suppressed_without_attribute_marker() -> None:
    """single_entity alone WITHOUT an attribute marker no
    longer fires entity-attr (was P=0.014 in the silver baseline)."""
    from mothrag.core.query_type_classifier import _hop_structure
    from mothrag.routing.semantic_features import SemanticFeatures

    f = SemanticFeatures(single_entity=0.8, attribute_marker=0.0)
    hop = _hop_structure(f)
    assert hop["is_1hop_entity_attr"] is False


def test_is_1hop_entity_attr_suppressed_by_two_entity() -> None:
    from mothrag.core.query_type_classifier import _hop_structure
    from mothrag.routing.semantic_features import SemanticFeatures

    # single_entity AND two_entity (overlap) -> NOT entity-attr
    # (two-entity dominates -> probably bridge).
    f = SemanticFeatures(single_entity=0.8, two_entity=0.8, attribute_marker=0.5)
    hop = _hop_structure(f)
    assert hop["is_1hop_entity_attr"] is False


def test_is_1hop_entity_attr_suppressed_by_multi_hop_marker() -> None:
    """entity-attr requires absence of subordinate-clause bridge."""
    from mothrag.core.query_type_classifier import _hop_structure
    from mothrag.routing.semantic_features import SemanticFeatures

    f = SemanticFeatures(
        single_entity=0.8, attribute_marker=0.5, multi_hop_marker=0.9,
    )
    hop = _hop_structure(f)
    assert hop["is_1hop_entity_attr"] is False


def test_is_general_multihop_positive_signal_required() -> None:
    """general_multihop is no longer pure residual. It requires
    a POSITIVE multi_hop_marker (subordinate-clause bridge "that X did
    Y") in addition to NOT matching any specific class.

    Rationale: the residual class in the silver baseline had P=0.138 (broke
    false fires on 499/1000 queries). Switching to positive multi_hop_marker
    turns residual into a signal-bearing predicate.
    """
    from mothrag.core.query_type_classifier import _hop_structure
    from mothrag.routing.semantic_features import SemanticFeatures

    # Empty features: NO positive signal -> general_multihop=False
    hop = _hop_structure(SemanticFeatures())
    assert hop["is_general_multihop"] is False

    # Positive multi_hop_marker, no other class -> general_multihop=True
    hop = _hop_structure(SemanticFeatures(multi_hop_marker=0.9))
    assert hop["is_general_multihop"] is True

    # 1hop_polar overrides general_multihop even with multi_hop_marker
    hop = _hop_structure(
        SemanticFeatures(is_polar_comparison=1.0, multi_hop_marker=0.9),
    )
    assert hop["is_general_multihop"] is False

    # 2hop_bridge fires with bridge_entity_marker -> general off
    hop = _hop_structure(SemanticFeatures(
        two_entity=0.8, bridge_entity_marker=0.6, multi_hop_marker=0.9,
    ))
    assert hop["is_general_multihop"] is False

    # 3hop_chain fires with all signals -> general off
    hop = _hop_structure(SemanticFeatures(
        chain_marker=0.7, two_entity=0.8, multi_hop_marker=0.9,
    ))
    assert hop["is_general_multihop"] is False


# ============================================================
# Soft multipliers — per-arm hop-conditioned weighting
# ============================================================

def test_v3bu_boosted_on_1hop_polar() -> None:
    """polar comparison gets v3bu x 2.0; a neutral query with
    no hop class fires (default 1.0) should yield strictly lower."""
    from mothrag.core.query_type_classifier import _score_v3bu_p_arm
    from mothrag.routing.semantic_features import SemanticFeatures

    # Polar query: v3bu multiplier = 2.0 via is_1hop_polar.
    f_polar = SemanticFeatures(
        is_polar_comparison=1.0, single_entity=0.5,
    )
    # Neutral: no hop class fires (no attribute_marker so no entity_attr,
    # no multi_hop_marker so no general; pure single_entity base).
    f_neutral = SemanticFeatures(
        is_polar_comparison=0.0, single_entity=0.5,
    )
    p_polar = _score_v3bu_p_arm(f_polar)
    p_neutral = _score_v3bu_p_arm(f_neutral)
    assert p_polar > p_neutral, (
        f"v3bu must be boosted on 1hop_polar: polar={p_polar:.4f} "
        f"neutral={p_neutral:.4f}"
    )


def test_v3bu_penalized_on_2hop_bridge() -> None:
    """2hop_bridge now requires bridge_entity_marker in addition
    to two_entity."""
    from mothrag.core.query_type_classifier import _score_v3bu_p_arm
    from mothrag.routing.semantic_features import SemanticFeatures

    # 2hop_bridge: v3bu multiplier = 0.1 (penalty). Compare to residual
    # (multiplier=1.0). Same base sigmoid inputs.
    f_bridge = SemanticFeatures(
        two_entity=0.8, single_entity=0.5, bridge_entity_marker=0.6,
    )
    f_residual = SemanticFeatures(single_entity=0.5)  # no bridge signal

    p_bridge = _score_v3bu_p_arm(f_bridge)
    p_residual = _score_v3bu_p_arm(f_residual)
    assert p_bridge < p_residual, (
        f"v3bu must be penalized on 2hop_bridge: bridge={p_bridge:.4f} "
        f"residual={p_residual:.4f}"
    )


def test_decompose_boosted_on_2hop_bridge() -> None:
    from mothrag.core.query_type_classifier import _score_decompose_p_arm
    from mothrag.routing.semantic_features import SemanticFeatures

    f_bridge = SemanticFeatures(two_entity=0.8, bridge_entity_marker=0.5)
    f_residual = SemanticFeatures(bridge_entity_marker=0.5)

    p_bridge = _score_decompose_p_arm(f_bridge)
    p_residual = _score_decompose_p_arm(f_residual)
    # bridge has multiplier 2.0; residual has 1.0. Bridge dominates even
    # though two_entity adds +0.3 to base which slightly raises bridge.
    assert p_bridge > p_residual


def test_decompose_boosted_on_3hop_chain() -> None:
    """3hop_chain now requires multi_hop_marker (subordinate
    bridge) in addition to chain_marker + two_entity."""
    from mothrag.core.query_type_classifier import _score_decompose_p_arm
    from mothrag.routing.semantic_features import SemanticFeatures

    f_chain = SemanticFeatures(
        chain_marker=0.7, two_entity=0.8, multi_hop_marker=0.9,
    )
    # residual: pure base signal, no hop class fires (no multi_hop_marker)
    f_residual = SemanticFeatures(bridge_entity_marker=0.5)

    p_chain = _score_decompose_p_arm(f_chain)
    p_residual = _score_decompose_p_arm(f_residual)
    assert p_chain > p_residual, (
        f"decompose should be boosted on chain by multiplier 2.0: "
        f"chain={p_chain:.4f} residual={p_residual:.4f}"
    )


def test_iter_boosted_on_3hop_chain() -> None:
    """chain query needs all three signals to fire."""
    from mothrag.core.query_type_classifier import _score_iter_p_arm
    from mothrag.routing.semantic_features import SemanticFeatures

    f_chain = SemanticFeatures(
        chain_marker=0.7, two_entity=0.8, multi_hop_marker=0.9,
    )
    # residual: NO hop class fires (no multi_hop_marker)
    f_residual = SemanticFeatures(temporal_marker=0.4)

    p_chain = _score_iter_p_arm(f_chain)
    p_residual = _score_iter_p_arm(f_residual)
    # chain: multiplier 2.0. residual: default 1.0. chain wins.
    assert p_chain > p_residual


def test_iter_boosted_on_general_multihop() -> None:
    """general_multihop requires multi_hop_marker positive
    signal (was: pure residual)."""
    from mothrag.core.query_type_classifier import _score_iter_p_arm, get_hop_weight, _hop_structure
    from mothrag.routing.semantic_features import SemanticFeatures

    # general_multihop requires multi_hop_marker > 0 + no other class.
    f_general = SemanticFeatures(multi_hop_marker=0.9)
    hop = _hop_structure(f_general)
    assert hop["is_general_multihop"] is True
    # iter multiplier on general_multihop = 2.0
    assert get_hop_weight("iter", hop) == 2.0


def test_iter_default_on_empty_features() -> None:
    """empty features no longer fire general_multihop. iter
    falls through to default multiplier 1.0 (was: 2.0 via general)."""
    from mothrag.core.query_type_classifier import get_hop_weight, _hop_structure
    from mothrag.routing.semantic_features import SemanticFeatures

    hop = _hop_structure(SemanticFeatures())
    assert hop["is_general_multihop"] is False
    assert get_hop_weight("iter", hop) == 1.0


def test_iter_penalized_on_1hop_polar() -> None:
    from mothrag.core.query_type_classifier import get_hop_weight, _hop_structure
    from mothrag.routing.semantic_features import SemanticFeatures

    f = SemanticFeatures(is_polar_comparison=1.0)
    hop = _hop_structure(f)
    # iter on polar gets penalty 0.1
    assert get_hop_weight("iter", hop) == 0.1


# ============================================================
# Pool-safety + variable-K
# ============================================================

def test_residual_query_all_arms_fire() -> None:
    """Pool-safety: on a residual (no specific hop) query with strong
    base signals, all 3 arms should produce non-zero scores so the
    variable-K threshold permits ensemble (not collapse to single arm).
    """
    from mothrag.core.query_type_classifier import (
        _score_v3bu_p_arm, _score_decompose_p_arm, _score_iter_p_arm,
    )
    from mothrag.routing.semantic_features import SemanticFeatures

    # Generic non-specific query with mid-range features
    f = SemanticFeatures(
        single_entity=0.5, attribute_marker=0.4, bridge_entity_marker=0.4,
        temporal_marker=0.4,
    )
    p_v3bu = _score_v3bu_p_arm(f)
    p_dec  = _score_decompose_p_arm(f)
    p_iter = _score_iter_p_arm(f)
    # Pool safety: all > 0 (no zeroed arm)
    assert p_v3bu > 0.0
    assert p_dec  > 0.0
    assert p_iter > 0.0


def test_hop_predicates_overlap_documented_via_priority() -> None:
    """overlap test now requires bridge_entity_marker to fire
    2hop_bridge alongside 1hop_polar."""
    from mothrag.core.query_type_classifier import (
        _hop_structure, get_hop_weight,
    )
    from mothrag.routing.semantic_features import SemanticFeatures

    # Polar query that ALSO has two_entity + bridge marker
    # (e.g., "Are the founders of X and Y both engineers?")
    f = SemanticFeatures(
        is_polar_comparison=1.0, two_entity=0.8, bridge_entity_marker=0.5,
    )
    hop = _hop_structure(f)
    # Both flags fire (overlap)
    assert hop["is_1hop_polar"] is True
    assert hop["is_2hop_bridge"] is True
    # For v3bu: is_1hop_polar listed FIRST -> 2.0 wins
    assert get_hop_weight("v3bu", hop) == 2.0
    # For decompose: is_2hop_bridge listed FIRST -> 2.0 wins
    # (NOT is_1hop_polar=0.1 which is later in decompose's dict).
    assert get_hop_weight("decompose", hop) == 2.0


def test_pam_lite_subset_under_variable_k_with_multipliers() -> None:
    """Variable-K threshold contract preserved under multipliers: with
    threshold=0.3, a 1hop_polar query should YIELD v3bu in subset (x 2.0
    multiplier amplifies; decompose / iter penalty 0.1 still expected to
    fail threshold for queries lacking decompose/iter base features).
    """
    from mothrag.core.query_type_classifier import arm_subset_pam_lite

    subset, probs = arm_subset_pam_lite(
        "Are Newton and Einstein both physicists?", threshold=0.3,
    )
    assert "v3bu" in subset, (
        f"1hop_polar query should include v3bu after multiplier x 2.0. "
        f"Got subset={subset} probs={probs}"
    )


# ============================================================
# get_hop_weight contract
# ============================================================

def test_get_hop_weight_returns_default_when_no_cohort_fires() -> None:
    from mothrag.core.query_type_classifier import get_hop_weight

    hop = {
        "is_1hop_polar": False, "is_2hop_bridge": False,
        "is_3hop_chain": False, "is_1hop_entity_attr": False,
        "is_general_multihop": False,
    }
    # No cohort fires. This is a real case (empty features
    # yield no hop class). All arms fall through to default=1.0.
    assert get_hop_weight("v3bu", hop) == 1.0
    assert get_hop_weight("decompose", hop) == 1.0
    assert get_hop_weight("iter", hop) == 1.0


def test_get_hop_weight_unknown_arm_raises() -> None:
    from mothrag.core.query_type_classifier import get_hop_weight

    with pytest.raises(KeyError):
        get_hop_weight("nonexistent_arm", {})

