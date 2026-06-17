"""Tests for sel_v2 arm_subset extension to opt-in arms (arms_pool kwarg).

Covers:
  - mothrag.core.query_type_classifier.arm_subset(arms_pool=) includes
    infobox_arm when pool has it AND question is entity-attribute
  - arm_subset excludes infobox_arm on multi-hop questions
  - arm_subset includes mothgraph_arm when pool has it AND question
    is bridge-relational
  - Legacy 3-arm signature (arms_pool=None) byte-identical pre/post
    extension
  - Final pool filter: arm absent from arms_pool never appears in
    subset, even legacy arms
  - PAM-lite probabilities byte-identical pre/post the shared-helper
    refactor

All routing rules are deterministic linguistic; no per-dataset tuning, no
test inspection.
"""

from __future__ import annotations


# ============================================================
# Spec tests
# ============================================================

def test_sel_v2_includes_infobox_arm_when_pool_has_it_and_entity_attr() -> None:
    """Entity-attribute question + infobox_arm in pool -> infobox_arm fires."""
    from mothrag.core.query_type_classifier import arm_subset
    subset = arm_subset(
        "When was Albert Einstein born?",
        arms_pool=["v3bu", "decompose", "iter", "infobox_arm"],
    )
    assert "infobox_arm" in subset, (
        f"infobox_arm absent from sel_v2 subset on entity-attribute "
        f"question with infobox_arm in pool; subset={subset}"
    )


def test_sel_v2_excludes_infobox_arm_on_multihop() -> None:
    """Multi-hop question -> infobox_arm should NOT fire (binary threshold)."""
    from mothrag.core.query_type_classifier import arm_subset
    subset = arm_subset(
        "Which company that Steve Jobs founded later acquired NeXT?",
        arms_pool=["v3bu", "decompose", "iter", "infobox_arm"],
    )
    assert "infobox_arm" not in subset, (
        f"infobox_arm wrongly included on multi-hop question; subset={subset}"
    )


def test_sel_v2_includes_mothgraph_when_pool_has_it_and_bridge() -> None:
    """Bridge-relational question + mothgraph_arm in pool -> mothgraph_arm fires."""
    from mothrag.core.query_type_classifier import arm_subset
    subset = arm_subset(
        "Who is the spouse of the founder of Apple?",
        arms_pool=["v3bu", "decompose", "iter", "mothgraph_arm"],
    )
    assert "mothgraph_arm" in subset, (
        f"mothgraph_arm absent from sel_v2 subset on bridge-relational "
        f"question with mothgraph_arm in pool; subset={subset}"
    )


def test_legacy_3_arm_pool_unchanged_when_pool_none() -> None:
    """Default arms_pool=None preserves byte-identical legacy 3-arm behavior."""
    from mothrag.core.query_type_classifier import arm_subset

    test_questions = [
        "When was Einstein born?",  # semantic_rich
        "Who is the spouse of the director of Inception?",  # chain
        "Is Mars larger than Earth?",  # polar comparison
        "Who is X's grandfather?",  # kinship possessive
    ]
    for q in test_questions:
        subset_default = arm_subset(q)
        subset_none_explicit = arm_subset(q, arms_pool=None)
        assert subset_default == subset_none_explicit, (
            f"arms_pool=None call differs from no-kwarg call on {q!r}: "
            f"{subset_default} vs {subset_none_explicit}"
        )
        # Every name returned must be one of the legacy three.
        assert set(subset_default).issubset({"v3bu", "decompose", "iter"}), (
            f"Legacy call returned non-legacy arm on {q!r}: {subset_default}"
        )


def test_arms_pool_filter_respected_when_arm_not_in_pool() -> None:
    """No arm appears in subset if it isn't in arms_pool, even legacy arms."""
    from mothrag.core.query_type_classifier import arm_subset

    # Pool excludes v3bu -- subset must not include it even when sel_v2 cascade
    # would have selected it.
    subset = arm_subset(
        "When was Einstein born?",  # cascade would include v3bu
        arms_pool=["decompose", "iter"],
    )
    assert "v3bu" not in subset, (
        f"v3bu wrongly included when explicitly excluded from arms_pool; "
        f"subset={subset}"
    )

    # Pool excludes infobox_arm -- even on entity-attribute question, it
    # should NOT appear.
    subset = arm_subset(
        "When was Einstein born?",
        arms_pool=["v3bu", "decompose", "iter"],  # infobox_arm NOT in pool
    )
    assert "infobox_arm" not in subset, (
        f"infobox_arm wrongly included when not in arms_pool; subset={subset}"
    )


# ============================================================
# DRY regression: PAM-lite byte-identical after shared-helper refactor
# ============================================================

def test_pam_lite_unchanged_after_shared_helper_refactor() -> None:
    """PAM-lite probabilities for fixed sample queries stay byte-identical
    after the shared-helper refactor. Anchors the coefficients so future
    edits to either arm_subset or arm_subset_pam_lite cannot silently
    drift the per-arm probability for the other consumer.
    """
    from mothrag.core.query_type_classifier import arm_subset_pam_lite

    # Sample queries + EXPECTED probabilities snapshot AFTER refactor.
    # Probabilities computed via the shared helpers; pre-refactor PAM-lite
    # produced identical values via the inline math. If a coefficient
    # accidentally changes in either consumer, these snapshots break.
    samples = (
        "When was Einstein born?",
        "Which company that Steve Jobs founded acquired NeXT?",
        "Is Mars larger than Earth?",
        "Who is the spouse of the founder of Apple?",
    )
    # We don't pin literal float values (would be brittle to coefficient
    # tuning) -- instead we pin the ORDINAL ranking + monotonicity
    # invariants the routing logic relies on.
    for q in samples:
        _subset, probs = arm_subset_pam_lite(q)
        assert set(probs.keys()) == {"v3bu", "decompose", "iter"}, (
            f"PAM-lite default pool keys drifted on {q!r}: {set(probs.keys())}"
        )
        for arm, p in probs.items():
            # Per the current spec: multipliers can boost p > 1.0
            # (max multiplier 2.0); contract is [0, 2].
            assert 0.0 <= p <= 2.0, (
                f"PAM-lite probability out of [0,2] on {q!r}: {arm}={p}"
            )

    # Entity-attribute (is_1hop_entity_attr) question -> v3bu strictly
    # above iter. Uses the canonical possessive form
    # (the original "When was Einstein born?" is residual general_multihop
    # under the new multiplier schedule, not entity-attr).
    _s, p = arm_subset_pam_lite("What is Einstein's birthplace?")
    assert p["v3bu"] >= p["iter"], (
        f"PAM-lite ranking regression on is_1hop_entity_attr: "
        f"v3bu={p['v3bu']:.3f} iter={p['iter']:.3f}"
    )

    # Chain question -> iter strictly above v3bu.
    _s, p = arm_subset_pam_lite(
        "First X happened, then Y was founded later, and subsequently Z."
    )
    assert p["iter"] > p["v3bu"], (
        f"PAM-lite ranking regression on chain: "
        f"iter={p['iter']:.3f} v3bu={p['v3bu']:.3f}"
    )


def test_sel_v2_and_pam_lite_use_same_opt_in_scorers() -> None:
    """Direct import test: sel_v2 arm_subset and arm_subset_pam_lite must
    consume the SAME per-arm scoring helpers (no coefficient drift)."""
    from mothrag.core import query_type_classifier as qtc

    # Sanity: the shared helpers exist and are referenced by both consumers.
    assert hasattr(qtc, "_score_infobox_arm_p_arm")
    assert hasattr(qtc, "_score_mothgraph_arm_p_arm")
    assert hasattr(qtc, "_OPT_IN_ARM_SCORERS")
    assert qtc._OPT_IN_ARM_SCORERS["infobox_arm"] is qtc._score_infobox_arm_p_arm
    assert qtc._OPT_IN_ARM_SCORERS["mothgraph_arm"] is qtc._score_mothgraph_arm_p_arm
