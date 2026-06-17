"""Tests for arbitrate_pam_lite.

Verifies the three modes (argmax / weighted_mix / subset), uncertain-pred
filter, and the pool-safety axiom (P_arm=0 excludes arm).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _block_sentence_transformers(monkeypatch) -> Iterator[None]:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    yield


# ============================================================
# Mode: argmax (default)
# ============================================================

def test_argmax_picks_highest_p_arm() -> None:
    from mothrag.core.arbitrate import arbitrate_pam_lite
    chosen, reason = arbitrate_pam_lite(
        preds={"v3bu": "Paris", "decompose": "Lyon", "iter": "Marseille"},
        p_arm={"v3bu": 0.9, "decompose": 0.5, "iter": 0.2},
    )
    assert chosen == "Paris"
    assert reason.startswith("pamlite:argmax_v3bu_P=")


def test_argmax_decompose_dominant() -> None:
    from mothrag.core.arbitrate import arbitrate_pam_lite
    chosen, reason = arbitrate_pam_lite(
        preds={"v3bu": "X1", "decompose": "X2", "iter": "X3"},
        p_arm={"v3bu": 0.1, "decompose": 0.9, "iter": 0.4},
    )
    assert chosen == "X2"
    assert reason.startswith("pamlite:argmax_decompose_P=")


def test_argmax_skips_uncertain_pred() -> None:
    """argmax arm's pred being uncertain falls back to next-highest-P arm."""
    from mothrag.core.arbitrate import arbitrate_pam_lite
    chosen, reason = arbitrate_pam_lite(
        preds={"v3bu": "not in passages", "decompose": "Lyon", "iter": "Marseille"},
        p_arm={"v3bu": 0.95, "decompose": 0.5, "iter": 0.4},
    )
    assert chosen == "Lyon"
    assert reason.startswith("pamlite:argmax_decompose_P=")


# ============================================================
# Mode: weighted_mix
# ============================================================

def test_weighted_mix_blends_predictions() -> None:
    """Two arms agree (same pred); their P_arms sum; beats a third arm's
    higher individual P. weighted_mix rewards consensus.
    """
    from mothrag.core.arbitrate import arbitrate_pam_lite
    # v3bu + decompose agree on "Paris" (combined P=0.4+0.4=0.8);
    # iter disagrees with "Lyon" (P=0.7). Consensus wins.
    chosen, reason = arbitrate_pam_lite(
        preds={"v3bu": "Paris", "decompose": "Paris", "iter": "Lyon"},
        p_arm={"v3bu": 0.4, "decompose": 0.4, "iter": 0.7},
        mode="weighted_mix",
    )
    assert chosen == "Paris"
    assert reason.startswith("pamlite:weighted_mix_")


def test_weighted_mix_argmax_when_no_consensus() -> None:
    """All preds distinct: weighted_mix degenerates to argmax."""
    from mothrag.core.arbitrate import arbitrate_pam_lite
    chosen, reason = arbitrate_pam_lite(
        preds={"v3bu": "P", "decompose": "Q", "iter": "R"},
        p_arm={"v3bu": 0.2, "decompose": 0.3, "iter": 0.85},
        mode="weighted_mix",
    )
    assert chosen == "R"
    assert reason.startswith("pamlite:weighted_mix_iter_")


# ============================================================
# Mode: subset
# ============================================================

def test_subset_mode_filters_by_threshold() -> None:
    """subset mode excludes arms with P_arm <= threshold, then argmax."""
    from mothrag.core.arbitrate import arbitrate_pam_lite
    # Threshold=0.5 excludes decompose (0.4) and iter (0.2); only v3bu passes.
    chosen, reason = arbitrate_pam_lite(
        preds={"v3bu": "Paris", "decompose": "Lyon", "iter": "Marseille"},
        p_arm={"v3bu": 0.9, "decompose": 0.4, "iter": 0.2},
        mode="subset",
        threshold=0.5,
    )
    assert chosen == "Paris"
    assert reason.startswith("pamlite:subset_v3bu_P=")


def test_subset_mode_falls_back_to_argmax_when_empty() -> None:
    """When threshold excludes ALL arms, fall back to global argmax
    (pool-safety: always return something).
    """
    from mothrag.core.arbitrate import arbitrate_pam_lite
    chosen, reason = arbitrate_pam_lite(
        preds={"v3bu": "Paris", "decompose": "Lyon", "iter": "Marseille"},
        p_arm={"v3bu": 0.2, "decompose": 0.1, "iter": 0.15},
        mode="subset",
        threshold=0.5,  # filters everything
    )
    assert chosen == "Paris"  # argmax fallback
    assert "subset_fallback_argmax_" in reason


# ============================================================
# Pool-safety axiom: P_arm=0 excludes arm
# ============================================================

def test_p_arm_zero_excludes_arm_from_arbitration() -> None:
    """When v3bu P_arm=0, it does NOT win argmax even if its pred is
    valid. The next-highest-P arm wins.
    """
    from mothrag.core.arbitrate import arbitrate_pam_lite
    chosen, reason = arbitrate_pam_lite(
        preds={"v3bu": "Paris", "decompose": "Lyon", "iter": "Marseille"},
        p_arm={"v3bu": 0.0, "decompose": 0.7, "iter": 0.4},
    )
    assert chosen == "Lyon"
    assert reason.startswith("pamlite:argmax_decompose_P=")


def test_pool_safety_weighted_mix_zero_arm_contributes_nothing() -> None:
    """In weighted_mix: an arm with P_arm=0 cannot win even if its pred
    agrees with another arm. The non-zero arms drive the sum.
    """
    from mothrag.core.arbitrate import arbitrate_pam_lite
    # v3bu agrees with decompose on "Paris" but P_v3bu=0 so contributes 0.
    # Sum for "Paris" = 0.0 + 0.5 = 0.5. iter's "Lyon" has P=0.6 alone.
    # iter wins.
    chosen, reason = arbitrate_pam_lite(
        preds={"v3bu": "Paris", "decompose": "Paris", "iter": "Lyon"},
        p_arm={"v3bu": 0.0, "decompose": 0.5, "iter": 0.6},
        mode="weighted_mix",
    )
    assert chosen == "Lyon"


# ============================================================
# Boundary cases
# ============================================================

def test_empty_preds_returns_empty() -> None:
    """No predictions at all -> empty string + no_preds reason."""
    from mothrag.core.arbitrate import arbitrate_pam_lite
    chosen, reason = arbitrate_pam_lite(
        preds={"v3bu": "", "decompose": "", "iter": None},
        p_arm={"v3bu": 0.5, "decompose": 0.5, "iter": 0.5},
    )
    assert chosen == ""
    assert "no_preds" in reason


def test_single_arm_only() -> None:
    """Only v3bu has a prediction; iter / decompose missing -> v3bu wins."""
    from mothrag.core.arbitrate import arbitrate_pam_lite
    chosen, reason = arbitrate_pam_lite(
        preds={"v3bu": "Paris"},
        p_arm={"v3bu": 0.5},
    )
    assert chosen == "Paris"


def test_all_p_arm_below_threshold_subset_fallback() -> None:
    """subset with high threshold: all excluded; argmax fallback yields
    SOMETHING (pool-safety always-non-empty contract).
    """
    from mothrag.core.arbitrate import arbitrate_pam_lite
    chosen, reason = arbitrate_pam_lite(
        preds={"v3bu": "X1", "decompose": "X2", "iter": "X3"},
        p_arm={"v3bu": 0.01, "decompose": 0.02, "iter": 0.03},
        mode="subset",
        threshold=0.9,
    )
    # All filtered; global argmax fallback picks iter (highest P=0.03).
    assert chosen == "X3"
    assert "subset_fallback_argmax_iter" in reason


def test_all_uncertain_falls_back_to_first_nonempty() -> None:
    """Every pred is uncertain (e.g., 'unknown', 'no answer'). The
    fallback returns a non-empty pred per priority order.
    """
    from mothrag.core.arbitrate import arbitrate_pam_lite
    chosen, reason = arbitrate_pam_lite(
        preds={"v3bu": "unknown", "decompose": "no answer", "iter": "not in passages"},
        p_arm={"v3bu": 0.5, "decompose": 0.5, "iter": 0.5},
    )
    # All uncertain -> fallback yields first non-empty pred per priority,
    # which is v3bu ("unknown").
    assert chosen == "unknown"
    assert "all_uncertain_fallback_v3bu" in reason


# ============================================================
# Contract violation: invalid mode
# ============================================================

def test_invalid_mode_raises() -> None:
    from mothrag.core.arbitrate import arbitrate_pam_lite
    with pytest.raises(ValueError):
        arbitrate_pam_lite(
            preds={"v3bu": "Paris"},
            p_arm={"v3bu": 0.5},
            mode="foobar",
        )


# ============================================================
# Tie-breaking (stable order: v3bu > decompose > iter)
# ============================================================

def test_argmax_tie_priority_v3bu() -> None:
    """When v3bu and decompose tie on P, v3bu wins (priority order)."""
    from mothrag.core.arbitrate import arbitrate_pam_lite
    chosen, reason = arbitrate_pam_lite(
        preds={"v3bu": "Paris", "decompose": "Lyon", "iter": "Marseille"},
        p_arm={"v3bu": 0.5, "decompose": 0.5, "iter": 0.3},
    )
    assert chosen == "Paris"


def test_pam_lite_arbitrator_exported_from_arbitrate_package() -> None:
    """arbitrate_pam_lite is re-exported from mothrag.core.arbitrate."""
    from mothrag.core.arbitrate import arbitrate_pam_lite
    assert callable(arbitrate_pam_lite)
