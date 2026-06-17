"""Tests for arbitrate_pam_lite_traced instrumentation."""

from __future__ import annotations

import pytest


# ============================================================
# Byte-compat: arbitrate_pam_lite unchanged
# ============================================================

def test_legacy_arbitrate_pam_lite_unchanged_byte_compat() -> None:
    """The original 2-tuple return contract is preserved."""
    from mothrag.core.arbitrate.pam_lite_arbitrator import (
        arbitrate_pam_lite,
    )

    pred, reason = arbitrate_pam_lite(
        {"v3bu": "Paris", "decompose": "Lyon", "iter": "Berlin"},
        {"v3bu": 0.6, "decompose": 0.4, "iter": 0.2},
    )
    assert pred == "Paris"
    assert "argmax_v3bu" in reason


# ============================================================
# Diagnostic trace shape
# ============================================================

def test_traced_returns_diagnostic_dataclass() -> None:
    from mothrag.core.arbitrate.pam_lite_arbitrator import (
        arbitrate_pam_lite_traced, PamLiteDiagnostic,
    )

    pred, reason, diag = arbitrate_pam_lite_traced(
        {"v3bu": "Paris", "decompose": "Lyon", "iter": "Berlin"},
        {"v3bu": 0.6, "decompose": 0.4, "iter": 0.2},
    )
    assert pred == "Paris"
    assert isinstance(diag, PamLiteDiagnostic)
    assert diag.mode == "argmax"
    assert diag.winner_arm == "v3bu"
    assert diag.winner_pred == "Paris"
    assert diag.winner_score == 0.6
    assert diag.fallback_fired is False
    assert diag.tie_break_strategy == "priority"


def test_diagnostic_dataclass_frozen() -> None:
    from mothrag.core.arbitrate.pam_lite_arbitrator import (
        arbitrate_pam_lite_traced,
    )
    from dataclasses import FrozenInstanceError

    _pred, _reason, diag = arbitrate_pam_lite_traced(
        {"v3bu": "A"}, {"v3bu": 0.5},
    )
    with pytest.raises(FrozenInstanceError):
        diag.winner_arm = "x"  # type: ignore[misc]


# ============================================================
# Tie-break strategies
# ============================================================

def test_tie_break_priority_default() -> None:
    from mothrag.core.arbitrate.pam_lite_arbitrator import (
        arbitrate_pam_lite_traced,
    )

    # All three arms have equal P_arm; priority order picks v3bu
    pred, _r, diag = arbitrate_pam_lite_traced(
        {"v3bu": "Paris", "decompose": "Lyon", "iter": "Berlin"},
        {"v3bu": 0.5, "decompose": 0.5, "iter": 0.5},
        tie_break="priority",
    )
    assert diag.winner_arm == "v3bu"
    assert diag.tie_break_fired is True


def test_tie_break_lexicographic() -> None:
    from mothrag.core.arbitrate.pam_lite_arbitrator import (
        arbitrate_pam_lite_traced,
    )

    # Lexicographic: "decompose" < "iter" < "v3bu" -> decompose wins ties
    pred, _r, diag = arbitrate_pam_lite_traced(
        {"v3bu": "Paris", "decompose": "Lyon", "iter": "Berlin"},
        {"v3bu": 0.5, "decompose": 0.5, "iter": 0.5},
        tie_break="lexicographic",
    )
    assert diag.winner_arm == "decompose"
    assert diag.tie_break_fired is True


def test_tie_break_first_insertion_order() -> None:
    from mothrag.core.arbitrate.pam_lite_arbitrator import (
        arbitrate_pam_lite_traced,
    )

    # Insertion order: iter first -> iter wins ties
    pred, _r, diag = arbitrate_pam_lite_traced(
        {"iter": "C", "decompose": "B", "v3bu": "A"},
        {"v3bu": 0.5, "decompose": 0.5, "iter": 0.5},
        tie_break="first",
    )
    assert diag.winner_arm == "iter"
    assert diag.tie_break_fired is True


def test_tie_break_unknown_strategy_raises() -> None:
    from mothrag.core.arbitrate.pam_lite_arbitrator import (
        arbitrate_pam_lite_traced,
    )

    with pytest.raises(ValueError, match="tie_break"):
        arbitrate_pam_lite_traced(
            {"v3bu": "A"}, {"v3bu": 0.5},
            tie_break="bogus_unknown",  # type: ignore[arg-type]
        )


def test_tie_break_not_fired_when_unique_max() -> None:
    from mothrag.core.arbitrate.pam_lite_arbitrator import (
        arbitrate_pam_lite_traced,
    )

    _pred, _r, diag = arbitrate_pam_lite_traced(
        {"v3bu": "A", "decompose": "B"},
        {"v3bu": 0.9, "decompose": 0.1},
    )
    assert diag.tie_break_fired is False


# ============================================================
# Subset mode + threshold
# ============================================================

def test_subset_mode_filters_below_threshold() -> None:
    from mothrag.core.arbitrate.pam_lite_arbitrator import (
        arbitrate_pam_lite_traced,
    )

    pred, _r, diag = arbitrate_pam_lite_traced(
        {"v3bu": "Paris", "decompose": "Lyon", "iter": "Berlin"},
        {"v3bu": 0.7, "decompose": 0.5, "iter": 0.1},
        mode="subset", threshold=0.3,
    )
    assert diag.mode == "subset"
    assert diag.subset_arms == ("v3bu", "decompose")
    assert diag.winner_arm == "v3bu"
    assert diag.fallback_fired is False


def test_subset_mode_empty_filter_falls_back_to_argmax() -> None:
    from mothrag.core.arbitrate.pam_lite_arbitrator import (
        arbitrate_pam_lite_traced,
    )

    pred, _r, diag = arbitrate_pam_lite_traced(
        {"v3bu": "Paris", "decompose": "Lyon"},
        {"v3bu": 0.1, "decompose": 0.05},
        mode="subset", threshold=0.5,
    )
    assert diag.fallback_fired is True
    assert "subset_empty_global_argmax" in diag.fallback_path
    assert diag.winner_arm == "v3bu"


def test_subset_mode_disable_fallback_returns_empty() -> None:
    from mothrag.core.arbitrate.pam_lite_arbitrator import (
        arbitrate_pam_lite_traced,
    )

    pred, _r, diag = arbitrate_pam_lite_traced(
        {"v3bu": "Paris", "decompose": "Lyon"},
        {"v3bu": 0.1, "decompose": 0.05},
        mode="subset", threshold=0.5,
        disable_fallback=True,
    )
    assert pred == ""
    assert diag.fallback_fired is True
    assert "subset_empty_no_fallback" in diag.fallback_path


# ============================================================
# Weighted-vote mode trace
# ============================================================

def test_weighted_mix_trace_records_buckets() -> None:
    from mothrag.core.arbitrate.pam_lite_arbitrator import (
        arbitrate_pam_lite_traced,
    )

    # Two arms agreeing on "Paris" (sum 0.8) beat single iter on "Lyon" (0.7)
    pred, _r, diag = arbitrate_pam_lite_traced(
        {"v3bu": "Paris", "decompose": "Paris", "iter": "Lyon"},
        {"v3bu": 0.4, "decompose": 0.4, "iter": 0.7},
        mode="weighted_mix",
    )
    assert pred == "Paris"
    assert diag.extra.get("bucket_count") == 2


# ============================================================
# Fallback control
# ============================================================

def test_disable_fallback_with_all_uncertain_returns_empty() -> None:
    from mothrag.core.arbitrate.pam_lite_arbitrator import (
        arbitrate_pam_lite_traced,
    )

    pred, _r, diag = arbitrate_pam_lite_traced(
        {"v3bu": "Not in passages", "decompose": "unknown"},
        {"v3bu": 0.5, "decompose": 0.5},
        disable_fallback=True,
    )
    assert pred == ""
    assert diag.fallback_fired is True
    assert diag.fallback_path == "no_fallback_flag"


def test_default_fallback_path_when_all_uncertain() -> None:
    """When NOT disabled, the all-uncertain fallback returns the first
    non-empty pred in _ARM_PRIORITY order."""
    from mothrag.core.arbitrate.pam_lite_arbitrator import (
        arbitrate_pam_lite_traced,
    )

    pred, _r, diag = arbitrate_pam_lite_traced(
        {"v3bu": "Not in passages", "decompose": "unknown"},
        {"v3bu": 0.5, "decompose": 0.5},
    )
    # Both uncertain, but legacy fallback returns the first non-empty pred
    assert diag.fallback_fired is True
    assert "all_uncertain_fallback" in diag.fallback_path


# ============================================================
# raw_p_arm snapshot includes all input keys
# ============================================================

def test_raw_p_arm_snapshot_covers_all_keys() -> None:
    from mothrag.core.arbitrate.pam_lite_arbitrator import (
        arbitrate_pam_lite_traced,
    )

    _pred, _r, diag = arbitrate_pam_lite_traced(
        {"v3bu": "A", "v3bu_dup_a": "A", "decompose": "B"},
        {"v3bu": 0.5, "v3bu_dup_a": 0.5, "decompose": 0.4},
    )
    assert set(diag.raw_p_arm.keys()) >= {"v3bu", "v3bu_dup_a", "decompose"}


# ============================================================
# Mechanism A/B scenarios
# ============================================================

def test_mechanism_ab_4arm_vs_3arm_trace_diff() -> None:
    """3-arm vs 4-arm (dup-v3bu): the trace contains extra arms in
    eligibility but the winner shouldn't change in pure argmax mode
    (P_arm-based). Mechanism shift happens via agreement (Stage 6 of
    DeterministicArbitrator) -- this test just verifies trace
    consistency under the arbitrate_pam_lite path."""
    from mothrag.core.arbitrate.pam_lite_arbitrator import (
        arbitrate_pam_lite_traced,
    )

    _p3, _r3, diag3 = arbitrate_pam_lite_traced(
        {"v3bu": "Paris", "decompose": "Lyon", "iter": "Lyon"},
        {"v3bu": 0.6, "decompose": 0.5, "iter": 0.4},
    )
    _p4, _r4, diag4 = arbitrate_pam_lite_traced(
        {
            "v3bu": "Paris", "v3bu_dup_a": "Paris",
            "decompose": "Lyon", "iter": "Lyon",
        },
        {
            "v3bu": 0.6, "v3bu_dup_a": 0.6,
            "decompose": 0.5, "iter": 0.4,
        },
    )

    # Pure argmax: same winner (v3bu has highest P_arm in both)
    assert diag3.winner_arm == "v3bu"
    assert diag4.winner_arm == "v3bu"
    # But the eligible pool grew
    assert len(diag4.eligible_arms) > len(diag3.eligible_arms)
