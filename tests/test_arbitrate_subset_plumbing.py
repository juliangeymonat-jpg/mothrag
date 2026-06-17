"""Tests for plumbing subset + p_arm through
arbitrate_with_c7 / arbitrate_excl_v3bu.

Verifies that upstream-provided subset + P_arm overrides the internal
classify_query_v2 routing; backward-compat preserved when subset=None.
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
# arbitrate_with_c7 subset / p_arm plumbing
# ============================================================

def test_arbitrate_uses_provided_subset_when_present() -> None:
    """When subset={'decompose', 'iter'} provided, v3bu_pred is NOT
    chosen even if internal classifier would select v3bu.
    """
    from mothrag.core.selective_ensemble import arbitrate_with_c7

    # "When was Einstein born?" is general_multihop / semantic_rich;
    # internal route_by_query_type_v2 would normally pick v3bu via
    # sel_v1 fallback. With subset={dec, iter}, v3bu must be skipped.
    chosen, reason, _ = arbitrate_with_c7(
        v3bu_pred="Paris",        # would normally be picked
        dec_pred="Lyon",
        question="When was Einstein born?",
        iter_pred="Marseille",
        use_router_v2=True,
        subset={"decompose", "iter"},
    )
    assert chosen != "Paris"
    assert chosen in ("Lyon", "Marseille")
    assert "plumb:" in reason


def test_arbitrate_uses_p_arm_for_pam_lite_argmax() -> None:
    """When subset + p_arm provided, argmax-by-p_arm picks within subset."""
    from mothrag.core.selective_ensemble import arbitrate_with_c7

    chosen, reason, _ = arbitrate_with_c7(
        v3bu_pred="Paris",
        dec_pred="Lyon",
        question="When was Einstein born?",
        iter_pred="Marseille",
        use_router_v2=True,
        subset={"v3bu", "decompose", "iter"},
        p_arm={"v3bu": 0.2, "decompose": 0.3, "iter": 0.9},
    )
    assert chosen == "Marseille"
    assert "argmax_iter" in reason


def test_arbitrate_falls_back_to_internal_classification_when_no_subset() -> None:
    """When subset=None, current behavior preserved (route_by_query_type_v2)."""
    from mothrag.core.selective_ensemble import arbitrate_with_c7

    # Baseline (no subset): should match legacy behavior.
    chosen_no_subset, reason_no_subset, _ = arbitrate_with_c7(
        v3bu_pred="Paris",
        dec_pred="Lyon",
        question="When was Einstein born?",
        iter_pred="Marseille",
        use_router_v2=True,
    )
    assert chosen_no_subset in ("Paris", "Lyon", "Marseille")
    # Routing reason should be LEGACY-style (router_v2 / sel_v1), not plumb.
    assert "plumb:" not in reason_no_subset


def test_arbitrate_subset_argmax_skips_uncertain_pred() -> None:
    """argmax arm with uncertain pred is skipped; next-highest wins."""
    from mothrag.core.selective_ensemble import arbitrate_with_c7

    chosen, reason, _ = arbitrate_with_c7(
        v3bu_pred="not in passages",  # uncertain
        dec_pred="Lyon",
        question="any question",
        iter_pred="Marseille",
        subset={"v3bu", "decompose", "iter"},
        p_arm={"v3bu": 0.95, "decompose": 0.5, "iter": 0.4},
    )
    assert chosen == "Lyon"


def test_arbitrate_subset_empty_in_subset_falls_back_pool_safety() -> None:
    """All in-subset preds uncertain -> fall back to first non-uncertain
    pred outside subset (pool safety).
    """
    from mothrag.core.selective_ensemble import arbitrate_with_c7

    chosen, reason, _ = arbitrate_with_c7(
        v3bu_pred="Paris",            # not in subset
        dec_pred="not in passages",   # in subset, uncertain
        question="any",
        iter_pred="unknown",          # in subset, uncertain
        subset={"decompose", "iter"},
        p_arm={"v3bu": 0.0, "decompose": 0.5, "iter": 0.4},
    )
    assert chosen == "Paris"
    assert "fallback_outside_subset" in reason


# ============================================================
# arbitrate_excl_v3bu subset / p_arm plumbing
# ============================================================

def test_arbitrate_excl_v3bu_with_subset_argmax() -> None:
    """When subset + p_arm provided in arbitrate_excl_v3bu, argmax over
    subset including v3bu_fallback (if "v3bu" in subset).
    """
    from mothrag.core.selective_ensemble import arbitrate_excl_v3bu

    # subset includes v3bu -> v3bu_fallback considered.
    chosen, reason = arbitrate_excl_v3bu(
        dec_pred="Lyon", iter_pred="Marseille", question="any",
        v3bu_fallback="Paris",
        subset={"v3bu", "decompose", "iter"},
        p_arm={"v3bu": 0.9, "decompose": 0.4, "iter": 0.2},
    )
    assert chosen == "Paris"
    assert "argmax_v3bu" in reason


def test_arbitrate_excl_v3bu_subset_excludes_v3bu_when_not_present() -> None:
    """When 'v3bu' NOT in subset, v3bu_fallback is filtered out even with
    highest P. Must pick from decompose / iter.
    """
    from mothrag.core.selective_ensemble import arbitrate_excl_v3bu

    chosen, reason = arbitrate_excl_v3bu(
        dec_pred="Lyon", iter_pred="Marseille", question="any",
        v3bu_fallback="Paris",
        subset={"decompose", "iter"},  # v3bu excluded
        p_arm={"v3bu": 0.9, "decompose": 0.4, "iter": 0.7},
    )
    assert chosen != "Paris"
    assert chosen == "Marseille"  # argmax over (dec=0.4, iter=0.7)


def test_select_via_subset_p_arm_helper_exposed() -> None:
    """The internal helper is importable for unit testing."""
    from mothrag.core.selective_ensemble import _select_via_subset_p_arm
    assert callable(_select_via_subset_p_arm)


def test_select_via_subset_p_arm_priority_tiebreak() -> None:
    """When p_arm ties, _ARM_PRIORITY_PLUMB order resolves it."""
    from mothrag.core.selective_ensemble import _select_via_subset_p_arm

    chosen, reason = _select_via_subset_p_arm(
        preds={"v3bu": "Paris", "decompose": "Lyon", "iter": "Marseille"},
        subset={"v3bu", "decompose", "iter"},
        p_arm={"v3bu": 0.5, "decompose": 0.5, "iter": 0.5},
    )
    assert chosen == "Paris"  # v3bu first in priority


def test_select_via_subset_p_arm_no_p_arm_picks_first_in_subset() -> None:
    """Without p_arm, picks first non-uncertain pred per priority."""
    from mothrag.core.selective_ensemble import _select_via_subset_p_arm

    chosen, reason = _select_via_subset_p_arm(
        preds={"v3bu": "Paris", "decompose": "Lyon", "iter": "Marseille"},
        subset={"decompose", "iter"},  # v3bu skipped
        p_arm=None,
    )
    assert chosen == "Lyon"  # first non-uncertain in subset by priority
    assert "first_in_subset_decompose" in reason


# ============================================================
# Backwards-compat: default-args == legacy behavior
# ============================================================

def test_arbitrate_excl_v3bu_no_subset_legacy_unchanged() -> None:
    """When subset=None (default), arbitrate_excl_v3bu output matches
    legacy semantics (uses classify_query_v2 internally).
    """
    from mothrag.core.selective_ensemble import arbitrate_excl_v3bu

    chosen, reason = arbitrate_excl_v3bu(
        dec_pred="Lyon", iter_pred="Marseille",
        question="When was Einstein born?",
        v3bu_fallback="Paris",
    )
    # Default semantic_rich path -> iter primary if available.
    assert chosen in ("Lyon", "Marseille", "Paris")
    # Legacy reasons start with "excl_v3bu:" but NOT "excl_v3bu:plumb:".
    assert reason.startswith("excl_v3bu:")
    assert "plumb:" not in reason


def test_arbitrate_with_c7_no_subset_legacy_unchanged() -> None:
    """When subset=None (default), arbitrate_with_c7 output uses legacy
    route_by_query_type_v2 / selective_arbitrate path.
    """
    from mothrag.core.selective_ensemble import arbitrate_with_c7

    chosen, reason, _ = arbitrate_with_c7(
        v3bu_pred="Paris", dec_pred="Lyon",
        question="When was Einstein born?",
        iter_pred="Marseille",
        use_router_v2=True,
    )
    # Legacy paths produce router_v2: / sel_v1: reasons.
    assert "plumb:" not in reason
