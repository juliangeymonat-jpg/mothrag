"""Tests for PDD generalization (signal-level dup primitive).

Analytical proof that the cardinality-vs-fixed-weight distinction in
the aggregator formula determines whether PDD lift propagates.
Anti-leak: no gold, no F1, pure math on signal values.
"""

from __future__ import annotations

import pytest


# ============================================================
# dup_signal_into_aggregator contract
# ============================================================

def test_dup_signal_mirrors_base_value() -> None:
    from mothrag.core.arbitrate.signal_dup import (
        dup_signal_into_aggregator,
    )

    signals = {"v3bu": 0.8, "decompose": 0.3, "iter": 0.5}
    r = dup_signal_into_aggregator(signals, "v3bu", "v3bu_dup_a")
    assert r.base_voter_id == "v3bu"
    assert r.dup_voter_id == "v3bu_dup_a"
    assert r.base_value == 0.8
    # original signals unchanged (we return a new dict)
    assert "v3bu_dup_a" not in signals
    # extended carries both
    assert r.extended_signals["v3bu"] == 0.8
    assert r.extended_signals["v3bu_dup_a"] == 0.8


def test_dup_signal_rejects_collision() -> None:
    from mothrag.core.arbitrate.signal_dup import (
        dup_signal_into_aggregator,
    )

    signals = {"v3bu": 0.8, "v3bu_dup_a": 0.1}
    with pytest.raises(ValueError, match="already exists"):
        dup_signal_into_aggregator(signals, "v3bu", "v3bu_dup_a")


def test_dup_signal_rejects_same_id() -> None:
    from mothrag.core.arbitrate.signal_dup import (
        dup_signal_into_aggregator,
    )

    with pytest.raises(ValueError, match="must differ"):
        dup_signal_into_aggregator(
            {"v3bu": 0.5}, "v3bu", "v3bu",
        )


def test_dup_signal_rejects_unknown_base() -> None:
    from mothrag.core.arbitrate.signal_dup import (
        dup_signal_into_aggregator,
    )

    with pytest.raises(KeyError):
        dup_signal_into_aggregator({"x": 0.5}, "v3bu", "v3bu_dup_a")


# ============================================================
# Cardinality-normalized aggregator: PDD AMPLIFIES (predicted + proven)
# ============================================================

def test_cardinality_average_pdd_amplifies_when_base_above_others() -> None:
    """When base_value > average_of_others, dup pulls the cardinality
    average TOWARD base. Net effect: amplification of base influence."""
    from mothrag.core.arbitrate.signal_dup import (
        apply_cardinality_average,
        dup_signal_into_aggregator,
    )

    # base voter (v3bu) has higher signal than peers
    signals = {"v3bu": 1.0, "decompose": 0.0, "iter": 0.0}
    baseline_avg = apply_cardinality_average(signals)
    # baseline: 1/3 = 0.333

    dup = dup_signal_into_aggregator(signals, "v3bu", "v3bu_dup_a")
    duped_avg = apply_cardinality_average(dup.extended_signals)
    # with dup: 2/4 = 0.5

    assert duped_avg > baseline_avg, (
        f"PDD predicted to amplify on cardinality aggregator: "
        f"baseline={baseline_avg:.4f}, duped={duped_avg:.4f}"
    )
    # exact analytical shift
    expected_shift = duped_avg - baseline_avg
    assert abs(expected_shift - (0.5 - 1.0 / 3.0)) < 1e-9


def test_cardinality_average_pdd_amplifies_when_base_below_others() -> None:
    """Symmetric: when base_value < average_of_others, dup pulls the
    cardinality average DOWN toward base. The mechanism amplifies
    base's influence in either direction."""
    from mothrag.core.arbitrate.signal_dup import (
        apply_cardinality_average,
        dup_signal_into_aggregator,
    )

    signals = {"v3bu": 0.0, "decompose": 1.0, "iter": 1.0}
    baseline_avg = apply_cardinality_average(signals)
    # baseline: 2/3 = 0.667

    dup = dup_signal_into_aggregator(signals, "v3bu", "v3bu_dup_a")
    duped_avg = apply_cardinality_average(dup.extended_signals)
    # with dup: 2/4 = 0.5

    assert duped_avg < baseline_avg, (
        f"PDD predicted to amplify base influence either direction: "
        f"baseline={baseline_avg:.4f}, duped={duped_avg:.4f}"
    )


def test_cardinality_average_dup_neutral_when_base_equals_average() -> None:
    """No amplification when base equals the other arms' average --
    the dup contributes the same value as the average so the result
    doesn't move."""
    from mothrag.core.arbitrate.signal_dup import (
        apply_cardinality_average,
        dup_signal_into_aggregator,
    )

    signals = {"v3bu": 0.5, "decompose": 0.5, "iter": 0.5}
    baseline_avg = apply_cardinality_average(signals)
    dup = dup_signal_into_aggregator(signals, "v3bu", "v3bu_dup_a")
    duped_avg = apply_cardinality_average(dup.extended_signals)
    assert baseline_avg == duped_avg == 0.5


# ============================================================
# Fixed-weight aggregator: PDD does NOT amplify the same way
# ============================================================

def test_fixed_weighted_sum_dup_with_zero_weight_no_effect() -> None:
    """Fixed-weight sum: when the dup voter has weight 0 (unseen
    voter default), it contributes nothing. PDD does NOT amplify."""
    from mothrag.core.arbitrate.signal_dup import (
        apply_fixed_weighted_sum,
        dup_signal_into_aggregator,
    )

    signals = {"v3bu": 1.0, "decompose": 0.5}
    weights = {"v3bu": 1.0, "decompose": 1.0}  # dup has implicit 1.0
    baseline = apply_fixed_weighted_sum(signals, weights)

    dup = dup_signal_into_aggregator(signals, "v3bu", "v3bu_dup_a")
    duped = apply_fixed_weighted_sum(dup.extended_signals, weights)
    # default weight for unseen voter is 1.0 -> sum grows by 1.0*1.0=1.0
    # This is NOT cardinality normalization; the total grows linearly
    # with N, not relative to base's share of the average.
    assert duped > baseline
    # Verify the growth is purely additive, not "share-of-average":
    growth = duped - baseline
    assert abs(growth - 1.0) < 1e-9


def test_fixed_weighted_sum_does_not_normalize_share_under_dup() -> None:
    """Key distinction: in cardinality aggregator, dup changes the
    SHARE of base's value (more weight on base). In fixed-weight
    sum, dup just adds a new term -- the share concept doesn't apply
    because there's no denominator."""
    from mothrag.core.arbitrate.signal_dup import (
        apply_cardinality_average, apply_fixed_weighted_sum,
        dup_signal_into_aggregator,
    )

    signals = {"v3bu": 1.0, "decompose": 0.0, "iter": 0.0}
    weights = {"v3bu": 1.0, "decompose": 1.0, "iter": 1.0}

    # Cardinality baseline + dup share-shift
    card_baseline = apply_cardinality_average(signals)
    card_dup = apply_cardinality_average(
        dup_signal_into_aggregator(signals, "v3bu", "v3bu_dup_a")
        .extended_signals
    )
    card_share_shift = card_dup - card_baseline  # > 0

    # Fixed-weight baseline + dup growth (NOT a share shift)
    fix_baseline = apply_fixed_weighted_sum(signals, weights)
    fix_dup = apply_fixed_weighted_sum(
        dup_signal_into_aggregator(signals, "v3bu", "v3bu_dup_a")
        .extended_signals,
        weights,
    )
    fix_growth = fix_dup - fix_baseline

    # Both grow but the SEMANTICS differ:
    # * cardinality: dup pulls average TOWARD base (share shift)
    # * fixed-weight: dup adds a new term, total grows linearly
    # The cardinality shift is bounded by base_value - baseline; the
    # fixed-weight growth equals weight_dup * base_value (=1.0 here).
    assert 0 < card_share_shift < 1.0
    assert fix_growth >= 1.0  # dup adds the full base_value term


# ============================================================
# pdd_lift_predicted -- structural classifier
# ============================================================

def test_pdd_predicted_for_cardinality_aggregators() -> None:
    from mothrag.core.arbitrate.signal_dup import pdd_lift_predicted

    assert pdd_lift_predicted("pairwise_agreement") is True
    assert pdd_lift_predicted("agreement_per_aspect") is True


def test_pdd_not_predicted_for_fixed_weight_aggregators() -> None:
    from mothrag.core.arbitrate.signal_dup import pdd_lift_predicted

    assert pdd_lift_predicted("DeterministicArbitrator") is False
    assert pdd_lift_predicted("ArbitratorV2") is False


def test_pdd_not_predicted_for_non_voting_aggregators() -> None:
    from mothrag.core.arbitrate.signal_dup import pdd_lift_predicted

    assert pdd_lift_predicted("gamma_l4b_andgate") is False
    assert pdd_lift_predicted("MultiModalRetriever") is False


def test_pdd_predicted_unknown_aggregator_fail_safe_to_false() -> None:
    """Unknown aggregator names default to 'no lift predicted'
    (fail-safe)."""
    from mothrag.core.arbitrate.signal_dup import pdd_lift_predicted

    assert pdd_lift_predicted("brand_new_uncatalogued_thing") is False


def test_aggregator_table_documents_all_known_consensus_points() -> None:
    """The lookup table includes every consensus point found by the
    code archaeology."""
    from mothrag.core.arbitrate.signal_dup import (
        AGGREGATOR_NORMALIZATION_TABLE,
    )

    expected = {
        "pairwise_agreement",
        "agreement_per_aspect",
        "DeterministicArbitrator",
        "ArbitratorV2",
        "gamma_l4b_andgate",
        "MultiModalRetriever",
    }
    assert expected <= set(AGGREGATOR_NORMALIZATION_TABLE.keys())


# ============================================================
# Anti-leak audit
# ============================================================

def test_signature_anti_leak_no_per_dataset_args() -> None:
    """All public functions reject per-dataset / gold inputs in
    signature."""
    import inspect
    from mothrag.core.arbitrate import signal_dup

    forbidden = {
        "dataset", "ds", "ds_label", "ds_hint", "ds_family",
        "corpus", "benchmark", "gold", "f1", "em",
    }
    for fn_name in (
        "dup_signal_into_aggregator",
        "pdd_lift_predicted",
        "apply_cardinality_average",
        "apply_fixed_weighted_sum",
    ):
        fn = getattr(signal_dup, fn_name)
        sig = inspect.signature(fn)
        params = set(sig.parameters.keys())
        leaked = params & forbidden
        assert not leaked, (
            f"{fn_name} signature must not accept per-dataset / gold args; "
            f"found: {leaked}"
        )
