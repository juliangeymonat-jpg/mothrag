"""Tests for ArbitratorV2 -- pool-safe composition algebra.

Pins the agreement-strategy design contract:

  Pool-safety axiom (formal):
      F1(pool ∪ {X}, C) ≡ F1(pool, C)    when    fire(X, C) = 0

In code: ``arbitrate_pool`` with an un-fired arm (None result, empty
pred, or is_fallback-tagged) produces a BYTE-IDENTICAL
:class:`ArbitrateResult` to ``arbitrate_pool`` without that arm.

Plus pluggable agreement-strategy contract:
  - "pairwise" (default) byte-identical to DeterministicArbitrator
  - "chain" raises NotImplementedError (placeholder hook)
  - custom callables accepted

No per-dataset tuning; deterministic constants + pluggable strategies.
"""

from __future__ import annotations

import pytest

from mothrag.arms.base import ArmResult
from mothrag.core.arbitrate import (
    ArbitrateResult,
    ArbitratorV2,
    DeterministicArbitrator,
)


# ============================================================
# Pool-safety axiom: F1(pool ∪ {X}, C) ≡ F1(pool, C) when fire(X, C) = 0
# ============================================================

def _result(pred: str, *, metadata: dict | None = None) -> ArmResult:
    return ArmResult(pred=pred, metadata=metadata or {})


def test_pool_safety_invariant_unfired_arm_zero_weight() -> None:
    """Adding an un-fired arm (None result) MUST yield byte-identical
    arbitration to the version without that arm.
    """
    arb = ArbitratorV2()
    pool_3arm = {
        "v3bu": _result("Berlin"),
        "decompose": _result("Paris"),
        "iter": _result("Berlin"),
    }
    pool_4arm = dict(pool_3arm)
    pool_4arm["infobox_arm"] = None  # un-fired

    out_3arm = arb.arbitrate_pool(pool_3arm)
    out_4arm = arb.arbitrate_pool(pool_4arm)

    assert out_3arm.selected_arm == out_4arm.selected_arm
    assert out_3arm.answer == out_4arm.answer
    assert out_3arm.arm_scores == out_4arm.arm_scores
    assert out_3arm.arbitrate_signal == out_4arm.arbitrate_signal


def test_pool_safety_invariant_empty_pred_skipped() -> None:
    """Empty pred -> arm not fired -> no contribution to arbitration."""
    arb = ArbitratorV2()
    pool_3arm = {
        "v3bu": _result("Berlin"),
        "decompose": _result("Paris"),
        "iter": _result("Berlin"),
    }
    pool_4arm = dict(pool_3arm)
    pool_4arm["infobox_arm"] = _result("")  # declined: empty pred

    out_3arm = arb.arbitrate_pool(pool_3arm)
    out_4arm = arb.arbitrate_pool(pool_4arm)

    assert out_3arm.arm_scores == out_4arm.arm_scores


def test_pool_safety_invariant_is_fallback_tagged_skipped() -> None:
    """is_fallback metadata -> arm not fired (matches the
    MQ regression fix)."""
    arb = ArbitratorV2()
    pool_3arm = {
        "v3bu": _result("Berlin"),
        "decompose": _result("Paris"),
        "iter": _result("Berlin"),
    }
    pool_4arm = dict(pool_3arm)
    # Fallback duplicate of v3bu's pred -- the regression shape.
    pool_4arm["mothgraph_arm"] = _result(
        "Berlin", metadata={"is_fallback": True}
    )

    out_3arm = arb.arbitrate_pool(pool_3arm)
    out_4arm = arb.arbitrate_pool(pool_4arm)

    # Spurious consensus boost would have arisen if mothgraph_arm
    # entered candidates with pred="Berlin". ArbitratorV2 filters it.
    assert out_3arm.arm_scores == out_4arm.arm_scores


def test_pool_safety_invariant_arm_applicability_false_skipped() -> None:
    """arm_applicability[X]=False -> arm not fired even with non-empty pred.

    Defensive: the caller passed an applicable-snapshot saying "this
    arm was not applicable" but the result somehow has content. The
    arbitrator respects the applicability flag.
    """
    arb = ArbitratorV2()
    pool_3arm = {
        "v3bu": _result("Berlin"),
        "decompose": _result("Paris"),
        "iter": _result("Berlin"),
    }
    pool_4arm = dict(pool_3arm)
    pool_4arm["infobox_arm"] = _result("Madrid")  # would-fire pred

    out_3arm = arb.arbitrate_pool(pool_3arm)
    out_4arm = arb.arbitrate_pool(
        pool_4arm,
        arm_applicability={"infobox_arm": False},  # but caller says NO
    )

    assert out_3arm.arm_scores == out_4arm.arm_scores


def test_pool_safety_invariant_legitimate_fire_changes_outcome() -> None:
    """Sanity check: when the opt-in arm legitimately fires with a
    different answer, the result DOES change (the axiom only applies
    to non-fire cases).
    """
    arb = ArbitratorV2()
    pool_3arm = {
        "v3bu": _result("Berlin"),
        "decompose": _result("Paris"),
        "iter": _result("Berlin"),
    }
    pool_4arm_legit = dict(pool_3arm)
    pool_4arm_legit["infobox_arm"] = _result("Madrid")  # legitimate

    out_3arm = arb.arbitrate_pool(pool_3arm)
    out_4arm = arb.arbitrate_pool(pool_4arm_legit)

    # Arm scores DICT should differ (4 entries vs 3).
    assert set(out_4arm.arm_scores.keys()) - set(out_3arm.arm_scores.keys()) == {"infobox_arm"}


# ============================================================
# Empty pool / all-unfired edge case
# ============================================================

def test_empty_pool_returns_fallback() -> None:
    arb = ArbitratorV2()
    out = arb.arbitrate_pool({})
    assert out.selected_arm == ""
    assert out.answer == ""
    assert out.arbitrate_signal == "fallback"


def test_all_arms_unfired_returns_fallback() -> None:
    arb = ArbitratorV2()
    out = arb.arbitrate_pool({
        "v3bu": None,
        "decompose": _result(""),
        "iter": _result("X", metadata={"is_fallback": True}),
    })
    assert out.selected_arm == ""
    assert out.answer == ""


# ============================================================
# Backward compat: arbitrate (legacy passthrough)
# ============================================================

def test_legacy_arbitrate_byte_identical_to_DeterministicArbitrator() -> None:
    """ArbitratorV2.arbitrate(answers) == DeterministicArbitrator.arbitrate(answers)."""
    answers = {"v3bu": "Berlin", "decompose": "Paris", "iter": "Berlin"}
    gamma = {"iter": 1.0}

    legacy = DeterministicArbitrator().arbitrate(
        answers=answers, gamma_signals=gamma,
    )
    v2 = ArbitratorV2().arbitrate(answers=answers, gamma_signals=gamma)

    assert legacy.selected_arm == v2.selected_arm
    assert legacy.arm_scores == v2.arm_scores
    assert legacy.arbitrate_signal == v2.arbitrate_signal


# ============================================================
# Agreement strategies
# ============================================================

def test_pairwise_agreement_strategy_default() -> None:
    """Default ``agreement_strategy='pairwise'`` is accepted + named."""
    arb = ArbitratorV2()
    assert arb.agreement_strategy_name == "pairwise"


def test_chain_agreement_strategy_raises_not_implemented() -> None:
    """'chain' agreement is a placeholder hook (raises NotImplementedError
    when invoked) until per-arm reasoning traces are wired."""
    arb = ArbitratorV2(agreement_strategy="chain")
    with pytest.raises(NotImplementedError, match="chain"):
        arb.arbitrate_pool({
            "v3bu": _result("X"),
            "decompose": _result("Y"),
        })


def test_unknown_agreement_strategy_rejected() -> None:
    with pytest.raises(ValueError, match="unknown agreement_strategy"):
        ArbitratorV2(agreement_strategy="not_a_real_strategy")


def test_custom_agreement_strategy_callable_accepted() -> None:
    """Caller-supplied callable matching :class:`AgreementStrategy`."""
    def _custom(answers, **_ctx):
        # Constant agreement signal -- toy strategy for the test.
        return {k: 0.42 for k in answers}

    arb = ArbitratorV2(agreement_strategy=_custom)
    assert arb.agreement_strategy_name == "custom"
    out = arb.arbitrate_pool({
        "v3bu": _result("X"),
        "decompose": _result("Y"),
    })
    # Strategy ran -- arbitration happened.
    assert out.selected_arm in {"v3bu", "decompose"}


def test_agreement_strategy_failure_defaults_to_zero() -> None:
    """When the agreement callable raises (other than NotImplementedError),
    arbitration falls back to zero agreement signals (no boost)."""
    def _raises(answers, **_ctx):  # noqa: ARG001
        raise RuntimeError("boom")

    arb = ArbitratorV2(agreement_strategy=_raises)
    out = arb.arbitrate_pool({
        "v3bu": _result("X"),
        "decompose": _result("Y"),
    })
    # Should not raise; should produce some result.
    assert out.selected_arm in {"v3bu", "decompose"}


# ============================================================
# Pool-safety invariant: signal-dict restriction to fired subset
# ============================================================

def test_signal_dicts_restricted_to_fired_arms() -> None:
    """gamma_signals / faith_signals / arm_probabilities for un-fired
    arms must NOT affect the fired-arm composition."""
    arb = ArbitratorV2()
    pool_3arm = {
        "v3bu": _result("Berlin"),
        "decompose": _result("Paris"),
        "iter": _result("Berlin"),
    }
    pool_4arm = dict(pool_3arm)
    pool_4arm["infobox_arm"] = None  # un-fired

    out_3arm = arb.arbitrate_pool(
        pool_3arm,
        gamma_signals={"v3bu": 0.5, "iter": 1.0},
    )
    out_4arm = arb.arbitrate_pool(
        pool_4arm,
        # Inject a spurious signal for the un-fired arm.
        gamma_signals={
            "v3bu": 0.5, "iter": 1.0,
            "infobox_arm": 0.9,  # SHOULD BE IGNORED
        },
    )
    assert out_3arm.arm_scores == out_4arm.arm_scores
