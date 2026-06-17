"""Tests for RerouteIterWithBoostStrategy."""

from __future__ import annotations

import sys
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _block_sentence_transformers(monkeypatch) -> Iterator[None]:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    yield


def _ctx(**overrides):
    """Build a RetryContext for RerouteIterWithBoost testing."""
    from mothrag.core.api import _HashEmbedder, _MemoryVectorStore
    from mothrag.core.retry import RetryContext

    emb = _HashEmbedder()
    vdb = _MemoryVectorStore()
    base = dict(
        question="What 4-D film attraction was based off a 2004 animated movie?",
        passages=["Some passage.", "Another passage."],
        q_emb=list(emb.embed_batch(["dummy"])[0]),
        top_k=5,
        arm_subset=["iter"],
        v3bu_pred="Not in passages",
        dec_pred="Not in passages",
        iter_pred="Not in passages",
        chosen="Not in passages",
        arbitrate_reason="h3_gamma_valid_disagree_iter",
        c7_info={"gamma_status": "valid", "h3_fires": True},
        abstention_signal="cross_arm_disagree",
        embedder=emb,
        vector_db=vdb,
    )
    base.update(overrides)
    return RetryContext(**base)


# ============================================================
# applicable / not-applicable
# ============================================================

def test_reroute_iter_fires_on_h3_signal_in_c7_info() -> None:
    from mothrag.core.retry.strategies.reroute_iter_with_boost import (
        RerouteIterWithBoostStrategy,
    )
    strat = RerouteIterWithBoostStrategy()
    ctx = _ctx(run_arm_iter=lambda **_: "answer")
    assert strat.applicable(ctx)


def test_reroute_iter_fires_on_h3_arbitrate_reason_marker() -> None:
    from mothrag.core.retry.strategies.reroute_iter_with_boost import (
        RerouteIterWithBoostStrategy,
    )
    strat = RerouteIterWithBoostStrategy()
    ctx = _ctx(
        c7_info=None,
        arbitrate_reason="h3_gamma_valid_disagree_iter",
        run_arm_iter=lambda **_: "answer",
    )
    assert strat.applicable(ctx)


def test_reroute_iter_fires_on_cross_arm_disagree_signal() -> None:
    from mothrag.core.retry.strategies.reroute_iter_with_boost import (
        RerouteIterWithBoostStrategy,
    )
    strat = RerouteIterWithBoostStrategy()
    ctx = _ctx(
        c7_info=None,
        arbitrate_reason="sel_v1:disagree-v3bu-wins",
        abstention_signal="cross_arm_disagree",
        run_arm_iter=lambda **_: "answer",
    )
    assert strat.applicable(ctx)


def test_reroute_iter_does_not_fire_without_iter_runner() -> None:
    from mothrag.core.retry.strategies.reroute_iter_with_boost import (
        RerouteIterWithBoostStrategy,
    )
    strat = RerouteIterWithBoostStrategy()
    ctx = _ctx(run_arm_iter=None)
    assert not strat.applicable(ctx)


def test_reroute_iter_does_not_fire_when_chosen_is_substantive() -> None:
    """The strategy targets ABSTAINED cases; substantive answers don't
    need a re-run."""
    from mothrag.core.retry.strategies.reroute_iter_with_boost import (
        RerouteIterWithBoostStrategy,
    )
    strat = RerouteIterWithBoostStrategy()
    ctx = _ctx(chosen="A substantive non-uncertain answer.",
               run_arm_iter=lambda **_: "x")
    assert not strat.applicable(ctx)


def test_reroute_iter_does_not_fire_without_h3_signal() -> None:
    from mothrag.core.retry.strategies.reroute_iter_with_boost import (
        RerouteIterWithBoostStrategy,
    )
    strat = RerouteIterWithBoostStrategy()
    ctx = _ctx(
        c7_info={"gamma_status": "refuse"},
        arbitrate_reason="sel_v1:both-uncertain",
        abstention_signal="gamma_refuse",
        run_arm_iter=lambda **_: "answer",
    )
    assert not strat.applicable(ctx)


# ============================================================
# try_recover behaviour
# ============================================================

def test_reroute_iter_passes_boost_factor_to_runner() -> None:
    """The runner must receive bottom_up_boost=1.5 on the first call."""
    from mothrag.core.retry.strategies.reroute_iter_with_boost import (
        RerouteIterWithBoostStrategy,
    )
    received_kwargs: list[dict] = []

    def _runner(**kwargs):
        received_kwargs.append(kwargs)
        return "Mickey's PhilharMagic"

    strat = RerouteIterWithBoostStrategy(boost_factor=1.5)
    ctx = _ctx(run_arm_iter=_runner)
    out = strat.try_recover(ctx)
    assert out == "Mickey's PhilharMagic"
    assert received_kwargs[0]["bottom_up_boost"] == 1.5


def test_reroute_iter_falls_back_to_higher_boost_on_first_pass_uncertain() -> None:
    """When first pass still returns uncertain, second pass with the
    fallback boost factor fires."""
    from mothrag.core.retry.strategies.reroute_iter_with_boost import (
        RerouteIterWithBoostStrategy,
    )
    received_boosts: list[float] = []

    def _runner(**kwargs):
        b = kwargs.get("bottom_up_boost")
        received_boosts.append(b)
        if b == 1.5:
            return "Not in passages"
        return "recovered-at-2.0"

    strat = RerouteIterWithBoostStrategy(
        boost_factor=1.5, fallback_boost_factor=2.0,
    )
    ctx = _ctx(run_arm_iter=_runner)
    out = strat.try_recover(ctx)
    assert out == "recovered-at-2.0"
    assert received_boosts == [1.5, 2.0]


def test_reroute_iter_returns_none_when_both_passes_uncertain() -> None:
    from mothrag.core.retry.strategies.reroute_iter_with_boost import (
        RerouteIterWithBoostStrategy,
    )

    def _runner(**kwargs):
        return "Not in passages"

    strat = RerouteIterWithBoostStrategy(
        boost_factor=1.5, fallback_boost_factor=2.0,
    )
    ctx = _ctx(run_arm_iter=_runner)
    assert strat.try_recover(ctx) is None


def test_reroute_iter_fallback_disabled_when_factor_is_none() -> None:
    """Setting fallback_boost_factor=None caps at one pass."""
    from mothrag.core.retry.strategies.reroute_iter_with_boost import (
        RerouteIterWithBoostStrategy,
    )
    boosts: list[float] = []

    def _runner(**kwargs):
        boosts.append(kwargs["bottom_up_boost"])
        return "Not in passages"

    strat = RerouteIterWithBoostStrategy(
        boost_factor=1.5, fallback_boost_factor=None,
    )
    ctx = _ctx(run_arm_iter=_runner)
    assert strat.try_recover(ctx) is None
    assert boosts == [1.5]


def test_reroute_iter_handles_legacy_shim_without_boost_kwarg() -> None:
    """Older iter runners may not accept bottom_up_boost; the strategy
    must catch the TypeError and fall back to a plain re-run."""
    from mothrag.core.retry.strategies.reroute_iter_with_boost import (
        RerouteIterWithBoostStrategy,
    )
    calls: list[dict] = []

    def _legacy_runner(*, question, passages, q_emb=None, top_k=None,
                      max_steps=None, l4b_anchor=None):
        # NB: deliberately does NOT accept bottom_up_boost
        calls.append({"question": question, "top_k": top_k})
        return "Not in passages"

    strat = RerouteIterWithBoostStrategy(
        boost_factor=1.5, fallback_boost_factor=None,
    )
    ctx = _ctx(run_arm_iter=_legacy_runner)
    out = strat.try_recover(ctx)
    # Strategy did NOT crash on the TypeError; the plain re-run still
    # returned uncertain so the strategy returns None.
    assert out is None
    assert calls, "Legacy fallback path was not exercised"


def test_reroute_iter_budget_guard_skips_when_exhausted() -> None:
    from mothrag.core.retry.strategies.reroute_iter_with_boost import (
        RerouteIterWithBoostStrategy,
    )
    ctx = _ctx(run_arm_iter=lambda **_: "x", budget_used=8, budget_limit=8)
    assert RerouteIterWithBoostStrategy().try_recover(ctx) is None


# ============================================================
# Orchestrator integration
# ============================================================

def test_reroute_iter_via_orchestrator_when_applicable() -> None:
    from mothrag.core.retry import build_default_orchestrator
    orch = build_default_orchestrator(
        ["reroute_iter_with_boost", "soft_fallback"],
    )

    def _runner(**kwargs):
        return "answer-via-boost-rerun"

    ctx = _ctx(run_arm_iter=_runner)
    out = orch.try_escalate(ctx)
    assert out.recovered_by == "reroute_iter_with_boost"
    assert out.answer == "answer-via-boost-rerun"
