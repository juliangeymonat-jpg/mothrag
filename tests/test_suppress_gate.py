"""Tests for SuppressGateStrategy (cascade short-circuit)."""

from __future__ import annotations

import sys
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _block_sentence_transformers(monkeypatch) -> Iterator[None]:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    yield


def _ctx(**overrides):
    """Build a RetryContext for SuppressGate testing."""
    from mothrag.core.api import _HashEmbedder, _MemoryVectorStore
    from mothrag.core.retry import RetryContext

    emb = _HashEmbedder()
    vdb = _MemoryVectorStore()
    base = dict(
        question="When was X born?",
        passages=["X is a person."],
        q_emb=list(emb.embed_batch(["dummy"])[0]),
        top_k=5,
        arm_subset=["v3bu"],
        v3bu_pred="14 March 1879",
        dec_pred="14 March 1879",
        iter_pred=None,
        chosen="14 March 1879",       # substantive answer
        arbitrate_reason="sel_v1:agree",
        c7_info={"gamma_status": "invalid"},  # gamma says invalid (FP)
        abstention_signal="gamma_refuse",
        embedder=emb,
        vector_db=vdb,
    )
    base.update(overrides)
    return RetryContext(**base)


# ============================================================
# applicable / not-applicable
# ============================================================

def test_suppress_gate_fires_on_gamma_invalid_substantive_pred() -> None:
    from mothrag.core.retry.strategies.suppress_gate import SuppressGateStrategy
    strat = SuppressGateStrategy()
    ctx = _ctx()
    assert strat.applicable(ctx)


def test_suppress_gate_does_not_fire_when_gamma_partial() -> None:
    from mothrag.core.retry.strategies.suppress_gate import SuppressGateStrategy
    strat = SuppressGateStrategy()
    ctx = _ctx(c7_info={"gamma_status": "partial"})
    assert not strat.applicable(ctx)


def test_suppress_gate_does_not_fire_when_gamma_refuse() -> None:
    from mothrag.core.retry.strategies.suppress_gate import SuppressGateStrategy
    strat = SuppressGateStrategy()
    ctx = _ctx(c7_info={"gamma_status": "refuse"})
    assert not strat.applicable(ctx)


def test_suppress_gate_does_not_fire_when_gamma_valid() -> None:
    from mothrag.core.retry.strategies.suppress_gate import SuppressGateStrategy
    strat = SuppressGateStrategy()
    ctx = _ctx(c7_info={"gamma_status": "valid"})
    assert not strat.applicable(ctx)


def test_suppress_gate_does_not_fire_on_uncertain_template_pred() -> None:
    """Even when gamma=invalid, do NOT short-circuit if the chosen pred
    itself is uncertain -- the cascade may still recover by re-retrieving."""
    from mothrag.core.retry.strategies.suppress_gate import SuppressGateStrategy
    strat = SuppressGateStrategy()
    for refuse_template in (
        "Not in passages", "Unknown", "I don't know", "no answer",
        "Insufficient information", "Cannot determine",
    ):
        ctx = _ctx(chosen=refuse_template)
        assert not strat.applicable(ctx), (
            f"SuppressGate fired on uncertain-template pred {refuse_template!r}; "
            f"should have deferred to cascade."
        )


def test_suppress_gate_does_not_fire_on_empty_pred() -> None:
    from mothrag.core.retry.strategies.suppress_gate import SuppressGateStrategy
    strat = SuppressGateStrategy()
    ctx = _ctx(chosen="")
    assert not strat.applicable(ctx)


def test_suppress_gate_does_not_fire_when_gamma_status_missing() -> None:
    """No c7_info dict, no abstention_signal == gamma_refuse -> no signal,
    gate does NOT fire."""
    from mothrag.core.retry.strategies.suppress_gate import SuppressGateStrategy
    strat = SuppressGateStrategy()
    ctx = _ctx(c7_info=None, abstention_signal="empty_answer")
    assert not strat.applicable(ctx)


# ============================================================
# try_recover behaviour
# ============================================================

def test_suppress_gate_returns_chosen_pred_unchanged() -> None:
    from mothrag.core.retry.strategies.suppress_gate import SuppressGateStrategy
    strat = SuppressGateStrategy()
    ctx = _ctx(chosen="Flavivirus")
    assert strat.try_recover(ctx) == "Flavivirus"


def test_suppress_gate_zero_cost_estimate() -> None:
    """SuppressGate must NOT spend cascade budget."""
    from mothrag.core.retry.strategies.suppress_gate import SuppressGateStrategy
    strat = SuppressGateStrategy()
    assert strat.cost_estimate == 0
    ctx = _ctx(budget_used=8, budget_limit=8)  # budget already saturated
    # Even with budget exhausted, try_recover should succeed (no spend).
    assert strat.try_recover(ctx) == ctx.chosen


# ============================================================
# c7_info key-name tolerance
# ============================================================

def test_suppress_gate_extracts_gamma_from_alternative_keys() -> None:
    """The gate must accept gamma_status keyed under any of the canonical
    aliases (gamma_status / gamma / gamma_final_status)."""
    from mothrag.core.retry.strategies.suppress_gate import SuppressGateStrategy
    strat = SuppressGateStrategy()
    for key in ("gamma_status", "gamma", "gamma_final_status"):
        ctx = _ctx(c7_info={key: "invalid"})
        assert strat.applicable(ctx), (
            f"SuppressGate could not read gamma signal from c7_info key {key!r}"
        )


# ============================================================
# Orchestrator integration -- when SuppressGate fires FIRST, the cascade
# short-circuits without invoking any downstream strategy.
# ============================================================

def test_suppress_gate_short_circuits_orchestrator_cascade() -> None:
    from mothrag.core.retry import build_default_orchestrator
    orch = build_default_orchestrator(
        ["suppress_gate", "iter_extension", "soft_fallback"],
    )

    invoked: list[str] = []

    def _iter_runner(**kwargs):
        invoked.append("iter")
        return "recovered-via-iter"

    ctx = _ctx(run_arm_iter=_iter_runner)
    out = orch.try_escalate(ctx)
    # Cascade short-circuited on suppress_gate; iter runner never called.
    assert out.recovered_by == "suppress_gate"
    assert out.answer == ctx.chosen
    assert "iter" not in invoked
    # No LLM budget consumed.
    assert out.budget_used == 0


def test_suppress_gate_yields_to_cascade_when_not_applicable() -> None:
    """When SuppressGate's predicate doesn't match, downstream strategies
    get a fair shot."""
    from mothrag.core.retry import build_default_orchestrator
    orch = build_default_orchestrator(
        ["suppress_gate", "iter_extension", "soft_fallback"],
    )

    invoked: list[str] = []

    def _iter_runner(**kwargs):
        invoked.append("iter")
        return "recovered-via-iter"

    # gamma=refuse (NOT invalid) so suppress_gate must NOT fire.
    ctx = _ctx(
        c7_info={"gamma_status": "refuse"},
        abstention_signal="gamma_refuse",
        chosen="Not in passages",
        run_arm_iter=_iter_runner,
    )
    out = orch.try_escalate(ctx)
    # Suppress should have deferred; iter_extension should have fired
    # OR soft_fallback as terminal. Either way, NOT suppress_gate.
    assert out.recovered_by != "suppress_gate"
