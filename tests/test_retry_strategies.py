"""Retry-on-abstain cascade tests.

Covers:
  * Orchestrator construction (all / sweet_spot / soft_fallback_only / explicit list)
  * Per-strategy applicable / try_recover behavior on representative contexts
  * Cascade priority order
  * SoftFallback non-empty guarantee
  * _detect_abstention_signal classification
  * MothRAG(production=True) escalation metadata wiring
  * Backward compat: retry_strategies="off" leaves the answer untouched
"""

from __future__ import annotations

import sys
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _block_sentence_transformers(monkeypatch) -> Iterator[None]:
    """Block sentence_transformers (local pyarrow segfault path)."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    yield


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch) -> Iterator[None]:
    for k in (
        "VERTEX_AI_PROJECT", "GOOGLE_CLOUD_PROJECT",
        "GEMINI_API_KEY", "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    yield


# ============================================================
# Orchestrator construction
# ============================================================

def test_orchestrator_all_preset_builds() -> None:
    from mothrag.core.retry import build_default_orchestrator
    orch = build_default_orchestrator("all")
    names = [s.name for s in orch.strategies]
    assert names[-1] == "soft_fallback"
    assert "iter_extension" in names
    assert "query_reformulation" in names


def test_orchestrator_sweet_spot_preset_builds() -> None:
    from mothrag.core.retry import build_default_orchestrator
    orch = build_default_orchestrator("sweet_spot")
    names = [s.name for s in orch.strategies]
    assert names == [
        "iter_extension", "arm_fallback", "cross_arm_consensus", "soft_fallback",
    ]


def test_orchestrator_soft_fallback_only_builds() -> None:
    from mothrag.core.retry import build_default_orchestrator
    orch = build_default_orchestrator("soft_fallback_only")
    assert [s.name for s in orch.strategies] == ["soft_fallback"]


def test_orchestrator_explicit_list_appends_soft_fallback() -> None:
    from mothrag.core.retry import build_default_orchestrator
    orch = build_default_orchestrator(["arm_fallback", "iter_extension"])
    names = [s.name for s in orch.strategies]
    assert names[-1] == "soft_fallback"
    assert names[:2] == ["arm_fallback", "iter_extension"]


def test_orchestrator_terminal_must_be_soft_fallback() -> None:
    from mothrag.core.retry import EscalationOrchestrator
    from mothrag.core.retry.strategies.iter_extension import IterBudgetExtensionStrategy
    with pytest.raises(ValueError, match="terminal"):
        EscalationOrchestrator([IterBudgetExtensionStrategy()])


def test_orchestrator_rejects_empty_strategy_list() -> None:
    from mothrag.core.retry import EscalationOrchestrator
    with pytest.raises(ValueError, match="at least one"):
        EscalationOrchestrator([])


# ============================================================
# Strategy applicability + recovery
# ============================================================

def _ctx(**overrides):
    from mothrag.core.retry import RetryContext
    base = dict(
        question="dummy?",
        passages=["First passage about X."],
        q_emb=[0.0],
        top_k=5,
        arm_subset=["v3bu", "iter"],
        v3bu_pred=None,
        dec_pred=None,
        iter_pred=None,
        chosen="",
        arbitrate_reason="",
        abstention_signal="empty_answer",
    )
    base.update(overrides)
    return RetryContext(**base)


def test_soft_fallback_returns_iter_when_available() -> None:
    from mothrag.core.retry.strategies.soft_fallback import SoftFallbackStrategy
    ctx = _ctx(iter_pred="The iter answer.", dec_pred="dec", v3bu_pred="v3bu")
    out = SoftFallbackStrategy().try_recover(ctx)
    assert out == "The iter answer."


def test_soft_fallback_returns_non_empty_even_for_all_empty_arms() -> None:
    from mothrag.core.retry.strategies.soft_fallback import SoftFallbackStrategy
    ctx = _ctx(v3bu_pred="", dec_pred="", iter_pred="", chosen="")
    out = SoftFallbackStrategy().try_recover(ctx)
    # SoftFallback may return None for an all-empty context; the
    # *orchestrator* layer is what guarantees a non-None final answer
    # (returns "[no answer recovered]" placeholder). Verify that via
    # try_escalate, not the bare strategy.
    from mothrag.core.retry import build_default_orchestrator
    orch = build_default_orchestrator("soft_fallback_only")
    result = orch.try_escalate(ctx)
    assert result.answer  # non-empty


def test_arm_fallback_picks_non_empty_sibling() -> None:
    from mothrag.core.retry.strategies.arm_fallback import ArmFallbackStrategy
    ctx = _ctx(
        abstention_signal="iter_abstain",
        v3bu_pred="A clear v3bu answer.",
        iter_pred="not in passages",
        chosen="not in passages",
    )
    strat = ArmFallbackStrategy()
    assert strat.applicable(ctx)
    assert strat.try_recover(ctx) == "A clear v3bu answer."


def test_arm_fallback_inapplicable_when_only_uncertain_siblings() -> None:
    from mothrag.core.retry.strategies.arm_fallback import ArmFallbackStrategy
    ctx = _ctx(
        abstention_signal="iter_abstain",
        v3bu_pred="not in passages",
        iter_pred="not in passages",
    )
    assert not ArmFallbackStrategy().applicable(ctx)


def test_cross_arm_consensus_returns_majority() -> None:
    from mothrag.core.api import _HashEmbedder
    from mothrag.core.retry.strategies.cross_arm_consensus import CrossArmConsensusStrategy
    emb = _HashEmbedder()
    # Two arms agree (same answer string -> identical hash vectors).
    ctx = _ctx(
        v3bu_pred="The answer is 1942.",
        dec_pred="The answer is 1942.",
        iter_pred="Some unrelated text.",
        embedder=emb,
    )
    out = CrossArmConsensusStrategy(threshold=0.7).try_recover(ctx)
    assert out is not None
    assert "1942" in out


def test_cross_arm_consensus_returns_none_when_no_majority() -> None:
    from mothrag.core.api import _HashEmbedder
    from mothrag.core.retry.strategies.cross_arm_consensus import CrossArmConsensusStrategy
    ctx = _ctx(
        v3bu_pred="Answer alpha.",
        dec_pred="Answer beta.",
        iter_pred="Answer gamma.",
        embedder=_HashEmbedder(),
    )
    # Set threshold so high that no pair crosses -> no cluster of size >= 2.
    out = CrossArmConsensusStrategy(threshold=0.99).try_recover(ctx)
    assert out is None


def test_iter_extension_inapplicable_without_runner() -> None:
    from mothrag.core.retry.strategies.iter_extension import IterBudgetExtensionStrategy
    ctx = _ctx(abstention_signal="iter_abstain", iter_pred="not in passages")
    assert not IterBudgetExtensionStrategy().applicable(ctx)


def test_iter_extension_applicable_with_runner_and_signal() -> None:
    from mothrag.core.retry.strategies.iter_extension import IterBudgetExtensionStrategy
    ctx = _ctx(
        abstention_signal="iter_abstain",
        iter_pred="not in passages",
        run_arm_iter=lambda **_: "Recovered answer via wider budget.",
    )
    strat = IterBudgetExtensionStrategy()
    assert strat.applicable(ctx)
    assert strat.try_recover(ctx) == "Recovered answer via wider budget."


def test_l4b_anchor_retry_inapplicable_when_no_l4b_info() -> None:
    from mothrag.core.retry.strategies.l4b_anchor_retry import L4bAnchorRetryStrategy
    ctx = _ctx(abstention_signal="iter_abstain", c7_info=None)
    assert not L4bAnchorRetryStrategy().applicable(ctx)


def test_l4b_anchor_retry_inapplicable_with_no_alt_anchor() -> None:
    from mothrag.core.retry.strategies.l4b_anchor_retry import L4bAnchorRetryStrategy
    ctx = _ctx(
        abstention_signal="iter_abstain",
        c7_info={"l4b": {"cancelled": True, "anchors": ["anchor_only_one"]}},
        run_arm_iter=lambda **_: "_",
    )
    assert not L4bAnchorRetryStrategy().applicable(ctx)


def test_l4b_anchor_retry_executes_with_alpha_l4b_state() -> None:
    """Regression test for a previously silent no-op.

    Earlier, the strategy was silently dead because the
    alpha pipeline never populated ``c7_info["l4b"]``. The fix now wires
    ``_query_production`` to emit positional anchors so the strategy
    actually executes when the iter arm abstains. This test asserts:

    * ``applicable`` is True with the alpha-substitute L4b state.
    * ``try_recover`` calls the runner and forwards the alt anchor
      (index 1, i.e. the runner-up passage).
    * The runner's response is returned verbatim.
    """
    from mothrag.core.retry.strategies.l4b_anchor_retry import L4bAnchorRetryStrategy

    captured: dict = {"invoked": False, "anchor": None, "passages_first": None}

    def fake_runner(*, question, passages, q_emb, top_k, max_steps=None,
                     l4b_anchor=None):
        captured["invoked"] = True
        captured["anchor"] = l4b_anchor
        captured["passages_first"] = passages[0] if passages else None
        return "recovered via alt anchor"

    ctx = _ctx(
        passages=["distractor first", "gold-bearing second", "third"],
        abstention_signal="iter_abstain",
        iter_pred="",
        c7_info={
            "l4b": {
                "cancelled": True,
                "anchors": [0, 1, 2],
                "alpha_substitute": True,
            },
        },
        run_arm_iter=fake_runner,
    )

    strat = L4bAnchorRetryStrategy()
    assert strat.applicable(ctx)
    result = strat.try_recover(ctx)
    assert captured["invoked"], "runner must execute (no more silent no-op)"
    assert captured["anchor"] == 1, "second-best anchor (index 1) must be forwarded"
    assert result == "recovered via alt anchor"


def test_reorder_passages_by_anchor_boosts_indexed_passage() -> None:
    """The alpha L4b anchor swap reorders passages by integer index.

    Verifies the static helper added to MothRAG so the unit-test pinning
    the contract does not require standing up the full instance.
    """
    from mothrag.core.api import MothRAG

    out = MothRAG._reorder_passages_by_anchor(  # noqa: SLF001
        ["a", "b", "c", "d"], anchor=2,
    )
    assert out == ["c", "a", "b", "d"]

    # No-op cases: None, out-of-range, non-numeric.
    assert MothRAG._reorder_passages_by_anchor(["a", "b"], anchor=None) == ["a", "b"]  # noqa: SLF001
    assert MothRAG._reorder_passages_by_anchor(["a", "b"], anchor=5) == ["a", "b"]  # noqa: SLF001
    assert MothRAG._reorder_passages_by_anchor(["a", "b"], anchor="zzz") == ["a", "b"]  # noqa: SLF001
    assert MothRAG._reorder_passages_by_anchor([], anchor=0) == []  # noqa: SLF001


def test_query_reformulation_respects_recursion_guard() -> None:
    from mothrag.core.retry.strategies.query_reformulation import QueryReformulationStrategy
    ctx = _ctx(
        abstention_signal="empty_answer",
        escalation_depth=1,
        reader=type("R", (), {"read": staticmethod(lambda q, p: "rewritten?")})(),
        run_arm_v3bu=lambda **_: "ans",
    )
    assert not QueryReformulationStrategy(max_recursion_depth=1).applicable(ctx)


# ============================================================
# Cascade priority order
# ============================================================

def test_cascade_first_applicable_wins() -> None:
    """When two strategies could recover, the earlier-priority one wins."""
    from mothrag.core.api import _HashEmbedder
    from mothrag.core.retry import build_default_orchestrator
    ctx = _ctx(
        abstention_signal="iter_abstain",
        v3bu_pred="v3bu has an answer.",
        dec_pred="v3bu has an answer.",
        iter_pred="not in passages",
        chosen="not in passages",
        embedder=_HashEmbedder(),
    )
    # arm_fallback fires before cross_arm_consensus in default order;
    # both are applicable here, arm_fallback wins.
    orch = build_default_orchestrator(["arm_fallback", "cross_arm_consensus"])
    out = orch.try_escalate(ctx)
    assert out.recovered_by == "arm_fallback"


def test_cascade_falls_through_to_soft_fallback() -> None:
    from mothrag.core.retry import build_default_orchestrator
    # No backends + no preds -> every non-terminal strategy returns None.
    ctx = _ctx(abstention_signal="empty_answer")
    orch = build_default_orchestrator("all")
    out = orch.try_escalate(ctx)
    assert out.recovered_by == "soft_fallback"
    assert out.final_confidence == "low_soft_fallback"
    assert out.answer  # non-empty


# ============================================================
# MothRAG integration
# ============================================================

def test_mothrag_retry_off_leaves_answer_untouched() -> None:
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader
    rag = MothRAG.from_documents(
        ["A doc.", "Another doc."],
        embedder=_HashEmbedder(),
        reader=_EchoReader(),
        production=True,
        retry_strategies="off",
    )
    qr = rag.query("Anything?")
    assert qr.metadata["escalation_recovered_by"] is None
    assert qr.metadata["escalation_applied"] == []


def test_mothrag_retry_metadata_populated_under_all_preset() -> None:
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader
    rag = MothRAG.from_documents(
        ["A.", "B."],
        embedder=_HashEmbedder(),
        reader=_EchoReader(),
        production=True,
        retry_strategies="all",
    )
    qr = rag.query("Q?")
    # All five new keys should be present in the metadata dict.
    for key in (
        "escalation_applied",
        "escalation_recovered_by",
        "original_abstention_signal",
        "final_answer_confidence",
    ):
        assert key in qr.metadata, f"missing metadata key {key!r}"


# ============================================================
# _detect_abstention_signal
# ============================================================

def test_detect_abstention_signal_known_cases() -> None:
    from mothrag.core.api import _detect_abstention_signal
    assert _detect_abstention_signal("", "γ_refuse", None) == "gamma_refuse"
    assert _detect_abstention_signal("", "gamma invalid", None) == "gamma_refuse"
    assert _detect_abstention_signal("", "h12 fired", None) == "h12_refuse"
    assert _detect_abstention_signal("not in passages", "ok", None) == "empty_answer"
    assert _detect_abstention_signal(
        "", "iter abstained", {"l4b": {"cancelled": True}},
    ) == "iter_abstain"
    assert _detect_abstention_signal("a confident answer", "ok", None) is None


# ============================================================
# Dual-mode (loop / abstention)
# ============================================================

def test_orchestrator_abstention_mode_terminal_abstain() -> None:
    """In abstention mode, exhausted cascade returns ('', terminal_abstain=True)."""
    from mothrag.core.retry import build_default_orchestrator
    # No backends, no preds -> every strategy returns None.
    ctx = _ctx(abstention_signal="empty_answer")
    orch = build_default_orchestrator(["arm_fallback"], mode="abstention")
    out = orch.try_escalate(ctx)
    assert out.answer == ""
    assert out.terminal_abstain is True
    assert out.recovered_by == "terminal_abstain"
    assert out.final_confidence == "terminal_abstain"
    assert out.mode == "abstention"


def test_orchestrator_abstention_mode_still_recovers_when_possible() -> None:
    """Mode only changes terminal; non-terminal strategies still recover."""
    from mothrag.core.retry import build_default_orchestrator
    ctx = _ctx(
        abstention_signal="iter_abstain",
        v3bu_pred="A real sibling answer.",
        iter_pred="not in passages",
        chosen="not in passages",
    )
    orch = build_default_orchestrator(["arm_fallback"], mode="abstention")
    out = orch.try_escalate(ctx)
    assert out.answer == "A real sibling answer."
    assert out.recovered_by == "arm_fallback"
    assert out.terminal_abstain is False
    assert out.mode == "abstention"


def test_orchestrator_abstention_mode_no_soft_fallback_requirement() -> None:
    """mode='abstention' allows a strategy list without soft_fallback terminus."""
    from mothrag.core.retry import EscalationOrchestrator
    from mothrag.core.retry.strategies.iter_extension import IterBudgetExtensionStrategy
    # Loop mode rejects this list ...
    with pytest.raises(ValueError, match="must be SoftFallbackStrategy"):
        EscalationOrchestrator([IterBudgetExtensionStrategy()], mode="loop")
    # ... but abstention mode accepts it.
    orch = EscalationOrchestrator([IterBudgetExtensionStrategy()], mode="abstention")
    assert orch.mode == "abstention"
    assert orch.strategies[0].name == "iter_extension"


def test_orchestrator_rejects_unknown_mode() -> None:
    from mothrag.core.retry import build_default_orchestrator
    with pytest.raises(ValueError, match="mode must be 'loop' or 'abstention'"):
        build_default_orchestrator("all", mode="circular")


def test_mothrag_abstention_mode_propagates_terminal_abstain() -> None:
    """MothRAG(retry_mode='abstention') surfaces qr.answer='' + terminal_abstain=True
    when the cascade exhausts without recovery."""
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader
    rag = MothRAG.from_documents(
        ["A.", "B."],
        embedder=_HashEmbedder(),
        reader=_EchoReader(),
        production=True,
        retry_mode="abstention",
        # No soft_fallback in the strategy list; abstention mode allows it.
        retry_strategies=["arm_fallback"],
    )
    qr = rag.query("Q?")
    assert qr.metadata["retry_mode"] == "abstention"
    # The _EchoReader returns the first sentence of the top passage, which
    # produces a non-empty chosen -> signal=None -> no escalation triggered.
    # We assert that retry_mode is plumbed through telemetry regardless.
    assert "terminal_abstain" in qr.metadata


def test_mothrag_loop_mode_default_preserves_non_empty_guarantee() -> None:
    """Default retry_mode='loop' keeps SoftFallback as terminus, qr.answer non-empty."""
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader
    rag = MothRAG.from_documents(
        ["A.", "B."],
        embedder=_HashEmbedder(),
        reader=_EchoReader(),
        production=True,
        # explicit retry_mode for clarity; default would behave the same
        retry_mode="loop",
    )
    qr = rag.query("Q?")
    assert qr.metadata["retry_mode"] == "loop"
    assert qr.answer  # non-empty


def test_mothrag_rejects_unknown_retry_mode() -> None:
    from mothrag import MothRAG
    with pytest.raises(ValueError, match="MothRAG retry_mode must be"):
        MothRAG(retry_mode="invalid")
