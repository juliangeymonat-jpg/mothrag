"""Integration tests: Strategies #8 + #9 composed with #1-#7 default cascade
and with the `retry_strategies=['default_7', 'active_gap_query',
'sub_question_reroute']` invocation style."""

from __future__ import annotations

import sys
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _block_sentence_transformers(monkeypatch) -> Iterator[None]:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    yield


# ============================================================
# Preset / list expansion -- "default_7" + opt-ins
# ============================================================

def test_orchestrator_default_7_plus_active_gap_query() -> None:
    """retry_strategies=['default_7', 'active_gap_query'] expands to the
    canonical 7 strategies followed by ActiveGapQuery (just before
    SoftFallback when in loop mode)."""
    from mothrag.core.retry import build_default_orchestrator
    orch = build_default_orchestrator(["default_7", "active_gap_query"])
    names = [s.name for s in orch.strategies]
    # All 7 default strategies present (in any order before the terminal).
    for required in (
        "iter_extension", "arm_fallback", "cross_arm_consensus",
        "bottom_up_boost", "l4b_anchor_retry", "query_reformulation",
    ):
        assert required in names, f"missing {required!r} in {names}"
    # active_gap_query present in the chain.
    assert "active_gap_query" in names
    # SoftFallback is terminal.
    assert names[-1] == "soft_fallback"


def test_orchestrator_default_7_plus_both_extensions() -> None:
    """The full opt-in stack composes cleanly."""
    from mothrag.core.retry import build_default_orchestrator
    orch = build_default_orchestrator([
        "default_7", "active_gap_query", "sub_question_reroute",
    ])
    names = [s.name for s in orch.strategies]
    assert "active_gap_query" in names
    assert "sub_question_reroute" in names
    assert names[-1] == "soft_fallback"


def test_explicit_single_extension_only() -> None:
    """retry_strategies=['active_gap_query'] -> just #8 + auto-appended #7."""
    from mothrag.core.retry import build_default_orchestrator
    orch = build_default_orchestrator(["active_gap_query"])
    names = [s.name for s in orch.strategies]
    assert names == ["active_gap_query", "soft_fallback"]


def test_strategy_8_not_in_default_priority() -> None:
    """ActiveGapQuery must NOT be in DEFAULT_PRIORITY (opt-in only)."""
    from mothrag.core.retry.orchestrator import DEFAULT_PRIORITY
    assert "active_gap_query" not in DEFAULT_PRIORITY
    assert "sub_question_reroute" not in DEFAULT_PRIORITY


# ============================================================
# Cascade priority: extensions fire after #1-#6, before #3 + #7
# ============================================================

def test_strategy_8_fires_after_cheaper_strategies_when_they_decline() -> None:
    """When an early strategy declines, #8 gets the chance."""
    from mothrag.core.api import _HashEmbedder, _MemoryVectorStore, Chunk
    from mothrag.core.retry import RetryContext, build_default_orchestrator

    emb = _HashEmbedder()
    vdb = _MemoryVectorStore()
    chunks = [Chunk(text=t, embedding=list(emb.embed_batch([t])[0]))
              for t in ("alpha", "beta", "gamma_text")]
    vdb.add(chunks)

    class _R:
        def read(self, q, p):
            return "alpha" if "GAP QUERY" in q else ""

    def runner(*, question, passages):
        return "recovered via active_gap_query"

    ctx = RetryContext(
        question="What is alpha?",
        passages=["alpha"],
        q_emb=list(emb.embed_batch(["What is alpha?"])[0]),
        top_k=3,
        arm_subset=["v3bu"],
        v3bu_pred="not in passages",
        dec_pred=None,
        iter_pred=None,
        chosen="not in passages",
        arbitrate_reason="γ_refuse",
        abstention_signal="gamma_refuse",
        embedder=emb,
        vector_db=vdb,
        reader=_R(),
        run_arm_v3bu=runner,
    )
    # Cascade: #1 iter_extension (no run_arm_iter -> skip), #2 arm_fallback
    # (no non-empty siblings -> skip), then #4 cross_arm_consensus (needs
    # >=2 non-empty arms -> skip), then #5 bottom_up_boost (uses run_arm_iter,
    # not wired here -> skip), #6 l4b_anchor_retry (no l4b -> skip), then
    # #8 active_gap_query -> fires.
    orch = build_default_orchestrator([
        "iter_extension", "arm_fallback", "cross_arm_consensus",
        "bottom_up_boost", "l4b_anchor_retry", "active_gap_query",
    ])
    out = orch.try_escalate(ctx)
    assert out.recovered_by == "active_gap_query"
    assert "recovered" in out.answer


def test_loop_termination_extensions_eventually_yield_to_soft_fallback() -> None:
    """If #8 + #9 can't recover, the cascade still terminates via
    SoftFallback in loop mode."""
    from mothrag.core.api import _HashEmbedder, _MemoryVectorStore
    from mothrag.core.retry import RetryContext, build_default_orchestrator

    class _EmptyReader:
        def read(self, q, p):
            return ""

    emb = _HashEmbedder()
    vdb = _MemoryVectorStore()
    ctx = RetryContext(
        question="Q?",
        passages=[],
        q_emb=list(emb.embed_batch(["Q?"])[0]),
        top_k=3,
        arm_subset=[],
        v3bu_pred=None,
        dec_pred=None,
        iter_pred=None,
        chosen="not in passages",
        arbitrate_reason="empty",
        abstention_signal="empty_answer",
        embedder=emb,
        vector_db=vdb,
        reader=_EmptyReader(),
        run_arm_v3bu=lambda **_: "",
    )
    orch = build_default_orchestrator([
        "active_gap_query", "sub_question_reroute",
    ])
    out = orch.try_escalate(ctx)
    # Whether the extensions fire or defer, the cascade terminates with
    # SoftFallback (loop mode is default).
    assert out.recovered_by == "soft_fallback"
    assert out.answer  # non-empty guarantee


# ============================================================
# MothRAG-level wiring smoke
# ============================================================

def test_mothrag_with_active_gap_extensions_does_not_crash() -> None:
    """MothRAG construction + a single query succeeds when extensions are
    enabled in retry_strategies."""
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader
    rag = MothRAG.from_documents(
        ["First doc.", "Second doc."],
        embedder=_HashEmbedder(),
        reader=_EchoReader(),
        production=True,
        mode="ensemble_arbitrate",
        retry_strategies=["default_7", "active_gap_query", "sub_question_reroute"],
    )
    qr = rag.query("Q?")
    # qr.metadata should have the ensemble-path telemetry, the retry escalation
    # metadata, AND the cascade was equipped with the extensions.
    assert qr.metadata["mode"] == "ensemble_arbitrate"
    assert "retry_mode" in qr.metadata


def test_mothrag_sub_question_max_depth_config_propagates() -> None:
    """sub_question_max_depth from MothRAG **config flows through to the
    strategy's runtime override."""
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader
    rag = MothRAG.from_documents(
        ["A.", "B."],
        embedder=_HashEmbedder(), reader=_EchoReader(),
        production=True,
        retry_strategies=["sub_question_reroute"],
        sub_question_max_depth=1,
        sub_question_layers=("syntactic",),
    )
    qr = rag.query("Q?")
    # Sanity: the config knobs survived into self.config, ready for the
    # strategy's runtime override.
    assert rag.config["sub_question_max_depth"] == 1
    assert tuple(rag.config["sub_question_layers"]) == ("syntactic",)
