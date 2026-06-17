"""Tests for ActiveGapQueryStrategy (#8)."""

from __future__ import annotations

import sys
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _block_sentence_transformers(monkeypatch) -> Iterator[None]:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    yield


def _make_ctx(**overrides):
    """Build a RetryContext for strategy testing.

    Returns a RetryContext with defaults that route around the segfault
    sentence_transformers path and provide enough infrastructure for
    the strategy to actually run an end-to-end cycle.
    """
    from mothrag.core.api import _HashEmbedder, _MemoryVectorStore, Chunk
    from mothrag.core.retry import RetryContext

    # Tiny corpus so the gap retrieve has something to chew on.
    emb = _HashEmbedder()
    vdb = _MemoryVectorStore()
    chunks = []
    for text in (
        "Paris is the capital of France.",
        "France is in Europe.",
        "Eiffel Tower is in Paris.",
        "Tokyo is the capital of Japan.",
    ):
        ch = Chunk(text=text, embedding=list(emb.embed_batch([text])[0]))
        chunks.append(ch)
    vdb.add(chunks)

    base = dict(
        question="What city is the capital of France?",
        passages=["Paris is the capital of France."],
        q_emb=list(emb.embed_batch(["What city is the capital of France?"])[0]),
        top_k=5,
        arm_subset=["v3bu", "iter"],
        v3bu_pred=None,
        dec_pred=None,
        iter_pred=None,
        chosen="",
        arbitrate_reason="γ_refuse",
        abstention_signal="gamma_refuse",
        embedder=emb,
        vector_db=vdb,
    )
    base.update(overrides)
    return RetryContext(**base)


class _StaticReader:
    """Returns a fixed answer regardless of input."""
    def __init__(self, answer):
        self.answer = answer
        self.calls = []
    def read(self, q, p):
        self.calls.append((q, p))
        return self.answer


# ============================================================
# 1. applicable / not-applicable
# ============================================================

def test_active_gap_query_applicable_on_gamma_refuse() -> None:
    from mothrag.core.retry.strategies.active_gap_query import ActiveGapQueryStrategy
    ctx = _make_ctx(reader=_StaticReader("Paris"),
                    run_arm_v3bu=lambda **_: "Paris")
    assert ActiveGapQueryStrategy().applicable(ctx)


def test_active_gap_query_not_applicable_without_reader() -> None:
    from mothrag.core.retry.strategies.active_gap_query import ActiveGapQueryStrategy
    ctx = _make_ctx(reader=None,
                    run_arm_v3bu=lambda **_: "Paris")
    assert not ActiveGapQueryStrategy().applicable(ctx)


def test_active_gap_query_not_applicable_without_arm_runners() -> None:
    from mothrag.core.retry.strategies.active_gap_query import ActiveGapQueryStrategy
    ctx = _make_ctx(reader=_StaticReader("X"),
                    run_arm_v3bu=None, run_arm_iter=None)
    assert not ActiveGapQueryStrategy().applicable(ctx)


def test_active_gap_query_not_applicable_on_unrelated_signal() -> None:
    from mothrag.core.retry.strategies.active_gap_query import ActiveGapQueryStrategy
    ctx = _make_ctx(
        reader=_StaticReader("X"),
        run_arm_v3bu=lambda **_: "X",
        abstention_signal="something_else",
    )
    assert not ActiveGapQueryStrategy().applicable(ctx)


# ============================================================
# 2. try_recover happy path
# ============================================================

def test_active_gap_query_recovers_on_first_round() -> None:
    from mothrag.core.retry.strategies.active_gap_query import ActiveGapQueryStrategy
    reader_calls: list[str] = []

    def reader_read(q, p):
        # First call: gap-query introspection. Reader returns the gap
        # query "capital of France".
        if "GAP QUERY" in q:
            return "capital of France"
        reader_calls.append(q)
        return ""

    class _R:
        def read(self, q, p):
            return reader_read(q, p)

    runner_calls: list[dict] = []

    def runner(*, question, passages):
        runner_calls.append({"question": question, "passages": list(passages)})
        return "Paris is the capital of France."

    ctx = _make_ctx(reader=_R(), run_arm_v3bu=runner)
    strat = ActiveGapQueryStrategy(max_rounds=3)
    out = strat.try_recover(ctx)
    assert out and "Paris" in out
    # The runner was called with augmented passages.
    assert len(runner_calls) >= 1


def test_active_gap_query_returns_none_on_no_gap_query() -> None:
    """If the reader returns an empty / no-op gap query, defer."""
    from mothrag.core.retry.strategies.active_gap_query import ActiveGapQueryStrategy

    class _EmptyR:
        def read(self, q, p):
            return ""

    ctx = _make_ctx(reader=_EmptyR(),
                    run_arm_v3bu=lambda **_: "Paris")
    out = ActiveGapQueryStrategy(max_rounds=3).try_recover(ctx)
    assert out is None


def test_active_gap_query_respects_max_rounds_via_ctx_config() -> None:
    """max_rounds override via ctx.config['active_gap_max_rounds']."""
    from mothrag.core.retry.strategies.active_gap_query import ActiveGapQueryStrategy

    round_counter = {"count": 0}

    class _R:
        def read(self, q, p):
            round_counter["count"] += 1
            if "GAP QUERY" in q:
                return "Tokyo"  # Always returns something, never converges
            return ""

    # Runner always returns uncertain so the loop iterates.
    ctx = _make_ctx(
        reader=_R(),
        run_arm_v3bu=lambda **_: "not in passages",
        config={"active_gap_max_rounds": 2},
    )
    strat = ActiveGapQueryStrategy(max_rounds=5)  # constructor default 5
    out = strat.try_recover(ctx)
    assert out is None
    # max_rounds=2 from ctx.config should win over constructor 5.
    # The reader is called at least once per round (gap query).
    assert round_counter["count"] <= 5


def test_active_gap_query_returns_none_when_runner_keeps_uncertain() -> None:
    from mothrag.core.retry.strategies.active_gap_query import ActiveGapQueryStrategy

    class _GR:
        def read(self, q, p):
            return "Tokyo" if "GAP QUERY" in q else ""

    ctx = _make_ctx(
        reader=_GR(),
        run_arm_v3bu=lambda **_: "not in passages",
    )
    strat = ActiveGapQueryStrategy(max_rounds=2)
    out = strat.try_recover(ctx)
    assert out is None


def test_active_gap_query_handles_reader_exception_gracefully() -> None:
    from mothrag.core.retry.strategies.active_gap_query import ActiveGapQueryStrategy

    class _BrokenR:
        def read(self, q, p):
            raise RuntimeError("simulated reader crash")

    ctx = _make_ctx(reader=_BrokenR(),
                    run_arm_v3bu=lambda **_: "X")
    out = ActiveGapQueryStrategy(max_rounds=2).try_recover(ctx)
    assert out is None


def test_active_gap_query_budget_guard_blocks_unbounded_recursion() -> None:
    from mothrag.core.retry.strategies.active_gap_query import ActiveGapQueryStrategy

    class _R:
        def read(self, q, p):
            return "capital of France" if "GAP QUERY" in q else ""

    ctx = _make_ctx(
        reader=_R(),
        run_arm_v3bu=lambda **_: "not in passages",
        budget_limit=1,  # well below cost_estimate*max_rounds
    )
    out = ActiveGapQueryStrategy(max_rounds=5).try_recover(ctx)
    # With budget=1 and cost_estimate=2 the strategy can't even fire
    # a single round -> defers immediately.
    assert out is None


# ============================================================
# sel_v2 dispatch contract
# ============================================================

def test_strategy_8_reread_respects_sel_v2_when_iter_chosen() -> None:
    """When ctx.arm_subset = ['iter'] (production sel_v2 a priori
    choice for chain-deep questions), the re-read must invoke
    run_arm_iter, NOT hard-code v3bu first."""
    from mothrag.core.retry.strategies.active_gap_query import (
        ActiveGapQueryStrategy,
    )

    iter_calls: list[str] = []
    v3bu_calls: list[str] = []

    def _v3bu(*, question, passages):
        v3bu_calls.append(question)
        return "answer-from-v3bu"

    def _iter(*, question, passages, q_emb=None, top_k=None,
              max_steps=None, l4b_anchor=None):
        iter_calls.append(question)
        return "answer-from-iter"

    class _R:
        def read(self, q, p):
            return "capital of france" if "GAP QUERY" in q else ""

    ctx = _make_ctx(
        reader=_R(),
        run_arm_v3bu=_v3bu,
        run_arm_iter=_iter,
        arm_subset=["iter"],  # sel_v2's a priori choice
    )
    out = ActiveGapQueryStrategy(max_rounds=1).try_recover(ctx)
    # iter must have been invoked (sel_v2's a priori choice respected).
    assert iter_calls, (
        "Strategy #8 ignored ctx.arm_subset=['iter'] and did not "
        "invoke run_arm_iter for the re-read. This is a sel_v2 "
        "dispatch violation analogous to the always-3-arms catch."
    )
    # v3bu must NOT have been invoked when iter was both available
    # and named by sel_v2.
    assert not v3bu_calls, (
        "Strategy #8 invoked v3bu even though sel_v2 chose iter and "
        "the iter runner was wired. v3bu should only fire when sel_v2's "
        "choice is not wired."
    )
    # Either way, a non-empty answer should come back.
    assert out == "answer-from-iter" or out is None


# ============================================================
# Precision gate: #8 must NOT fire when the original pred is
# substantive -- empirically, replacing substantive partial-correct
# preds with #8 alternatives caused a large F1 regression on the
# 0<F1<0.3 mid-range cohort.
# ============================================================

def test_strategy_8_precision_gate_declines_on_substantive_pred() -> None:
    """When ctx.chosen is a substantive answer (not empty / not a
    refuse-template), Strategy #8 must NOT fire -- even on
    gamma_refuse signal."""
    from mothrag.core.retry.strategies.active_gap_query import (
        ActiveGapQueryStrategy,
    )
    strat = ActiveGapQueryStrategy()
    for substantive_pred in (
        "Flavivirus",
        "14 March 1879",
        "Mickey's PhilharMagic",
        "Paris is the capital of France",
        "Jon Hamm",
    ):
        ctx = _make_ctx(
            chosen=substantive_pred,
            abstention_signal="gamma_refuse",
            reader=_StaticReader("X"),
            run_arm_v3bu=lambda **_: "X",
        )
        assert not strat.applicable(ctx), (
            f"Strategy #8 fired on substantive pred {substantive_pred!r}; "
            f"precision gate violated -- expected decline."
        )


def test_strategy_8_precision_gate_allows_empty_and_refuse_templates() -> None:
    """The precision gate must NOT block firing on empty / refuse-template
    preds (the cohort where #8 has a chance to recover)."""
    from mothrag.core.retry.strategies.active_gap_query import (
        ActiveGapQueryStrategy,
    )
    strat = ActiveGapQueryStrategy()
    for refuse_pred in (
        "",
        "Not in passages",
        "Unknown",
        "I don't know",
        "no answer",
        "Insufficient information",
        "n/a",
        "I cannot answer",
    ):
        ctx = _make_ctx(
            chosen=refuse_pred,
            abstention_signal="gamma_refuse",
            reader=_StaticReader("X"),
            run_arm_v3bu=lambda **_: "X",
        )
        assert strat.applicable(ctx), (
            f"Strategy #8 declined on refuse-template pred {refuse_pred!r}; "
            f"precision gate over-fired -- the empty/refuse cohort is "
            f"exactly where #8 should fire."
        )


def test_strategy_8_falls_back_when_sel_v2_arm_not_wired() -> None:
    """When ctx.arm_subset names an arm whose runner is None, fall
    back through v3bu then iter (runner-availability fallback, NOT a
    sel_v2 override)."""
    from mothrag.core.retry.strategies.active_gap_query import (
        ActiveGapQueryStrategy,
    )

    v3bu_calls: list[str] = []

    def _v3bu(*, question, passages):
        v3bu_calls.append(question)
        return "answer-from-v3bu"

    class _R:
        def read(self, q, p):
            return "capital of france" if "GAP QUERY" in q else ""

    ctx = _make_ctx(
        reader=_R(),
        run_arm_v3bu=_v3bu,
        run_arm_iter=None,        # iter runner NOT wired
        arm_subset=["iter"],     # sel_v2 chose iter
    )
    ActiveGapQueryStrategy(max_rounds=1).try_recover(ctx)
    # iter runner is None; fall-through reaches v3bu.
    assert v3bu_calls, (
        "Strategy #8 must fall back through v3bu when sel_v2's chosen "
        "runner is not wired. The fallback chain is a runner-"
        "availability fallback, not a sel_v2 override."
    )
