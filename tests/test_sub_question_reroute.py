"""Tests for SubQuestionRerouteCascadeStrategy (#9, 3-layer)."""

from __future__ import annotations

import sys
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _block_sentence_transformers(monkeypatch) -> Iterator[None]:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    yield


def _make_ctx(**overrides):
    """Build a RetryContext for sub_question_reroute testing."""
    from mothrag.core.api import _HashEmbedder, _MemoryVectorStore
    from mothrag.core.retry import RetryContext

    emb = _HashEmbedder()
    vdb = _MemoryVectorStore()

    base = dict(
        question="What is the capital of France and the capital of Japan?",
        passages=["Paris is the capital of France.", "Tokyo is the capital of Japan."],
        q_emb=list(emb.embed_batch(["dummy"])[0]),
        top_k=5,
        arm_subset=["v3bu", "iter"],
        v3bu_pred="Paris and Tokyo",
        dec_pred="Paris.",
        iter_pred="Tokyo.",
        chosen="not in passages",
        arbitrate_reason="γ_refuse",
        abstention_signal="gamma_refuse",
        embedder=emb,
        vector_db=vdb,
    )
    base.update(overrides)
    return RetryContext(**base)


class _DummyReader:
    """Reader that returns a configurable answer per query type."""
    def __init__(self, *, on_gap=None, on_compose=None, on_decompose=None, default=""):
        self.on_gap = on_gap
        self.on_compose = on_compose
        self.on_decompose = on_decompose
        self.default = default
        self.calls = []

    def read(self, q, p):
        self.calls.append(q)
        if "Decompose the following" in q:
            return self.on_decompose or ""
        if "FINAL ANSWER" in q.upper() or "Synthesise" in q:
            return self.on_compose or ""
        if "re-phrase" in q.lower() or "rephrase" in q.lower():
            return self.on_gap or ""
        return self.default


# ============================================================
# applicable / not-applicable
# ============================================================

def test_sub_question_reroute_applicable_on_supported_signals() -> None:
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )
    strat = SubQuestionRerouteCascadeStrategy()
    ctx = _make_ctx(reader=_DummyReader(default="Paris"),
                    run_arm_v3bu=lambda **_: "Paris")
    assert strat.applicable(ctx)


def test_sub_question_reroute_not_applicable_without_reader() -> None:
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )
    ctx = _make_ctx(reader=None, run_arm_v3bu=lambda **_: "X")
    assert not SubQuestionRerouteCascadeStrategy().applicable(ctx)


# ============================================================
# Layer 1 syntactic
# ============================================================

def test_layer1_syntactic_splits_conjunction() -> None:
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )
    strat = SubQuestionRerouteCascadeStrategy(layers=("syntactic",))
    ctx = _make_ctx(reader=_DummyReader(default="Paris"))
    sub_qs = strat._layer1_syntactic(ctx)
    # The question has "and" splitting two halves -> at least 2 sub-qs.
    assert len(sub_qs) >= 2


def test_layer1_syntactic_single_clause_returns_empty() -> None:
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )
    strat = SubQuestionRerouteCascadeStrategy(layers=("syntactic",))
    ctx = _make_ctx(
        question="What is Paris?",
        reader=_DummyReader(default="X"),
    )
    sub_qs = strat._layer1_syntactic(ctx)
    # Single-clause question: regex split yields 0 or 1, both treated as no split.
    assert len(sub_qs) == 0


# ============================================================
# Layer 2 spectral
# ============================================================

def test_layer2_spectral_flags_low_signal_aspects() -> None:
    """When chosen answer has aspects + the global signal is invalid,
    Layer 2 should flag at least one aspect as low-signal."""
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )
    strat = SubQuestionRerouteCascadeStrategy(layers=("spectral",))
    ctx = _make_ctx(
        chosen="Paris is the capital of France.",
        abstention_signal="gamma_refuse",  # -> gamma_status = "invalid"
        c7_info={"gamma_status": "invalid"},
        reader=_DummyReader(default="X"),
        run_arm_v3bu=lambda **_: "X",
    )
    sub_qs = strat._layer2_spectral(ctx)
    # With gamma=invalid (0.0), every aspect is below 0.5 threshold.
    assert len(sub_qs) >= 1
    # Default template form.
    assert any("Verify:" in q for q in sub_qs)


def test_layer2_spectral_no_low_signal_returns_empty() -> None:
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )
    strat = SubQuestionRerouteCascadeStrategy(layers=("spectral",))
    ctx = _make_ctx(
        chosen="Paris is the capital of France.",
        abstention_signal="empty_answer",
        c7_info={"gamma_status": "valid"},
        v3bu_pred="Paris is the capital of France.",
        dec_pred="Paris is the capital of France.",
        iter_pred="Paris is the capital of France.",
        reader=_DummyReader(default="X"),
    )
    # gamma=valid (1.0), agreement high (identical arm answers),
    # l4b default 1.0 -> no aspect below 0.5 -> no sub-questions.
    sub_qs = strat._layer2_spectral(ctx)
    assert sub_qs == []


def test_layer2_spectral_no_aspects_returns_empty() -> None:
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )
    strat = SubQuestionRerouteCascadeStrategy(layers=("spectral",))
    ctx = _make_ctx(
        chosen="",  # nothing for aspect extractor to chew on
        reader=_DummyReader(default="X"),
    )
    assert strat._layer2_spectral(ctx) == []


def test_layer2_spectral_with_use_llm_calls_reader() -> None:
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )
    reader = _DummyReader(on_gap="Is Paris really the capital?", default="")
    strat = SubQuestionRerouteCascadeStrategy(
        layers=("spectral",), use_llm_in_spectral=True,
    )
    ctx = _make_ctx(
        chosen="Paris is the capital of France.",
        abstention_signal="gamma_refuse",
        c7_info={"gamma_status": "invalid"},
        reader=reader,
    )
    sub_qs = strat._layer2_spectral(ctx)
    assert sub_qs
    # At least one sub-question should be the LLM-phrased version.
    assert any("Paris" in q or "really" in q.lower() for q in sub_qs)


def test_layer2_spectral_extracts_gamma_status_from_c7_info() -> None:
    """Verify the helper picks up gamma_status from c7_info dict."""
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )
    ctx = _make_ctx(c7_info={"gamma_status": "partial"})
    g = SubQuestionRerouteCascadeStrategy._extract_gamma_status_from_ctx(ctx)
    assert g == "partial"


def test_layer2_spectral_extracts_l4b_cancelled_from_c7_info() -> None:
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )
    ctx = _make_ctx(c7_info={"l4b": {"cancelled": True}})
    l = SubQuestionRerouteCascadeStrategy._extract_l4b_cancelled_from_ctx(ctx)
    assert l is True


# ============================================================
# Layer 3 LLM fallback
# ============================================================

def test_layer3_llm_fires_when_layers_1_2_yield_few() -> None:
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )
    reader = _DummyReader(
        on_decompose="What is the capital of France?\nWhat is the capital of Japan?",
        default="",
    )
    strat = SubQuestionRerouteCascadeStrategy(
        layers=("llm",), min_sub_questions_before_llm=2,
    )
    ctx = _make_ctx(reader=reader, run_arm_v3bu=lambda **_: "Paris")
    sub_qs = strat._generate_sub_questions(ctx)
    assert len(sub_qs) >= 2


def test_layer3_llm_does_not_fire_when_layers_satisfy_threshold() -> None:
    """When Layer 1 already produced >= min_sub_questions_before_llm, Layer 3
    should NOT call the reader (cost guard)."""
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )
    reader = _DummyReader(on_decompose="LLM-decomp", default="")
    strat = SubQuestionRerouteCascadeStrategy(
        layers=("syntactic", "llm"),
        min_sub_questions_before_llm=2,
    )
    ctx = _make_ctx(reader=reader)
    sub_qs = strat._generate_sub_questions(ctx)
    # The conjunction-question yields >= 2 sub-qs via Layer 1.
    assert len(sub_qs) >= 2
    # Layer 3 didn't fire -> reader.on_decompose not in the result.
    assert "LLM-decomp" not in sub_qs


def test_layer3_llm_handles_reader_exception_gracefully() -> None:
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )

    class _BrokenR:
        def read(self, q, p):
            raise RuntimeError("boom")

    strat = SubQuestionRerouteCascadeStrategy(
        layers=("llm",), min_sub_questions_before_llm=99,
    )
    ctx = _make_ctx(reader=_BrokenR())
    sub_qs = strat._layer3_llm_fallback(ctx)
    assert sub_qs == []


# ============================================================
# Composition + try_recover end-to-end
# ============================================================

def test_compose_template_substitution_for_conjunction() -> None:
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )
    strat = SubQuestionRerouteCascadeStrategy()
    ctx = _make_ctx(reader=_DummyReader(default=""))
    sub_qs = ["What is the capital of France?", "What is the capital of Japan?"]
    answers = ["Paris", "Tokyo"]
    out = strat._compose(ctx, sub_qs, answers)
    assert "Paris" in out and "Tokyo" in out


def test_compose_llm_synthesis_when_no_conjunction() -> None:
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )
    reader = _DummyReader(on_compose="Paris is the capital of France.")
    strat = SubQuestionRerouteCascadeStrategy()
    ctx = _make_ctx(
        question="What is Paris known for?",  # no conjunction
        reader=reader,
    )
    sub_qs = ["Is Paris a city?"]
    answers = ["Yes"]
    out = strat._compose(ctx, sub_qs, answers)
    assert "Paris" in out


def test_try_recover_end_to_end_returns_composed_answer() -> None:
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )

    def runner(*, question, passages):
        if "France" in question:
            return "Paris"
        if "Japan" in question:
            return "Tokyo"
        return ""

    strat = SubQuestionRerouteCascadeStrategy(layers=("syntactic",))
    ctx = _make_ctx(
        reader=_DummyReader(default=""),
        run_arm_v3bu=runner,
    )
    out = strat.try_recover(ctx)
    assert out
    # Template substitution joins answers from the two clausal sub-qs.
    assert "Paris" in out or "Tokyo" in out


def test_try_recover_respects_max_depth_via_ctx_config() -> None:
    """ctx.config['sub_question_max_depth']=1 caps the loop at one pass."""
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )

    # Runner always returns uncertain so the strategy never converges.
    strat = SubQuestionRerouteCascadeStrategy(layers=("syntactic",))
    ctx = _make_ctx(
        reader=_DummyReader(default=""),
        run_arm_v3bu=lambda **_: "not in passages",
        config={"sub_question_max_depth": 1, "sub_question_layers": ("syntactic",)},
    )
    out = strat.try_recover(ctx)
    assert out is None  # loop exhausted; no recovery


def test_try_recover_budget_guard() -> None:
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )
    strat = SubQuestionRerouteCascadeStrategy()
    ctx = _make_ctx(
        reader=_DummyReader(default=""),
        run_arm_v3bu=lambda **_: "Paris",
        budget_limit=1,  # below cost_estimate=4
    )
    out = strat.try_recover(ctx)
    assert out is None


def test_try_recover_returns_none_when_no_sub_questions_generated() -> None:
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )
    strat = SubQuestionRerouteCascadeStrategy(layers=("syntactic",))
    ctx = _make_ctx(
        question="What is Paris?",  # single-clause
        reader=_DummyReader(default=""),
        run_arm_v3bu=lambda **_: "Paris",
    )
    out = strat.try_recover(ctx)
    assert out is None


# ============================================================
# sel_v2 dispatch + state accumulation contract
# ============================================================

def test_strategy_9_dispatches_via_sel_v2_same_arm_allowed() -> None:
    """Per-sub-question dispatch consults sel_v2 (``arm_subset``) afresh
    and uses whatever arm sel_v2 picks. There is NO "different arm"
    forcing -- the strategy must invoke the SAME arm as the original
    failing path when sel_v2 says so for the sub-question shape."""
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )

    # Force sel_v2 to always return ["v3bu"] for any sub-question.
    # If the implementation forces a "different arm", this monkey-patch
    # would be ignored and a non-v3bu runner would be called.
    import mothrag.core.query_type_classifier as qtc

    calls_v3bu: list[str] = []
    calls_decompose: list[str] = []
    calls_iter: list[str] = []

    def _v3bu(*, question, passages):
        calls_v3bu.append(question)
        return "answer-from-v3bu"

    def _decompose(*, question, passages):
        calls_decompose.append(question)
        return "answer-from-decompose"

    def _iter(*, question, passages, q_emb=None, top_k=None,
              max_steps=None, l4b_anchor=None):
        calls_iter.append(question)
        return "answer-from-iter"

    strat = SubQuestionRerouteCascadeStrategy(layers=("syntactic",))
    ctx = _make_ctx(
        reader=_DummyReader(default=""),
        run_arm_v3bu=_v3bu,
        run_arm_decompose=_decompose,
        run_arm_iter=_iter,
        # ctx.arm_subset names the original chosen path (here: iter).
        # The strategy must NOT exclude it from sub-question dispatch.
        arm_subset=["iter"],
    )

    original_arm_subset = qtc.arm_subset
    try:
        # Pin sel_v2 to v3bu for sub-questions so the test is deterministic.
        qtc.arm_subset = lambda q, **kw: ["v3bu"]
        strat.try_recover(ctx)
    finally:
        qtc.arm_subset = original_arm_subset

    # sel_v2's choice for sub-questions was v3bu; v3bu MUST have fired.
    assert calls_v3bu, (
        "Strategy #9 did not invoke v3bu even though sel_v2 chose it for "
        "every sub-question. This indicates a sel_v2 dispatch violation."
    )
    # Decompose / iter MUST NOT fire when sel_v2 chose v3bu, regardless
    # of what arm the original failing chosen path was.
    assert not calls_decompose
    assert not calls_iter


def test_strategy_9_context_accumulates_across_sub_questions() -> None:
    """Subsequent sub-questions see the running ``(sub_q, sub_a)`` trail.
    The third sub-question's prompt MUST mention the first two
    sub-answers (state preservation contract)."""
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )

    seen_passages_per_call: list[list[str]] = []
    seen_questions_per_call: list[str] = []

    def _v3bu(*, question, passages):
        seen_questions_per_call.append(question)
        seen_passages_per_call.append(list(passages))
        # Return a distinct token per call so accumulated state is
        # observable in the next sub-question's prompt.
        return f"answer-{len(seen_questions_per_call)}"

    # The original question splits into 3 syntactic clauses so Layer 1
    # produces 3 sub-questions deterministically.
    strat = SubQuestionRerouteCascadeStrategy(layers=("syntactic",))
    ctx = _make_ctx(
        question="What is alpha and what is beta and what is gamma?",
        passages=["base passage A", "base passage B"],
        reader=_DummyReader(default=""),
        run_arm_v3bu=_v3bu,
    )
    strat.try_recover(ctx)

    assert len(seen_questions_per_call) >= 2, (
        "Expected at least 2 sub-questions to fire on a 3-clause "
        f"conjunction; got {len(seen_questions_per_call)}."
    )

    # State-accumulation contract: each subsequent sub-question prompt
    # must include the prior sub-answers OR a "Prior sub-question facts"
    # passage. Either suffices to prove state was passed forward.
    second_call_q = seen_questions_per_call[1]
    second_call_passages = seen_passages_per_call[1]
    accumulated_in_q = "answer-1" in second_call_q \
        or "Context from prior" in second_call_q
    accumulated_in_passages = any(
        "answer-1" in p or "Prior sub-question" in p
        for p in second_call_passages
    )
    assert accumulated_in_q or accumulated_in_passages, (
        "Strategy #9 did not propagate the first sub-answer into the "
        "second sub-question's context. State accumulation contract "
        "violated."
    )


# ============================================================
# Precision gate: #9 must NOT fire when the original pred is
# substantive -- empirically, replacing substantive partial-correct
# preds with composed sub-answer synthesis caused -16 to -23pp F1
# regression on the 0<F1<0.3 mid-range cohort.
# ============================================================

def test_strategy_9_precision_gate_declines_on_substantive_pred() -> None:
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )
    strat = SubQuestionRerouteCascadeStrategy()
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
            reader=_DummyReader(default="X"),
            run_arm_v3bu=lambda **_: "X",
        )
        assert not strat.applicable(ctx), (
            f"Strategy #9 fired on substantive pred {substantive_pred!r}; "
            f"precision gate violated -- expected decline."
        )


def test_strategy_9_precision_gate_allows_empty_and_refuse_templates() -> None:
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )
    strat = SubQuestionRerouteCascadeStrategy()
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
            reader=_DummyReader(default="X"),
            run_arm_v3bu=lambda **_: "X",
        )
        assert strat.applicable(ctx), (
            f"Strategy #9 declined on refuse-template pred {refuse_pred!r}; "
            f"precision gate over-fired."
        )
