"""ctx-completeness for #8/#9 + dense_plus_infobox CLI hooks.

Validates that the new pipeline-ctx adapters in
``scripts/route_prospective.py`` produce RetryContexts on which the
opt-in Strategy #8 ActiveGapQuery + #9 SubQuestionRerouteCascade are
:meth:`applicable` (= "INVALID standalone test" gap closed), and that
the dense_plus_infobox augmentation injects synthetic infobox chunks
into the pipeline's dense index when triples are harvestable.

These are unit-level tests using stubbed pipeline / iter_runner /
reader_client surfaces rather than the real cloud pipelines (those
remain gated behind a separate eval run).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest


# Make the scripts/ dir importable as a module for direct symbol access.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture(autouse=True)
def _block_sentence_transformers(monkeypatch) -> Iterator[None]:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    yield


# ============================================================
# Stubs mirroring the surface route_prospective.py consumes
# ============================================================

class _StubReaderClient:
    """Mimics openai.OpenAI's `chat.completions.create` surface."""

    def __init__(self, on_call=None):
        self._on_call = on_call or (lambda **_: "stub answer")
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self._on_call(**kwargs)
        # Build a minimal OpenAI-shape response.
        class _Msg:
            def __init__(self, c): self.content = c
        class _Choice:
            def __init__(self, c): self.message = _Msg(c)
        class _Usage:
            prompt_tokens = 10
            completion_tokens = 5
        class _Resp:
            def __init__(self, c):
                self.choices = [_Choice(c)]
                self.usage = _Usage()
        return _Resp(content)


class _StubPipeline:
    """Minimal MothRAGPipeline-like stub for adapter unit tests."""

    def __init__(self, chunks: dict[str, str], reader_client=None):
        import numpy as np
        self.chunks_by_id = {
            cid: {"text": text, "chunk_id": cid, "metadata": {}}
            for cid, text in chunks.items()
        }
        self.chunk_ids = list(chunks.keys())
        # Tiny deterministic embeddings: hash-bucketed unit-norm vectors.
        dim = 32

        def _embed(text: str):
            v = np.zeros(dim, dtype=np.float32)
            for i, ch in enumerate(text[:dim]):
                v[i % dim] += float(ord(ch) % 13)
            n = np.linalg.norm(v)
            return v / n if n > 0 else v

        self._embed = _embed
        self.query_embedder = _embed
        self.chunk_vecs = (
            np.stack([_embed(t) for t in chunks.values()], axis=0)
            if chunks else np.zeros((0, dim), dtype=np.float32)
        )
        self.reader_client = reader_client or _StubReaderClient()

    def retrieve(self, question: str, entity_seeds=None):
        import numpy as np
        qv = self._embed(question)
        scores = self.chunk_vecs @ qv
        top = np.argsort(-scores)[:5].tolist()
        return top, "stub_route", float(scores[top[0]]) if len(top) else 0.0


# ============================================================
# Adapter unit tests
# ============================================================

def test_pipeline_reader_shim_exposes_read_surface() -> None:
    import route_prospective as rp
    client = _StubReaderClient(on_call=lambda **kw: "stubbed answer")
    pipeline = _StubPipeline({"c1": "Some passage."}, reader_client=client)
    shim = rp._PipelineReaderShim(pipeline, reader_model="stub-model")
    out = shim.read("question?", ["passage"])
    assert isinstance(out, str)
    # The shim must have invoked the underlying client at least once.
    assert client.calls


def test_pipeline_vdb_shim_returns_chunks_with_text() -> None:
    import route_prospective as rp
    pipeline = _StubPipeline({
        "c1": "Paris is the capital of France.",
        "c2": "Tokyo is the capital of Japan.",
        "c3": "Berlin is in Germany.",
    })
    shim = rp._PipelineVdbShim(pipeline)
    q_emb = pipeline._embed("capital of France")
    chunks = shim.retrieve(q_emb, top_k=2)
    assert len(chunks) == 2
    assert all(hasattr(c, "text") and hasattr(c, "chunk_id") for c in chunks)


def test_pipeline_vdb_shim_handles_empty_index() -> None:
    import route_prospective as rp
    pipeline = _StubPipeline({})
    shim = rp._PipelineVdbShim(pipeline)
    assert shim.retrieve([0.0] * 32, top_k=5) == []


def test_resolve_arm_subset_returns_nonempty_for_normal_question() -> None:
    import route_prospective as rp
    out = rp._resolve_arm_subset("When was Albert Einstein born?")
    assert isinstance(out, list)
    assert out


def test_resolve_arm_subset_falls_back_to_v3bu_on_exception() -> None:
    import route_prospective as rp
    # Pass an empty question; arm_subset returns [] for trivial inputs.
    out = rp._resolve_arm_subset("")
    assert "v3bu" in out


# ============================================================
# RetryContext completeness via _maybe_run_escalation (mock pipeline)
# ============================================================

class _Args:
    """Mimics the argparse.Namespace surface that _maybe_run_escalation reads."""

    def __init__(self, **kwargs):
        defaults = dict(
            retry_strategies="active_gap_query",
            retry_mode="loop",
            retry_budget_limit=8,
            sub_question_layers="syntactic,llm",
            sub_question_max_depth=2,
            sub_question_max_sub_questions=4,
            active_gap_max_rounds=2,
            active_gap_max_passages_per_round=3,
            use_spectral=False,
            top_k_chunks=5,
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


def test_strategy_8_applicable_with_pipeline_ctx_adapters() -> None:
    """Contract: with the real pipeline-ctx adapters wired,
    Strategy #8 ActiveGapQuery's :meth:`applicable` must return True
    (the prior placeholder plumbing made it return False because
    ctx.reader was None)."""
    from mothrag.core.retry.strategies.active_gap_query import (
        ActiveGapQueryStrategy,
    )
    import route_prospective as rp

    pipeline = _StubPipeline({"c1": "Some context.", "c2": "More context."})

    # Reconstruct the ctx the way _maybe_run_escalation does (without
    # actually running the orchestrator; we only need the ctx for the
    # applicable() check).
    embedder = rp._PipelineEmbedderShim(pipeline)
    reader_shim = rp._PipelineReaderShim(pipeline, reader_model="stub")
    vdb_shim = rp._PipelineVdbShim(pipeline)

    from mothrag.core.retry import RetryContext

    ctx = RetryContext(
        question="When was X born?",
        passages=["Some context.", "More context."],
        q_emb=list(pipeline._embed("When was X born?")),
        top_k=5,
        arm_subset=rp._resolve_arm_subset("When was X born?"),
        v3bu_pred=None,
        dec_pred=None,
        iter_pred=None,
        chosen="Not in passages",
        arbitrate_reason="γ_refuse",
        abstention_signal="gamma_refuse",
        embedder=embedder,
        reader=reader_shim,
        vector_db=vdb_shim,
        run_arm_v3bu=lambda **_: "v3bu-stub",
        run_arm_iter=lambda **_: "iter-stub",
    )
    assert ActiveGapQueryStrategy().applicable(ctx), (
        "ActiveGapQuery declined to fire even with the real pipeline-ctx "
        "reader + vector_db adapters wired. Context-completeness contract violated."
    )


def test_strategy_9_applicable_with_pipeline_ctx_adapters() -> None:
    """Contract: Strategy #9 SubQuestionRerouteCascade must similarly
    be :meth:`applicable` with the real adapters wired."""
    from mothrag.core.retry.strategies.sub_question_reroute import (
        SubQuestionRerouteCascadeStrategy,
    )
    import route_prospective as rp

    pipeline = _StubPipeline({"c1": "ctx"})
    reader_shim = rp._PipelineReaderShim(pipeline, reader_model="stub")
    vdb_shim = rp._PipelineVdbShim(pipeline)

    from mothrag.core.retry import RetryContext

    ctx = RetryContext(
        question="Where was X born and what is the capital of Y?",
        passages=["ctx"],
        q_emb=list(pipeline._embed("Where was X born?")),
        top_k=5,
        arm_subset=["v3bu", "decompose"],
        v3bu_pred="X1",
        dec_pred="X2",
        iter_pred=None,
        chosen="Not in passages",
        arbitrate_reason="sel_v1:both-uncertain",
        abstention_signal="gamma_refuse",
        embedder=rp._PipelineEmbedderShim(pipeline),
        reader=reader_shim,
        vector_db=vdb_shim,
        run_arm_v3bu=lambda **_: "v3bu-stub",
        run_arm_iter=lambda **_: "iter-stub",
    )
    assert SubQuestionRerouteCascadeStrategy().applicable(ctx), (
        "SubQuestionRerouteCascade declined to fire even with the real "
        "pipeline-ctx adapters wired. Context-completeness contract violated."
    )


# ============================================================
# all_8 preset alias
# ============================================================

def test_all_8_preset_expands_to_canonical_seven_plus_8_9_plus_terminal() -> None:
    from mothrag.core.retry.orchestrator import _expand_preset_names
    names = _expand_preset_names(["all_8"])
    # Canonical 7 strategies are present
    for n in (
        "iter_extension", "arm_fallback", "cross_arm_consensus",
        "bottom_up_boost", "l4b_anchor_retry", "query_reformulation",
    ):
        assert n in names, f"all_8 missing canonical strategy {n!r}"
    # Active-learning extensions present
    assert "active_gap_query" in names
    assert "sub_question_reroute" in names
    # Terminal soft_fallback present
    assert "soft_fallback" in names


def test_all_8_preset_can_build_orchestrator() -> None:
    from mothrag.core.retry import build_default_orchestrator
    orch = build_default_orchestrator("all_8")
    names = [s.name for s in orch.strategies]
    assert names[-1] == "soft_fallback"
    assert "active_gap_query" in names
    assert "sub_question_reroute" in names


# ============================================================
# dense_plus_infobox augmentation
# ============================================================

def test_augment_pipeline_with_infobox_extends_chunk_vecs() -> None:
    """Synthetic infobox chunks must be appended to pipeline.chunk_vecs
    / chunk_ids / chunks_by_id."""
    import route_prospective as rp

    wikitext = (
        "{{Infobox person\n"
        "| name = Alan Turing\n"
        "| born = 23 June 1912\n"
        "| nationality = British\n"
        "}}"
    )
    pipeline = _StubPipeline({"a1": wikitext, "a2": "Unrelated."})
    n_before = len(pipeline.chunk_ids)
    n_added = rp._augment_pipeline_with_infobox(pipeline, top_n_boost=3)
    assert n_added >= 1, "No infobox chunks injected from a valid template"
    assert len(pipeline.chunk_ids) == n_before + n_added
    assert pipeline.chunk_vecs.shape[0] == len(pipeline.chunk_ids)
    # Every new chunk_id starts with the infobox: prefix.
    new_ids = pipeline.chunk_ids[n_before:]
    assert all(cid.startswith("infobox:") for cid in new_ids)
    # And every new chunk_id is registered in chunks_by_id with text.
    for cid in new_ids:
        assert cid in pipeline.chunks_by_id
        assert pipeline.chunks_by_id[cid]["text"]


def test_augment_pipeline_no_triples_is_noop() -> None:
    """Corpus with no infobox / fact patterns -> 0 added chunks, no crash."""
    import route_prospective as rp
    pipeline = _StubPipeline({"x": "Just plain prose without any structure."})
    n_before = len(pipeline.chunk_ids)
    added = rp._augment_pipeline_with_infobox(pipeline)
    assert added == 0
    assert len(pipeline.chunk_ids) == n_before


def test_augment_pipeline_handles_empty_corpus_gracefully() -> None:
    import route_prospective as rp
    pipeline = _StubPipeline({})
    assert rp._augment_pipeline_with_infobox(pipeline) == 0
