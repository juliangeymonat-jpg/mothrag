# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""gemini-embedding-2 corpus-embedder routing fix.

The PROD corpus embedder is ``gemini-embedding-2``
with NO SentenceTransformer workaround. ``mothrag/eval/pipeline.py``
previously routed only the bare string ``"gemini"`` to the Gemini corpus
path; passing the canonical model id ``"gemini-embedding-2"`` fell
through to ``SentenceTransformerEmbedder("gemini-embedding-2")`` (wrong
model / crash).

These tests assert all three aliases (``gemini`` / ``gemini-2`` /
``gemini-embedding-2``) route to the Gemini corpus branch, proven by the
branch's distinctive ``FileNotFoundError`` on the missing
``chunk_vecs_gemini_doc.npy`` doc-vector cache (raised before any network
call, so the test is offline + fast).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mothrag.eval.pipeline import MothRAGPipeline, PipelineConfig


def _make_corpus(tmp: Path) -> Path:
    (tmp / "chunks.jsonl").write_text(
        json.dumps({"id": "c1", "text": "Paris is the capital of France."}) + "\n",
        encoding="utf-8",
    )
    (tmp / "entities.json").write_text("{}", encoding="utf-8")
    (tmp / "edges.json").write_text("[]", encoding="utf-8")
    return tmp


@pytest.mark.parametrize("alias", ["gemini", "gemini-2", "gemini-embedding-2"])
def test_gemini_aliases_route_to_corpus_branch(tmp_path, monkeypatch, alias):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-test-key")
    corpus = _make_corpus(tmp_path)
    with pytest.raises(FileNotFoundError) as exc:
        MothRAGPipeline.from_corpus(
            corpus, embedding=alias, config=PipelineConfig(embedding=alias),
        )
    # The gemini corpus branch is the ONLY place that looks up this cache.
    assert "chunk_vecs_gemini_doc.npy" in str(exc.value)


def test_non_gemini_takes_sentence_transformer_branch(tmp_path, monkeypatch):
    """A non-gemini embedding must take the SentenceTransformer branch.

    Stub ``SentenceTransformerEmbedder`` (imported at function scope inside
    ``from_corpus``) with a sentinel so the test stays offline + fast and
    proves the else-branch is taken (NOT the gemini-cache path).
    """
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-test-key")
    corpus = _make_corpus(tmp_path)

    import mothrag.retrieval.embeddings as emb_mod

    class _Sentinel(Exception):
        pass

    def _boom(*_a, **_k):
        raise _Sentinel("ST-branch-taken")

    monkeypatch.setattr(emb_mod, "SentenceTransformerEmbedder", _boom)

    with pytest.raises(_Sentinel):
        MothRAGPipeline.from_corpus(
            corpus, embedding="bge-base",
            config=PipelineConfig(embedding="bge-base"),
        )
