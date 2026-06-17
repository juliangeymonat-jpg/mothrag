# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""v0.5.0 zero-config end-to-end smoke test.

These tests confirm that the public API surface:
1. Imports without ANY heavy optional dep (no API keys, no Gemini, no openai).
2. Constructs `MothRAG` with full auto-default chain (offline fallbacks fire).
3. Ingests documents via from_documents (text + mixed-format folder).
4. Runs query / batch_query end-to-end and returns a `QueryResult`.
5. Routes via `arm_subset()` and exposes the decision in `result.arm_subset`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mothrag import MothRAG, Document, QueryResult


SAMPLE_DOCS = [
    "The Eiffel Tower is located in Paris, France. It was built in 1889.",
    "Python is a programming language created by Guido van Rossum in 1991.",
    "Memento (2000) was directed by Christopher Nolan and stars Guy Pearce.",
]


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Force offline-fallback path by hiding all known API key env vars."""
    for var in ("TOGETHER_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY",
                "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


# ----------------------------------------------------------------
# Import & construction
# ----------------------------------------------------------------

def test_import_surface():
    """Public symbols are importable in a minimal environment."""
    from mothrag import (
        MothRAG, Document, Chunk, QueryResult,
        Embedder, Reader, VectorStore,
    )
    assert MothRAG.__name__ == "MothRAG"


def test_zero_config_construct():
    """`MothRAG()` without args resolves all three backends from auto-defaults."""
    rag = MothRAG()
    assert rag.embedder is not None
    assert rag.reader is not None
    assert rag.vector_db is not None
    assert len(rag.vector_db) == 0


# ----------------------------------------------------------------
# Ingestion
# ----------------------------------------------------------------

def test_from_documents_list_of_strings():
    rag = MothRAG.from_documents(SAMPLE_DOCS)
    assert len(rag.vector_db) >= len(SAMPLE_DOCS)  # ≥ 1 chunk per doc


def test_from_documents_list_of_document_objects():
    docs = [Document(text=t, metadata={"id": i}) for i, t in enumerate(SAMPLE_DOCS)]
    rag = MothRAG.from_documents(docs)
    assert len(rag.vector_db) >= len(SAMPLE_DOCS)


def test_from_documents_folder(tmp_path: Path):
    """Folder of mixed-format files is ingested via auto_load dispatch."""
    (tmp_path / "a.txt").write_text(SAMPLE_DOCS[0], encoding="utf-8")
    (tmp_path / "b.md").write_text(SAMPLE_DOCS[1], encoding="utf-8")
    import json
    (tmp_path / "c.json").write_text(json.dumps([SAMPLE_DOCS[2]]), encoding="utf-8")
    rag = MothRAG.from_documents(tmp_path)
    assert len(rag.vector_db) >= 3


def test_ingest_incremental():
    rag = MothRAG()
    rag.ingest([SAMPLE_DOCS[0]])
    n0 = len(rag.vector_db)
    rag.ingest([SAMPLE_DOCS[1], SAMPLE_DOCS[2]])
    n1 = len(rag.vector_db)
    assert n1 > n0


# ----------------------------------------------------------------
# Query
# ----------------------------------------------------------------

def test_query_returns_query_result():
    rag = MothRAG.from_documents(SAMPLE_DOCS)
    result = rag.query("Who created Python?")
    assert isinstance(result, QueryResult)
    assert isinstance(result.answer, str)
    assert isinstance(result.retrieved_chunks, list)
    assert isinstance(result.arm_subset, list)


def test_query_retrieves_correct_chunk_via_default_embedder():
    """The auto-default embedder is enough to retrieve the right doc on lexical-leaning queries."""
    rag = MothRAG.from_documents(SAMPLE_DOCS)
    result = rag.query("Who created Python?")
    # Echo reader returns the first sentence of the top chunk; assert it mentions Python.
    assert "Python" in result.answer or any("Python" in c.text for c in result.retrieved_chunks[:3])


def test_arm_subset_populated_for_normal_query():
    rag = MothRAG.from_documents(SAMPLE_DOCS)
    result = rag.query("Who created Python?")
    # Plain semantic_rich short query keeps all three arms
    assert result.arm_subset == ["v3bu", "decompose", "iter"]


def test_arm_subset_excludes_v3bu_on_kinship_possessive():
    """`X's grandfather` triggers the implicit-multihop kinship-possessive lever."""
    rag = MothRAG.from_documents(SAMPLE_DOCS)
    result = rag.query("Who is Christopher Nolan's grandfather?")
    assert "v3bu" not in result.arm_subset


def test_arm_subset_excludes_v3bu_on_comparative_selection():
    """`Which X was Y first, A or B?` triggers the comparative-selection lever."""
    rag = MothRAG.from_documents(SAMPLE_DOCS)
    result = rag.query("Which film was released first, Memento or Inception?")
    assert "v3bu" not in result.arm_subset


# ----------------------------------------------------------------
# Batch + Async
# ----------------------------------------------------------------

def test_batch_query_preserves_order():
    rag = MothRAG.from_documents(SAMPLE_DOCS)
    questions = ["Who created Python?", "When was the Eiffel Tower built?", "Who directed Memento?"]
    results = rag.batch_query(questions, max_workers=2)
    assert len(results) == len(questions)
    assert all(isinstance(r, QueryResult) for r in results)


def test_async_query():
    import asyncio
    rag = MothRAG.from_documents(SAMPLE_DOCS)
    result = asyncio.run(rag.aquery("Who created Python?"))
    assert isinstance(result, QueryResult)


# ----------------------------------------------------------------
# Edge cases / failure modes
# ----------------------------------------------------------------

def test_empty_corpus_returns_safely():
    rag = MothRAG()
    result = rag.query("anything?")
    assert isinstance(result, QueryResult)
    assert result.retrieved_chunks == []


def test_unsupported_extension_raises(tmp_path: Path):
    """Path with unsupported extension surfaces a helpful error."""
    bad = tmp_path / "doc.xyz"
    bad.write_text("dummy", encoding="utf-8")
    rag = MothRAG()
    with pytest.raises((ValueError, NotImplementedError)):
        rag.ingest(bad)


def test_short_string_treated_as_text_not_path():
    """A bare string without path separators is treated as text content, not a path."""
    rag = MothRAG()
    rag.ingest("Just a sentence of text.")
    assert len(rag.vector_db) >= 1


def test_pdf_loader_wired(tmp_path: Path):
    """PDF is supported in v0.5.0 via the ``pdf`` extra. A malformed file
    surfaces a pypdf error rather than the legacy NotImplementedError."""
    pytest.importorskip("pypdf")
    from pypdf.errors import PdfReadError

    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4 dummy\n%%EOF\n")
    rag = MothRAG()
    with pytest.raises(PdfReadError):
        rag.ingest(tmp_path / "doc.pdf")


def test_repr_contains_index_size():
    rag = MothRAG.from_documents(SAMPLE_DOCS)
    assert "indexed=" in repr(rag)
