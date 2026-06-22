"""Incremental update / delete.

A fact can change or be retracted without rebuilding the index. Covers the
low-level _MemoryVectorStore mutation API and the high-level
MothRAG.update()/delete() convenience, plus the honest capability guards
(append-only store, non-dense retrieval). Runs fully offline via the hash
embedder and echo reader fallbacks.
"""

from __future__ import annotations

import sys
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _block_sentence_transformers(monkeypatch) -> Iterator[None]:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    yield


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch) -> Iterator[None]:
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY",
              "GROQ_API_KEY", "TOGETHER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    yield


def _doc(source, text):
    from mothrag.core.api import Document
    return Document(text=text, metadata={"source": source})


def _chunk(cid, doc_id, text="x"):
    from mothrag.core.api import Chunk
    return Chunk(text=text, doc_id=doc_id, chunk_id=cid, embedding=[1.0, 0.0])


def _rag():
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader
    return MothRAG(embedder=_HashEmbedder(), reader=_EchoReader())


# ------------------------------------------------------------
# Low-level store mutation
# ------------------------------------------------------------

def test_memory_store_delete_by_id():
    from mothrag.core.api import _MemoryVectorStore
    store = _MemoryVectorStore()
    store.add([_chunk("d1#chunk0", "d1"), _chunk("d2#chunk0", "d2")])
    assert len(store) == 2
    removed = store.delete(["d1#chunk0"])
    assert removed == 1
    assert len(store) == 1
    assert [c.chunk_id for c in store._chunks] == ["d2#chunk0"]
    # embeddings stay aligned with chunks
    assert len(store._embeddings) == len(store._chunks)


def test_memory_store_delete_by_doc():
    from mothrag.core.api import _MemoryVectorStore
    store = _MemoryVectorStore()
    store.add([_chunk("d1#chunk0", "d1"), _chunk("d1#chunk1", "d1"),
               _chunk("d2#chunk0", "d2")])
    removed = store.delete_by_doc("d1")
    assert removed == 2
    assert {c.doc_id for c in store._chunks} == {"d2"}


def test_memory_store_upsert_replaces():
    from mothrag.core.api import _MemoryVectorStore
    store = _MemoryVectorStore()
    store.add([_chunk("d1#chunk0", "d1", text="old")])
    store.upsert([_chunk("d1#chunk0", "d1", text="new")])
    assert len(store) == 1
    assert store._chunks[0].text == "new"
    assert len(store._embeddings) == 1


def test_memory_store_delete_unknown_is_noop():
    from mothrag.core.api import _MemoryVectorStore
    store = _MemoryVectorStore()
    store.add([_chunk("d1#chunk0", "d1")])
    assert store.delete(["nope"]) == 0
    assert len(store) == 1


def test_memory_store_is_mutable_protocol():
    from mothrag.core.api import _MemoryVectorStore, MutableVectorStore, VectorStore
    store = _MemoryVectorStore()
    assert isinstance(store, VectorStore)
    assert isinstance(store, MutableVectorStore)


# ------------------------------------------------------------
# High-level MothRAG.delete / update
# ------------------------------------------------------------

def test_mothrag_delete_removes_doc():
    rag = _rag()
    rag.ingest([_doc("a", "Alpha is the first letter."),
                _doc("b", "Beta is the second letter.")])
    n0 = len(rag.vector_db)
    removed = rag.delete("a")
    assert removed >= 1
    assert len(rag.vector_db) == n0 - removed
    assert all(c.doc_id != "a" for c in rag.vector_db._chunks)


def test_mothrag_update_replaces_content():
    rag = _rag()
    rag.ingest([_doc("price", "The price is 10 dollars.")])
    rag.update("price", "The price is 20 dollars.")
    texts = " ".join(c.text for c in rag.vector_db._chunks)
    assert "20 dollars" in texts
    assert "10 dollars" not in texts
    # still exactly one doc under that id
    assert {c.doc_id for c in rag.vector_db._chunks} == {"price"}


def test_mothrag_update_on_missing_doc_acts_as_insert():
    rag = _rag()
    removed = rag.update("new", "Fresh content.")
    assert removed == 0
    assert any(c.doc_id == "new" for c in rag.vector_db._chunks)


def test_mothrag_query_follows_update():
    rag = _rag()
    rag.ingest([_doc("cap", "The capital of Atlantis is Marisburg.")])
    rag.update("cap", "The capital of Atlantis is Coralton.")
    qr = rag.query("What is the capital of Atlantis?")
    # retrieved evidence reflects the update, never the stale fact
    joined = " ".join(c.text for c in qr.retrieved_chunks)
    assert "Marisburg" not in joined
    assert "Coralton" in joined


# ------------------------------------------------------------
# Honest capability guards
# ------------------------------------------------------------

def test_delete_raises_on_append_only_store():
    class AppendOnly:
        def __init__(self):
            self._c = []
        def add(self, chunks):
            self._c.extend(chunks)
        def retrieve(self, q, top_k=10):
            return self._c[:top_k]
        def __len__(self):
            return len(self._c)

    rag = _rag()
    rag.vector_db = AppendOnly()  # store that does not support mutation
    with pytest.raises(NotImplementedError, match="append-only"):
        rag.delete("a")


def test_update_raises_on_non_dense_retrieval():
    rag = _rag()
    rag.retrieval = "dense_plus_infobox"  # exercise the guard without full setup
    with pytest.raises(NotImplementedError, match="retrieval='dense'"):
        rag.update("a", "x")
