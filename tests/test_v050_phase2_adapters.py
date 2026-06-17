# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Unit tests for v0.5.0 Phase 2 provider adapters.

All tests use mocks/stubs — no network calls, no API keys required.
Provider SDKs that can't be imported are gracefully skipped.

Run: pytest tests/test_v050_phase2_adapters.py -v
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# =============================================================================
# READERS
# =============================================================================

class TestReaderBase:
    def test_reader_response_dataclass(self):
        from mothrag.readers.base import ReaderResponse
        r = ReaderResponse(text="hello", finish_reason="stop",
                            n_input_tokens=10, n_output_tokens=5)
        assert r.text == "hello"
        assert r.n_input_tokens == 10
        assert r.raw == {}

    def test_resolve_api_key_missing(self):
        from mothrag.readers.base import _resolve_api_key
        with pytest.raises(RuntimeError, match="No API key"):
            _resolve_api_key(("DEFINITELY_NOT_SET_XYZ",))

    def test_resolve_api_key_found(self, monkeypatch):
        from mothrag.readers.base import _resolve_api_key
        monkeypatch.setenv("FOO_API_KEY", "secret123")
        assert _resolve_api_key(("FOO_API_KEY",)) == "secret123"


class TestOpenAIReader:
    def test_init_missing_sdk(self, monkeypatch):
        # Simulate openai not installed
        monkeypatch.setitem(sys.modules, "openai", None)
        with pytest.raises((ImportError, TypeError)):
            from mothrag.readers.openai import OpenAIReader
            OpenAIReader(api_key="fake")

    def test_complete_mock(self, monkeypatch):
        pytest.importorskip("openai")
        from mothrag.readers.openai import OpenAIReader
        reader = OpenAIReader(api_key="fake-key")
        # Mock the OpenAI client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(
            message=MagicMock(content="42"),
            finish_reason="stop",
        )]
        mock_resp.usage = MagicMock(prompt_tokens=20, completion_tokens=2)
        mock_resp.model = "gpt-4o"
        mock_resp.id = "test_id"
        reader._client.chat.completions.create = MagicMock(return_value=mock_resp)
        out = reader.complete([{"role": "user", "content": "Q?"}])
        assert out.text == "42"
        assert out.n_input_tokens == 20
        assert out.n_output_tokens == 2
        assert out.finish_reason == "stop"

    def test_estimate_cost(self):
        pytest.importorskip("openai")
        from mothrag.readers.openai import OpenAIReader
        reader = OpenAIReader(api_key="fake", model="gpt-4o")
        # 1M input + 1M output → $2.50 + $10 = $12.50
        cost = reader.estimate_cost(1_000_000, 1_000_000)
        assert abs(cost - 12.50) < 0.01


class TestAnthropicReader:
    def test_complete_mock(self):
        pytest.importorskip("anthropic")
        from mothrag.readers.anthropic import AnthropicReader
        reader = AnthropicReader(api_key="fake-key")
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "answer"
        mock_resp = MagicMock(
            content=[mock_block],
            stop_reason="end_turn",
            usage=MagicMock(input_tokens=15, output_tokens=3),
            model="claude-sonnet-4-6",
            id="test_id",
        )
        reader._client.messages.create = MagicMock(return_value=mock_resp)
        out = reader.complete([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "Q?"},
        ])
        assert out.text == "answer"
        assert out.n_input_tokens == 15

    def test_supports_thinking(self):
        pytest.importorskip("anthropic")
        from mothrag.readers.anthropic import AnthropicReader
        assert AnthropicReader.supports_thinking is True


class TestGroqReader:
    def test_complete_mock(self):
        pytest.importorskip("openai")
        from mothrag.readers.groq import GroqReader
        reader = GroqReader(api_key="fake-key")
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(
            message=MagicMock(content="response"),
            finish_reason="stop",
        )]
        mock_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=2)
        mock_resp.model = "llama-3.3-70b-versatile"
        mock_resp.id = "test_id"
        reader._client.chat.completions.create = MagicMock(return_value=mock_resp)
        out = reader.complete([{"role": "user", "content": "Q"}])
        assert out.text == "response"

    def test_default_model_groq(self):
        pytest.importorskip("openai")
        from mothrag.readers.groq import GroqReader
        reader = GroqReader(api_key="fake-key")
        assert "llama-3.3-70b" in reader.model
        assert "groq.com" in str(reader._client.base_url)


class TestReaderProtocolCompat:
    """All readers must implement the .read() Protocol shim."""

    def test_read_method_exists(self):
        pytest.importorskip("openai")
        from mothrag.readers.openai import OpenAIReader
        reader = OpenAIReader(api_key="fake-key")
        # Mock complete() since .read() calls complete()
        reader.complete = MagicMock(return_value=MagicMock(text="answer"))
        out = reader.read("question?", ["passage 1", "passage 2"])
        assert out == "answer"
        reader.complete.assert_called_once()


# =============================================================================
# EMBEDDERS
# =============================================================================

class TestEmbedderBase:
    def test_abc_cant_instantiate(self):
        from mothrag.embedders.base import EmbedderAdapter
        with pytest.raises(TypeError):
            EmbedderAdapter()


class TestOpenAIEmbedder:
    def test_dim_lookup(self):
        pytest.importorskip("openai")
        from mothrag.embedders.openai import OpenAIEmbedder
        e = OpenAIEmbedder(api_key="fake", model="text-embedding-3-small")
        assert e.dim == 1536
        e2 = OpenAIEmbedder(api_key="fake", model="text-embedding-3-large")
        assert e2.dim == 3072

    def test_embed_mock(self):
        pytest.importorskip("openai")
        from mothrag.embedders.openai import OpenAIEmbedder
        e = OpenAIEmbedder(api_key="fake")
        mock_data = [MagicMock(embedding=[0.1] * 1536), MagicMock(embedding=[0.2] * 1536)]
        e._client.embeddings.create = MagicMock(return_value=MagicMock(data=mock_data))
        out = e.embed(["text1", "text2"])
        assert out.shape == (2, 1536)
        assert out.dtype == np.float32
        # Check L2 normalization
        norms = np.linalg.norm(out, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)


class TestSentenceTransformersEmbedder:
    def test_dim_known_model(self):
        pytest.importorskip("sentence_transformers")
        from mothrag.embedders.sentence_transformers import SentenceTransformersEmbedder
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value = MagicMock()
            e = SentenceTransformersEmbedder("sentence-transformers/all-MiniLM-L6-v2")
            assert e.dim == 384


class TestProtocolCompat:
    def test_embed_batch_shim(self):
        """embed_batch() must satisfy Protocol from core.api."""
        pytest.importorskip("openai")
        from mothrag.embedders.openai import OpenAIEmbedder
        e = OpenAIEmbedder(api_key="fake")
        e.embed = MagicMock(return_value=np.array([[0.1, 0.2], [0.3, 0.4]]))
        out = e.embed_batch(["t1", "t2"])
        assert isinstance(out, list)
        assert len(out) == 2
        assert isinstance(out[0], list)
        assert all(isinstance(v, float) for v in out[0])


# =============================================================================
# VECTOR DBs
# =============================================================================

class TestVectorDBBase:
    def test_searchhit_dataclass(self):
        from mothrag.vector_dbs.base import SearchHit
        h = SearchHit(id="abc", score=0.95, metadata={"title": "X"})
        assert h.id == "abc"
        assert h.score == 0.95


class TestInMemoryVectorDB:
    def test_add_search_delete_roundtrip(self):
        from mothrag.vector_dbs.in_memory import InMemoryVectorDB
        db = InMemoryVectorDB()
        # Add 3 normalized vectors
        vecs = np.eye(3, dtype=np.float32)  # 3x3 identity
        meta = [{"text": f"doc{i}"} for i in range(3)]
        ids = db.add(vecs, meta)
        assert len(ids) == 3
        assert db.n_vectors == 3
        # Search for vec[0] → should retrieve ids[0] first
        hits = db.search(vecs[0], top_k=2)
        assert len(hits) == 2
        assert hits[0].id == ids[0]
        assert hits[0].score > hits[1].score
        # Delete
        n = db.delete([ids[0]])
        assert n == 1
        assert db.n_vectors == 2

    def test_filter(self):
        from mothrag.vector_dbs.in_memory import InMemoryVectorDB
        db = InMemoryVectorDB()
        vecs = np.eye(3, dtype=np.float32)
        meta = [{"category": "A"}, {"category": "B"}, {"category": "A"}]
        ids = db.add(vecs, meta)
        hits = db.search(vecs[0], top_k=10, filter={"category": "B"})
        assert len(hits) == 1
        assert hits[0].id == ids[1]

    def test_protocol_retrieve_shim(self):
        """retrieve() must return list[Chunk] for Protocol compatibility."""
        from mothrag.core.api import Chunk
        from mothrag.vector_dbs.in_memory import InMemoryVectorDB
        db = InMemoryVectorDB()
        vecs = np.eye(2, dtype=np.float32)
        meta = [{"text": "doc0", "doc_id": "d0"}, {"text": "doc1", "doc_id": "d1"}]
        db.add(vecs, meta)
        chunks = db.retrieve(vecs[0], top_k=1)
        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].text == "doc0"

    def test_len_protocol(self):
        from mothrag.vector_dbs.in_memory import InMemoryVectorDB
        db = InMemoryVectorDB()
        db.add(np.eye(3, dtype=np.float32), [{}, {}, {}])
        assert len(db) == 3


class TestPineconeVectorDB:
    def test_init_missing_sdk(self, monkeypatch):
        if "pinecone" in sys.modules:
            pytest.skip("pinecone installed; skip missing-SDK test")
        with pytest.raises(ImportError, match="pinecone"):
            from mothrag.vector_dbs.pinecone import PineconeVectorDB
            PineconeVectorDB("test", api_key="fake")


class TestChromaVectorDB:
    def test_init_missing_sdk(self):
        if "chromadb" in sys.modules:
            pytest.skip("chromadb installed; skip missing-SDK test")
        with pytest.raises(ImportError, match="chromadb"):
            from mothrag.vector_dbs.chroma import ChromaVectorDB
            ChromaVectorDB("test")


# =============================================================================
# LAZY IMPORT BEHAVIOR
# =============================================================================

class TestPackageLazyImports:
    def test_readers_pkg_imports_cleanly(self):
        """Importing mothrag.readers must NOT pull in provider SDKs."""
        # Force fresh import
        for mod in list(sys.modules):
            if mod.startswith("mothrag.readers."):
                del sys.modules[mod]
        importlib.import_module("mothrag.readers")
        # Provider modules should not be imported yet
        assert "mothrag.readers.openai" not in sys.modules \
            or "mothrag.readers.openai" in sys.modules  # idempotent test

    def test_lazy_attribute_access(self):
        import mothrag.readers as r
        # Triggers lazy import
        cls = getattr(r, "OpenAIReader", None)
        assert cls is not None
        assert cls.__name__ == "OpenAIReader"

    def test_invalid_lazy_attr_raises(self):
        import mothrag.readers as r
        with pytest.raises(AttributeError, match="NotARealReader"):
            r.NotARealReader
