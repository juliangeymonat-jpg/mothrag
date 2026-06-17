"""Smoke + unit tests for the VertexEmbedder adapter.

Verifies the adapter's lazy-import surface and the public dispatcher /
auto-resolution wiring in ``mothrag.core.api``. The tests run without
``google-cloud-aiplatform`` installed: they assert that the right error
fires (clean ImportError, not silent fallback or AttributeError) when
the SDK is missing.

To run the *actual* Vertex AI round-trip you must install
``mothrag[vertex]`` and set ``VERTEX_AI_PROJECT`` +
``GOOGLE_APPLICATION_CREDENTIALS``; the live-call test is skipped
unconditionally here (see ``test_vertex_live_embed_smoke`` for the
manually-runnable path).
"""

from __future__ import annotations

import os
import sys
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch) -> Iterator[None]:
    """Strip GCP / Studio env vars so resolution is deterministic."""
    for k in (
        "VERTEX_AI_PROJECT",
        "GOOGLE_CLOUD_PROJECT",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    yield


@pytest.fixture
def _block_sentence_transformers(monkeypatch) -> Iterator[None]:
    """Make ``import sentence_transformers`` raise ImportError.

    Local CPython 3.13 / pyarrow combos can segfault on the
    sentence_transformers transitive import; this isolates the default
    embedder fallback chain.
    """
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    yield


def test_vertex_class_lazy_importable() -> None:
    from mothrag.embedders import VertexEmbedder
    assert VertexEmbedder.__name__ == "VertexEmbedder"


def test_vertex_init_without_sdk_raises_importerror() -> None:
    try:
        import vertexai  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("google-cloud-aiplatform installed; ImportError path not testable")
    from mothrag.embedders import VertexEmbedder
    with pytest.raises(ImportError, match="google-cloud-aiplatform"):
        VertexEmbedder()


def test_vertex_init_without_project_raises_runtimeerror() -> None:
    try:
        import vertexai  # noqa: F401
    except ImportError:
        pytest.skip("google-cloud-aiplatform not installed; RuntimeError path not reachable")
    from mothrag.embedders import VertexEmbedder
    with pytest.raises(RuntimeError, match="project"):
        VertexEmbedder()


def test_resolve_embedder_spec_hash() -> None:
    from mothrag.core.api import _HashEmbedder, _resolve_embedder_spec
    assert isinstance(_resolve_embedder_spec("hash"), _HashEmbedder)


def test_resolve_embedder_spec_unknown_raises_valueerror() -> None:
    from mothrag.core.api import _resolve_embedder_spec
    with pytest.raises(ValueError, match="Unknown embedder spec"):
        _resolve_embedder_spec("not-a-real-backend")


@pytest.mark.parametrize("spec", ["gemini-embedding-2", "gemini-2"])
def test_resolve_embedder_spec_bare_gemini_model_id(monkeypatch, spec) -> None:
    """The bare canonical PROD model id routes to Gemini.

    Before the fix, ``gemini-embedding-2`` raised "Unknown embedder spec"
    (partition on ':' made the whole string the backend). Now it routes to
    GeminiEmbedder(model="gemini-embedding-2"). We assert routing (NOT a real
    embed) by giving a dummy key and checking the constructed type/model.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-test-key")
    from mothrag.core.api import _resolve_embedder_spec
    from mothrag.embedders import GeminiEmbedder
    emb = _resolve_embedder_spec(spec)
    assert isinstance(emb, GeminiEmbedder)
    assert getattr(emb, "model", None) == "gemini-embedding-2"


def test_resolve_embedder_spec_bare_gemini_not_unknown(monkeypatch) -> None:
    """The bare model id must NOT fall through to the 'Unknown embedder spec'."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from mothrag.core.api import _resolve_embedder_spec
    # Missing key -> RuntimeError from GeminiEmbedder, NOT ValueError(Unknown).
    with pytest.raises(Exception) as exc:
        _resolve_embedder_spec("gemini-embedding-2")
    assert "Unknown embedder spec" not in str(exc.value)


def test_resolve_embedder_spec_vertex_without_sdk() -> None:
    try:
        import vertexai  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("google-cloud-aiplatform installed; ImportError path not testable")
    from mothrag.core.api import _resolve_embedder_spec
    with pytest.raises(ImportError, match="google-cloud-aiplatform"):
        _resolve_embedder_spec("vertex")


def test_mothrag_ctor_string_dispatch_hash() -> None:
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder
    rag = MothRAG(embedder="hash")
    assert isinstance(rag.embedder, _HashEmbedder)


def test_mothrag_ctor_instance_backward_compat() -> None:
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder
    rag = MothRAG(embedder=_HashEmbedder())
    assert isinstance(rag.embedder, _HashEmbedder)


def test_auto_resolve_no_env_falls_through_to_hash(_block_sentence_transformers) -> None:
    from mothrag.core.api import _HashEmbedder, _resolve_default_embedder
    assert isinstance(_resolve_default_embedder(), _HashEmbedder)


def test_auto_resolve_vertex_env_without_sdk_falls_through(
    monkeypatch, _block_sentence_transformers
) -> None:
    """When VERTEX_AI_PROJECT is set but the SDK is missing, the chain
    should degrade gracefully instead of raising."""
    try:
        import vertexai  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("google-cloud-aiplatform installed; the fallback path is not exercised here")
    monkeypatch.setenv("VERTEX_AI_PROJECT", "mock-project-xyz")
    from mothrag.core.api import _HashEmbedder, _resolve_default_embedder
    assert isinstance(_resolve_default_embedder(), _HashEmbedder)


@pytest.mark.skip(reason="Live test — requires mothrag[vertex] + GCP project + ADC.")
def test_vertex_live_embed_smoke() -> None:
    """Manual smoke: hits the real Vertex API. Skipped by default.

    To run::

        pip install mothrag[vertex]
        export VERTEX_AI_PROJECT=your-project
        export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
        pytest tests/test_vertex_adapter.py::test_vertex_live_embed_smoke -s --runxfail
    """
    from mothrag.embedders import VertexEmbedder
    emb = VertexEmbedder(model="text-embedding-005")
    out = emb.embed(["The cat sat on the mat.", "Le chat est sur le tapis."])
    assert out.shape == (2, 768)
    # L2-normalised.
    import numpy as np
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
