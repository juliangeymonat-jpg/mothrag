# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""MothRAG embedder adapter package.

All embedders subclass :class:`EmbedderAdapter` and implement :meth:`embed`.
Lazy imports: provider SDKs loaded only on instantiation.

Available adapters:

- :class:`SentenceTransformersEmbedder` — local ST models (default st-mini)
- :class:`OpenAIEmbedder` — text-embedding-3-small / -large
- :class:`CohereEmbedder` — embed-english-v3
- :class:`GeminiEmbedder` — gemini-embedding-2 (MOTHRAG production default, 3072-d)
- :class:`VertexEmbedder` — Vertex AI text-embedding-005 (768-d, GDPR-region
  GCP enterprise backend). Note: gemini-embedding-2 is not on Vertex AI yet —
  use Studio (GeminiEmbedder) for the -2 model.
"""

from __future__ import annotations

from mothrag.embedders.base import EmbedderAdapter


def _lazy_import(name: str):
    if name == "OpenAIEmbedder":
        from mothrag.embedders.openai import OpenAIEmbedder
        return OpenAIEmbedder
    if name == "CohereEmbedder":
        from mothrag.embedders.cohere import CohereEmbedder
        return CohereEmbedder
    if name == "SentenceTransformersEmbedder":
        from mothrag.embedders.sentence_transformers import SentenceTransformersEmbedder
        return SentenceTransformersEmbedder
    if name == "GeminiEmbedder":
        from mothrag.embedders.gemini import GeminiEmbedder
        return GeminiEmbedder
    if name == "VertexEmbedder":
        from mothrag.embedders.vertex import VertexEmbedder
        return VertexEmbedder
    raise AttributeError(f"module 'mothrag.embedders' has no attribute {name!r}")


def __getattr__(name: str):
    return _lazy_import(name)


__all__ = [
    "EmbedderAdapter",
    "OpenAIEmbedder",
    "CohereEmbedder",
    "SentenceTransformersEmbedder",
    "GeminiEmbedder",
    "VertexEmbedder",
]
