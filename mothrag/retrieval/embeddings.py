# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Embedding + cross-encoder wrappers used by the MothRAG retrieval pipeline.

The reference defaults are:
  - bi-encoder: ``sentence-transformers/all-MiniLM-L6-v2`` ("st-mini")
  - cross-encoder: ``BAAI/bge-reranker-v2-m3``

These choices match the empirical configuration documented in the paper. Pass
any sentence-transformers-compatible model name to use a different encoder
(e.g. ``BAAI/bge-base-en-v1.5`` for "st-base").
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


_BIENCODER_ALIASES = {
    "st-mini": "sentence-transformers/all-MiniLM-L6-v2",
    "st-base": "BAAI/bge-base-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
}

_CROSSENCODER_ALIASES = {
    "bge-rerank": "BAAI/bge-reranker-v2-m3",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
}


def _resolve_biencoder(name: str) -> str:
    return _BIENCODER_ALIASES.get(name, name)


def _resolve_crossencoder(name: str) -> str:
    return _CROSSENCODER_ALIASES.get(name, name)


class SentenceTransformerEmbedder:
    """Lazy wrapper around ``sentence_transformers.SentenceTransformer``.

    Offers a single ``encode(...)`` method returning L2-normalised float32
    vectors, and ``__call__`` for use as the ``embedder`` callable expected by
    :func:`mothrag.core.build_anchor_registry`.
    """

    def __init__(self, model_name: str = "st-mini", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = _resolve_biencoder(model_name)
        self.model = SentenceTransformer(self.model_name, device=device)

    def encode(self, texts: str | Iterable[str], batch_size: int = 32,
               show_progress_bar: bool = False) -> np.ndarray:
        single = isinstance(texts, str)
        arr = self.model.encode(
            [texts] if single else list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        return arr[0] if single else arr

    def __call__(self, text: str) -> np.ndarray:
        return self.encode(text)


class CrossEncoderReranker:
    """Lazy wrapper around ``sentence_transformers.CrossEncoder``."""

    def __init__(self, model_name: str = "bge-rerank", max_length: int = 512,
                 device: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model_name = _resolve_crossencoder(model_name)
        self.model = CrossEncoder(self.model_name, max_length=max_length, device=device)

    def predict(self, pairs: list[tuple[str, str]],
                show_progress_bar: bool = False) -> np.ndarray:
        scores = self.model.predict(pairs, show_progress_bar=show_progress_bar)
        return np.asarray(scores, dtype=np.float32)


def cosine_topk(query_vec: np.ndarray, chunk_vecs: np.ndarray, k: int):
    """Return ``(indices, scores)`` for top-k chunks by cosine similarity.

    Assumes ``chunk_vecs`` and ``query_vec`` are L2-normalised so that
    dot product == cosine.
    """
    scores = chunk_vecs @ query_vec
    if k >= len(scores):
        idx = np.argsort(-scores)
    else:
        idx = np.argpartition(-scores, k)[:k]
        idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]
