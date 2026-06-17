# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Sentence-Transformers embedder adapter — formalizes existing st-mini.

Wraps :class:`mothrag.retrieval.embeddings.SentenceTransformerEmbedder` (or
falls back to direct sentence-transformers SDK) under the EmbedderAdapter ABC.
"""

from __future__ import annotations

import numpy as np

from mothrag.embedders.base import EmbedderAdapter

_DIMS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-base-en-v1.5":                  768,
    "BAAI/bge-large-en-v1.5":                 1024,
    "BAAI/bge-m3":                            1024,
}


class SentenceTransformersEmbedder(EmbedderAdapter):
    """Local sentence-transformers embedder (CPU/GPU)."""

    name = "sentence-transformers"

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SentenceTransformersEmbedder requires `sentence-transformers`. "
                "Install via `pip install mothrag[st]`."
            ) from e
        self.model = model
        self._st_model = SentenceTransformer(model, device=device) if device \
            else SentenceTransformer(model)

    @property
    def dim(self) -> int:
        if self.model in _DIMS:
            return _DIMS[self.model]
        return int(self._st_model.get_sentence_embedding_dimension())

    def embed(self, texts, *, batch_size: int = 32):
        embs = self._st_model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embs, dtype=np.float32)


__all__ = ["SentenceTransformersEmbedder"]
