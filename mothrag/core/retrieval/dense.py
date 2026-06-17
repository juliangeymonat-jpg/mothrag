# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""DenseRetriever -- adapter wrapping the legacy Embedder + VectorStore
pair to satisfy the :class:`Retriever` Protocol.

This is the default retrieval path; it preserves the v0.5.0 alpha
behaviour exactly. The retriever embeds the question via the configured
:class:`mothrag.core.api.Embedder` and dispatches to the configured
:class:`mothrag.core.api.VectorStore.retrieve` cosine-similarity index.
"""

from __future__ import annotations

from typing import Sequence


class DenseRetriever:
    """Adapter: Embedder + VectorStore -> Retriever.

    Parameters
    ----------
    embedder
        Anything with ``.embed_batch(list[str]) -> list[list[float]]``.
    vector_db
        Anything with ``.add(chunks)``, ``.retrieve(q_emb, top_k)`` and
        ``__len__``. The :class:`mothrag.core.api._MemoryVectorStore` is
        the default in-memory cosine index.
    """

    name = "dense"

    def __init__(self, embedder, vector_db) -> None:
        self.embedder = embedder
        self.vector_db = vector_db

    def index(self, chunks: Sequence) -> None:
        # Embeddings populated by the caller (MothRAG.ingest); pass through.
        self.vector_db.add(list(chunks))

    def retrieve(self, question: str, *, top_k: int = 10) -> list:
        q_emb = self.embedder.embed_batch([question])[0]
        return list(self.vector_db.retrieve(q_emb, top_k=top_k))

    def __len__(self) -> int:
        return len(self.vector_db)


__all__ = ["DenseRetriever"]
