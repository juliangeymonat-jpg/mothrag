# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""FAISS-backed vector store adapter.

Drop-in replacement for the default ``_MemoryVectorStore`` for corpora
in the 100k–10M chunk range. CPU-only by default (``faiss-cpu``).
Embeddings are L2-normalized at insertion time so ``IndexFlatIP``
provides cosine similarity ranking with no extra normalization at
query time.

Usage::

    from mothrag import MothRAG
    from mothrag.vector_dbs import FaissVectorStore

    store = FaissVectorStore(dim=384)
    rag = MothRAG(vector_db=store)
    rag.ingest(["doc 1...", "doc 2..."])
    print(rag.query("question").answer)

If ``faiss`` is not installed, the import raises ``ImportError`` with
the install hint ``pip install mothrag[faiss]``. Auto-detection: if you
construct ``FaissVectorStore()`` without ``dim``, the index is created
lazily on the first ``.add()`` call using the first embedding's
dimensionality.
"""

from __future__ import annotations

import logging
from typing import Sequence

from mothrag.core.api import Chunk

logger = logging.getLogger(__name__)


class FaissVectorStore:
    """FAISS IndexFlatIP-backed cosine-similarity store."""

    def __init__(self, dim: int | None = None) -> None:
        """Construct an empty FAISS store.

        Parameters
        ----------
        dim
            Embedding dimensionality. If ``None``, inferred from the
            first ``.add()`` call.
        """
        try:
            import faiss  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "FaissVectorStore requires `faiss-cpu` — install via "
                "`pip install mothrag[faiss]` or `pip install faiss-cpu`"
            ) from exc

        self._dim: int | None = dim
        self._index = None  # lazy-init on first add
        self._chunks: list[Chunk] = []
        if dim is not None:
            self._init_index(dim)

    def _init_index(self, dim: int) -> None:
        import faiss
        self._dim = dim
        # IndexFlatIP = inner-product (cosine when embeddings L2-normalized).
        self._index = faiss.IndexFlatIP(dim)
        logger.info("FaissVectorStore initialized with dim=%d", dim)

    def add(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        import numpy as np
        # Determine dim from first chunk if not yet set
        first_emb = chunks[0].embedding
        if first_emb is None:
            raise ValueError(f"Chunk {chunks[0].chunk_id} has no embedding")
        if self._index is None:
            self._init_index(len(first_emb))
        # Stack + L2-normalize for cosine via IP
        arr = np.asarray([c.embedding for c in chunks], dtype=np.float32)
        if arr.shape[1] != self._dim:
            raise ValueError(
                f"Chunk embedding dim {arr.shape[1]} does not match "
                f"FAISS index dim {self._dim}"
            )
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        arr = arr / norms
        self._index.add(arr)
        self._chunks.extend(chunks)

    def retrieve(self, query_embedding: list[float], top_k: int = 10) -> list[Chunk]:
        if self._index is None or len(self._chunks) == 0:
            return []
        import numpy as np
        q = np.asarray([query_embedding], dtype=np.float32)
        qn = np.linalg.norm(q, axis=1, keepdims=True)
        qn[qn == 0.0] = 1.0
        q = q / qn
        k = min(top_k, len(self._chunks))
        _scores, idx = self._index.search(q, k)
        return [self._chunks[int(i)] for i in idx[0] if int(i) >= 0]

    def __len__(self) -> int:
        return len(self._chunks)

    def __repr__(self) -> str:
        return f"FaissVectorStore(dim={self._dim}, indexed={len(self._chunks)})"


__all__ = ["FaissVectorStore"]
