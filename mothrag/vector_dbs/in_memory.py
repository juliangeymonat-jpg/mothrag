# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""In-memory numpy-backed vector DB (default for MothRAG v0.5.0).

Zero external deps beyond numpy. Suitable for ≲500K chunks. For larger
corpora use FAISS/Qdrant/Pinecone adapters.
"""

from __future__ import annotations

import uuid

import numpy as np

from mothrag.vector_dbs.base import SearchHit, VectorDBAdapter


class InMemoryVectorDB(VectorDBAdapter):
    """Numpy cosine-similarity index (vectors must be L2-normalised)."""

    name = "memory"

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vectors: list[np.ndarray] = []
        self._metadata: list[dict] = []
        self._matrix: np.ndarray | None = None  # cached stack
        self._dirty = True

    def add(self, vectors, metadata, ids=None) -> list[str]:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        n = vectors.shape[0]
        if len(metadata) != n:
            raise ValueError(f"metadata len ({len(metadata)}) != vectors n ({n})")
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]
        elif len(ids) != n:
            raise ValueError(f"ids len ({len(ids)}) != vectors n ({n})")
        for i in range(n):
            self._ids.append(ids[i])
            self._vectors.append(vectors[i])
            self._metadata.append(dict(metadata[i]))
        self._dirty = True
        return list(ids)

    def search(self, query_vector, top_k=10, filter=None) -> list[SearchHit]:
        if not self._ids:
            return []
        if self._dirty or self._matrix is None:
            self._matrix = np.stack(self._vectors, axis=0)
            self._dirty = False
        q = np.asarray(query_vector, dtype=np.float32)
        # Assume normalised: cosine = dot
        scores = self._matrix @ q
        if filter:
            mask = np.array([self._matches_filter(m, filter) for m in self._metadata])
            scores = np.where(mask, scores, -np.inf)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            SearchHit(
                id=self._ids[int(i)],
                score=float(scores[int(i)]),
                metadata=self._metadata[int(i)],
            )
            for i in top_idx if scores[int(i)] != -np.inf
        ]

    @staticmethod
    def _matches_filter(meta: dict, flt: dict) -> bool:
        for k, v in flt.items():
            if meta.get(k) != v:
                return False
        return True

    def delete(self, ids) -> int:
        ids_set = set(ids)
        removed = 0
        new_ids, new_vecs, new_meta = [], [], []
        for i, _id in enumerate(self._ids):
            if _id in ids_set:
                removed += 1
            else:
                new_ids.append(_id)
                new_vecs.append(self._vectors[i])
                new_meta.append(self._metadata[i])
        self._ids, self._vectors, self._metadata = new_ids, new_vecs, new_meta
        self._dirty = True
        return removed

    @property
    def n_vectors(self) -> int:
        return len(self._ids)


__all__ = ["InMemoryVectorDB"]
