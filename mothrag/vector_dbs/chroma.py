# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Chroma vector DB adapter."""

from __future__ import annotations

import uuid

import numpy as np

from mothrag.vector_dbs.base import SearchHit, VectorDBAdapter


class ChromaVectorDB(VectorDBAdapter):
    """Chroma collection (in-memory by default; persistent via persist_directory)."""

    name = "chroma"

    def __init__(
        self,
        collection: str,
        *,
        persist_directory: str | None = None,
        host: str | None = None,
        port: int = 8000,
    ) -> None:
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "ChromaVectorDB requires `chromadb`. "
                "Install via `pip install mothrag[chroma]`."
            ) from e
        if host:
            self._client = chromadb.HttpClient(host=host, port=port)
        elif persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, vectors, metadata, ids=None) -> list[str]:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        n = vectors.shape[0]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]
        self._collection.add(
            ids=list(ids),
            embeddings=vectors.tolist(),
            metadatas=[dict(m) for m in metadata],
            documents=[m.get("text", "") for m in metadata],
        )
        return list(ids)

    def search(self, query_vector, top_k=10, filter=None) -> list[SearchHit]:
        query_vector = np.asarray(query_vector, dtype=np.float32)
        result = self._collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            where=filter,
        )
        hits = []
        for i, _id in enumerate(result["ids"][0]):
            # Chroma returns squared L2 by default; convert to cosine if cosine space
            dist = float(result["distances"][0][i])
            score = 1.0 - dist  # cosine_similarity = 1 - cosine_distance
            meta = dict(result["metadatas"][0][i] or {})
            hits.append(SearchHit(id=str(_id), score=score, metadata=meta))
        return hits

    def delete(self, ids) -> int:
        self._collection.delete(ids=list(ids))
        return len(ids)

    @property
    def n_vectors(self) -> int:
        return int(self._collection.count())


__all__ = ["ChromaVectorDB"]
