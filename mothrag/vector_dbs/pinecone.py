# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Pinecone vector DB adapter."""

from __future__ import annotations

import uuid

import numpy as np

from mothrag.vector_dbs.base import SearchHit, VectorDBAdapter, _resolve_api_key


class PineconeVectorDB(VectorDBAdapter):
    """Pinecone serverless / dedicated index."""

    name = "pinecone"

    def __init__(
        self,
        index_name: str,
        *,
        api_key: str | None = None,
        namespace: str = "default",
    ) -> None:
        try:
            from pinecone import Pinecone
        except ImportError as e:
            raise ImportError(
                "PineconeVectorDB requires `pinecone`. "
                "Install via `pip install mothrag[pinecone]`."
            ) from e
        self.index_name = index_name
        self.namespace = namespace
        client = Pinecone(api_key=api_key or _resolve_api_key(("PINECONE_API_KEY",)))
        self._index = client.Index(index_name)

    def add(self, vectors, metadata, ids=None) -> list[str]:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        n = vectors.shape[0]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]
        items = [
            {"id": ids[i], "values": vectors[i].tolist(), "metadata": metadata[i]}
            for i in range(n)
        ]
        self._index.upsert(vectors=items, namespace=self.namespace)
        return list(ids)

    def search(self, query_vector, top_k=10, filter=None) -> list[SearchHit]:
        query_vector = np.asarray(query_vector, dtype=np.float32)
        resp = self._index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter,
            namespace=self.namespace,
        )
        return [
            SearchHit(id=m.id, score=float(m.score), metadata=dict(m.metadata or {}))
            for m in resp.matches
        ]

    def delete(self, ids) -> int:
        self._index.delete(ids=list(ids), namespace=self.namespace)
        return len(ids)  # Pinecone delete returns no count; assume success

    @property
    def n_vectors(self) -> int:
        stats = self._index.describe_index_stats()
        ns = stats.namespaces.get(self.namespace) if hasattr(stats, "namespaces") else None
        return int(ns.vector_count) if ns else 0


__all__ = ["PineconeVectorDB"]
