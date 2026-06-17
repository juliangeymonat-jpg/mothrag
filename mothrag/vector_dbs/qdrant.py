# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Qdrant vector DB adapter."""

from __future__ import annotations

import uuid

import numpy as np

from mothrag.vector_dbs.base import SearchHit, VectorDBAdapter


class QdrantVectorDB(VectorDBAdapter):
    """Qdrant collection (local or cloud)."""

    name = "qdrant"

    def __init__(
        self,
        collection: str,
        *,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        dim: int | None = None,
        distance: str = "Cosine",
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as e:
            raise ImportError(
                "QdrantVectorDB requires `qdrant-client`. "
                "Install via `pip install mothrag[qdrant]`."
            ) from e
        self.collection = collection
        self._client = QdrantClient(url=url, api_key=api_key)
        # Lazy collection creation if dim provided
        if dim is not None:
            existing = {c.name for c in self._client.get_collections().collections}
            if collection not in existing:
                self._client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=dim,
                        distance=getattr(Distance, distance.upper()),
                    ),
                )
        self._models_cache = None

    def _models(self):
        if self._models_cache is None:
            from qdrant_client import models
            self._models_cache = models
        return self._models_cache

    def add(self, vectors, metadata, ids=None) -> list[str]:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        n = vectors.shape[0]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]
        models = self._models()
        points = [
            models.PointStruct(
                id=ids[i],
                vector=vectors[i].tolist(),
                payload=metadata[i],
            )
            for i in range(n)
        ]
        self._client.upsert(collection_name=self.collection, points=points)
        return list(ids)

    def search(self, query_vector, top_k=10, filter=None) -> list[SearchHit]:
        query_vector = np.asarray(query_vector, dtype=np.float32)
        models = self._models()
        q_filter = None
        if filter:
            q_filter = models.Filter(
                must=[
                    models.FieldCondition(key=k, match=models.MatchValue(value=v))
                    for k, v in filter.items()
                ]
            )
        results = self._client.search(
            collection_name=self.collection,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=q_filter,
        )
        return [
            SearchHit(id=str(p.id), score=float(p.score), metadata=dict(p.payload or {}))
            for p in results
        ]

    def delete(self, ids) -> int:
        models = self._models()
        self._client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(points=list(ids)),
        )
        return len(ids)

    @property
    def n_vectors(self) -> int:
        info = self._client.get_collection(collection_name=self.collection)
        return int(info.points_count or 0)


__all__ = ["QdrantVectorDB"]
