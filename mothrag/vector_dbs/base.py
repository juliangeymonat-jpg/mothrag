# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""VectorDBAdapter ABC + SearchHit for MothRAG v0.5.0 vector store plugins.

All concrete vector DBs subclass :class:`VectorDBAdapter` and implement
:meth:`add` / :meth:`search` / :meth:`delete`. The legacy
:class:`mothrag.core.api.VectorStore` Protocol is satisfied via the inherited
:meth:`retrieve` shim that adapts SearchHit list to Chunk list.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class SearchHit:
    """Provider-agnostic search result."""
    id: str
    score: float
    metadata: dict = field(default_factory=dict)


class VectorDBAdapter(ABC):
    """Pluggable vector database. Subclasses implement add / search / delete."""

    name: str = ""

    @abstractmethod
    def add(self, vectors, metadata: list[dict],
            ids: list[str] | None = None) -> list[str]:
        """Add vectors with metadata; returns assigned IDs."""

    @abstractmethod
    def search(self, query_vector, top_k: int = 10,
               filter: dict | None = None) -> list[SearchHit]:
        """Cosine-similarity (or provider-native) top-K search."""

    @abstractmethod
    def delete(self, ids: list[str]) -> int:
        """Delete by ID; returns count actually deleted."""

    def upsert(self, ids, vectors, metadata) -> None:
        """Idempotent upsert (delete-then-add)."""
        self.delete(ids)
        self.add(vectors, metadata, ids)

    @property
    @abstractmethod
    def n_vectors(self) -> int:
        """Current count."""

    # ------ core.api.VectorStore Protocol shim ------
    def retrieve(self, query_embedding, top_k: int = 10):
        """Translate VectorStore Protocol .retrieve(qe, k) → list[Chunk]."""
        from mothrag.core.api import Chunk
        hits = self.search(query_embedding, top_k=top_k)
        return [
            Chunk(
                text=h.metadata.get("text", ""),
                doc_id=h.metadata.get("doc_id", ""),
                chunk_id=h.id,
                metadata={k: v for k, v in h.metadata.items() if k not in ("text",)},
            )
            for h in hits
        ]

    def __len__(self) -> int:
        return self.n_vectors


def _resolve_api_key(env_keys: tuple[str, ...]) -> str:
    for k in env_keys:
        v = os.environ.get(k)
        if v:
            return v
    raise RuntimeError(f"No API key found in env. Tried: {env_keys}.")


__all__ = ["VectorDBAdapter", "SearchHit", "_resolve_api_key"]
