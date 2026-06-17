# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Retriever Protocol -- the contract every concrete retrieval backend
implements for the MothRag query pipeline."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

# Avoid the import cycle: type-only reference to Chunk via TYPE_CHECKING.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mothrag.core.api import Chunk  # pragma: no cover


@runtime_checkable
class Retriever(Protocol):
    """Question -> top-K passages.

    Retriever generalises the legacy dense-only
    :class:`mothrag.core.api.VectorStore` Protocol by taking the raw
    question text (a graph retriever needs entity extraction over the
    text; a hybrid retriever needs both the embedding and the text;
    a BM25 retriever needs only the text). The
    :class:`DenseRetriever` adapter wraps an existing
    ``(Embedder, VectorStore)`` pair to satisfy this Protocol with zero
    behavioural change.
    """

    def index(self, chunks: Sequence["Chunk"]) -> None:
        """Add chunks to the underlying index. Idempotent on chunk_id."""
        ...

    def retrieve(self, question: str, *, top_k: int = 10) -> list["Chunk"]:
        """Return the top-K chunks relevant to the question, ranked."""
        ...

    def __len__(self) -> int:
        """Number of chunks currently indexed."""
        ...


__all__ = ["Retriever"]
