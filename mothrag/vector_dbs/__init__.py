# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Vector store adapters for MothRAG.

The default in-memory store ships inside :mod:`mothrag.core.api`
(``_MemoryVectorStore``) — zero deps, good for corpora up to ~100k
chunks. For larger corpora, plug in one of the optional adapters below.

All adapters subclass :class:`VectorDBAdapter` (Phase 2 ABC) and
satisfy the legacy :class:`mothrag.core.api.VectorStore` Protocol via
the inherited :meth:`retrieve` / :meth:`__len__` shims.

Available adapters:

- :class:`InMemoryVectorDB` — numpy in-memory (Phase 2 default, zero deps)
- :class:`PineconeVectorDB` — Pinecone serverless / dedicated
- :class:`QdrantVectorDB` — Qdrant local / cloud
- :class:`ChromaVectorDB` — Chroma ephemeral / persistent / HTTP
- :class:`FaissVectorStore` — FAISS CPU/GPU (planned, deferred to v0.5.1)

Lazy imports: provider SDKs loaded only when an adapter is instantiated.
"""

from __future__ import annotations

from mothrag.vector_dbs.base import SearchHit, VectorDBAdapter
from mothrag.vector_dbs.in_memory import InMemoryVectorDB


def _lazy_import(name: str):
    if name == "PineconeVectorDB":
        from mothrag.vector_dbs.pinecone import PineconeVectorDB
        return PineconeVectorDB
    if name == "QdrantVectorDB":
        from mothrag.vector_dbs.qdrant import QdrantVectorDB
        return QdrantVectorDB
    if name == "ChromaVectorDB":
        from mothrag.vector_dbs.chroma import ChromaVectorDB
        return ChromaVectorDB
    if name == "FaissVectorStore":
        # Scaffolding — lazy until faiss_adapter.py lands
        from mothrag.vector_dbs.faiss_adapter import FaissVectorStore
        return FaissVectorStore
    raise AttributeError(f"module 'mothrag.vector_dbs' has no attribute {name!r}")


def __getattr__(name: str):
    return _lazy_import(name)


__all__ = [
    "VectorDBAdapter",
    "SearchHit",
    "InMemoryVectorDB",
    "PineconeVectorDB",
    "QdrantVectorDB",
    "ChromaVectorDB",
    "FaissVectorStore",
]
