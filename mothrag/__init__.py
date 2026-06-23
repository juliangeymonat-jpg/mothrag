# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""MothRAG — adaptive multi-arm subset routing for multi-hop RAG.

Quick start (v0.5.0+, high-level API):

    from mothrag import MothRAG
    rag = MothRAG.from_documents(["doc 1 text...", "doc 2 text..."])
    print(rag.query("What is mentioned?").answer)

Legacy plug-and-play interface (v0.3.x, paper reproduction):

    from mothrag import simple
    results = simple.run(
        questions=["..."],
        corpus_path="hotpotqa-mini",
        reader="llama-3.3-70b",
        config="default",
    )

For library-level access, import from submodules:
    from mothrag.core import Anchor, EntryPointClassifier
    from mothrag.plugins import WikipediaDomainPlugin
    from mothrag.eval import metrics, soft_em
"""

__version__ = "0.6.0"

from mothrag.core.api import (
    Chunk,
    Document,
    Embedder,
    MothRAG,
    QueryResult,
    Reader,
    VectorStore,
)
# v0.5.0 Phase 2 adapter packages — lazy module-level access. Importing
# ``mothrag`` does NOT pull provider SDKs (openai, anthropic, etc.); the
# proxy below resolves names like ``mothrag.OpenAIReader`` only on access.

_LAZY_MAP = {
    # Readers
    "OpenAIReader":      ("mothrag.readers.openai", "OpenAIReader"),
    "AnthropicReader":   ("mothrag.readers.anthropic", "AnthropicReader"),
    "GroqReader":        ("mothrag.readers.groq", "GroqReader"),
    "GeminiReader":      ("mothrag.readers.gemini", "GeminiReader"),
    "CohereReader":      ("mothrag.readers.cohere", "CohereReader"),
    # Embedders
    "OpenAIEmbedder":                 ("mothrag.embedders.openai", "OpenAIEmbedder"),
    "CohereEmbedder":                 ("mothrag.embedders.cohere", "CohereEmbedder"),
    "SentenceTransformersEmbedder":   ("mothrag.embedders.sentence_transformers",
                                        "SentenceTransformersEmbedder"),
    "GeminiEmbedder":                 ("mothrag.embedders.gemini", "GeminiEmbedder"),
    # Vector DBs
    "InMemoryVectorDB":  ("mothrag.vector_dbs.in_memory", "InMemoryVectorDB"),
    "PineconeVectorDB":  ("mothrag.vector_dbs.pinecone", "PineconeVectorDB"),
    "QdrantVectorDB":    ("mothrag.vector_dbs.qdrant", "QdrantVectorDB"),
    "ChromaVectorDB":    ("mothrag.vector_dbs.chroma", "ChromaVectorDB"),
    "FaissVectorStore":  ("mothrag.vector_dbs.faiss_adapter", "FaissVectorStore"),
}


def __getattr__(name):
    """PEP 562 module-level lazy attr lookup."""
    if name in _LAZY_MAP:
        mod_path, cls_name = _LAZY_MAP[name]
        import importlib
        return getattr(importlib.import_module(mod_path), cls_name)
    raise AttributeError(f"module 'mothrag' has no attribute {name!r}")
from mothrag.core.anchor import Anchor
from mothrag.core.domain_plugin import DomainPlugin
from mothrag.core.mothrag import (
    EntryPointClassifier,
    HotPathCache,
    ContextGraphBuilder,
    NavigationPolicyHeuristic,
    build_anchor_registry,
)
from mothrag.api import simple  # noqa: F401  (re-exported, legacy)

__all__ = [
    "__version__",
    # v0.5.0+ high-level API
    "MothRAG",
    "Document",
    "Chunk",
    "QueryResult",
    "Embedder",
    "Reader",
    "VectorStore",
    # v0.5.0 Phase 2 reader adapters (lazy)
    "OpenAIReader",
    "AnthropicReader",
    "GroqReader",
    "GeminiReader",
    "CohereReader",
    # v0.5.0 Phase 2 embedder adapters (lazy)
    "OpenAIEmbedder",
    "CohereEmbedder",
    "SentenceTransformersEmbedder",
    "GeminiEmbedder",
    # v0.5.0 Phase 2 vector DB adapters (lazy)
    "InMemoryVectorDB",
    "PineconeVectorDB",
    "QdrantVectorDB",
    "ChromaVectorDB",
    "FaissVectorStore",
    # legacy / library-level
    "Anchor",
    "DomainPlugin",
    "EntryPointClassifier",
    "HotPathCache",
    "ContextGraphBuilder",
    "NavigationPolicyHeuristic",
    "build_anchor_registry",
    "simple",
]
