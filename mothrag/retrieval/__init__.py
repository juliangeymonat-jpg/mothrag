# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Retrieval primitives: embeddings, OpenIE (NER + triples), NER cache."""

from mothrag.retrieval.embeddings import (
    SentenceTransformerEmbedder,
    CrossEncoderReranker,
    cosine_topk,
)
from mothrag.retrieval.openie import OpenIEClient, OpenIEResult
from mothrag.retrieval.ner import (
    build_ner_cache,
    load_cache,
    save_cache,
    link_query_entities_with_cache,
)

__all__ = [
    "SentenceTransformerEmbedder",
    "CrossEncoderReranker",
    "cosine_topk",
    "OpenIEClient",
    "OpenIEResult",
    "build_ner_cache",
    "load_cache",
    "save_cache",
    "link_query_entities_with_cache",
]
