# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Pluggable retrieval-layer Protocol for the MothRag query pipeline.

Generalises the legacy dense-only :class:`mothrag.core.api.VectorStore`
to accept the raw question text (not just an embedding), so non-vector
retrievers (graph-based, hybrid dense+graph, BM25 + dense fusion,
structured-infobox fusion, ...) compose alongside the in-memory cosine
index.

Public surface:

- :class:`Retriever` -- the Protocol every concrete retriever implements.
- :class:`DenseRetriever` -- adapter wrapping the existing
  :class:`mothrag.core.api.Embedder` + :class:`mothrag.core.api.VectorStore`
  pair. This is the default; it preserves v0.5.0 alpha behaviour exactly.
- :class:`HybridGraphRetriever` -- wraps the OSU-NLP-Group HippoRAG2 SDK
  (Apache 2.0) for dense + personalized-PageRank graph fusion. Optional
  dependency via ``mothrag[hybrid-graph]``.
- :class:`MultiModalRetriever` -- blends a dense retriever with a
  structured :class:`InfoboxIndex` for entity-attribute questions.
  Higher precision on questions like "when was X born?" /
  "who is the spouse of Y?" without giving up dense recall on
  open-ended Wh-questions.
- :class:`InfoboxIndex` -- subject-attribute keyed lookup over
  ``(subject, attribute, value)`` triples harvested from chunks
  (wikitext-template parsing + conservative natural-language patterns)
  or fed from external KG sources.

All retrievers expose ``index(chunks)`` and ``retrieve(question, top_k)``;
``DenseRetriever`` is also an iterable of the underlying chunks for
backward compatibility with the v0.5.0 alpha ``len(rag.vector_db)``
introspection idiom.
"""

from __future__ import annotations

from mothrag.core.retrieval.protocol import Retriever
from mothrag.core.retrieval.dense import DenseRetriever
from mothrag.core.retrieval.hybrid_graph import HybridGraphRetriever
from mothrag.core.retrieval.infobox import (
    InfoboxIndex,
    InfoboxTriple,
    build_infobox_index_from_chunks,
    extract_natural_facts,
    extract_wikitext_infobox,
)
from mothrag.core.retrieval.multimodal import (
    MultiModalRetriever,
    extract_question_hints,
)

__all__ = [
    "Retriever",
    "DenseRetriever",
    "HybridGraphRetriever",
    "MultiModalRetriever",
    "InfoboxIndex",
    "InfoboxTriple",
    "build_infobox_index_from_chunks",
    "extract_question_hints",
    "extract_natural_facts",
    "extract_wikitext_infobox",
]
