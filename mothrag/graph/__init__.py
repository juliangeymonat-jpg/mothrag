# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Deterministic graph-retrieval primitives for MothRag.

The sub-package implements :class:`MothGraphArm`-style anchor-driven
graph traversal over an :class:`InfoboxIndex`-style triple corpus.
Three layers compose:

1. :mod:`mothrag.graph.openie` -- extraction. Deterministic regex /
   dependency-parse harvest of ``(subject, predicate, object)`` triples
   from free-text chunks. Spacy is an OPTIONAL accelerator; the
   default extractor uses only stdlib ``re`` so the module is unit-
   testable in CI without external dependencies.

2. :mod:`mothrag.graph.index` -- storage + traversal. A
   :class:`GraphIndex` keyed by ``normalize_surface`` entity strings
   maps each entity to its adjacent ``(subject, predicate, object)``
   edges. Traversal is anchor-driven (start at an entity, expand
   neighbourhoods up to a configurable depth) and emits a deterministic
   L4b-style stability hash so the consumer can detect path-set
   convergence across iterations.

3. :mod:`mothrag.graph.mothgraph_arm` (lives under :mod:`mothrag.arms`
   to keep the Arm Protocol contract co-located with the rest of the
   opt-in arm pool).

Per :data:`feedback_no_dataset_specific_training_general_purpose_only_2026_05_20`:
all primitives in this package are GENERAL-PURPOSE linguistic / graph
algorithms. No per-dataset tuning, no gold-derived patterns, no
training. Wikipedia infobox markup is supported because it is a
ubiquitous public format -- not because any specific test corpus uses
it.
"""

from __future__ import annotations

from mothrag.graph.index import (
    GraphEdge,
    GraphIndex,
    build_graph_index_from_chunks,
    traversal_hash,
)
from mothrag.graph.openie import (
    extract_triples,
    extract_triples_from_text,
)

__all__ = [
    "GraphEdge",
    "GraphIndex",
    "build_graph_index_from_chunks",
    "extract_triples",
    "extract_triples_from_text",
    "traversal_hash",
]
