# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""DomainPlugin protocol — adapts MothRAG core to a specific knowledge domain.

A DomainPlugin encapsulates everything domain-specific:
  - How to build hierarchical anchors from the raw entities/edges
  - How to link mentions in a query to entity_ids
  - How to detect relational intent and map to edge types
  - Layer specificity factors used by the entity-linking boost

The reference implementation ships with:
  - WikipediaDomainPlugin (in ``mothrag.plugins.wikipedia``) — used for
    HotpotQA / 2WikiMultiHopQA / MuSiQue benchmarks.

Custom domains add their own plugin and pass it to the framework. Nothing in
``mothrag.core`` should know about any specific domain.
"""

from typing import Protocol, runtime_checkable

import numpy as np

from mothrag.core.anchor import Anchor


@runtime_checkable
class DomainPlugin(Protocol):
    """Domain-specific behavior plugged into the MothRAG core."""

    name: str  # e.g. "wikipedia" | "your-domain"

    def build_anchors(
        self,
        entities: list[dict],
        edges: list[dict],
        chunk_vecs: np.ndarray,
        chunks_by_id: dict,
        chunk_ids: list[str],
        embedder,  # callable: text -> np.ndarray (used as fallback for empty-anchor scope_vec)
    ) -> list[Anchor]:
        """Build the full set of anchors for this domain (all layers)."""
        ...

    def link_query_entities(
        self,
        q_text: str,
        entities_by_id: dict,
    ) -> list[str]:
        """Return up to N entity_ids referenced in the query text.

        Used by the EntryPointClassifier to boost anchors containing those
        entities. Reference plugins use NER + name matching; alternatives
        include regex over a closed-vocabulary domain or learned linkers.
        """
        ...

    def detect_relational_intent(
        self,
        q_text: str,
    ) -> tuple[bool, str | None]:
        """Detect if a query is relational + what edge type it targets.

        Returns ``(needs_hop_expansion, wanted_edge_type or None)``.
        Used by NavigationPolicy to enable EXPAND_HOP and to score graph
        traversal hits.
        """
        ...

    def layer_specificity(self) -> dict[int, float]:
        """Boost factor per layer for the entity-linking signal in the classifier.

        Higher means more specific (= deserves bigger boost when an entity
        from the query is found inside an anchor of this layer).
        """
        ...
