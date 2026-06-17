# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Symbolic memory layer for MothRAG bottom-up retrieval.

Version 1 (current):
- Store of triples (src_entity, predicate_text, dst_entity, confidence, source_chunk_ids)
- Lookup by entity (neighbors within N-hop with aggregated confidence)
- No learned predicate-text matching: graph traversal + entity proximity
- Confidence: log-product of edge weights along the path (default edge weight 1.0)

Future:
- Predicate text matching against wh-question (LLM classifier or learned embedding)
- Per-tenant scoping
- Online updates for personalized memory

Minimal API::

    from mothrag.core.symbolic_memory import SymbolicMemoryStore
    store = SymbolicMemoryStore.from_edges(edges)
    matches = store.lookup_neighbors("ent_clement-attlee", max_hops=2, top_k=5)
    # -> [(target_entity_id, confidence, path), ...]
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Triple:
    src: str
    predicate: str
    dst: str
    confidence: float = 1.0
    source_chunks: list[str] = field(default_factory=list)


class SymbolicMemoryStore:
    """KB triple store with graph-traversal lookup.

    Triples are bidirectional for traversal (graph is undirected) but we keep
    the original src->dst direction for predicate interpretation later.
    """

    def __init__(self):
        self.triples: list[Triple] = []
        self._by_src: dict[str, list[int]] = defaultdict(list)
        self._by_dst: dict[str, list[int]] = defaultdict(list)
        self._adj: dict[str, dict[str, float]] = defaultdict(dict)
        self._entity_set: set[str] = set()

    def add_triple(self, src: str, predicate: str, dst: str,
                   confidence: float = 1.0,
                   source_chunks: list[str] | None = None) -> None:
        idx = len(self.triples)
        t = Triple(src=src, predicate=predicate, dst=dst,
                   confidence=confidence, source_chunks=source_chunks or [])
        self.triples.append(t)
        self._by_src[src].append(idx)
        self._by_dst[dst].append(idx)
        self._entity_set.add(src)
        self._entity_set.add(dst)
        prev_a = self._adj[src].get(dst, 0.0)
        prev_b = self._adj[dst].get(src, 0.0)
        self._adj[src][dst] = max(prev_a, confidence)
        self._adj[dst][src] = max(prev_b, confidence)

    def has_entity(self, eid: str) -> bool:
        return eid in self._entity_set

    @property
    def n_entities(self) -> int:
        return len(self._entity_set)

    @property
    def n_triples(self) -> int:
        return len(self.triples)

    def neighbors(self, src: str) -> dict[str, float]:
        """Direct neighbors (1-hop) with edge confidence."""
        return dict(self._adj.get(src, {}))

    def predicates_between(self, src: str, dst: str) -> list[str]:
        """Predicates of triples between src and dst (in either direction)."""
        out = []
        for idx in self._by_src.get(src, []):
            t = self.triples[idx]
            if t.dst == dst:
                out.append(t.predicate)
        for idx in self._by_dst.get(src, []):
            t = self.triples[idx]
            if t.src == dst:
                out.append(t.predicate)
        return out

    def lookup_neighbors(self, src: str, max_hops: int = 2,
                         top_k: int = 10,
                         min_confidence: float = 0.0
                         ) -> list[tuple[str, float, list[str]]]:
        """BFS bottom-up: return top-k ``(target, conf, path)`` within ``max_hops``.

        Confidence of a path = product of edge weights along the path.
        If multiple paths reach the same target, take the BEST (max conf).
        """
        if src not in self._entity_set:
            return []

        best: dict[str, tuple[float, list[str]]] = {src: (1.0, [src])}
        frontier = [src]
        for _hop in range(max_hops):
            next_frontier = []
            for node in frontier:
                node_conf, node_path = best[node]
                for nbr, edge_conf in self._adj.get(node, {}).items():
                    new_conf = node_conf * edge_conf
                    if new_conf < min_confidence:
                        continue
                    if nbr in best and best[nbr][0] >= new_conf:
                        continue
                    best[nbr] = (new_conf, node_path + [nbr])
                    next_frontier.append(nbr)
            frontier = next_frontier
            if not frontier:
                break

        results = [(eid, conf, path) for eid, (conf, path) in best.items() if eid != src]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def evidence_chunks(self, path: list[str]) -> list[str]:
        """Source chunk_ids that 'witness' the edges along a path.

        For each consecutive pair ``(a, b)`` in path, take the union of
        source_chunks across triples between a and b.
        """
        out: list[str] = []
        seen = set()
        for a, b in zip(path, path[1:]):
            for idx in self._by_src.get(a, []):
                t = self.triples[idx]
                if t.dst == b:
                    for c in t.source_chunks:
                        if c not in seen:
                            seen.add(c)
                            out.append(c)
            for idx in self._by_dst.get(a, []):
                t = self.triples[idx]
                if t.src == b:
                    for c in t.source_chunks:
                        if c not in seen:
                            seen.add(c)
                            out.append(c)
        return out

    @classmethod
    def from_edges(cls, edges: list[dict],
                   confidence_default: float = 1.0) -> "SymbolicMemoryStore":
        """Build from ``edges.json`` shape: ``[{src, dst, predicate, source_chunk?}, ...]``.

        If ``source_chunk_ids`` not present in edges, ``source_chunks=[]``.
        Confidence currently flat (1.0). Later: scale by triple frequency or LLM logprob.
        """
        store = cls()
        for ed in edges:
            store.add_triple(
                src=ed["src"],
                predicate=ed.get("predicate", ""),
                dst=ed["dst"],
                confidence=ed.get("confidence", confidence_default),
                source_chunks=ed.get("source_chunk_ids", []),
            )
        return store
