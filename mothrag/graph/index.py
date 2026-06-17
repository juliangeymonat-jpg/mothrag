# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""GraphIndex -- entity-keyed triple store + anchor-driven traversal.

Storage model::

    _table[normalized_entity] -> list[GraphEdge]

where each edge carries ``(subject_surface, predicate, object_surface,
source_chunk_id, confidence)`` and is bidirectional (both subject and
object enter the adjacency list keyed under the OTHER endpoint, so a
forward traversal from any node touches all incident edges).

Anchor-driven traversal:

    traverse_from_anchor(anchor, *, depth, top_k)
      -> list[Path]

walks the graph in breadth-first order starting from ``anchor`` (a
surface-form entity string, normalised internally), expanding up to
``depth`` hops. Returns at most ``top_k`` paths sorted by total path
confidence (product of per-edge confidences), descending.

L4b stability hash:

    traversal_hash(paths)  -> str

emits a deterministic SHA1 over the sorted set of canonical edge IDs
in ``paths``. Used by :class:`mothrag.arms.MothGraphArm` to detect
across-iteration path-set convergence cheaply (O(n log n) vs. O(n^2)
full set comparison).

Per :data:`feedback_no_dataset_specific_training_general_purpose_only_2026_05_20`:
the index + traversal are GENERIC graph primitives. No corpus-
specific tuning; pure structural reasoning over the triple store.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from mothrag.graph.openie import (
    RawTriple,
    extract_triples,
    normalize_entity,
)

logger = logging.getLogger(__name__)


# ---- Edge / path data model ------------------------------------------------

@dataclass(frozen=True)
class GraphEdge:
    """One ``(subject, predicate, object)`` edge in :class:`GraphIndex`."""

    subject: str
    predicate: str
    object: str
    source_chunk_id: str = ""
    confidence: float = 0.7

    @property
    def edge_id(self) -> str:
        """Canonical, normalised ID -- used for L4b stability hashing.

        Order-insensitive on (subject, object) so reversing the
        traversal direction of an undirected edge does NOT change the
        hash. This is the right semantics for the MothGraphArm: we
        care about which facts surfaced, not which direction the walk
        encountered them in.
        """
        a = normalize_entity(self.subject)
        b = normalize_entity(self.object)
        # Order-insensitive endpoint pair, predicate kept.
        lo, hi = sorted((a, b))
        return f"{lo}|{self.predicate}|{hi}"


@dataclass(frozen=True)
class Path:
    """Anchored walk of one or more :class:`GraphEdge` objects.

    Sequence preserved as discovered (BFS); :attr:`anchor` is the
    surface-form node the traversal started from.
    """

    anchor: str
    edges: tuple[GraphEdge, ...] = field(default_factory=tuple)

    @property
    def confidence(self) -> float:
        """Path confidence = product of per-edge confidences."""
        c = 1.0
        for e in self.edges:
            c *= max(0.0, min(1.0, e.confidence))
        return c

    @property
    def endpoints(self) -> tuple[str, ...]:
        """Distinct entity surface forms touched by the path."""
        ordered: list[str] = [self.anchor]
        seen_norm: set[str] = {normalize_entity(self.anchor)}
        for e in self.edges:
            for ep in (e.subject, e.object):
                nk = normalize_entity(ep)
                if nk and nk not in seen_norm:
                    ordered.append(ep)
                    seen_norm.add(nk)
        return tuple(ordered)


# ---- The index -------------------------------------------------------------

class GraphIndex:
    """Entity-keyed undirected multigraph of ``(subj, pred, obj)`` triples.

    Build-once / read-many. Insertion order preserved for deterministic
    traversal across Python versions / dict-iteration semantics.
    """

    def __init__(self) -> None:
        self._table: dict[str, list[GraphEdge]] = defaultdict(list)
        self._edges: list[GraphEdge] = []
        self._edge_keys: set[tuple[str, str, str]] = set()

    # ---- Mutation --------------------------------------------------------

    def add_edge(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        source_chunk_id: str = "",
        confidence: float = 0.7,
    ) -> None:
        """Insert one edge. Idempotent on (norm_subject, predicate, norm_object)."""
        s_norm = normalize_entity(subject)
        o_norm = normalize_entity(obj)
        if not s_norm or not predicate or not o_norm:
            return
        # Order-insensitive dedup key (mirrors GraphEdge.edge_id semantics).
        lo, hi = sorted((s_norm, o_norm))
        key = (lo, predicate, hi)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        edge = GraphEdge(
            subject=subject,
            predicate=predicate,
            object=obj,
            source_chunk_id=source_chunk_id,
            confidence=confidence,
        )
        self._edges.append(edge)
        self._table[s_norm].append(edge)
        if o_norm != s_norm:
            self._table[o_norm].append(edge)

    def add_raw_triples(self, triples: Iterable[RawTriple]) -> None:
        for t in triples:
            self.add_edge(
                t.subject, t.predicate, t.object,
                source_chunk_id=t.source_chunk_id,
                confidence=t.confidence,
            )

    # ---- Read ------------------------------------------------------------

    def neighbours(self, entity: str) -> list[GraphEdge]:
        """Return all edges incident to ``entity`` (subject OR object).

        Surface form is normalised internally; callers pass the raw
        question-side surface and get back canonical edges.
        """
        return list(self._table.get(normalize_entity(entity), ()))

    def __contains__(self, entity: str) -> bool:
        return normalize_entity(entity) in self._table

    def __len__(self) -> int:
        return len(self._edges)

    @property
    def n_entities(self) -> int:
        return len(self._table)

    # ---- Traversal -------------------------------------------------------

    def traverse_from_anchor(
        self,
        anchor: str,
        *,
        depth: int = 2,
        top_k: int = 8,
    ) -> list[Path]:
        """BFS from ``anchor`` up to ``depth`` hops; return top-K paths.

        Paths are ranked by confidence (descending). Ties broken by
        canonical edge-id concatenation (deterministic).

        ``depth=0`` returns a single empty-edge path (the anchor as a
        zero-length path) iff the anchor exists in the index, else
        empty.
        """
        a_norm = normalize_entity(anchor)
        if not a_norm or a_norm not in self._table:
            return []
        if depth <= 0:
            return [Path(anchor=anchor, edges=())]

        # BFS frontier: (current_entity_norm, path_so_far)
        seen_paths: set[str] = set()
        out: list[Path] = []
        # We emit paths of every length >= 1 up to depth.
        frontier: deque[tuple[str, tuple[GraphEdge, ...]]] = deque()
        frontier.append((a_norm, ()))
        while frontier:
            cur, edges_so_far = frontier.popleft()
            if len(edges_so_far) >= depth:
                continue
            for edge in self._table.get(cur, ()):
                # Avoid traversing the same edge twice within one path.
                if any(e.edge_id == edge.edge_id for e in edges_so_far):
                    continue
                next_edges = edges_so_far + (edge,)
                # Identify the "other" endpoint to continue walking from.
                s_norm = normalize_entity(edge.subject)
                o_norm = normalize_entity(edge.object)
                other = o_norm if cur == s_norm else s_norm
                if not other:
                    continue
                path = Path(anchor=anchor, edges=next_edges)
                # Dedup by canonical edge-id concatenation.
                key = "->".join(e.edge_id for e in next_edges)
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                out.append(path)
                if len(next_edges) < depth:
                    frontier.append((other, next_edges))

        out.sort(
            key=lambda p: (-p.confidence,
                           "->".join(e.edge_id for e in p.edges)),
        )
        return out[:top_k]


# ---- Stability hash --------------------------------------------------------

def traversal_hash(paths: Sequence[Path]) -> str:
    """SHA1 over the sorted set of canonical edge IDs in ``paths``.

    Two consecutive iterations of the same anchor traversal that
    surface the SAME path set hash identically -- this is the L4b
    stability signal :class:`MothGraphArm` uses to break out of the
    refinement loop.

    Order-insensitive over paths AND over edges within a path: the
    hash answers "which facts were touched?" not "in what order".
    """
    if not paths:
        return hashlib.sha1(b"").hexdigest()
    edge_ids: list[str] = []
    for p in paths:
        for e in p.edges:
            edge_ids.append(e.edge_id)
    edge_ids.sort()
    h = hashlib.sha1()
    for eid in edge_ids:
        h.update(eid.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# ---- Corpus-level builder --------------------------------------------------

def build_graph_index_from_chunks(
    chunks: Sequence,
    *,
    chunk_id_attr: str = "chunk_id",
    chunk_text_attr: str = "text",
    max_triples_per_chunk: int = 64,
    use_spacy: bool = False,
    extra_triples: Iterable[RawTriple] | None = None,
) -> GraphIndex:
    """Build a :class:`GraphIndex` from a sequence of chunk-like objects.

    Convenience wrapper: runs
    :func:`mothrag.graph.openie.extract_triples` on ``chunks``, then
    materialises a :class:`GraphIndex` over the union with
    ``extra_triples`` (allows seeding from an external KG / Wikidata
    dump alongside the corpus-harvested triples).
    """
    index = GraphIndex()
    if chunks:
        raw_triples = extract_triples(
            chunks,
            chunk_id_attr=chunk_id_attr,
            chunk_text_attr=chunk_text_attr,
            max_triples_per_chunk=max_triples_per_chunk,
            use_spacy=use_spacy,
        )
        index.add_raw_triples(raw_triples)
    if extra_triples:
        index.add_raw_triples(extra_triples)
    logger.info(
        "build_graph_index_from_chunks: %d edges over %d entities from %d chunks",
        len(index), index.n_entities, len(chunks) if chunks else 0,
    )
    return index


__all__ = [
    "GraphEdge",
    "GraphIndex",
    "Path",
    "build_graph_index_from_chunks",
    "traversal_hash",
]
