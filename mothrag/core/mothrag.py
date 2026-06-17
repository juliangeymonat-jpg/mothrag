# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""MothRAG core — domain-agnostic.

Domain-specific behavior is delegated to a DomainPlugin
(see :mod:`mothrag.core.domain_plugin`). This file contains:
  - EntryPointClassifier (cross-domain, uses plugin for entity-linking)
  - HotPathCache (cross-domain)
  - ContextGraphBuilder (cross-domain)
  - NavigationPolicyHeuristic (cross-domain, uses plugin for relational-intent detection)

The Anchor data class lives in :mod:`mothrag.core.anchor`.
The Wikipedia plugin (HotpotQA/2Wiki/MuSiQue benchmarks) lives in
:mod:`mothrag.plugins.wikipedia`.
"""

import hashlib
import re

import numpy as np

from mothrag.core.anchor import Anchor
from mothrag.core.domain_plugin import DomainPlugin


# ---- AnchorRegistry build (delegated to plugin) ----

def build_anchor_registry(entities, edges, chunk_vecs, chunks_by_id, chunk_ids,
                          embedder, plugin: DomainPlugin):
    """Build anchors via the supplied :class:`DomainPlugin`.

    ``embedder`` is a callable ``text -> np.ndarray`` used as fallback when an
    anchor has no member chunks (so we embed the ``scope_text`` instead).
    """
    return plugin.build_anchors(entities, edges, chunk_vecs, chunks_by_id, chunk_ids, embedder)


# ---- EntryPointClassifier (cross-domain) ----

class EntryPointClassifier:
    """Hybrid: cosine similarity (vs anchor scope_vec) + entity-linking boost.

    Entity-linking is delegated to the :class:`DomainPlugin`. A production
    deployment can substitute a sentence-transformer fine-tuned on
    ``(query, anchor)`` pairs; the reference implementation uses cosine
    similarity plus a plugin-provided entity linker.
    """

    def __init__(self, anchors: list[Anchor], plugin: DomainPlugin,
                 entities_by_id: dict | None = None):
        self.anchors = anchors
        self.plugin = plugin
        self.entities_by_id = entities_by_id or {}
        self.scope_matrix = np.stack([a.scope_vec for a in anchors])
        _norms = np.linalg.norm(self.scope_matrix, axis=1, keepdims=True)
        _norms[_norms == 0.0] = 1.0
        self.scope_matrix = (self.scope_matrix / _norms).astype(np.float32)
        self.anchors_by_member: dict[str, list[int]] = {}
        for i, a in enumerate(anchors):
            for eid in a.members:
                self.anchors_by_member.setdefault(eid, []).append(i)
        self._layer_specificity = plugin.layer_specificity()

    def classify(self, qv: np.ndarray, q_text: str | None = None,
                 top_k: int = 3) -> list[tuple[Anchor, float]]:
        scores = self.scope_matrix @ qv

        if q_text is not None:
            seeds = self.plugin.link_query_entities(q_text, self.entities_by_id)
            if seeds:
                boost = np.zeros_like(scores)
                for seed in seeds:
                    for ai in self.anchors_by_member.get(seed, []):
                        anc = self.anchors[ai]
                        layer_boost = self._layer_specificity.get(anc.layer, 0.1)
                        boost[ai] += layer_boost
                scores = scores + boost
                # Broadcast small boost to L1 if seed not in any anchor (rare)
                for seed in seeds:
                    if not self.anchors_by_member.get(seed):
                        for i, a in enumerate(self.anchors):
                            if a.layer == 1:
                                scores[i] += 0.05

        top_idx = np.argsort(-scores)[:top_k]
        return [(self.anchors[i], float(scores[i])) for i in top_idx]


# ---- HotPathCache ----

class HotPathCache:
    def __init__(self):
        self.cache: dict[str, list[tuple[Anchor, float]]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def normalize(text: str) -> str:
        STOPS = {"the", "a", "an", "is", "are", "of", "in", "and", "or", "to", "for", "with", "what", "who", "which"}
        toks = [t for t in re.findall(r"[A-Za-z0-9-]+", text.lower()) if t not in STOPS]
        return " ".join(sorted(toks))

    def key(self, text: str, tenant_id: str = "default") -> str:
        norm = self.normalize(text)
        return hashlib.sha1(f"{tenant_id}::{norm}".encode()).hexdigest()

    def get(self, text: str, tenant_id: str = "default"):
        k = self.key(text, tenant_id)
        if k in self.cache:
            self.hits += 1
            return self.cache[k]
        self.misses += 1
        return None

    def put(self, text: str, value, tenant_id: str = "default"):
        k = self.key(text, tenant_id)
        self.cache[k] = value


# ---- ContextGraphBuilder ----

class ContextGraphBuilder:
    def __init__(self, anchors_by_id: dict[str, Anchor],
                 chunks_per_entity: dict[str, list[int]],
                 chunk_vecs: np.ndarray):
        self.anchors_by_id = anchors_by_id
        self.chunks_per_entity = chunks_per_entity
        self.chunk_vecs = chunk_vecs

    def build(self, anchor: Anchor) -> list[int]:
        out: list[int] = []
        for eid in anchor.members:
            out.extend(self.chunks_per_entity.get(eid, []))
        return out


# ---- Navigation Policy (heuristic) ----

class NavigationPolicyHeuristic:
    """Decides ANSWER / EXPAND_HOP / ESCALATE actions.

    Heuristic policy:
      conf >= 0.7: enter top-1, no escalation
      0.4 <= conf < 0.7: bring top-1, top-2, top-3 (warm-up)
      conf < 0.4: fallback to L1 broadest anchor

    EXPAND_HOP and edge-type detection are delegated to the
    :class:`DomainPlugin`.
    """

    def __init__(self, plugin: DomainPlugin):
        self.plugin = plugin

    def needs_hop_expansion(self, q_text: str) -> bool:
        needs, _ = self.plugin.detect_relational_intent(q_text)
        return needs

    def wanted_edge_type(self, q_text: str) -> str | None:
        _, wanted = self.plugin.detect_relational_intent(q_text)
        return wanted

    def decide(self, top3: list[tuple[Anchor, float]],
               anchors_by_id: dict[str, Anchor],
               builder: ContextGraphBuilder) -> tuple[list[Anchor], str]:
        top1, conf1 = top3[0]
        if conf1 >= 0.7:
            anchor_chunks = builder.build(top1)
            if len(anchor_chunks) < 50 and top1.parent and top1.parent in anchors_by_id:
                parent = anchors_by_id[top1.parent]
                return [parent, top1], "escalate-parent"
            return [top1], "high-conf"
        elif conf1 >= 0.4:
            return [top3[0][0], top3[1][0], top3[2][0]], "warm-up"
        else:
            l1 = [a for a in anchors_by_id.values() if a.layer == 1]
            if not l1:
                return [top1], "low-conf-no-l1"
            cur = top1
            while cur.parent and cur.parent in anchors_by_id:
                cur = anchors_by_id[cur.parent]
                if cur.layer == 1:
                    break
            if cur.layer == 1:
                return [cur, top1], "fallback-l1"
            return [l1[0], top1], "fallback-l1"
