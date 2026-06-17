# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Custom domain plugin — minimal example.

Implement :class:`mothrag.DomainPlugin` for your knowledge graph. The plugin
defines:
  * how to build hierarchical anchors from your entities + edges,
  * how to link query mentions to entity_ids,
  * whether a query needs hop expansion (and which edge type to follow),
  * the layer-specificity factors used by the entity-linking boost.

Run with::

    python examples/02_custom_domain.py
"""

import numpy as np

from mothrag import (
    Anchor,
    EntryPointClassifier,
    NavigationPolicyHeuristic,
    ContextGraphBuilder,
    build_anchor_registry,
)


class TinyDomainPlugin:
    """Trivial example: a 2-document corpus about authors and their books."""

    name = "tiny-authors"

    def build_anchors(self, entities, edges, chunk_vecs, chunks_by_id,
                      chunk_ids, embedder):
        anchors = []
        # One anchor per document (layer 2)
        for e in entities:
            if e["type"] != "document":
                continue
            members = {e["id"]} | {edge["dst"] for edge in edges if edge["src"] == e["id"]}
            scope_text = f"Document {e['name']}: {e.get('summary', '')[:120]}"
            anchors.append(Anchor(
                anchor_id=f"L2_{e['id']}", layer=2, members=members, scope_text=scope_text,
            ))
        # One global topic anchor (layer 1)
        all_ents = {e["id"] for e in entities}
        anchors.append(Anchor(
            anchor_id="L1_root", layer=1, members=all_ents,
            scope_text="All documents in the corpus.",
        ))
        # Embed scope text
        for a in anchors:
            a.scope_vec = embedder(a.scope_text)
        return anchors

    def link_query_entities(self, q_text, entities_by_id):
        ql = q_text.lower()
        seeds = []
        for eid, e in entities_by_id.items():
            name = (e.get("name") or "").lower()
            if name and len(name) >= 3 and name in ql:
                seeds.append(eid)
        return seeds

    def detect_relational_intent(self, q_text):
        ql = q_text.lower()
        return ("written by" in ql or "author of" in ql), None

    def layer_specificity(self):
        return {1: 0.05, 2: 0.30}


def demo():
    # Toy 2-document corpus
    entities = [
        {"id": "doc_orwell", "name": "George Orwell",
         "type": "document", "summary": "British author of 1984 and Animal Farm."},
        {"id": "doc_huxley", "name": "Aldous Huxley",
         "type": "document", "summary": "British author of Brave New World."},
        {"id": "ent_1984", "name": "1984", "type": "entity"},
        {"id": "ent_brave_new_world", "name": "Brave New World", "type": "entity"},
    ]
    edges = [
        {"src": "doc_orwell", "dst": "ent_1984", "type": "triple", "predicate": "wrote"},
        {"src": "doc_huxley", "dst": "ent_brave_new_world", "type": "triple", "predicate": "wrote"},
    ]
    chunks = [
        {"id": "c0", "text": "George Orwell (1903-1950) was a British author known for 1984 and Animal Farm.",
         "entity_id": "doc_orwell", "mentions": ["ent_1984"]},
        {"id": "c1", "text": "Aldous Huxley (1894-1963) was a British author best known for Brave New World.",
         "entity_id": "doc_huxley", "mentions": ["ent_brave_new_world"]},
    ]

    # Trivial in-memory embedder for the demo (lowercased character n-gram hash)
    def embedder(text):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=128).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    chunk_vecs = np.stack([embedder(c["text"]) for c in chunks])
    chunks_by_id = {c["id"]: c for c in chunks}
    chunk_ids = [c["id"] for c in chunks]
    chunks_per_entity: dict = {}
    for i, cid in enumerate(chunk_ids):
        chunks_per_entity.setdefault(chunks[i]["entity_id"], []).append(i)
        for m in chunks[i]["mentions"]:
            chunks_per_entity.setdefault(m, []).append(i)

    plugin = TinyDomainPlugin()
    anchors = build_anchor_registry(entities, edges, chunk_vecs, chunks_by_id,
                                     chunk_ids, embedder, plugin)
    classifier = EntryPointClassifier(anchors, plugin,
                                      entities_by_id={e["id"]: e for e in entities})
    builder = ContextGraphBuilder({a.anchor_id: a for a in anchors},
                                   chunks_per_entity, chunk_vecs)
    policy = NavigationPolicyHeuristic(plugin)

    question = "Who wrote 1984?"
    qv = embedder(question)
    top3 = classifier.classify(qv, q_text=question, top_k=3)
    chosen, route = policy.decide(top3, {a.anchor_id: a for a in anchors}, builder)
    print("Top anchors:", [(a.anchor_id, score) for a, score in top3])
    print("Route:", route)
    print("Chosen anchors:", [a.anchor_id for a in chosen])


if __name__ == "__main__":
    demo()
