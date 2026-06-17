# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""WikipediaDomainPlugin — adapts MothRAG to a Wikipedia corpus
(HotpotQA / 2WikiMultiHopQA / MuSiQue).

Pipeline (executed once per corpus, results cached on disk):
  1. For each document: chunk into paragraphs (or sentences)
  2. For each chunk: run OpenIE (NER + triples) via an LLM
  3. Build entity registry from union of all NER entities (deduped + canonical-cased)
  4. Build edge list from triples (subject -- predicate --> object)
  5. Build chunks list (one per chunk text, linked to entities mentioned)

Hierarchical anchor schema (5 layers):
  L1 = topic cluster (k-means on document embedding centroids)
  L2 = single document (a Wikipedia page or HotpotQA paragraph collection)
  L3 = paragraph cluster within document (sub-topical grouping)
  L4 = NER entity ego-graph (1-hop subgraph from this entity via triples)
  L5 = triple-specific anchors (specific (s, p, o) atomic facts)

Layers are populated only when meaningful (skipped if too few members).
"""

import re
from collections import defaultdict

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

from mothrag.core.anchor import Anchor


_QUERY_RELATIONAL_PATTERNS = [
    # Multi-hop indicators (HotpotQA/2Wiki/MuSiQue style questions)
    r"\bwhich\b.*\b(also|both|same)\b",
    r"\bwho\b.*\bof\b.*\bwhich\b",
    r"\b(who|what)\b.*\bafter\b",
    r"\bbetween\b.*\band\b",
    r"\bcommon\b",
    r"\bsame\b.*\bas\b",
    r"\bborn in\b",
]

_GENERIC_RELATIONAL_KEYWORDS = [
    "located in", "born in", "founded by", "directed by", "written by",
    "produced by", "starring", "married to", "father of", "child of",
    "company of", "employer of", "nationality of", "capital of", "part of",
]


class WikipediaDomainPlugin:
    """Plugin for Wikipedia-style multi-hop QA (HotpotQA / 2Wiki / MuSiQue)."""

    name = "wikipedia"

    def __init__(self):
        # Layer specificity: L4 (entity ego-graph) is the most specific for Wikipedia QA
        # L5 triple anchors are extremely specific but small; L3 paragraph clusters
        # are useful for broad context, L2 documents for single-doc lookups.
        self._layer_specificity = {
            1: 0.05,  # broad topic cluster
            2: 0.10,  # single document
            3: 0.15,  # paragraph cluster
            4: 0.30,  # NER entity ego-graph (highest)
            5: 0.25,  # triple-specific
        }

    # ---- Anchor building ----

    def build_anchors(self, entities, edges, chunk_vecs, chunks_by_id,
                      chunk_ids, embedder) -> list[Anchor]:
        """Build hierarchical Wikipedia anchors.

        entities: list of {id, name, type='entity'|'document', mentions_chunk_ids}
        edges:    list of {src, dst, type='triple_predicate', predicate=str}
        chunk_vecs: (N, D) embeddings
        chunks_by_id: dict[chunk_id, {text, entity_id (=document_id), title, idx}]
        chunk_ids: list[str] aligned with chunk_vecs
        embedder: callable(text) -> np.ndarray (used as fallback for empty anchors)
        """
        entities_by_id = {e["id"]: e for e in entities}

        chunks_per_entity: dict[str, list[int]] = defaultdict(list)
        for i, cid in enumerate(chunk_ids):
            chunk = chunks_by_id[cid]
            if "entity_id" in chunk:
                chunks_per_entity[chunk["entity_id"]].append(i)
            for ent_id in chunk.get("mentions", []):
                chunks_per_entity[ent_id].append(i)

        G_ner = nx.Graph()
        for e in entities:
            if e.get("type") == "entity":
                G_ner.add_node(e["id"])
        for ed in edges:
            if ed["src"] in G_ner.nodes and ed["dst"] in G_ner.nodes:
                G_ner.add_edge(ed["src"], ed["dst"], predicate=ed.get("predicate", ""))

        anchors: list[Anchor] = []

        # ---- L2: single documents ----
        doc_entities = [e for e in entities if e.get("type") == "document"]
        for doc in doc_entities:
            members = {doc["id"]}
            doc_chunk_ids = [cid for cid in chunk_ids
                             if chunks_by_id[cid].get("entity_id") == doc["id"]]
            for cid in doc_chunk_ids:
                for ent_id in chunks_by_id[cid].get("mentions", []):
                    members.add(ent_id)
            if len(members) < 2:
                continue
            anchors.append(Anchor(
                anchor_id=f"L2_doc_{doc['id']}",
                layer=2,
                members=members,
                scope_text=f"Document {doc.get('name', doc['id'])}: {doc.get('summary', '')[:200]}",
            ))

        # ---- L1: topic clusters (k-means on doc embeddings) ----
        if doc_entities:
            doc_vecs = []
            doc_ids = []
            for doc in doc_entities:
                idxs = chunks_per_entity.get(doc["id"], [])
                if idxs:
                    v = chunk_vecs[idxs].mean(axis=0)
                    n = np.linalg.norm(v)
                    if n > 0:
                        v = v / n
                    doc_vecs.append(v)
                    doc_ids.append(doc["id"])
            if doc_vecs:
                doc_vecs_arr = np.stack(doc_vecs)
                k = max(2, min(20, int(np.sqrt(len(doc_vecs)))))
                if k >= 2 and len(doc_vecs) > k:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = km.fit_predict(doc_vecs_arr)
                    cluster_to_docs: dict[int, list[str]] = defaultdict(list)
                    for did, lbl in zip(doc_ids, labels):
                        cluster_to_docs[lbl].append(did)
                    for cid, doc_list in cluster_to_docs.items():
                        members = set(doc_list)
                        for did in doc_list:
                            for ed in edges:
                                if ed["src"] == did or ed["dst"] == did:
                                    members.add(ed["src"])
                                    members.add(ed["dst"])
                        if len(members) < 5:
                            continue
                        anchors.append(Anchor(
                            anchor_id=f"L1_topic_{cid}",
                            layer=1,
                            members=members,
                            scope_text=f"Topic cluster #{cid}: {len(doc_list)} documents covering related subjects.",
                        ))

        # ---- L4: NER entity ego-graphs ----
        ner_entities = [e for e in entities if e.get("type") == "entity"]
        ner_degree = [(e["id"], G_ner.degree(e["id"]) if e["id"] in G_ner else 0)
                      for e in ner_entities]
        ner_degree.sort(key=lambda x: -x[1])
        top_ner_count = min(200, len(ner_entities))
        for ent_id, _ in ner_degree[:top_ner_count]:
            if ent_id not in G_ner:
                continue
            members = {ent_id}
            members |= set(G_ner.neighbors(ent_id))
            if len(members) < 2:
                continue
            ent = entities_by_id[ent_id]
            anchors.append(Anchor(
                anchor_id=f"L4_ent_{ent_id}",
                layer=4,
                members=members,
                scope_text=f"Entity {ent.get('name', ent_id)} and its 1-hop relations.",
            ))

        # ---- L3: paragraph clusters within document ----
        for doc in doc_entities:
            doc_chunk_idxs = chunks_per_entity.get(doc["id"], [])
            if len(doc_chunk_idxs) < 4:
                continue
            sub_vecs = chunk_vecs[doc_chunk_idxs]
            sub_k = min(3, max(2, len(doc_chunk_idxs) // 3))
            if sub_k < 2:
                continue
            km = KMeans(n_clusters=sub_k, random_state=42, n_init=5)
            sub_labels = km.fit_predict(sub_vecs)
            for sub_cid in range(sub_k):
                chunk_idxs_in_cluster = [doc_chunk_idxs[i] for i, l in enumerate(sub_labels) if l == sub_cid]
                members = {doc["id"]}
                for ci in chunk_idxs_in_cluster:
                    cid_str = chunk_ids[ci]
                    for ent_id in chunks_by_id[cid_str].get("mentions", []):
                        members.add(ent_id)
                if len(members) < 3:
                    continue
                anchors.append(Anchor(
                    anchor_id=f"L3_doc_{doc['id']}_para_{sub_cid}",
                    layer=3,
                    members=members,
                    scope_text=f"Paragraph cluster #{sub_cid} of {doc.get('name', doc['id'])}.",
                    parent=f"L2_doc_{doc['id']}",
                ))

        # ---- Compute scope_vec for each anchor ----
        for anc in anchors:
            member_chunk_idxs: list[int] = []
            for eid in anc.members:
                member_chunk_idxs.extend(chunks_per_entity.get(eid, []))
            member_chunk_idxs = list(set(member_chunk_idxs))
            if member_chunk_idxs:
                centroid = chunk_vecs[member_chunk_idxs].mean(axis=0)
                n = np.linalg.norm(centroid)
                if n > 0:
                    centroid = centroid / n
                anc.scope_vec = centroid.astype(np.float32)
            else:
                anc.scope_vec = embedder(anc.scope_text)

        return anchors

    # ---- Query-side ----

    def link_query_entities(self, q_text: str, entities_by_id: dict) -> list[str]:
        """Match query mentions to graph entities by exact-name (case-insensitive).

        For Wikipedia: a proper NER on the query helps, but exact-match works as a
        baseline (HotpotQA queries usually mention entity names verbatim). For
        higher-recall linking, use :mod:`mothrag.retrieval.ner` to build an
        LLM-NER cache and route through ``link_query_entities_with_cache``.
        """
        ql = q_text.lower()
        seeds = []
        sorted_ents = sorted(
            [(eid, ent) for eid, ent in entities_by_id.items()
             if ent.get("type") == "entity" and ent.get("name")],
            key=lambda x: -len(x[1]["name"]),
        )
        for eid, ent in sorted_ents:
            name = ent["name"].lower()
            if len(name) < 3:
                continue
            if re.search(r"\b" + re.escape(name) + r"\b", ql):
                seeds.append(eid)
                if len(seeds) >= 8:
                    break
        return seeds

    def detect_relational_intent(self, q_text: str) -> tuple[bool, str | None]:
        ql = q_text.lower()
        needs_hop = (
            any(re.search(p, ql) for p in _QUERY_RELATIONAL_PATTERNS)
            or any(k in ql for k in _GENERIC_RELATIONAL_KEYWORDS)
        )
        # For Wikipedia we rarely have "wanted edge type" — predicates are too varied.
        # Pass needs_hop=True so EXPAND_HOP fires, with no edge filter (None = generic boost).
        return needs_hop, None

    def layer_specificity(self) -> dict[int, float]:
        return self._layer_specificity
