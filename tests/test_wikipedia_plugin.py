# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for the Wikipedia domain plugin (no network, no LLM)."""

import numpy as np

from mothrag.plugins.wikipedia import WikipediaDomainPlugin


def test_relational_intent_detection():
    p = WikipediaDomainPlugin()
    needs, _ = p.detect_relational_intent("What does the city located in this region offer?")
    assert needs is True  # contains "located in"
    needs2, _ = p.detect_relational_intent("Where was Karen Joy Fowler born in?")
    assert needs2 is True  # "born in" matches the relational keyword list
    needs3, _ = p.detect_relational_intent("Hello world")
    assert needs3 is False


def test_layer_specificity_keys():
    p = WikipediaDomainPlugin()
    spec = p.layer_specificity()
    assert set(spec.keys()) == {1, 2, 3, 4, 5}
    # L4 (entity ego-graph) is the most specific layer for Wikipedia
    assert spec[4] == max(spec.values())


def test_link_query_entities_by_name():
    p = WikipediaDomainPlugin()
    entities_by_id = {
        "ent_paris": {"id": "ent_paris", "name": "Paris", "type": "entity"},
        "ent_rome": {"id": "ent_rome", "name": "Rome", "type": "entity"},
        "doc_x": {"id": "doc_x", "name": "France", "type": "document"},
    }
    seeds = p.link_query_entities("Where is Paris located?", entities_by_id)
    assert "ent_paris" in seeds
    assert "ent_rome" not in seeds


def test_build_anchors_minimal():
    p = WikipediaDomainPlugin()
    entities = [
        {"id": "doc_a", "name": "Alpha", "type": "document", "summary": "About Alpha"},
        {"id": "doc_b", "name": "Beta", "type": "document", "summary": "About Beta"},
        {"id": "ent_x", "name": "X", "type": "entity"},
        {"id": "ent_y", "name": "Y", "type": "entity"},
    ]
    edges = [{"src": "ent_x", "dst": "ent_y", "type": "triple", "predicate": "rel"}]
    chunks = [
        {"id": "c0", "text": "Alpha mentions X",
         "entity_id": "doc_a", "mentions": ["ent_x"]},
        {"id": "c1", "text": "Beta mentions Y",
         "entity_id": "doc_b", "mentions": ["ent_y"]},
    ]
    chunk_ids = ["c0", "c1"]
    chunks_by_id = {c["id"]: c for c in chunks}
    rng = np.random.default_rng(0)
    chunk_vecs = rng.normal(size=(2, 16)).astype(np.float32)
    chunk_vecs /= np.linalg.norm(chunk_vecs, axis=1, keepdims=True)

    def embedder(text):
        return rng.normal(size=16).astype(np.float32)

    anchors = p.build_anchors(entities, edges, chunk_vecs, chunks_by_id, chunk_ids, embedder)
    assert any(a.layer == 2 for a in anchors)  # L2 documents always present in this corpus
    for a in anchors:
        assert a.scope_vec is not None
