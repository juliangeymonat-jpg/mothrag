# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for the symbolic memory store."""

from mothrag.core.symbolic_memory import SymbolicMemoryStore


def _toy_edges():
    return [
        {"src": "alice", "predicate": "knows", "dst": "bob", "confidence": 1.0},
        {"src": "bob", "predicate": "knows", "dst": "carol", "confidence": 1.0},
        {"src": "carol", "predicate": "knows", "dst": "dave", "confidence": 0.5},
    ]


def test_from_edges_indexes_entities():
    store = SymbolicMemoryStore.from_edges(_toy_edges())
    assert store.n_entities == 4
    assert store.n_triples == 3
    assert store.has_entity("alice")
    assert not store.has_entity("eve")


def test_lookup_neighbors_2hop():
    store = SymbolicMemoryStore.from_edges(_toy_edges())
    results = store.lookup_neighbors("alice", max_hops=2, top_k=10)
    targets = {tgt for tgt, _, _ in results}
    assert "bob" in targets   # 1-hop
    assert "carol" in targets  # 2-hop


def test_lookup_neighbors_confidence_decays_along_path():
    store = SymbolicMemoryStore.from_edges(_toy_edges())
    results = store.lookup_neighbors("alice", max_hops=3, top_k=10)
    by_target = {tgt: conf for tgt, conf, _ in results}
    assert by_target["bob"] == 1.0
    assert by_target["carol"] == 1.0  # 1.0 * 1.0
    assert by_target["dave"] == 0.5   # 1.0 * 1.0 * 0.5
