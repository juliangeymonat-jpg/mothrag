# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for _rerank_accum_passages passage embedding cache.

Verifies 5-10× speedup: only NEW passages embedded each iter; cached texts
skipped. Anti-leak: pure memoization, no gold info in cache key.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from mothrag.eval import iterative_pipeline as ip


@pytest.fixture(autouse=True)
def _clear_cache():
    ip._PASSAGE_EMB_CACHE.clear()
    yield
    ip._PASSAGE_EMB_CACHE.clear()


def _make_mock_embed(dim: int = 8):
    """Returns (embed_fn, call_log) where call_log records each batch's texts."""
    call_log: list[list[str]] = []

    def embed_fn(texts):
        call_log.append(list(texts))
        # Deterministic vec per text via hash → reproducible
        rng = np.random.default_rng(abs(hash("|".join(texts))) % (2**32))
        return rng.standard_normal((len(texts), dim)).astype(np.float32)
    return embed_fn, call_log


def test_first_call_embeds_query_plus_all_passages():
    embed, log = _make_mock_embed()
    ids = ["p1", "p2", "p3"]
    texts = ["alpha", "beta", "gamma"]
    ip._rerank_accum_passages(ids, texts, "what?", embed_fn=embed, top_k=3)
    assert len(log) == 1
    # First call: query + 3 NEW passages
    assert log[0] == ["what?", "alpha", "beta", "gamma"]


def test_second_call_same_texts_only_embeds_query():
    embed, log = _make_mock_embed()
    ids, texts = ["p1", "p2"], ["alpha", "beta"]
    ip._rerank_accum_passages(ids, texts, "q1", embed_fn=embed, top_k=2)
    log.clear()
    # Same texts, different query
    ip._rerank_accum_passages(ids, texts, "q2", embed_fn=embed, top_k=2)
    assert len(log) == 1
    assert log[0] == ["q2"]  # ONLY query embedded; passages from cache


def test_partial_new_text_only_embeds_new():
    embed, log = _make_mock_embed()
    # iter 1: 2 passages
    ip._rerank_accum_passages(["p1", "p2"], ["alpha", "beta"],
                              "q1", embed_fn=embed, top_k=2)
    log.clear()
    # iter 2: 3 passages, one new ("gamma")
    ip._rerank_accum_passages(["p1", "p2", "p3"], ["alpha", "beta", "gamma"],
                              "q1", embed_fn=embed, top_k=3)
    assert len(log) == 1
    assert log[0] == ["q1", "gamma"]


def test_cache_hit_count_iter_simulation():
    """Simulates 5-iter loop with growing accum — measure embed call savings."""
    embed, log = _make_mock_embed()
    texts_grown = ["a", "b", "c", "d", "e"]
    # Pre-fill: simulate 5 iterations where accum accumulates same texts
    for k in range(1, 6):
        ip._rerank_accum_passages([f"p{i}" for i in range(k)], texts_grown[:k],
                                  "q", embed_fn=embed, top_k=20)
    # Expected: 5 calls, each = [q] + new_texts where new_texts grows by 1
    # iter1: [q, a], iter2: [q, b], iter3: [q, c], iter4: [q, d], iter5: [q, e]
    total_passage_embeds = sum(
        len([t for t in call if t != "q"]) for call in log
    )
    assert total_passage_embeds == 5  # not 1+2+3+4+5 = 15
    # Without cache it would be 5+5+5+5+5 = 25 passage embeds (or 15 if accum grows)


def test_lru_eviction():
    embed, _ = _make_mock_embed()
    original = ip._PASSAGE_EMB_CACHE_MAX
    try:
        ip._PASSAGE_EMB_CACHE_MAX = 3  # type: ignore[misc]
        for i in range(5):
            ip._rerank_accum_passages([f"p{i}"], [f"text_{i}"],
                                      "q", embed_fn=embed, top_k=1)
        assert len(ip._PASSAGE_EMB_CACHE) == 3
    finally:
        ip._PASSAGE_EMB_CACHE_MAX = original  # type: ignore[misc]


def test_empty_accum_passthrough():
    embed, log = _make_mock_embed()
    ids, texts = ip._rerank_accum_passages([], [], "q",
                                            embed_fn=embed, top_k=5)
    assert ids == [] and texts == []
    assert log == []  # no embed call


def test_no_embed_fn_passthrough():
    ids, texts = ip._rerank_accum_passages(["p1", "p2"], ["a", "b"],
                                            "q", embed_fn=None, top_k=5)
    assert ids == ["p1", "p2"] and texts == ["a", "b"]


def test_embed_fn_failure_falls_through():
    def bad_embed(texts):
        raise RuntimeError("API down")
    ids, texts = ip._rerank_accum_passages(["p1", "p2"], ["a", "b"], "q",
                                            embed_fn=bad_embed, top_k=2)
    # Falls back to FIFO truncation
    assert ids == ["p1", "p2"] and texts == ["a", "b"]


def test_top_k_truncation_preserved():
    embed, _ = _make_mock_embed()
    ids = [f"p{i}" for i in range(10)]
    texts = [f"text_{i}" for i in range(10)]
    out_ids, out_texts = ip._rerank_accum_passages(ids, texts, "q",
                                                    embed_fn=embed, top_k=3)
    assert len(out_ids) == 3 and len(out_texts) == 3


# Anti-leak signature audit
_FORBIDDEN = {"gold", "f1", "em", "dataset", "ds", "label", "answer", "gold_doc_ids"}


def test_rerank_signature_clean():
    sig = inspect.signature(ip._rerank_accum_passages)
    leaked = set(sig.parameters.keys()) & _FORBIDDEN
    assert not leaked


def test_passage_cache_helpers_clean():
    for fn in (ip._passage_cache_key, ip._passage_cache_get, ip._passage_cache_put):
        sig = inspect.signature(fn)
        leaked = set(sig.parameters.keys()) & _FORBIDDEN
        assert not leaked


# ============================================================
# Model-namespaced cache key
# ============================================================

def test_cache_key_different_models_diverge():
    """Same text under different embedder models → different keys."""
    k1 = ip._passage_cache_key("alpha beta", model="gemini-embedding-001")
    k2 = ip._passage_cache_key("alpha beta", model="bge-large-en-v1.5")
    assert k1 != k2


def test_cache_key_same_model_same_key():
    """Sanity: deterministic per (model, text) pair."""
    k1 = ip._passage_cache_key("x y z", model="m1")
    k2 = ip._passage_cache_key("x y z", model="m1")
    assert k1 == k2


def test_cache_key_empty_model_backward_compat():
    """Empty model keeps legacy SHA256(text)-only behavior."""
    import hashlib
    legacy = hashlib.sha256("hello world".encode("utf-8")).hexdigest()
    assert ip._passage_cache_key("hello world", model="") == legacy
    # Default arg also empty
    assert ip._passage_cache_key("hello world") == legacy


def test_rerank_passes_embed_model_to_cache(monkeypatch):
    """When ``embed_model`` provided, cache hit requires model match."""
    embed, _ = _make_mock_embed()
    ids, texts = ["p1"], ["alpha"]
    # iter1: embed under model_a
    ip._rerank_accum_passages(ids, texts, "q",
                              embed_fn=embed, top_k=1, embed_model="model_a")
    # iter2 same text under model_b — should NOT cache-hit (different namespace)
    embed_b, log_b = _make_mock_embed()
    ip._rerank_accum_passages(ids, texts, "q",
                              embed_fn=embed_b, top_k=1, embed_model="model_b")
    # log_b shows passage WAS embedded (cache miss under new model)
    assert any("alpha" in call for call in log_b)
