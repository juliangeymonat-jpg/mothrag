# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for GeminiEmbedder file cache + judge disk cache.

Anti-leak: no real Gemini calls — mocks injected. Anti-leak signature tests.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from mothrag.eval.faithfulness import (
    _judge_disk_cache_path,
    faithfulness_score,
)


# ============================================================
# GeminiEmbedder file cache
# ============================================================

def test_gemini_embedder_cache_dir_env_var_pickup(monkeypatch, tmp_path):
    """Env var MOTHRAG_GEMINI_CACHE_DIR is respected when cache_dir param None."""
    # GeminiEmbedder __init__ tries to import google.genai — too heavy for unit;
    # verify only the cache-dir-pickup logic via a stub.
    from mothrag.embedders import gemini as gmod

    class _StubClient:
        class _Models:
            def embed_content(self, **kw):
                class _R:
                    embeddings = [type("e", (), {"values": [1.0]*3072})()]
                return _R()
        models = _Models()

    monkeypatch.setattr(gmod, "_resolve_api_key", lambda envs: "stub")
    monkeypatch.setenv("MOTHRAG_GEMINI_CACHE_DIR", str(tmp_path / "cache"))
    # Mock google.genai
    import sys, types as _types
    fake_genai = _types.ModuleType("google.genai")
    fake_genai.Client = lambda **kw: _StubClient()  # noqa: E731
    fake_pkg = _types.ModuleType("google")
    fake_pkg.genai = fake_genai
    fake_types = _types.ModuleType("google.genai.types")
    fake_types.EmbedContentConfig = lambda **kw: None  # noqa: E731
    monkeypatch.setitem(sys.modules, "google", fake_pkg)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    emb = gmod.GeminiEmbedder()
    assert emb._cache_dir is not None
    assert emb._cache_dir == (tmp_path / "cache")


def test_gemini_embedder_no_cache_when_unset(monkeypatch):
    """No cache_dir + no env var → _cache_dir is None (legacy live-only)."""
    from mothrag.embedders import gemini as gmod

    monkeypatch.delenv("MOTHRAG_GEMINI_CACHE_DIR", raising=False)
    monkeypatch.setattr(gmod, "_resolve_api_key", lambda envs: "stub")
    import sys, types as _types
    fake_genai = _types.ModuleType("google.genai")
    fake_genai.Client = lambda **kw: object()  # noqa: E731
    fake_pkg = _types.ModuleType("google")
    fake_pkg.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_pkg)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    emb = gmod.GeminiEmbedder()
    assert emb._cache_dir is None


# ============================================================
# judge file cache key + hit/miss
# ============================================================

def test_judge_disk_cache_path_none_when_env_unset(monkeypatch):
    monkeypatch.delenv("MOTHRAG_JUDGE_CACHE_DIR", raising=False)
    assert _judge_disk_cache_path("q", "p", "m") is None


def test_judge_disk_cache_path_under_env_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MOTHRAG_JUDGE_CACHE_DIR", str(tmp_path))
    p = _judge_disk_cache_path("Who?", "Paris", "gemini-2.5-flash")
    assert p is not None
    assert str(p).startswith(str(tmp_path))
    assert p.suffix == ".json"


def test_judge_disk_cache_key_changes_with_inputs(monkeypatch, tmp_path):
    monkeypatch.setenv("MOTHRAG_JUDGE_CACHE_DIR", str(tmp_path))
    p1 = _judge_disk_cache_path("Q1", "P", "m")
    p2 = _judge_disk_cache_path("Q2", "P", "m")
    p3 = _judge_disk_cache_path("Q1", "P2", "m")
    p4 = _judge_disk_cache_path("Q1", "P", "m2")
    assert len({p1, p2, p3, p4}) == 4


def test_judge_disk_cache_normalizes_q_and_pred(monkeypatch, tmp_path):
    monkeypatch.setenv("MOTHRAG_JUDGE_CACHE_DIR", str(tmp_path))
    p1 = _judge_disk_cache_path("Who founded Microsoft?", "Bill Gates", "m")
    p2 = _judge_disk_cache_path("  WHO FOUNDED MICROSOFT?  ", "BILL GATES", "m")
    assert p1 == p2  # case + whitespace normalized


# ============================================================
# faithfulness_score uses disk cache
# ============================================================

class _MockGeminiClient:
    """Mocks genai.Client.models.generate_content"""
    class _Models:
        def __init__(self, response="yes"):
            self._r = response
        def generate_content(self, **kw):
            class _R:
                text = self._r if False else None  # noqa
            obj = _R()
            obj.text = self._r
            obj.candidates = []
            return obj
    def __init__(self, response="yes"):
        self.models = self._Models(response)


def test_faithfulness_disk_cache_hit(monkeypatch, tmp_path):
    monkeypatch.setenv("MOTHRAG_JUDGE_CACHE_DIR", str(tmp_path))
    # Pre-populate cache file
    p = _judge_disk_cache_path("Who founded?", "Bill Gates", "test-model")
    p.write_text(json.dumps({"score": 1.0, "label": "yes"}), encoding="utf-8")
    # Pass a client whose call would BLOW UP — so any cache miss fails the test
    bad_client = type("X", (), {})()
    score, label = faithfulness_score(
        bad_client, "test-model", "Who founded?", ["passage"], "Bill Gates",
        provider="gemini",
    )
    assert score == 1.0
    assert label == "yes"


def test_faithfulness_disk_cache_miss_then_writes(monkeypatch, tmp_path):
    monkeypatch.setenv("MOTHRAG_JUDGE_CACHE_DIR", str(tmp_path))
    client = _MockGeminiClient(response="yes")
    score, label = faithfulness_score(
        client, "test-model", "Who?", ["passage"], "Bill Gates",
        provider="gemini",
    )
    assert score == 1.0
    # Verify cache file was written
    p = _judge_disk_cache_path("Who?", "Bill Gates", "test-model")
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["score"] == 1.0


def test_faithfulness_abstain_bypasses_judge(monkeypatch):
    monkeypatch.setenv("MOTHRAG_JUDGE_CACHE_DIR", "/tmp/should-not-be-used")
    # is_abstain("Not in passages") → True → no judge call, no cache hit
    bad_client = type("X", (), {})()
    score, label = faithfulness_score(
        bad_client, "test-model", "Q", ["passage"], "Not in passages",
        provider="gemini",
    )
    assert score == 0.0
    assert label == "no"


def test_faithfulness_corrupt_cache_falls_through(monkeypatch, tmp_path):
    monkeypatch.setenv("MOTHRAG_JUDGE_CACHE_DIR", str(tmp_path))
    p = _judge_disk_cache_path("Q", "P", "m")
    p.write_text("not valid json", encoding="utf-8")
    client = _MockGeminiClient(response="yes")
    # Should fall through to live call, not raise
    score, _ = faithfulness_score(
        client, "m", "Q", ["passage"], "P", provider="gemini",
    )
    assert score in (0.0, 0.5, 1.0)


# ============================================================
# Anti-leak signatures
# ============================================================

_FORBIDDEN = {"gold", "f1", "em", "dataset", "ds", "ds_label",
              "corpus", "benchmark", "label", "answer_label", "gold_doc_ids"}


def test_judge_disk_cache_path_signature_clean():
    sig = inspect.signature(_judge_disk_cache_path)
    leaked = set(sig.parameters.keys()) & _FORBIDDEN
    assert not leaked


def test_faithfulness_score_signature_clean():
    sig = inspect.signature(faithfulness_score)
    leaked = set(sig.parameters.keys()) & _FORBIDDEN
    assert not leaked
