# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Layer-2 wiring — pipelined hand-off debt repair (NOT flag-gated).

FIX #1: bridge → ChainFilter. The bridge rank rides forward as a multiplicative
        prior (``bridge_score``) on the chain-density. Absent/1.0 → legacy.
FIX #2: graph-aware-iter → reformulation. Cross-iter accumulated entities are
        woven into the γ-loop reformulation text. Empty → legacy reformulation.

Both are UNCONDITIONAL baseline fixes (no A/B flag) but SIGNAL-gated: they only
change behaviour when the upstream signal exists (a producer-attached
bridge_score / a non-empty accumulated_entities), so graph-aware-OFF and
non-bridge paths stay byte-identical.
"""
from __future__ import annotations

import importlib.util as _u
import pathlib
import sys
from types import SimpleNamespace

import pytest

import mothrag.eval.iterative_pipeline as ip
import mothrag.retrieval.chain_filter.chain_filter as cfmod


# --------------------------------------------------------------------------- #
# FIX #1 — ChainFilter consumes the bridge_score prior
# --------------------------------------------------------------------------- #
@pytest.fixture
def _deterministic_chain(monkeypatch):
    # every kept fact supports every candidate, γ=1.0 → one HIGH-band unit each,
    # so the legacy chain-density is a clean constant we can scale-check.
    monkeypatch.setattr(cfmod, "_candidate_supports", lambda fact, text: True)
    f = cfmod.ChainFilter()
    f.gamma_scorer = lambda q, fact, text: 1.0
    return f


def test_chain_filter_uses_bridge_score_prior(_deterministic_chain):
    f = _deterministic_chain
    kept = [["a", "rel", "b"]]
    legacy = f._chain_score("q", SimpleNamespace(text="x"), kept)
    boosted = f._chain_score("q", SimpleNamespace(text="x", bridge_score=1.5), kept)
    assert legacy > 0.0
    assert boosted == pytest.approx(legacy * 1.5)          # multiplicative prior
    assert f.counters["bridge_prior_applied"] == 1         # counted (prior > 1.0)


def test_chain_filter_legacy_no_bridge_score(_deterministic_chain):
    f = _deterministic_chain
    kept = [["a", "rel", "b"]]
    legacy = f._chain_score("q", SimpleNamespace(text="x"), kept)
    # None (attribute absent) and an explicit neutral 1.0 are both byte-identical.
    none_prior = f._chain_score("q", SimpleNamespace(text="x"), kept)
    neutral = f._chain_score("q", SimpleNamespace(text="x", bridge_score=1.0), kept)
    assert none_prior == legacy
    assert neutral == legacy
    assert f.counters["bridge_prior_applied"] == 0         # never counted when ≤ 1.0


def test_chain_filter_bridge_prior_non_numeric_degrades_to_legacy(_deterministic_chain):
    # defensive: a junk prior must degrade to legacy, never raise.
    f = _deterministic_chain
    kept = [["a", "rel", "b"]]
    legacy = f._chain_score("q", SimpleNamespace(text="x"), kept)
    junk = f._chain_score("q", SimpleNamespace(text="x", bridge_score="oops"), kept)
    assert junk == legacy


# --------------------------------------------------------------------------- #
# FIX #2 — accumulated entities woven into the reformulation
# --------------------------------------------------------------------------- #
def test_refuse_loop_uses_accumulated_entities():
    q = "Who founded the maker of the Model S?"
    ents = ["Tesla", "Elon Musk", "SpaceX"]
    # partial/invalid path (a cue is present) → focus on the ungrounded step
    cued = ip._accumulated_entity_next_query(q, ents, cue="founder of Tesla")
    assert "entities seen: Tesla, Elon Musk, SpaceX" in cued
    assert "focus: founder of Tesla" in cued
    # refuse path (no cue) → pure broadening, still carrying the entity context
    refused = ip._accumulated_entity_next_query(q, ents)
    assert "entities seen: Tesla, Elon Musk, SpaceX" in refused
    assert "alternative phrasings" in refused
    # cap caps the prompt budget
    capped = ip._accumulated_entity_next_query(q, [f"E{i}" for i in range(20)])
    assert "E0" in capped and "E7" in capped and "E8" not in capped


def test_refuse_loop_legacy_when_no_entities():
    # Backwards-compat: both injection sites GUARD on `if accumulated_entities`
    # and keep their legacy reformulation f-strings in the else-branch, so a
    # graph-aware-OFF run (accumulated_entities == []) is byte-identical.
    src = pathlib.Path(ip.__file__).read_text(encoding="utf-8")
    assert src.count("if accumulated_entities:") >= 2          # both sites guarded
    assert "(alternative phrasings, related entities, " in src  # legacy refuse string
    assert "Find passages that " in src                         # legacy partial string
    # the helper is only ever reached under the guard (never on the empty path).
    start = src.index("def _accumulated_entity_next_query")
    assert "accumulated_entities[:cap]" in src[start:start + 800]


# --------------------------------------------------------------------------- #
# FIX #1 producer wiring — proves the hand-off is REAL (not a silent no-op)
# --------------------------------------------------------------------------- #
def _load_rp():
    path = pathlib.Path(ip.__file__).resolve().parents[2] / "scripts" / "route_prospective.py"
    spec = _u.spec_from_file_location("rp_layer2", path)
    mod = _u.module_from_spec(spec)
    saved = sys.argv
    sys.argv = ["route_prospective"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod, path.read_text(encoding="utf-8")


def test_bridge_producer_attaches_bridge_score():
    rp, src = _load_rp()
    # the gain constant exists and is a gentle (<1.0) boost
    assert 0.0 < rp._BRIDGE_PRIOR_GAIN <= 1.0
    # _apply_chain_filter attaches the normalized boost-only prior to candidates
    start = src.index("def _apply_chain_filter")
    end = src.index("def _gate_allows", start)
    body = src[start:end]
    assert "bridge_score=(1.0 + _BRIDGE_PRIOR_GAIN" in body
    assert "ann_score=float(n - r)" in body                    # late channel preserved too
