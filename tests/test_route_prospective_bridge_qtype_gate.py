# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Qtype-gated bridge substrate.

Cross-DS evidence: the bridge substrate HELPS ``semantic_rich`` (+2..+4pp) and
HURTS ``bridge_entity`` (-3.7..-14.8pp). The gate confines the bridge to the
cohort it helps. Input-feature gating only (``classify_query_v2`` on the
question text — anti-leak safe, NEVER a DS signal). Default ``none`` is
byte-identical to the ungated substrate.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "route_prospective",
    Path(__file__).resolve().parents[1] / "scripts" / "route_prospective.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def _sub(qtype_gate="none", scope="primary"):
    pipe = mod._StubDensePipeline()
    sub = mod._BridgeSubstrate(pipe, judge_model="claude-haiku-4-5",
                               max_cost_usd=10.0, require_backend=False,
                               scope=scope, qtype_gate=qtype_gate)
    return pipe, sub


def _force_qtype(monkeypatch, qtype):
    monkeypatch.setattr(mod, "classify_query_v2", lambda q: qtype)


# ---- gate='none' is the ungated default ------------------------------------

def test_gate_none_bridges_everything(monkeypatch):
    _force_qtype(monkeypatch, "bridge_entity")   # would be skipped if gated
    pipe, sub = _sub(qtype_gate="none")
    prep = sub.prepare("anything")
    assert prep["fired"] is True
    assert prep["bridge_qtype_skipped"] is False
    _i, route, _ = pipe.retrieve("anything")
    assert route == "bridge_substrate"


# ---- gate='semantic_rich_only' --------------------------------------------

def test_semantic_rich_only_skips_non_semantic(monkeypatch):
    _force_qtype(monkeypatch, "bridge_entity")
    pipe, sub = _sub(qtype_gate="semantic_rich_only")
    prep = sub.prepare("a bridge question")
    assert prep["fired"] is False
    assert prep["bridge_qtype_skipped"] is True
    assert prep["reason"] == "qtype_gated:bridge_entity"
    assert sub._gate_skip is True
    _i, route, _ = pipe.retrieve("a bridge question")
    assert route == "dense"
    # no bridge run was spent on a gated-out question.
    assert sub.n_bridge_runs == 0


def test_semantic_rich_only_allows_semantic(monkeypatch):
    _force_qtype(monkeypatch, "semantic_rich")
    pipe, sub = _sub(qtype_gate="semantic_rich_only")
    prep = sub.prepare("a semantic question")
    assert prep["fired"] is True
    assert prep["bridge_qtype_skipped"] is False
    assert prep["bridge_qtype"] == "semantic_rich"
    _i, route, _ = pipe.retrieve("a semantic question")
    assert route == "bridge_substrate"


def test_chain_deep_also_skipped_under_semantic_rich_only(monkeypatch):
    _force_qtype(monkeypatch, "chain_deep")
    pipe, sub = _sub(qtype_gate="semantic_rich_only")
    prep = sub.prepare("a chain question")
    assert prep["bridge_qtype_skipped"] is True


# ---- gate='exclude_bridge_entity' -----------------------------------------

def test_exclude_bridge_entity_skips_only_bridge_entity(monkeypatch):
    _force_qtype(monkeypatch, "bridge_entity")
    _pipe, sub = _sub(qtype_gate="exclude_bridge_entity")
    assert sub.prepare("q")["bridge_qtype_skipped"] is True

    _force_qtype(monkeypatch, "chain_deep")
    _pipe2, sub2 = _sub(qtype_gate="exclude_bridge_entity")
    assert sub2.prepare("q")["bridge_qtype_skipped"] is False  # chain_deep allowed


# ---- scope='all' + gate: a gated question bridges NOTHING (incl sub-Q) ------

def test_gate_suppresses_all_scope_subqueries(monkeypatch):
    _force_qtype(monkeypatch, "bridge_entity")
    pipe, sub = _sub(qtype_gate="semantic_rich_only", scope="all")
    sub.prepare("gated top-level")
    # even under scope='all', a gated-out question bridges no retrieval.
    _i, prim_route, _ = pipe.retrieve("gated top-level")
    _j, subq_route, _ = pipe.retrieve("some sub question")
    assert prim_route == "dense"
    assert subq_route == "dense"


# ---- fail-open on classifier error ----------------------------------------

def test_classifier_failure_fails_open(monkeypatch):
    def _boom(_q):
        raise RuntimeError("classifier down")
    monkeypatch.setattr(mod, "classify_query_v2", _boom)
    pipe, sub = _sub(qtype_gate="semantic_rich_only")
    prep = sub.prepare("q")
    assert prep["bridge_qtype_skipped"] is False   # bridge proceeds (fail open)
    _i, route, _ = pipe.retrieve("q")
    assert route == "bridge_substrate"


# ---- validation + telemetry ------------------------------------------------

def test_invalid_gate_raises():
    pipe = mod._StubDensePipeline()
    with pytest.raises(ValueError):
        mod._BridgeSubstrate(pipe, judge_model="claude-haiku-4-5",
                             max_cost_usd=10.0, require_backend=False,
                             qtype_gate="bogus")


def test_gate_in_stats():
    _pipe, sub = _sub(qtype_gate="semantic_rich_only")
    assert sub.stats()["qtype_gate"] == "semantic_rich_only"


# ---- CLI flag default + choices -------------------------------------------

def test_cli_qtype_gate_flag():
    orig = argparse.ArgumentParser.parse_args
    box: dict = {}

    class _Stop(Exception):
        pass

    def _cap(self, *a, **k):
        box["p"] = self
        raise _Stop()

    argparse.ArgumentParser.parse_args = _cap
    try:
        mod.main()
    except _Stop:
        pass
    finally:
        argparse.ArgumentParser.parse_args = orig
    p = box["p"]
    assert orig(p, []).bridge_substrate_qtype_gate == "none"
    assert orig(p, ["--bridge-substrate-qtype-gate", "semantic_rich_only"]
                ).bridge_substrate_qtype_gate == "semantic_rich_only"
    with pytest.raises(SystemExit):
        orig(p, ["--bridge-substrate-qtype-gate", "nope"])
