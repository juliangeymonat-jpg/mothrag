# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Bridge substrate scope (primary vs all).

Extends the substrate from PRIMARY-only (top-level seed-free retrieval)
to optionally ALL retrievals (primary + sub-Q + iter-refinement). Verifies the
scope rule, cross-query cache dedup (no double-spend), cost-cap degradation,
and that ``primary`` (default) reproduces the prior primary-only behaviour
exactly.
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


def _sub(scope: str):
    pipe = mod._StubDensePipeline()
    sub = mod._BridgeSubstrate(pipe, judge_model="claude-haiku-4-5",
                               max_cost_usd=10.0, require_backend=False,
                               scope=scope)
    return pipe, sub


# ---- scope='all' bridges sub-Q + seeded iter paths ------------------------

def test_scope_all_bridges_distinct_subquery():
    pipe, sub = _sub("all")
    sub.prepare("alpha beta")
    # a DISTINCT sub-question (≠ prepared primary, no seeds) is bridged too.
    _idx, route, _ = pipe.retrieve("gamma gold")
    assert route == "bridge_substrate"


def test_scope_all_bridges_seeded_iter_path():
    pipe, sub = _sub("all")
    sub.prepare("alpha beta")
    # iter-refinement passes entity_seeds; under 'all' it is still bridged.
    _idx, route, _ = pipe.retrieve("alpha beta", ["some_entity"])
    assert route == "bridge_substrate"


# ---- scope='primary' (default) reproduces primary-only behaviour ----------

def test_scope_primary_keeps_subquery_and_seeds_dense():
    pipe, sub = _sub("primary")
    sub.prepare("alpha beta")
    _i, subq_route, _ = pipe.retrieve("gamma gold")        # distinct sub-Q
    _j, seed_route, _ = pipe.retrieve("alpha beta", ["x"])  # seeded iter-path
    assert subq_route == "dense"
    assert seed_route == "dense"
    # the prepared primary still gets the substrate.
    _k, prim_route, _ = pipe.retrieve("alpha beta")
    assert prim_route == "bridge_substrate"


def test_default_scope_is_primary():
    pipe = mod._StubDensePipeline()
    sub = mod._BridgeSubstrate(pipe, judge_model="claude-haiku-4-5",
                               max_cost_usd=10.0, require_backend=False)
    assert sub._scope == "primary"


# ---- cross-query cache dedup (no double-spend) ----------------------------

def test_cross_query_cache_dedup():
    pipe, sub = _sub("all")
    pipe.retrieve("alpha beta")
    pipe.retrieve("alpha beta")   # identical -> cache hit, no new bridge run
    assert sub.n_bridge_runs == 1
    pipe.retrieve("gamma gold")   # distinct -> one more run
    assert sub.n_bridge_runs == 2
    assert sub.stats()["n_distinct_queries"] == 2


# ---- cost cap degrades subsequent distinct queries to dense ----------------

def test_scope_all_cost_cap_reverts_new_queries_to_dense():
    pipe, sub = _sub("all")
    pipe.retrieve("alpha beta")       # one real run
    sub.total_cost_usd = 99.0         # force past the cap
    _idx, route, _ = pipe.retrieve("delta filler")   # new query -> capped
    assert route == "dense"
    assert sub.n_cost_capped == 1
    # the already-cached query still serves from cache (no new spend).
    _i, route2, _ = pipe.retrieve("alpha beta")
    assert route2 == "bridge_substrate"


# ---- invalid scope rejected -----------------------------------------------

def test_invalid_scope_raises():
    pipe = mod._StubDensePipeline()
    with pytest.raises(ValueError):
        mod._BridgeSubstrate(pipe, judge_model="claude-haiku-4-5",
                             max_cost_usd=10.0, require_backend=False,
                             scope="everything")


# ---- dry-run covers both scopes -------------------------------------------

def test_dry_run_all_scope_semantics():
    prim = mod._dry_run_bridge_substrate(scope="primary")
    allx = mod._dry_run_bridge_substrate(scope="all")
    assert prim["ok"] is True and allx["ok"] is True
    assert prim["n_seed_path_bridged"] == 0 and prim["subq_bridged"] is False
    assert allx["n_seed_path_bridged"] == 5 and allx["subq_bridged"] is True
    assert allx["substrate_stats"]["scope"] == "all"


# ---- CLI flag: default primary + choices ----------------------------------

def test_cli_scope_flag_default_and_choices():
    orig = argparse.ArgumentParser.parse_args
    captured: dict = {}

    class _Stop(Exception):
        pass

    def _cap(self, *a, **k):
        captured["p"] = self
        raise _Stop()

    argparse.ArgumentParser.parse_args = _cap
    try:
        mod.main()
    except _Stop:
        pass
    finally:
        argparse.ArgumentParser.parse_args = orig
    parser = captured["p"]
    assert orig(parser, []).bridge_substrate_scope == "primary"
    assert orig(parser, ["--bridge-substrate-scope", "all"]).bridge_substrate_scope == "all"
    with pytest.raises(SystemExit):
        orig(parser, ["--bridge-substrate-scope", "bogus"])
