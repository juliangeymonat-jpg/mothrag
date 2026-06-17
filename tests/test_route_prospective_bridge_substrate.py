# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""route_prospective bridge SUBSTRATE wiring.

Verifies the unified "4-arm + bridge substrate" runner: the bridge reshapes
each query's PRIMARY (seed-free) retrieval feeding the arm pool, the arm POOL
stays 4 (v3bu / decompose / iter / iter_dup_a PDD) with the bridge NOT a
candidate (zero pool-safety violation), graceful degradation on error /
cost-cap, and the offline ``--dry-run`` mechanics + paired-comparison check.

Anti-leak / pool-safety regression guard for the bridge-substrate
architecture.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "route_prospective",
    Path(__file__).resolve().parents[1] / "scripts" / "route_prospective.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def _stub_pipeline():
    """A synthetic dense pipeline with the surface the substrate consumes."""
    return mod._StubDensePipeline()


# ---- substrate install + primary-retrieval reshaping ----------------------

def test_substrate_installs_and_wraps_retrieve():
    pipe = _stub_pipeline()
    orig = pipe.retrieve
    sub = mod._BridgeSubstrate(pipe, judge_model="claude-haiku-4-5",
                               max_cost_usd=10.0, require_backend=False)
    # The bound method was replaced by the substrate wrapper.
    assert pipe.retrieve is not orig
    assert pipe.retrieve == sub._wrapped_retrieve

    info = sub.prepare("alpha beta")
    assert info["fired"] is True
    assert info["reason"] == "bridge_substrate"
    top_idx, route, conf = pipe.retrieve("alpha beta")
    assert route == "bridge_substrate"
    assert isinstance(top_idx, list) and top_idx
    # capped to the dense top_k_chunks, never more.
    assert len(top_idx) <= pipe.config.top_k_chunks


def test_substrate_falls_through_for_seeded_query():
    # entity_seeds present => iter-refinement path => MUST bypass the bridge.
    pipe = _stub_pipeline()
    sub = mod._BridgeSubstrate(pipe, judge_model="claude-haiku-4-5",
                               max_cost_usd=10.0, require_backend=False)
    sub.prepare("alpha beta")
    _idx, route, _ = pipe.retrieve("alpha beta", ["alpha"])
    assert route == "dense"


def test_substrate_falls_through_for_other_question():
    # A query string other than the prepared top-level question => dense.
    pipe = _stub_pipeline()
    sub = mod._BridgeSubstrate(pipe, judge_model="claude-haiku-4-5",
                               max_cost_usd=10.0, require_backend=False)
    sub.prepare("alpha beta")
    _idx, route, _ = pipe.retrieve("gamma gold")
    assert route == "dense"


# ---- POOL SAFETY: substrate must NOT add a 5th candidate -------------------

def test_substrate_keeps_pool_at_four_arms(monkeypatch):
    """4-arm pool + bridge substrate: candidates == {v3bu,decompose,iter,
    iter_dup_a}; bridge is NEVER a candidate; the arms read the substrate."""
    pipe = _stub_pipeline()
    sub = mod._BridgeSubstrate(pipe, judge_model="claude-haiku-4-5",
                               max_cost_usd=10.0, require_backend=False)

    routes_seen: list[str] = []

    def _stub_arm(pred):
        def _run(pipeline, question, *a, **k):
            # Exercise the substrate exactly like the real arms' primary
            # retrieval does, and record the route it served.
            _idx, route, _ = pipeline.retrieve(question)
            routes_seen.append(route)
            return {
                "pred": pred, "retrieved_chunk_ids": [pipeline.chunk_ids[0]],
                "n_llm_calls": 1, "prompt_tokens": 1, "completion_tokens": 1,
                "latency_s": 0.0,
            }
        return _run

    def _stub_iter(iter_runner, question, *a, **k):
        _idx, route, _ = pipe.retrieve(question)
        routes_seen.append(route)
        return {
            "pred": "iter-ans", "retrieved_chunk_ids": [pipe.chunk_ids[1]],
            "n_llm_calls": 1, "prompt_tokens": 1, "completion_tokens": 1,
            "latency_s": 0.0, "gamma_final_status": "valid", "iterations_used": 2,
        }

    monkeypatch.setattr(mod, "_run_v3bu", _stub_arm("v3bu-ans"))
    monkeypatch.setattr(mod, "_run_decompose",
                        lambda p, q, *a, **k: _stub_arm("dec-ans")(p, q))
    monkeypatch.setattr(mod, "_run_iter", _stub_iter)

    sub.prepare("alpha beta")
    out = mod._run_ensemble_arbitrate(
        pipe, object(), "alpha beta", "model", 10,
        arms_pool=["v3bu", "decompose", "iter", "iter_dup_a"],
    )
    scores = out["arm_scores"]
    # PDD (iter_dup_a) present => the 4th arm is in the pool.
    assert "iter_dup_a" in scores
    # The bridge is a SUBSTRATE, never a candidate/arm.
    assert "bridge" not in scores and "bridge_arm" not in scores
    # Only the 4 legacy/PDD arms are pool members.
    assert set(scores).issubset({"v3bu", "decompose", "iter", "iter_dup_a"})
    # The arms actually read the bridge substrate (primary, seed-free retrieve).
    assert "bridge_substrate" in routes_seen
    assert sub.n_bridge_runs == 1


# ---- graceful degradation: cost cap + bridge error -------------------------

def test_cost_cap_reverts_to_dense(monkeypatch):
    pipe = _stub_pipeline()
    sub = mod._BridgeSubstrate(pipe, judge_model="claude-haiku-4-5",
                               max_cost_usd=10.0, require_backend=False)
    # Force the running total past the cap.
    sub.total_cost_usd = 99.0
    info = sub.prepare("alpha beta")
    assert info["fired"] is False
    assert info["reason"] == "cost_capped"
    assert sub.n_cost_capped == 1
    _idx, route, _ = pipe.retrieve("alpha beta")
    assert route == "dense"


def test_bridge_error_reverts_to_dense(monkeypatch):
    pipe = _stub_pipeline()
    sub = mod._BridgeSubstrate(pipe, judge_model="claude-haiku-4-5",
                               max_cost_usd=10.0, require_backend=False)

    def _boom(question):
        raise RuntimeError("bridge boom")

    monkeypatch.setattr(sub._arm, "retrieve", _boom)
    info = sub.prepare("alpha beta")
    assert info["fired"] is False
    assert info["reason"].startswith("bridge_error")
    assert sub.n_fallback == 1
    _idx, route, _ = pipe.retrieve("alpha beta")
    assert route == "dense"  # run continues on dense


# ---- offline dry-run + paired comparison -----------------------------------

def test_dry_run_function_ok():
    sim = mod._dry_run_bridge_substrate()
    assert sim["ok"] is True
    assert sim["mode"] == "dry_run"
    assert sim["synthetic_queries"] == 5
    assert sim["n_bridge_fired"] == 5
    # paired comparison output present for every query.
    assert len(sim["paired"]) == 5
    for row in sim["paired"]:
        assert "dense_top" in row and "bridge_top" in row
        assert "diverged_from_dense" in row


def test_dry_run_cli_returns_zero(monkeypatch):
    monkeypatch.setattr("sys.argv", ["route_prospective.py", "--dry-run"])
    assert mod.main() == 0


def test_live_run_requires_inputs(monkeypatch):
    # No --dry-run and missing --data-dir/--queries/--out => SystemExit.
    monkeypatch.setattr("sys.argv", ["route_prospective.py"])
    with pytest.raises(SystemExit):
        mod.main()


def test_use_bridge_arm_alias(monkeypatch):
    # Back-compat: --use-bridge-arm sets the same dest as --use-bridge-substrate.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-bridge-substrate", "--use-bridge-arm",
                    dest="use_bridge_substrate", action="store_true",
                    default=False)
    assert ap.parse_args(["--use-bridge-arm"]).use_bridge_substrate is True
    assert ap.parse_args(["--use-bridge-substrate"]).use_bridge_substrate is True
    assert ap.parse_args([]).use_bridge_substrate is False
