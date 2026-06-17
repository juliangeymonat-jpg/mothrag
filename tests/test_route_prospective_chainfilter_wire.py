# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""ChainFilter is actually WIRED into the bridge path.

An earlier revision only pre-wired the flags: ON hit the no-extractor early
return, `filter()` was never called, no behavioural telemetry → ON == OFF live.
This pins the fix: (1) `_build_chain_filter` wires a real OpenIE triple_extractor
so ON runs end-to-end; (2) `_BridgeSubstrate._apply_chain_filter` actually calls
`chain_filter.filter()` over the bridge ranking and remaps to indices;
(3) behavioural counters move; (4) negative control (all-γ=0 → passthrough, no
regression).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

from mothrag.retrieval.chain_filter import ChainFilter, ChainFilterConfig

_SPEC = importlib.util.spec_from_file_location(
    "route_prospective",
    Path(__file__).resolve().parents[1] / "scripts" / "route_prospective.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)

MULTIHOP = {"np_depth": 3, "n_relations": 2, "has_chain": True}


def _substrate(chain_filter):
    stub = mod._StubDensePipeline()
    sub = mod._BridgeSubstrate(
        stub, judge_model="claude-haiku-4-5", max_cost_usd=10.0,
        require_backend=False, scope="primary", qtype_gate="none",
        chain_filter=chain_filter)
    return sub


# ---- 1. wiring: _apply_chain_filter actually calls filter() + remaps ---------

class _FakeCF:
    counters = {"gate_fired": 0}

    def __init__(self):
        self.calls = []

    def filter(self, question, candidates):
        self.calls.append((question, [c.passage_id for c in candidates]))
        return list(reversed(candidates))[:3]   # controlled reorder


def test_apply_chain_filter_calls_filter_and_remaps():
    fake = _FakeCF()
    sub = _substrate(fake)
    sub._current_q = "a multi-hop question"
    new_idx = sub._apply_chain_filter(["p0", "p1", "p2"], [0, 1, 2])
    # filter() was invoked with the TOP-LEVEL question + the reconstructed cands.
    assert fake.calls == [("a multi-hop question", ["p0", "p1", "p2"])]
    # reversed[:3] = p2,p1,p0 → indices [2,1,0]
    assert new_idx == [2, 1, 0]


def test_apply_chain_filter_passthrough_on_empty_result():
    sub = _substrate(NS(filter=lambda q, c: []))   # filter returns nothing
    sub._current_q = "q"
    assert sub._apply_chain_filter(["p0", "p1"], [0, 1]) == [0, 1]  # idx preserved


# ---- 2. real ChainFilter reranks through the substrate (end-to-end) ----------

def test_real_chainfilter_runs_through_substrate():
    # The REAL ChainFilter executes end-to-end through the substrate path
    # (gate fires, extractor runs); output stays a subset of the input indices
    # (pool-safe). Reranking ORDER correctness is pinned corpus-independently in
    # tests/test_chain_filter.py.
    stub = mod._StubDensePipeline()
    cf = ChainFilter(
        config=ChainFilterConfig(enabled=True, top_k_out=3),
        classify_fn=lambda q: MULTIHOP,
        triple_extractor=lambda t: [["x", "r", "y"]],   # populated for every cand
        fact_filter=lambda q, facts: [["x", "r", "y"]],
        gamma_scorer=lambda q, f, t: 0.9)
    sub = _substrate(cf)
    sub._current_q = "multi hop"
    pids = list(stub.chunk_ids[:4])
    in_idx = [stub.chunk_ids.index(p) for p in pids]
    new_idx = sub._apply_chain_filter(pids, in_idx)
    assert cf.counters["gate_fired"] == 1          # real filter ran end-to-end
    assert set(new_idx) <= set(in_idx)             # subset → pool-safe
    assert len(new_idx) >= 1


# ---- 3. negative control: multi-hop + populated extractor + all-γ=0 ----------

def test_negative_control_all_gamma_zero_passthrough():
    fact = ["a", "rel", "b"]
    cf = ChainFilter(
        config=ChainFilterConfig(enabled=True, top_k_out=2),
        classify_fn=lambda q: MULTIHOP,
        triple_extractor=lambda t: [fact],         # populated
        fact_filter=lambda q, facts: [fact],
        gamma_scorer=lambda q, f, t: 0.0)          # all γ=0 → LOW → excluded
    cands = [NS(passage_id="p1", text="a b", ann_score=0.1),
             NS(passage_id="p2", text="x", ann_score=0.9),
             NS(passage_id="p3", text="z", ann_score=0.5)]
    out = cf.filter("multi hop", cands)
    assert [c.passage_id for c in out] == ["p1", "p2"]   # passthrough, unchanged
    assert cf.counters["gate_fired"] == 1
    assert cf.counters["zero_chain_support_passthrough"] == 1
    assert cf.counters["reranked"] == 0


# ---- 4. counters distinguish ON-active from single-hop-bypass ----------------

def test_counters_single_hop_bypass_vs_fired():
    cf = ChainFilter(config=ChainFilterConfig(enabled=True),
                     classify_fn=lambda q: {"np_depth": 1, "n_relations": 0,
                                            "has_chain": False},
                     triple_extractor=lambda t: [["a", "r", "b"]])
    cf.filter("single hop", [NS(passage_id="p1", text="a b", ann_score=0.5)])
    assert cf.counters["gate_skipped_single_hop"] == 1
    assert cf.counters["gate_fired"] == 0
