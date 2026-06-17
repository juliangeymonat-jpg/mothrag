# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""ChainFilter chain_deep cohort opt-out.

MQ chain_deep is extraction-bound: the OpenIE triple-extract can discard
answer-bearing chunks, so ChainFilter's rerank can hurt there. The cohort gate
(``--use-chainfilter-cohort-gate``) makes ChainFilter a pure passthrough on the
chain_deep cohort while every other cohort keeps the legacy rerank. Default OFF
⇒ byte-identical.
"""
from __future__ import annotations

import importlib.util as _u
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

import mothrag.retrieval.chain_filter.chain_filter as cfmod


def _cf(qtype, *, cohort_gate, enabled=True):
    cfg = cfmod.ChainFilterConfig(
        enabled=enabled, cohort_gate_skip_chain_deep=cohort_gate)
    return cfmod.ChainFilter(
        config=cfg,
        # never extract for real; if the legacy path runs it degrades to
        # passthrough via the no-extractor branch (distinct counter).
        triple_extractor=None,
        classify_fn=lambda q: {"label_v2": qtype, "has_chain": True,
                               "np_depth": 3, "n_relations": 2},
    )


def _cands(n=6):
    return [SimpleNamespace(passage_id=f"p{i}", text=f"t{i}", ann_score=float(n - i))
            for i in range(n)]


def test_chain_deep_plus_flag_skips_chainfilter():
    cf = _cf("chain_deep", cohort_gate=True)
    out = cf.filter("deep multi-hop q", _cands(6))
    assert [c.passage_id for c in out] == ["p0", "p1", "p2", "p3", "p4"]  # top_k_out passthrough
    assert cf.counters["cohort_skipped_chain_deep"] == 1
    assert cf.counters["gate_fired"] == 0          # never reached the rerank body


def test_non_chain_deep_plus_flag_runs_legacy():
    cf = _cf("semantic_rich", cohort_gate=True)
    cf.filter("a semantic q", _cands(6))
    assert cf.counters["cohort_skipped_chain_deep"] == 0   # cohort gate did NOT fire
    # legacy path WAS entered (gate fired on has_chain), then degraded (no extractor)
    assert cf.counters["gate_fired"] == 1


def test_flag_off_is_legacy_even_on_chain_deep():
    cf = _cf("chain_deep", cohort_gate=False)
    cf.filter("deep multi-hop q", _cands(6))
    assert cf.counters["cohort_skipped_chain_deep"] == 0   # no cohort opt-out
    assert cf.counters["gate_fired"] == 1                  # legacy ChainFilter ran


# --- telemetry surfaces in per-q JSON + aggregate summary.counters ----------- #
class _AnyArgs:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, _):
        return None


def _load_rp():
    path = pathlib.Path(cfmod.__file__).resolve().parents[3] / "scripts" / "route_prospective.py"
    spec = _u.spec_from_file_location("rp_cf_cohort", path)
    mod = _u.module_from_spec(spec)
    saved = sys.argv
    sys.argv = ["route_prospective"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


def test_chainfilter_cohort_telemetry_in_json(tmp_path):
    rp = _load_rp()
    per_q = [
        {"qid": "q1", "qtype": "chain_deep", "em": 1.0, "f1": 0.5, "r_at_10": 0.5,
         "iterations_used": 1, "arm_used": "x", "n_llm_calls": 1,
         "chainfilter_active": False, "chainfilter_skipped_chain_deep": True},
        {"qid": "q2", "qtype": "semantic_rich", "em": 1.0, "f1": 0.5, "r_at_10": 0.5,
         "iterations_used": 1, "arm_used": "x", "n_llm_calls": 1,
         "chainfilter_active": True, "chainfilter_skipped_chain_deep": False},
    ]
    out = tmp_path / "cf.json"
    rp._write_partial(out, per_q, _AnyArgs(use_chainfilter=True,
                                           use_chainfilter_cohort_gate=True),
                      partial=False)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "chainfilter_active" in data["per_question"][0]
    c = data["summary"]["counters"]
    assert c["chainfilter_active_total"] == 1
    assert c["chainfilter_skipped_chain_deep_total"] == 1
