# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""CONSERVATIVE γ-aware bridge cohort 2-pass.

A first-pass DENSE iter probe yields a γ proxy; the bridge expansion is skipped
ONLY for the EASIEST cohort — γ=valid AND qtype==semantic_rich AND hop_count==1 —
and the probe's iter result is REUSED by the pool (no double iter). Multi-hop
semantic_rich, chain_deep, and bridge_entity ALWAYS keep the bridge (an earlier
broader skip dropped ~60%% of γ=valid queries and regressed on the bulk cohort).
Default OFF ⇒ legacy.
"""
from __future__ import annotations

import importlib.util as _u
import json
import pathlib
import sys

import pytest


def _load_rp():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "route_prospective.py"
    spec = _u.spec_from_file_location("rp_bridge_cohort", path)
    mod = _u.module_from_spec(spec)
    saved = sys.argv
    sys.argv = ["route_prospective"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


rp = _load_rp()
_skip = rp._bridge_cohort_should_skip


# --- the cohort gate predicate (the 4 dispatch cases) ----------------------- #
def test_skips_only_easy_single_hop_semantic_rich():
    # cohort MATCH: γ=valid + semantic_rich + single-hop → skip
    assert _skip("valid", "semantic_rich", enabled=True, hop_count=1) is True


def test_multi_hop_semantic_rich_keeps_bridge():
    # cohort NO-MATCH (multi-hop): the bulk cohort that v1 wrongly skipped
    assert _skip("valid", "semantic_rich", enabled=True, hop_count=2) is False


def test_chain_deep_and_bridge_entity_keep_bridge_even_when_valid():
    # cohort NO-MATCH (non-semantic_rich): both always keep the bridge
    assert _skip("valid", "chain_deep", enabled=True, hop_count=1) is False
    assert _skip("valid", "bridge_entity", enabled=True, hop_count=1) is False


def test_gamma_partial_or_none_keeps_bridge_active():
    assert _skip("partial", "semantic_rich", enabled=True, hop_count=1) is False
    assert _skip("invalid", "semantic_rich", enabled=True, hop_count=1) is False
    assert _skip(None, "semantic_rich", enabled=True, hop_count=1) is False  # verifier fail


def test_flag_off_is_legacy():
    assert _skip("valid", "semantic_rich", enabled=False, hop_count=1) is False


def test_hop_count_proxy_classifies_single_vs_multi_hop():
    # single-hop: ≤1 relation, no chain cue
    assert rp._hop_count_proxy({"n_relations": 1, "has_chain": False}) == 1
    assert rp._hop_count_proxy({"n_relations": 0, "has_chain": False}) == 1
    # multi-hop: chain cue OR ≥2 relations
    assert rp._hop_count_proxy({"n_relations": 2, "has_chain": False}) == 2
    assert rp._hop_count_proxy({"n_relations": 1, "has_chain": True}) == 2


# --- the 2-pass optimisation: probe iter is REUSED (no double iter) ---------- #
def _stub_arm(pred):
    return {"pred": pred, "retrieved_chunk_ids": [], "n_llm_calls": 1,
            "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0,
            "iterations_used": 1, "gamma_final_status": "valid"}


def test_precomputed_iter_reused_no_double_iter(monkeypatch):
    calls = {"iter": 0}

    def _fake_iter(_runner, _q):
        calls["iter"] += 1
        return _stub_arm("freshly-run-iter")

    monkeypatch.setattr(rp, "_run_iter", _fake_iter)
    monkeypatch.setattr(rp, "_run_v3bu", lambda *a, **k: _stub_arm("v3"))
    monkeypatch.setattr(rp, "_run_decompose", lambda *a, **k: _stub_arm("dec"))
    # _run_ensemble_arbitrate does `from ...query_type_classifier import arm_subset
    # as _arm_subset` at call time → patch the source module attribute.
    import mothrag.core.query_type_classifier as _qtc
    monkeypatch.setattr(_qtc, "arm_subset",
                        lambda q, **k: ["v3bu", "decompose", "iter"])
    monkeypatch.setattr(rp, "arm_subset",
                        lambda q, **k: ["v3bu", "decompose", "iter"])
    monkeypatch.setattr(rp, "_arbitrate_candidates", lambda *a, **k: {
        "pred": "X", "retrieved_chunk_ids": [], "n_llm_calls": 0,
        "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0,
        "selected_arm": "iter", "arbitrate_signal": "s", "arm_scores": {},
        "pdd_active": False, "pdd_skipped_chain_deep_valid": False,
        "pdd_preserved_semantic_rich": False})

    probe = _stub_arm("probe-iter")
    out = rp._run_ensemble_arbitrate(
        object(), object(), "q", "model", 5, precomputed_iter=probe)
    assert calls["iter"] == 0                       # iter NOT re-run — probe reused
    assert out["iter_pred"] == "probe-iter"
    # and without a precomputed probe, iter DOES run
    rp._run_ensemble_arbitrate(object(), object(), "q", "model", 5)
    assert calls["iter"] == 1


# --- telemetry surfaces in per-q JSON + aggregate --------------------------- #
class _AnyArgs:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, _):
        return None


def test_bridge_cohort_telemetry_in_json(tmp_path):
    per_q = [
        {"qid": "q1", "qtype": "semantic_rich", "em": 1.0, "f1": 0.5, "r_at_10": 0.5,
         "iterations_used": 1, "arm_used": "x", "n_llm_calls": 1,
         "bridge_active": False, "bridge_skipped_easy_semantic_rich": True},
        {"qid": "q2", "qtype": "bridge_entity", "em": 1.0, "f1": 0.5, "r_at_10": 0.5,
         "iterations_used": 1, "arm_used": "x", "n_llm_calls": 1,
         "bridge_active": True, "bridge_skipped_easy_semantic_rich": False},
    ]
    out = tmp_path / "bc.json"
    rp._write_partial(out, per_q, _AnyArgs(use_gamma_aware_bridge_cohort=True),
                      partial=False)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "bridge_skipped_easy_semantic_rich" in data["per_question"][0]
    c = data["summary"]["counters"]
    assert c["bridge_active_total"] == 1
    assert c["bridge_skipped_easy_semantic_rich_total"] == 1
