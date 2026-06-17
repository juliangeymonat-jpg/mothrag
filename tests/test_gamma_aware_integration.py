# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""γ-aware gating — end-to-end integration.

Exercises the EVAL wiring (``route_prospective._arbitrate_candidates`` →
``arms_runner.arbitrate_pool`` → real ``DeterministicArbitrator``) with the
γ-aware-PDD flag, and writes a sample run-summary JSON so the per-q telemetry
fields + config provenance are demonstrably present in on-disk output.

Only ``pairwise_agreement`` is stubbed (deterministic, embedder-free); the
arbitrator + the dup-drop in ``arbitrate_pool`` run for real. The both-flags
live dry-run (with --use-gamma-aware-bridge) lands with the bridge fast-follow;
the live JSON with these same fields is produced by a separate eval run.
"""
from __future__ import annotations

import importlib.util as _u
import json
import pathlib
import sys

import pytest

_RP_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "route_prospective.py"


def _load_rp():
    spec = _u.spec_from_file_location("route_prospective_intg", _RP_PATH)
    mod = _u.module_from_spec(spec)
    _saved = sys.argv
    sys.argv = ["route_prospective"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = _saved
    return mod


rp = _load_rp()


class _StubPipeline:
    """embedder unused once pairwise_agreement is stubbed."""
    embedder_model = None


@pytest.fixture
def _deterministic_agreement(monkeypatch):
    # agreement = 1.0 for any answer equal to the iter answer, else 0.0 — so the
    # dup (a copy of iter) measurably amplifies iter's agreement UNLESS dropped.
    import mothrag.core.arbitrate as arb_mod

    def _fake_pairwise(answers, **_kw):
        iter_ans = answers.get("iter", "")
        return {k: (1.0 if v == iter_ans else 0.0) for k, v in answers.items()}

    monkeypatch.setattr(arb_mod, "pairwise_agreement", _fake_pairwise)


def _candidates():
    return {
        "iter": {"pred": "Frank Herbert", "retrieved_chunk_ids": ["c1"],
                 "n_llm_calls": 1, "prompt_tokens": 10, "completion_tokens": 5,
                 "latency_s": 0.1},
        "iter_dup_a": {"pred": "Frank Herbert", "retrieved_chunk_ids": ["c1"],
                       "n_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                       "latency_s": 0.0, "metadata": {"dup_of": "iter"}},
        "decompose": {"pred": "Brian Herbert", "retrieved_chunk_ids": ["c2"],
                      "n_llm_calls": 1, "prompt_tokens": 8, "completion_tokens": 4,
                      "latency_s": 0.1},
    }


def test_eval_wiring_pdd_skipped_when_chain_deep_valid(_deterministic_agreement):
    out = rp._arbitrate_candidates(
        _StubPipeline(), candidates=_candidates(),
        iter_gamma_status="valid", qtype="chain_deep", use_gamma_aware_pdd=True)
    assert out["pdd_skipped_chain_deep_valid"] is True
    assert out["pdd_active"] is False


def test_eval_wiring_pdd_preserved_when_semantic_rich_valid(_deterministic_agreement):
    # v2 bulk-safe: semantic_rich keeps the dup even at γ=valid + flag ON.
    out = rp._arbitrate_candidates(
        _StubPipeline(), candidates=_candidates(),
        iter_gamma_status="valid", qtype="semantic_rich", use_gamma_aware_pdd=True)
    assert out["pdd_preserved_semantic_rich"] is True
    assert out["pdd_skipped_chain_deep_valid"] is False
    assert out["pdd_active"] is True


def test_eval_wiring_pdd_active_when_flag_off(_deterministic_agreement):
    out = rp._arbitrate_candidates(
        _StubPipeline(), candidates=_candidates(),
        iter_gamma_status="valid", qtype="chain_deep", use_gamma_aware_pdd=False)
    assert out["pdd_active"] is True
    assert out["pdd_skipped_chain_deep_valid"] is False


def test_dryrun_summary_json_carries_telemetry_and_config(tmp_path,
                                                          _deterministic_agreement):
    # Assemble a route-style run summary and round-trip it through disk so the
    # config flag + per-q telemetry are demonstrably present in output JSON.
    rows = []
    for gamma, qtype, flag in (("valid", "chain_deep", True),
                               ("partial", "chain_deep", True),
                               ("valid", "chain_deep", False)):
        arb = rp._arbitrate_candidates(
            _StubPipeline(), candidates=_candidates(),
            iter_gamma_status=gamma, qtype=qtype, use_gamma_aware_pdd=flag)
        rows.append({
            "qid": f"q_{gamma}_{flag}", "gamma_final_status": gamma,
            "pdd_active": arb["pdd_active"],
            "pdd_skipped_chain_deep_valid": arb["pdd_skipped_chain_deep_valid"],
        })
    summary = {
        "summary": {"config": {"use_gamma_aware_pdd": True,
                               "mode": "ensemble_arbitrate"}},
        "per_question": rows,
    }
    out_path = tmp_path / "gamma_aware_pdd_dryrun.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["summary"]["config"]["use_gamma_aware_pdd"] is True
    fields = {(r["qid"], r["pdd_skipped_chain_deep_valid"]) for r in data["per_question"]}
    # chain_deep valid+flag → skipped; partial+flag → not; valid+no-flag → not.
    assert ("q_valid_True", True) in fields
    assert ("q_partial_True", False) in fields
    assert ("q_valid_False", False) in fields
