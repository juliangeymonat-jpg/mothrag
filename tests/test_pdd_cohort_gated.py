# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""REVERSED qtype-conditional γ-aware PDD.

This reverses an earlier cohort gate that dropped the dup
on every non-``chain_deep`` cohort when iter was γ=valid, which hit the
semantic_rich BULK (the bulk of queries) and regressed accuracy. The current
gate drops the dup ONLY on ``chain_deep`` (where the ensemble vote is noise
once iter is γ-confident) and PRESERVES it on the semantic_rich bulk.

Telemetry: ``pdd_skipped_chain_deep_valid`` (drop fired on chain_deep) +
``pdd_preserved_semantic_rich`` (drop would have fired but the non-chain_deep
cohort gate kept it) + ``pdd_active``. Exactly one of {skipped, preserved} fires
when the gate engages.
"""
from __future__ import annotations

import importlib.util as _u
import json
import pathlib
import sys

import pytest

from mothrag.core.arms_runner import gamma_aware_pdd_should_skip


_RP_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "route_prospective.py"


def _load_rp():
    spec = _u.spec_from_file_location("rp_pdd_cohort", _RP_PATH)
    mod = _u.module_from_spec(spec)
    saved = sys.argv
    sys.argv = ["route_prospective"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


rp = _load_rp()


class _StubPipeline:
    embedder_model = None


@pytest.fixture
def _captured_pool(monkeypatch):
    """Patch the arbitrate core; record the answers (pool) the arbitrator scores."""
    import mothrag.core.arbitrate as arb_mod
    cap: dict = {}

    class _FakeRes:
        def __init__(self, answers):
            self.answer = next(iter(answers.values()), "")
            self.selected_arm = next(iter(answers), "")
            self.arbitrate_signal = "stub"
            self.arm_scores = {k: 0.0 for k in answers}

    class _FakeArb:
        def __init__(self, **_):
            pass

        def arbitrate(self, *, answers, gamma_signals, agreement_signals,
                      arm_probabilities=None):
            cap["pool"] = set(answers)
            return _FakeRes(answers)

    monkeypatch.setattr(arb_mod, "DeterministicArbitrator", _FakeArb)
    monkeypatch.setattr(arb_mod, "pairwise_agreement",
                        lambda answers, **_: {k: 0.0 for k in answers})
    return cap


def _cands():
    return {
        "iter": {"pred": "Frank Herbert", "retrieved_chunk_ids": []},
        "iter_dup_a": {"pred": "Frank Herbert", "retrieved_chunk_ids": [],
                       "metadata": {"dup_of": "iter"}},
        "decompose": {"pred": "Brian Herbert", "retrieved_chunk_ids": []},
    }


def _arb(qtype, gamma, flag, cap):
    out = rp._arbitrate_candidates(
        _StubPipeline(), candidates=_cands(), iter_gamma_status=gamma,
        qtype=qtype, use_gamma_aware_pdd=flag)
    return out, cap["pool"]


# 1 — chain_deep + γ=valid + flag ON → dup DROPPED + skipped counter (v2 REVERSED)
def test_pdd_skipped_when_chain_deep_gamma_valid(_captured_pool):
    out, pool = _arb("chain_deep", "valid", True, _captured_pool)
    assert "iter_dup_a" not in pool                   # dropped (3-arm effective)
    assert out["pdd_skipped_chain_deep_valid"] is True
    assert out["pdd_preserved_semantic_rich"] is False
    assert out["pdd_active"] is False
    assert gamma_aware_pdd_should_skip(_cands(), "valid", "chain_deep",
                                       enabled=True) is True


# 2 — semantic_rich (bulk) + γ=valid + flag ON → dup PRESERVED + preserved counter
def test_pdd_preserved_when_semantic_rich_gamma_valid(_captured_pool):
    out, pool = _arb("semantic_rich", "valid", True, _captured_pool)
    assert "iter_dup_a" in pool                       # preserved (PDD kept on bulk)
    assert out["pdd_preserved_semantic_rich"] is True
    assert out["pdd_skipped_chain_deep_valid"] is False
    assert out["pdd_active"] is True                  # dup did participate
    assert gamma_aware_pdd_should_skip(_cands(), "valid", "semantic_rich",
                                       enabled=True) is False
    # bridge_entity is also non-chain_deep ⇒ preserved
    out2, pool2 = _arb("bridge_entity", "valid", True, _captured_pool)
    assert "iter_dup_a" in pool2
    assert out2["pdd_preserved_semantic_rich"] is True


# 3 — γ=partial (any qtype) + flag ON → dup KEPT + active counter, no fire
def test_pdd_active_when_gamma_partial(_captured_pool):
    out, pool = _arb("chain_deep", "partial", True, _captured_pool)
    assert "iter_dup_a" in pool
    assert out["pdd_active"] is True
    assert out["pdd_skipped_chain_deep_valid"] is False
    assert out["pdd_preserved_semantic_rich"] is False   # preserve only on γ=valid
    # also for a semantic_rich partial
    out2, pool2 = _arb("semantic_rich", "partial", True, _captured_pool)
    assert "iter_dup_a" in pool2 and out2["pdd_active"] is True


# 4 — flag OFF (legacy) → dup KEPT, γ-aware-specific counters zero
def test_pdd_active_when_flag_off(_captured_pool):
    out, pool = _arb("chain_deep", "valid", False, _captured_pool)
    assert "iter_dup_a" in pool                       # legacy 4-arm preserved
    assert out["pdd_skipped_chain_deep_valid"] is False
    assert out["pdd_preserved_semantic_rich"] is False
    assert out["pdd_active"] is True                  # dup active in legacy
    # semantic_rich + flag off is also legacy (no preserve "fire")
    out2, _ = _arb("semantic_rich", "valid", False, _captured_pool)
    assert out2["pdd_preserved_semantic_rich"] is False


# 5 — telemetry counters land in per-q JSON + aggregate summary.counters
class _AnyArgs:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, _):
        return None


def _row(qid, **pdd):
    r = {"qid": qid, "em": 1.0, "f1": 0.5, "r_at_10": 0.5, "iterations_used": 1,
         "arm_used": "ensemble_arbitrate", "qtype": "x", "n_llm_calls": 1,
         "pdd_active": False, "pdd_skipped_chain_deep_valid": False,
         "pdd_preserved_semantic_rich": False}
    r.update(pdd)
    return r


def test_telemetry_counters_in_per_query_json_output(tmp_path):
    per_q = [
        _row("q1", pdd_active=True, pdd_preserved_semantic_rich=True),  # sr bulk valid
        _row("q2", pdd_skipped_chain_deep_valid=True),                  # chain_deep valid
        _row("q3", pdd_active=True),                                    # partial
    ]
    out = tmp_path / "pdd.json"
    rp._write_partial(out, per_q, _AnyArgs(use_gamma_aware_pdd=True), partial=False)
    data = json.loads(out.read_text(encoding="utf-8"))
    # per_query carries the three fields
    pq0 = data["per_question"][0]
    assert {"pdd_active", "pdd_skipped_chain_deep_valid",
            "pdd_preserved_semantic_rich"} <= set(pq0)
    # aggregate summary.counters totals
    c = data["summary"]["counters"]
    assert c["pdd_active_total"] == 2
    assert c["pdd_skipped_chain_deep_valid_total"] == 1
    assert c["pdd_preserved_semantic_rich_total"] == 1
    # config dump still carries the flag
    assert data["summary"]["config"]["use_gamma_aware_pdd"] is True
