# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Conservative-gating contract.

An earlier eval showed an earlier gating variant was too aggressive on the
semantic_rich BULK (HP +1.3pp but MQ -3.6pp / 2W -3.1pp). v2 re-derives each
gate to PRESERVE the bulk cohort. This file pins the 4-cases-per-fix contract
(ON cohort-match / ON cohort-no-match / multi-condition / OFF legacy) for the
four revised gates; FIX A (ChainFilter chain_deep opt-out) is unchanged and
covered by ``test_chainfilter_cohort_gate.py``.

All assertions hit the pure predicates / helpers (no network, no LLM).
"""
from __future__ import annotations

import importlib.util as _u
import pathlib
import sys

import pytest

from mothrag.core.arms_runner import gamma_aware_pdd_should_skip
from mothrag.eval.iterative_pipeline import (
    IterativeConfig, _effective_gamma_retrigger_cap, _faithfulness_gamma_coord_skip)


def _load_rp():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "route_prospective.py"
    spec = _u.spec_from_file_location("rp_route_prospective", path)
    mod = _u.module_from_spec(spec)
    saved = sys.argv
    sys.argv = ["route_prospective"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


rp = _load_rp()
_bridge_skip = rp._bridge_cohort_should_skip
_hop = rp._hop_count_proxy


def _pool():
    return {"iter": {"pred": "A"}, "iter_dup_a": {"pred": "A"},
            "decompose": {"pred": "B"}}


# ===================================================================== FIX B-v2
# Bridge skipped ONLY for γ=valid + semantic_rich + single-hop (easiest cohort).
def test_b_v2_cohort_match_skips_bridge():
    assert _bridge_skip("valid", "semantic_rich", enabled=True, hop_count=1) is True


def test_b_v2_multi_hop_semantic_rich_keeps_bridge():
    # the bulk cohort v1 wrongly skipped — v2 keeps the bridge for it
    assert _bridge_skip("valid", "semantic_rich", enabled=True, hop_count=2) is False


def test_b_v2_non_semantic_rich_cohort_keeps_bridge():
    assert _bridge_skip("valid", "chain_deep", enabled=True, hop_count=1) is False
    assert _bridge_skip("valid", "bridge_entity", enabled=True, hop_count=1) is False


def test_b_v2_flag_off_and_non_valid_are_legacy():
    assert _bridge_skip("valid", "semantic_rich", enabled=False, hop_count=1) is False
    assert _bridge_skip("partial", "semantic_rich", enabled=True, hop_count=1) is False
    # hop-count proxy: chain cue OR ≥2 relations ⇒ multi-hop
    assert _hop({"n_relations": 1, "has_chain": False}) == 1
    assert _hop({"n_relations": 3, "has_chain": False}) == 2
    assert _hop({"n_relations": 0, "has_chain": True}) == 2


# ===================================================================== FIX C-v2
# Only chain_deep gets the extra retry (3); the bulk stays at baseline 2.
def _cap(qtype, *, adaptive=True, composite=False, fixed=2, composite_cap=3):
    cfg = IterativeConfig(use_adaptive_gamma_retrigger=adaptive,
                          gamma_max_retrigger=fixed,
                          composite_gamma_max_retrigger=composite_cap)
    return _effective_gamma_retrigger_cap(cfg, qtype, composite)


def test_c_v2_chain_deep_gets_extra_retry():
    assert _cap("chain_deep") == 3


def test_c_v2_semantic_rich_stays_at_baseline_2():
    # v1 used 1 here → bulk starvation; v2 keeps it at the baseline
    assert _cap("semantic_rich") == 2


def test_c_v2_all_other_cohorts_baseline_2():
    assert _cap("bridge_entity") == 2
    assert _cap("anything_else") == 2
    assert _cap(None) == 2


def test_c_v2_flag_off_is_legacy_fixed_cap():
    assert _cap("chain_deep", adaptive=False, fixed=2) == 2
    assert _cap("semantic_rich", adaptive=False, fixed=2) == 2
    assert _cap("chain_deep", adaptive=False, composite=True, composite_cap=3) == 3


# ===================================================================== FIX D-v2
# Two safe-skip branches: clean_valid (γ=valid) | exhausted_safe (exhausted ∧
# qtype!=chain_deep ∧ iter>=2). chain_deep cap-hit KEEPS the faithfulness gate.
def _faith(cfg, gamma, **kw):
    return _faithfulness_gamma_coord_skip(cfg, gamma, **kw)


def test_d_v2_clean_valid_branch():
    cfg = IterativeConfig(use_faithfulness_gamma_coord=True)
    assert _faith(cfg, "valid") == "clean_valid"


def test_d_v2_exhausted_safe_branch():
    cfg = IterativeConfig(use_faithfulness_gamma_coord=True)
    assert _faith(cfg, "invalid", gamma_refuse_loop_exhausted=True,
                  qtype="semantic_rich", iter_count=2) == "exhausted_safe"
    assert _faith(cfg, "partial", gamma_refuse_loop_exhausted=True,
                  qtype="bridge_entity", iter_count=3) == "exhausted_safe"


def test_d_v2_chain_deep_and_early_iter_keep_faithfulness():
    cfg = IterativeConfig(use_faithfulness_gamma_coord=True)
    # chain_deep cap-hit = recovery cohort → KEEP the gate
    assert _faith(cfg, "invalid", gamma_refuse_loop_exhausted=True,
                  qtype="chain_deep", iter_count=3) is None
    # iter<2 too early → KEEP
    assert _faith(cfg, "invalid", gamma_refuse_loop_exhausted=True,
                  qtype="semantic_rich", iter_count=1) is None
    # not exhausted + not valid → KEEP (real check runs)
    assert _faith(cfg, "invalid", gamma_refuse_loop_exhausted=False,
                  qtype="semantic_rich", iter_count=3) is None


def test_d_v2_flag_off_is_legacy():
    cfg = IterativeConfig(use_faithfulness_gamma_coord=False)
    assert _faith(cfg, "valid") is None
    assert _faith(cfg, "invalid", gamma_refuse_loop_exhausted=True,
                  qtype="semantic_rich", iter_count=5) is None


# ============================================================ NEW telemetry (v2)
# γ-valid yield per cohort + per DS (the MQ-38%% cascade diagnostic).
def test_gamma_valid_rate_per_qtype_and_ds():
    per_q = [
        {"qtype": "semantic_rich", "gamma_final_status": "valid"},
        {"qtype": "semantic_rich", "gamma_final_status": "invalid"},
        {"qtype": "chain_deep", "gamma_final_status": "valid"},
        {"qtype": "chain_deep", "gamma_final_status": None},  # cascade-fail counts
    ]
    by_q = rp._gamma_valid_rate_by(per_q, key="qtype")
    assert by_q["semantic_rich"] == {"valid": 1, "total": 2, "rate": 0.5}
    assert by_q["chain_deep"] == {"valid": 1, "total": 2, "rate": 0.5}
    by_ds = rp._gamma_valid_rate_by(per_q, const="data_wiki_musiqueFULL")
    assert by_ds["data_wiki_musiqueFULL"] == {"valid": 2, "total": 4, "rate": 0.5}


def test_ds_name_from_args_basename():
    class _A:
        data_dir = "/home/ubuntu/sandbox/data_wiki_musiqueFULL"
    assert rp._ds_name_from_args(_A()) == "data_wiki_musiqueFULL"


# =================================================================== PDD-v2
# Dup dropped ONLY on chain_deep+γ=valid; preserved on the semantic_rich bulk.
def test_pdd_v2_skip_on_chain_deep_valid():
    assert gamma_aware_pdd_should_skip(
        _pool(), "valid", "chain_deep", enabled=True) is True


def test_pdd_v2_preserve_on_semantic_rich_bulk():
    assert gamma_aware_pdd_should_skip(
        _pool(), "valid", "semantic_rich", enabled=True) is False
    # unclassified default (pip path) is also preserved (safe legacy 4-arm)
    assert gamma_aware_pdd_should_skip(_pool(), "valid", None, enabled=True) is False


def test_pdd_v2_kept_when_not_valid():
    assert gamma_aware_pdd_should_skip(
        _pool(), "partial", "chain_deep", enabled=True) is False
    assert gamma_aware_pdd_should_skip(
        _pool(), "invalid", "chain_deep", enabled=True) is False


def test_pdd_v2_flag_off_is_legacy():
    assert gamma_aware_pdd_should_skip(
        _pool(), "valid", "chain_deep", enabled=False) is False
