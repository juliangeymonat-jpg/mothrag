# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Faithfulness ↔ γ coordination (revised).

Two SAFE-skip branches: ``clean_valid`` (γ=valid ⇒ already verified) and
``exhausted_safe`` (γ-loop exhausted AND qtype != chain_deep AND iter>=2 ⇒ a
non-chain_deep cap-hit answer is safe). chain_deep cap-hit answers KEEP the
faithfulness gate (recovery cohort — an earlier variant skipped 60.7% on MQ,
too aggressive). Either skip treats the answer as grounded (fscore = judge_min_score)
so loop-exit is byte-identical to a faithfulness PASS. Default OFF ⇒ legacy.
"""
from __future__ import annotations

import importlib.util as _u
import inspect
import json
import pathlib
import sys

import pytest

from mothrag.eval.iterative_pipeline import (
    IterativeConfig, IterativeMothRAG, _faithfulness_gamma_coord_skip)


def test_clean_valid_branch():
    cfg = IterativeConfig(use_faithfulness_gamma_coord=True)
    assert _faithfulness_gamma_coord_skip(cfg, "valid") == "clean_valid"


def test_exhausted_safe_branch():
    cfg = IterativeConfig(use_faithfulness_gamma_coord=True)
    assert _faithfulness_gamma_coord_skip(
        cfg, "invalid", gamma_refuse_loop_exhausted=True,
        qtype="semantic_rich", iter_count=2) == "exhausted_safe"


def test_chain_deep_recovery_keeps_faithfulness():
    # chain_deep cap-hit = recovery cohort → KEEP the gate
    cfg = IterativeConfig(use_faithfulness_gamma_coord=True)
    assert _faithfulness_gamma_coord_skip(
        cfg, "invalid", gamma_refuse_loop_exhausted=True,
        qtype="chain_deep", iter_count=3) is None
    # iter<2 (too early) and not-exhausted also KEEP
    assert _faithfulness_gamma_coord_skip(
        cfg, "partial", gamma_refuse_loop_exhausted=True,
        qtype="semantic_rich", iter_count=1) is None
    assert _faithfulness_gamma_coord_skip(
        cfg, "partial", gamma_refuse_loop_exhausted=False,
        qtype="semantic_rich", iter_count=3) is None


def test_legacy_when_flag_off():
    cfg = IterativeConfig(use_faithfulness_gamma_coord=False)
    assert _faithfulness_gamma_coord_skip(cfg, "valid") is None
    assert _faithfulness_gamma_coord_skip(
        cfg, "invalid", gamma_refuse_loop_exhausted=True,
        qtype="semantic_rich", iter_count=9) is None


def test_skip_path_keeps_loop_exit_identical_source():
    # The skip path must NOT change loop-exit behaviour: it sets fscore to
    # judge_min_score (a faithfulness PASS) instead of calling the LLM judge.
    src = inspect.getsource(IterativeMothRAG.answer)
    assert "_faithfulness_gamma_coord_skip(" in src
    assert '"skipped_" + _faith_skip' in src
    assert "faithfulness_skipped_clean_valid = True" in src
    assert "faithfulness_skipped_exhausted_safe = True" in src
    # the real check is still the else-branch (legacy path preserved)
    assert "self._check_faithfulness(" in src


# --- telemetry surfaces in per-q JSON + aggregate --------------------------- #
class _AnyArgs:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, _):
        return None


def _load_rp():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "route_prospective.py"
    spec = _u.spec_from_file_location("rp_faith_coord", path)
    mod = _u.module_from_spec(spec)
    saved = sys.argv
    sys.argv = ["route_prospective"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


def test_faithfulness_coord_telemetry_in_json(tmp_path):
    rp = _load_rp()
    per_q = [
        {"qid": "q1", "qtype": "chain_deep", "em": 1.0, "f1": 0.5, "r_at_10": 0.5,
         "iterations_used": 3, "arm_used": "x", "n_llm_calls": 1, "gamma_final_status": "valid",
         "faithfulness_active": False, "faithfulness_skipped_clean_valid": True,
         "faithfulness_skipped_exhausted_safe": False},
        {"qid": "q2", "qtype": "semantic_rich", "em": 1.0, "f1": 0.5, "r_at_10": 0.5,
         "iterations_used": 3, "arm_used": "x", "n_llm_calls": 1, "gamma_final_status": "invalid",
         "faithfulness_active": False, "faithfulness_skipped_clean_valid": False,
         "faithfulness_skipped_exhausted_safe": True},
        {"qid": "q3", "qtype": "semantic_rich", "em": 1.0, "f1": 0.5, "r_at_10": 0.5,
         "iterations_used": 1, "arm_used": "x", "n_llm_calls": 1, "gamma_final_status": "invalid",
         "faithfulness_active": True, "faithfulness_skipped_clean_valid": False,
         "faithfulness_skipped_exhausted_safe": False},
    ]
    out = tmp_path / "fc.json"
    rp._write_partial(out, per_q,
                      _AnyArgs(use_faithfulness_gamma_coord=True,
                               data_dir="/data/data_wiki_musiqueFULL"),
                      partial=False)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "faithfulness_skipped_clean_valid" in data["per_question"][0]
    c = data["summary"]["counters"]
    assert c["faithfulness_active_total"] == 1
    assert c["faithfulness_skipped_clean_valid_total"] == 1
    assert c["faithfulness_skipped_exhausted_safe_total"] == 1
    # NEW v2 telemetry present
    assert c["gamma_valid_rate_per_qtype"]["chain_deep"]["rate"] == 1.0
    assert c["gamma_valid_rate_per_DS"]["data_wiki_musiqueFULL"]["total"] == 3
