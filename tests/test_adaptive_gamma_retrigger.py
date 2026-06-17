# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Conservative per-cohort γ-loop cap.

ONLY chain_deep=3 (deep chains were abandoned too early at the fixed 2); every
other cohort stays at the baseline 2 (semantic_rich=2 / bridge_entity=2 /
other=2). An earlier semantic_rich=1 variant starved the bulk cohort
(MQ -3.6pp / 2W -3.1pp). Default OFF ⇒ the legacy composite-aware fixed cap.
"""
from __future__ import annotations

import importlib.util as _u
import json
import pathlib
import sys

import pytest

from mothrag.eval.iterative_pipeline import IterativeConfig, _effective_gamma_retrigger_cap


def _cap(qtype, *, adaptive, composite=False, fixed=2, composite_cap=3):
    cfg = IterativeConfig(use_adaptive_gamma_retrigger=adaptive,
                          gamma_max_retrigger=fixed,
                          composite_gamma_max_retrigger=composite_cap)
    return _effective_gamma_retrigger_cap(cfg, qtype, composite)


def test_chain_deep_cap_is_3():
    assert _cap("chain_deep", adaptive=True) == 3


def test_semantic_rich_cap_is_2_baseline():
    # v2-CONSERVATIVE: the bulk cohort stays at baseline 2 (v1 used 1 → regression)
    assert _cap("semantic_rich", adaptive=True) == 2


def test_only_chain_deep_gets_extra_retry():
    # every non-chain_deep cohort resolves to the baseline 2
    assert _cap("bridge_entity", adaptive=True) == 2
    assert _cap("semantic_rich", adaptive=True) == 2
    assert _cap("something_else", adaptive=True) == 2     # fallback default
    assert _cap(None, adaptive=True) == 2                 # classifier failure
    assert _cap("chain_deep", adaptive=True) == 3         # the ONLY cohort lifted


def test_flag_off_is_legacy_fixed_cap():
    # adaptive OFF → the fixed cap regardless of cohort (byte-identical legacy)
    assert _cap("chain_deep", adaptive=False, fixed=2) == 2
    assert _cap("semantic_rich", adaptive=False, fixed=2) == 2
    # composite path is preserved when adaptive OFF
    assert _cap("chain_deep", adaptive=False, composite=True, composite_cap=3) == 3


# --- telemetry: per-q cap + aggregate distribution -------------------------- #
class _AnyArgs:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, _):
        return None


def _load_rp():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "route_prospective.py"
    spec = _u.spec_from_file_location("rp_adaptive", path)
    mod = _u.module_from_spec(spec)
    saved = sys.argv
    sys.argv = ["route_prospective"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


def _row(qid, cap):
    return {"qid": qid, "qtype": "x", "em": 1.0, "f1": 0.5, "r_at_10": 0.5,
            "iterations_used": 1, "arm_used": "x", "n_llm_calls": 1,
            "gamma_retrigger_cap_used": cap}


def test_retrigger_distribution_in_json(tmp_path):
    rp = _load_rp()
    # v2 caps: chain_deep=3, everything else=2 → a 3-vs-2 distribution
    per_q = [_row("q1", 3), _row("q2", 3), _row("q3", 2), _row("q4", 2)]
    out = tmp_path / "ad.json"
    rp._write_partial(out, per_q, _AnyArgs(use_adaptive_gamma_retrigger=True),
                      partial=False)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["per_question"][0]["gamma_retrigger_cap_used"] == 3
    dist = data["summary"]["counters"]["gamma_retrigger_cap_distribution"]
    # JSON object keys are strings; chain_deep=3 vs others=2
    assert dist == {"3": 2, "2": 2}
