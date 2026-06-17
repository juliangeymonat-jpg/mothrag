# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Object-in-span relaxation (--use-relaxed-object-span).

The γ-verifier's object grounding required an object token in the narrow cited
SPAN — too strict for descriptive / boolean objects ("a fictional private
detective", "true") whose words aren't in the quoted fragment. That was the
dominant secondary γ-invalid mode observed in a small dry-run. RELAXED grounds
the object's tokens against the PASSAGE (the cited doc) instead. Default OFF ⇒
byte-identical legacy.
"""
from __future__ import annotations

import importlib.util as _u
import json
import pathlib
import sys

import pytest

from mothrag.aurora.adapter import parse_reader_prooftree_json as _parse
from mothrag.aurora.verifier import verify_proof_tree


# passage HAS the descriptive words; the cited span supports the relation but
# does NOT contain the object words → exact-in-span fails, passage-coverage passes.
_PASSAGES = [{"doc_id": "d",
              "text": "Sherlock Holmes is a fictional private detective. "
                      "He appears in The Adventure of the Seven Clocks."}]


def _tree(obj: str, span: str = "He appears in The Adventure of the Seven Clocks"):
    j = ('{"steps": [{"step": 1, "rule": "lookup", "subject": "Holmes",'
         ' "predicate": "is", "object": "' + obj + '", "claim_text": "c",'
         ' "sources": [{"doc_id": "d", "span_text": "' + span + '"}]}],'
         ' "naturalized_answer": "x", "is_complete": true}')
    return _parse(j)


def _status(obj, *, relaxed, span="He appears in The Adventure of the Seven Clocks"):
    t = _tree(obj, span)
    verify_proof_tree(t, _PASSAGES, use_relaxed_object_span=relaxed)
    return t.steps[0]


# 1 — descriptive phrase valid WITH flag (tokens in passage, not in span)
def test_descriptive_phrase_valid_with_flag():
    s = _status("a fictional private detective", relaxed=True)
    assert s.verifier_status == "valid"
    assert s.object_match_mode == "relaxed"


# 2 — same input invalid WITHOUT flag (legacy preserved)
def test_descriptive_phrase_invalid_without_flag():
    s = _status("a fictional private detective", relaxed=False)
    assert s.verifier_status == "invalid"
    assert "not grounded in span" in s.verifier_reason


# 3 — exact-in-span object valid either way (backwards-compat)
def test_exact_substring_valid_either_way():
    # object word IS in the cited span → both modes pass
    span = "Sherlock Holmes is a fictional private detective"
    off = _status("detective", relaxed=False, span=span)
    on = _status("detective", relaxed=True, span=span)
    assert off.verifier_status == "valid" and off.object_match_mode == "exact"
    assert on.verifier_status == "valid" and on.object_match_mode == "relaxed"


# 4 — orthogonal object invalid either way (relaxation is not a free pass)
def test_orthogonal_object_invalid_either_way():
    assert _status("quantum chromodynamics", relaxed=False).verifier_status == "invalid"
    assert _status("quantum chromodynamics", relaxed=True).verifier_status == "invalid"


# 5 — telemetry counters surfaced in per-q JSON + aggregate
class _AnyArgs:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, _):
        return None


def _load_rp():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "route_prospective.py"
    spec = _u.spec_from_file_location("rp_r2", path)
    mod = _u.module_from_spec(spec)
    saved = sys.argv
    sys.argv = ["route_prospective"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


def test_telemetry_counters_surfaced(tmp_path):
    rp = _load_rp()
    per_q = [
        {"qid": "q1", "qtype": "x", "em": 1.0, "f1": 0.5, "r_at_10": 0.5,
         "iterations_used": 1, "arm_used": "x", "n_llm_calls": 1,
         "object_relaxed_match_count": 2, "object_exact_match_count": 0},
        {"qid": "q2", "qtype": "x", "em": 1.0, "f1": 0.5, "r_at_10": 0.5,
         "iterations_used": 1, "arm_used": "x", "n_llm_calls": 1,
         "object_relaxed_match_count": 0, "object_exact_match_count": 3},
    ]
    out = tmp_path / "r2.json"
    rp._write_partial(out, per_q, _AnyArgs(use_relaxed_object_span=True),
                      partial=False)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "object_relaxed_match_count" in data["per_question"][0]
    c = data["summary"]["counters"]
    assert c["object_relaxed_match_total"] == 2
    assert c["object_exact_match_total"] == 3
