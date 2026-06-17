# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Crash hardening — proof-tree parser tolerates string-shaped provenance.

Observed failure mode: the reader emitted ``"sources": ["Cesar Millan"]`` — a
list of bare STRINGS — and ``_source_from_dict`` called ``.get`` on a str →
AttributeError → the whole query died to ``pred=""``/``f1=0``. Same family as
the singular-"source" schema drift (see test_gamma_source_key_fix.py): LLM
provenance shapes drift; the parser must degrade to γ-invalid, never crash the
query.

Covers: str entries in sources, bare-str "sources" value, single-object
"sources" value, non-dict junk in "steps", and non-dict junk source entries.
"""
from __future__ import annotations

from mothrag.aurora.adapter import (_source_from_dict, _step_from_dict,
                                    prooftree_from_dict)


def _step(sources_payload) -> dict:
    return {"step": 1, "rule": "lookup", "subject": "Daddy",
            "object": "Cesar Millan",
            "claim_text": "Daddy worked with Cesar Millan",
            "sources": sources_payload}


def test_string_source_entry_becomes_doc_id():
    # the exact observed shape: list of bare strings
    step = _step_from_dict(_step(["Cesar Millan"]))
    assert len(step.sources) == 1
    assert step.sources[0].doc_id == "Cesar Millan"
    assert step.sources[0].span_text == ""          # unverifiable, NOT a crash


def test_bare_string_sources_value_wrapped_not_char_iterated():
    # "sources": "Cesar Millan" (no list at all) must NOT explode into
    # one source per character
    step = _step_from_dict(_step("Cesar Millan"))
    assert len(step.sources) == 1
    assert step.sources[0].doc_id == "Cesar Millan"


def test_single_object_sources_value_wrapped():
    step = _step_from_dict(_step({"doc_id": "doc_x", "span_text": "spam"}))
    assert len(step.sources) == 1
    assert step.sources[0].doc_id == "doc_x"
    assert step.sources[0].span_text == "spam"


def test_non_dict_junk_source_entry_coerced():
    step = _step_from_dict(_step([42]))
    assert step.sources[0].doc_id == "42"


def test_source_from_dict_str_direct():
    s = _source_from_dict("  doc_y  ")
    assert s.doc_id == "doc_y"
    assert s.char_offset is None


def test_non_dict_steps_dropped_not_crashed():
    tree = prooftree_from_dict({
        "steps": ["a stray narration string", _step(["Cesar Millan"])],
        "naturalized_answer": "Cesar Millan",
        "is_complete": True,
    })
    assert len(tree.steps) == 1                      # junk step dropped
    assert tree.steps[0].sources[0].doc_id == "Cesar Millan"
    assert tree.naturalized_answer == "Cesar Millan"


def test_dict_sources_unchanged_no_regression():
    step = _step_from_dict(_step([{"doc_id": "doc_z",
                                   "span_text": "verbatim span"}]))
    assert step.sources[0].doc_id == "doc_z"
    assert step.sources[0].span_text == "verbatim span"
