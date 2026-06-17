# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Parser accepts singular "source" OR plural "sources" in a proof-tree
lookup step.

The active `full` proof-tree prompt emits the SINGULAR "source" key; the parser
previously read only the plural "sources" → every lookup got sources=[] →
verifier "no source for lookup step" → γ=invalid ~100%. This fix makes the
parser tolerate both. The `llama` prompt (plural) path is unchanged.
"""
from __future__ import annotations

import json

from mothrag.aurora.adapter import _step_from_dict, parse_reader_prooftree_json
from mothrag.aurora.verifier import verify_proof_tree


_PASSAGES = [{"doc_id": "doc_inception",
              "text": "Inception is a 2010 science fiction film written and "
                      "directed by Christopher Nolan."}]


def _lookup(source_key: str) -> str:
    prov = ('"source": {"doc_id": "doc_inception", "span_text": "directed by '
            'Christopher Nolan"}') if source_key == "source" else (
            '"sources": [{"doc_id": "doc_inception", "span_text": "directed by '
            'Christopher Nolan"}]')
    return ('{"steps": [{"step": 1, "rule": "lookup", "predicate": "directed_by",'
            ' "subject": "Inception", "object": "Christopher Nolan",'
            ' "claim_text": "Inception was directed by Christopher Nolan", '
            + prov + '}], "naturalized_answer": "Christopher Nolan",'
            ' "is_complete": true}')


def test_singular_source_now_parsed():
    # the full-prompt singular "source" is wrapped into the sources list
    step = _step_from_dict(json.loads(_lookup("source"))["steps"][0])
    assert len(step.sources) == 1
    assert step.sources[0].doc_id == "doc_inception"


def test_full_style_tree_verifies_valid_after_fix():
    tree = parse_reader_prooftree_json(_lookup("source"))
    verify_proof_tree(tree, _PASSAGES)
    assert tree.steps[0].verifier_status == "valid"          # was invalid before the fix
    assert tree.steps[0].verifier_reason != "no source for lookup step"
    assert tree.overall_status == "valid"                    # was invalid before the fix


def test_llama_style_plural_sources_unchanged():
    tree = parse_reader_prooftree_json(_lookup("sources"))
    verify_proof_tree(tree, _PASSAGES)
    assert len(tree.steps[0].sources) == 1
    assert tree.overall_status == "valid"                    # no regression


def test_no_provenance_key_degrades_gracefully():
    # a lookup with neither key → empty sources → invalid (legacy behaviour, no raise)
    d = {"step": 1, "rule": "lookup", "subject": "x", "object": "y",
         "claim_text": "c"}
    step = _step_from_dict(d)
    assert step.sources == []


def test_negation_step_no_source_needed():
    # pure refusal tree (no lookup) → "refuse", unaffected by the source key fix
    raw = ('{"steps": [{"step": 1, "rule": "negation_as_failure",'
           ' "claim_text": "no passage supports the requested fact"}],'
           ' "naturalized_answer": "REFUSE_NO_PROOF", "is_complete": false}')
    tree = parse_reader_prooftree_json(raw)
    verify_proof_tree(tree, _PASSAGES)
    assert tree.overall_status == "refuse"
