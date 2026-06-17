# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""M8 DecomposeArm 2.0 (chain-coherent decomposition).

Detection of the compositional cluster, the chain-coherence validator (the
"2.0" novelty: a decomposition that drifts → fallback, not a confidently-wrong
recompose), placeholder chaining, and the full backend-agnostic arm.
"""
from __future__ import annotations

import pytest

from mothrag.retrieval.specialist.compare_arm import Candidate
from mothrag.retrieval.specialist.decompose_arm_v2 import (
    DecomposeArmV2,
    DecomposeConfig,
    HopResult,
    _resolve_placeholders,
    chain_key_terms,
    contains_compositional_markers,
    needs_decomposition,
    validate_chain_coherence,
)


# ---- detection -------------------------------------------------------------

COMPOSITIONAL = [
    "What year did a director of a North Korean cinema film get kidnapped?",
    "Who is the spouse of the director of a Pixar film?",
    "In which country was the author of a Booker-winning novel born?",
    "What is the capital of the country where the inventor of the telephone was born?",
]
SINGLE_HOP = [
    "What is the capital of France?",
    "Who directed Inception?",
    "When was Einstein born?",
]


@pytest.mark.parametrize("q", COMPOSITIONAL)
def test_compositional_detected(q):
    assert contains_compositional_markers(q) is True
    assert needs_decomposition(q, gold_n_estimate=1) is True


@pytest.mark.parametrize("q", SINGLE_HOP)
def test_single_hop_not_compositional(q):
    assert contains_compositional_markers(q) is False


def test_gold_n_estimate_primary_signal():
    assert needs_decomposition("What is the capital of France?", gold_n_estimate=3) is True
    assert needs_decomposition("What is the capital of France?", gold_n_estimate=1) is False


# ---- chain key-term extraction ---------------------------------------------

def test_chain_key_terms_titlecase_numbers_content():
    terms = chain_key_terms("Shin Sang-ok was kidnapped in 1978")
    assert "shin sang-ok" in terms
    assert "1978" in terms
    assert "kidnapped" in terms          # significant content token
    assert "was" not in terms            # stopword excluded


def test_chain_key_terms_injected_ner():
    terms = chain_key_terms("anything", ner=lambda t: ["Pulgasari"])
    assert "pulgasari" in terms


# ---- chain coherence validator (the 2.0 novelty) ---------------------------

def _hop(sq, ans, ctx):
    return HopResult(sub_question=sq, answer=ans, passage_ids=["p"], context_texts=ctx)


def test_coherent_chain_all_links_hold():
    hops = [
        _hop("q1", "Pulgasari", ["Pulgasari is a North Korean film."]),
        _hop("q2", "Shin Sang-ok", ["Pulgasari was directed by Shin Sang-ok."]),
        _hop("q3", "1978", ["Shin Sang-ok was kidnapped in 1978."]),
    ]
    c = validate_chain_coherence(hops)
    assert c.coherent is True
    assert c.links == [True, True]
    assert c.broken_at is None


def test_broken_chain_flagged_at_first_break():
    hops = [
        _hop("q1", "Pulgasari", ["Pulgasari is a North Korean film."]),
        _hop("q2", "Christopher Nolan", ["Inception was directed by Christopher Nolan."]),
        _hop("q3", "2010", ["Inception was released in 2010."]),
    ]
    c = validate_chain_coherence(hops)
    assert c.coherent is False
    assert c.links[0] is False           # Pulgasari never appears in hop2 context
    assert c.broken_at == 0


def test_single_hop_trivially_coherent():
    assert validate_chain_coherence([_hop("q", "a", ["ctx"])]).coherent is True
    assert validate_chain_coherence([]).coherent is True


def test_min_overlap_threshold():
    hops = [
        _hop("q1", "Pulgasari", ["A film named Pulgasari exists."]),
        _hop("q2", "x", ["Pulgasari is referenced once here."]),
    ]
    assert validate_chain_coherence(hops, min_overlap=1).coherent is True
    assert validate_chain_coherence(hops, min_overlap=5).coherent is False


# ---- placeholder chaining --------------------------------------------------

def test_resolve_placeholders():
    assert _resolve_placeholders("Who directed {prev}?", ["Pulgasari"]) == "Who directed Pulgasari?"
    assert _resolve_placeholders("link {1} and {2}", ["A", "B"]) == "link A and B"
    assert _resolve_placeholders("no placeholder", ["A"]) == "no placeholder"
    assert _resolve_placeholders("Who directed {prev}?", []) == "Who directed ?"


# ---- the arm ---------------------------------------------------------------

def _retr(q, k):
    if "directed" in q.lower() or "Pulgasari" in q:
        return [Candidate("p2", "Pulgasari was directed by Shin Sang-ok", 1.0)]
    return [Candidate("p1", "Pulgasari is a North Korean film", 1.0)]


def _ans(q, ctx):
    return "Pulgasari" if "film" in q.lower() else "Shin Sang-ok"


def _deco(q):
    return ["What is a North Korean film?", "Who directed {prev}?"]


def test_arm_chains_and_recomposes_coherently():
    arm = DecomposeArmV2(_retr, decomposer=_deco, answerer=_ans,
                         config=DecomposeConfig(per_subq_top_k=2, final_top_k=4))
    r = arm.retrieve("What year did a director of a NK film get kidnapped?")
    assert r.hops[1].sub_question == "Who directed Pulgasari?"   # {prev} resolved
    assert r.coherence.coherent is True
    assert r.fallback is False
    assert r.ranked_passage_ids == ["p1", "p2"]                  # balanced union


def test_arm_signals_fallback_on_broken_chain():
    def bad_ans(q, ctx):
        return "Pulgasari" if "film" in q.lower() else "Unrelated Person"

    def bad_retr(q, k):
        # hop2 context never mentions hop1's answer -> chain breaks
        if "directed" in q.lower():
            return [Candidate("p9", "Some unrelated movie by Unrelated Person", 1.0)]
        return [Candidate("p1", "Pulgasari is a North Korean film", 1.0)]

    arm = DecomposeArmV2(bad_retr, decomposer=_deco, answerer=bad_ans)
    r = arm.retrieve("compositional q")
    assert r.coherence.coherent is False
    assert r.fallback is True


def test_arm_fallback_when_undecomposable():
    # no decomposer -> single hop -> fallback (let M7 handle it).
    arm = DecomposeArmV2(_retr)
    r = arm.retrieve("What is the capital of France?")
    assert r.sub_questions == ["What is the capital of France?"]
    assert r.fallback is True


def test_arm_applicable_and_requires_retriever():
    assert DecomposeArmV2(_retr).applicable("anything", gold_n_estimate=3) is True
    with pytest.raises(ValueError):
        DecomposeArmV2(None)


def test_arm_resilient_to_hop_retrieval_failure():
    def flaky(q, k):
        if "directed" in q.lower():
            raise RuntimeError("dense down")
        return [Candidate("p1", "Pulgasari is a North Korean film", 1.0)]

    arm = DecomposeArmV2(flaky, decomposer=_deco, answerer=_ans)
    r = arm.retrieve("compositional q")        # must not raise
    assert "p1" in r.ranked_passage_ids
