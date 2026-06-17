# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""ChainFilter v0.1 (γ-weighted, hop-gated post-retrieval filter).

Pins the two IP twists + the safety contract, all offline (no LLM): the gate
fires only on input-feature multi-hop (iii), γ-band weighting promotes
confident-fact-supported passages and excludes anti-context facts (i), the
output is always a SUBSET of the input (reranker, never a 5th arm / never
fabricates), and every failure / non-fire path is a safe passthrough.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from mothrag.retrieval.chain_filter import ChainFilter, ChainFilterConfig
from mothrag.retrieval.chain_filter.chain_filter import (
    default_fact_filter,
    default_gamma_scorer,
    _candidate_supports,
)

MULTIHOP = {"np_depth": 3, "n_relations": 2, "has_chain": True}
SINGLEHOP = {"np_depth": 1, "n_relations": 0, "has_chain": False}


def _c(pid, text, ann):
    return NS(passage_id=pid, text=text, ann_score=ann)


def _extractor(mapping):
    return lambda text: mapping.get(text, [])


# ---- twist (iii): hop gate + default-OFF -----------------------------------

def test_default_off_is_passthrough():
    cf = ChainFilter(config=ChainFilterConfig(enabled=False, top_k_out=2),
                     classify_fn=lambda q: MULTIHOP)
    cands = [_c("p1", "a", 0.1), _c("p2", "b", 0.9), _c("p3", "c", 0.5)]
    assert [x.passage_id for x in cf.filter("q", cands)] == ["p1", "p2"]  # unchanged top-2


def test_single_hop_bypasses_even_when_enabled():
    fired = {"n": 0}

    def _ext(text):
        fired["n"] += 1
        return [["a", "r", "b"]]

    cf = ChainFilter(config=ChainFilterConfig(enabled=True, top_k_out=2),
                     classify_fn=lambda q: SINGLEHOP, triple_extractor=_ext)
    cands = [_c("p1", "a b", 0.1), _c("p2", "x", 0.9)]
    out = cf.filter("What color is the sky?", cands)
    assert [x.passage_id for x in out] == ["p1", "p2"]   # passthrough
    assert fired["n"] == 0                                # no extraction fired


def test_gate_fires_on_multihop():
    cf = ChainFilter(config=ChainFilterConfig(enabled=True),
                     classify_fn=lambda q: MULTIHOP)
    assert cf.gate_fires("q")
    cf_off = ChainFilter(config=ChainFilterConfig(enabled=False),
                         classify_fn=lambda q: MULTIHOP)
    assert not cf_off.gate_fires("q")


def test_hop_count_is_max_of_features():
    cf = ChainFilter(config=ChainFilterConfig(enabled=True))
    assert cf.hop_count({"np_depth": 3, "n_relations": 1}) == 3
    assert cf.hop_count({"np_depth": 0, "n_relations": 2}) == 2
    assert cf.hop_count({}) == 0


def test_gate_fires_on_has_chain_even_if_low_hopcount():
    cf = ChainFilter(config=ChainFilterConfig(enabled=True, hop_gate_min=5),
                     classify_fn=lambda q: {"np_depth": 1, "n_relations": 1,
                                            "has_chain": True})
    assert cf.gate_fires("q")   # has_chain is the secondary trigger


# ---- twist (i): γ-band weighting -------------------------------------------

def test_low_gamma_fact_excluded_high_gamma_promotes():
    # p_high supports a fact with HIGH γ; p_low supports the same fact but the
    # γ-scorer rates its support LOW → excluded → p_low not promoted by it.
    fact = ["paris", "capital of", "france"]
    cf = ChainFilter(
        config=ChainFilterConfig(enabled=True, top_k_out=1, alpha_ann=0.0),
        classify_fn=lambda q: MULTIHOP,
        triple_extractor=_extractor({
            "Paris capital of France": [fact],
            "Paris France": [fact],
        }),
        fact_filter=lambda q, facts: [fact],
        gamma_scorer=lambda q, f, text: 0.9 if "capital" in text else 0.1,
    )
    cands = [_c("p_low", "Paris France", 0.5),
             _c("p_high", "Paris capital of France", 0.5)]
    out = cf.filter("Paris capital France?", cands)
    assert out[0].passage_id == "p_high"   # HIGH-γ support wins, LOW-γ excluded


def test_band_weight_table():
    cf = ChainFilter(config=ChainFilterConfig(enabled=True))
    assert cf._band_weight(0.9) == 1.0    # HIGH
    assert cf._band_weight(0.5) == 0.5    # MID
    assert cf._band_weight(0.1) == 0.0    # LOW (anti-context, excluded)


# ---- deterministic helpers --------------------------------------------------

def test_default_gamma_scorer_bounded():
    g = default_gamma_scorer("q", ["paris", "x", "france"],
                             "Paris is the capital of France")
    assert 0.0 <= g <= 1.0 and g > 0.0
    assert default_gamma_scorer("q", ["zzz", "x", "yyy"], "nothing") == 0.0


def test_default_fact_filter_keeps_relevant_dedups_drops_zero():
    facts = [["paris", "capital", "france"], ["paris", "capital", "france"],
             ["tokyo", "capital", "japan"], ["foo", "bar", "baz"]]
    kept = default_fact_filter(2)("where is paris in france", facts)
    assert ["paris", "capital", "france"] in kept
    assert ["foo", "bar", "baz"] not in kept        # zero question overlap dropped
    assert len(kept) <= 2


def test_candidate_supports_requires_subject_and_object():
    assert _candidate_supports(["paris", "in", "france"],
                               "Paris is in France")
    assert not _candidate_supports(["paris", "in", "france"], "Paris only")


# ---- end-to-end + safety contract ------------------------------------------

def test_filter_reranks_supporting_passage_above_high_ann():
    fact = ["eiffel tower", "in", "paris"]
    cf = ChainFilter(
        config=ChainFilterConfig(enabled=True, top_k_out=2, alpha_ann=0.1),
        classify_fn=lambda q: MULTIHOP,
        triple_extractor=_extractor({"The Eiffel Tower is in Paris": [fact]}),
        fact_filter=lambda q, facts: [fact],
        gamma_scorer=lambda q, f, text: 0.9,
    )
    cands = [_c("noise", "irrelevant high-dense passage", 0.99),
             _c("hit", "The Eiffel Tower is in Paris", 0.05)]
    out = cf.filter("Where is the Eiffel Tower located in Paris?", cands)
    assert out[0].passage_id == "hit"           # chain validity beat raw dense


def test_output_is_always_a_subset_of_input():
    fact = ["a", "r", "b"]
    cf = ChainFilter(
        config=ChainFilterConfig(enabled=True, top_k_out=3),
        classify_fn=lambda q: MULTIHOP,
        triple_extractor=_extractor({"a b": [fact]}),
        fact_filter=lambda q, facts: [fact], gamma_scorer=lambda q, f, t: 0.8)
    cands = [_c(f"p{i}", "a b" if i == 0 else "x", 0.5) for i in range(5)]
    out = cf.filter("a then b?", cands)
    in_ids = {c.passage_id for c in cands}
    assert all(o.passage_id in in_ids for o in out)   # reranker: never fabricates
    assert len(out) <= 3


@pytest.mark.parametrize("extractor,ff", [
    (lambda t: (_ for _ in ()).throw(RuntimeError()), None),  # extractor raises
    (lambda t: [], None),                                     # no triples
    (lambda t: [["a", "r", "b"]], lambda q, f: []),           # no kept facts
])
def test_failure_paths_are_safe_passthrough(extractor, ff):
    cf = ChainFilter(
        config=ChainFilterConfig(enabled=True, top_k_out=2),
        classify_fn=lambda q: MULTIHOP, triple_extractor=extractor, fact_filter=ff)
    cands = [_c("p1", "a b", 0.1), _c("p2", "y", 0.9), _c("p3", "z", 0.5)]
    out = cf.filter("multi hop q", cands)
    assert [x.passage_id for x in out] == ["p1", "p2"]   # original top-2 preserved


def test_no_triple_extractor_passes_through():
    cf = ChainFilter(config=ChainFilterConfig(enabled=True, top_k_out=1),
                     classify_fn=lambda q: MULTIHOP, triple_extractor=None)
    cands = [_c("p1", "a", 0.1), _c("p2", "b", 0.9)]
    assert [x.passage_id for x in cf.filter("q", cands)] == ["p1"]


def test_classify_fn_sees_only_question():
    seen = []
    cf = ChainFilter(config=ChainFilterConfig(enabled=True),
                     classify_fn=lambda q: seen.append(q) or MULTIHOP)
    cf.gate_fires("only the question text")
    assert seen == ["only the question text"]   # anti-leak: no DS/gold input
