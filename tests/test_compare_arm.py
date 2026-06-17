# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""M8 CompareArm (boolean-intersection retrieval).

Detection of the comparison_yes_no cluster, deterministic entity/attribute
extraction, and the balanced-union retrieval that guarantees BOTH compared
entities are covered. $0 LLM (rule-based + injected dense ann).
"""
from __future__ import annotations

import pytest

from mothrag.retrieval.specialist.compare_arm import (
    Candidate,
    CompareArm,
    CompareConfig,
    extract_comparison_attributes,
    extract_compared_entities,
    is_comparison_query,
)

# 20 comparison_yes_no queries (2W / MQ style) the cluster detector must fire on.
COMPARISON_QUERIES = [
    "Are the directors of both The Knockout Kid and One Nine Nine Four from the same country?",
    "Are Reign of Fire and The Last Witch Hunter both American films?",
    "Do both Albert Einstein and Niels Bohr have the same nationality?",
    "Are the authors of Pride and Prejudice and Jane Eyre both British?",
    "Were both World War One and World War Two started in the same decade?",
    "Is the director of Inception older than the director of Interstellar?",
    "Are both Paris and London capital cities?",
    "Did both The Beatles and The Rolling Stones form in the same year?",
    "Are Mount Everest and K2 located in the same country?",
    "Do The Godfather and Goodfellas share the same director?",
    "Were both Steve Jobs and Bill Gates born in the same year?",
    "Is Lake Superior larger than Lake Victoria?",
    "Are both Toyota and Honda Japanese companies?",
    "Do both Canada and Australia have the same head of state?",
    "Are the writers of Hamlet and Macbeth the same person?",
    "Were The Eiffel Tower and The Statue of Liberty built in the same century?",
    "Are both Serena Williams and Venus Williams professional tennis players?",
    "Did both Apollo Eleven and Apollo Thirteen launch from the same site?",
    "Are Real Madrid and Barcelona based in the same country?",
    "Do both Mercury and Venus orbit closer to the Sun than Earth?",
]

NON_COMPARISON_QUERIES = [
    "Who wrote Hamlet?",
    "What year was Albert Einstein born?",
    "Are dolphins mammals?",                 # boolean but no marker / 1 entity
    "Is the Eiffel Tower in Paris?",         # boolean, 2 entities, but no marker
    "Describe the plot of Inception.",       # not a question
    "Both of them left early.",              # marker but no boolean / no entities
]


@pytest.mark.parametrize("q", COMPARISON_QUERIES)
def test_comparison_cluster_detected(q):
    assert is_comparison_query(q) is True


@pytest.mark.parametrize("q", NON_COMPARISON_QUERIES)
def test_non_comparison_not_detected(q):
    assert is_comparison_query(q) is False


def test_all_comparison_queries_yield_two_plus_entities():
    for q in COMPARISON_QUERIES:
        ents = extract_compared_entities(q)
        assert len(ents) >= 2, f"<2 entities for: {q} -> {ents}"


# ---- entity / attribute extraction -----------------------------------------

def test_entity_extraction_canonical():
    q = "Are the directors of both The Knockout Kid and One Nine Nine Four from the same country?"
    assert extract_compared_entities(q) == ["The Knockout Kid", "One Nine Nine Four"]


def test_entity_extraction_quoted_and_between():
    assert extract_compared_entities('Are "alpha one" and "beta two" the same thing?') \
        == ["alpha one", "beta two"]
    ents = extract_compared_entities("Is there a difference between Alpha Co and Beta Co?")
    assert "Alpha Co" in ents and "Beta Co" in ents


def test_entity_extraction_injected_ner_wins():
    def ner(_q):
        return ["Entity X", "Entity Y", "Entity Z"]
    assert extract_compared_entities("anything at all?", ner=ner) == \
        ["Entity X", "Entity Y", "Entity Z"]


def test_attribute_extraction():
    q = "Are the directors of both A B and C D from the same country?"
    attrs = extract_comparison_attributes(q)
    assert "director" in attrs and "country" in attrs


# ---- the arm ---------------------------------------------------------------

def _entity_keyed_ann(question, k):
    base = "kid" if "Knockout" in question else ("nnf" if "Nine" in question else "x")
    return [Candidate(f"{base}-{i}", question, 1.0 - 0.05 * i) for i in range(k)]


def test_balanced_union_interleaves_both_entities():
    q = "Are the directors of both The Knockout Kid and One Nine Nine Four from the same country?"
    arm = CompareArm(_entity_keyed_ann, config=CompareConfig(per_entity_top_k=5, final_top_k=6))
    res = arm.retrieve(q)
    assert res.fired is True
    # round-robin: both entities appear before either's 2nd doc.
    assert res.ranked_passage_ids[:2] == ["kid-0", "nnf-0"]
    assert "kid" in res.ranked_passage_ids[0] and "nnf" in res.ranked_passage_ids[1]
    assert len(res.ranked_passage_ids) == 6
    assert set(res.per_entity_passage_ids) == {"The Knockout Kid", "One Nine Nine Four"}


def test_arm_does_not_fire_below_two_entities():
    arm = CompareArm(lambda q, k: [Candidate("x", "", 1.0)])
    res = arm.retrieve("Is Paris a city?")     # <2 entities
    assert res.fired is False
    assert res.ranked_passage_ids == []


def test_arm_applicable_matches_detector():
    arm = CompareArm(_entity_keyed_ann)
    assert arm.applicable(COMPARISON_QUERIES[0]) is True
    assert arm.applicable("Who wrote Hamlet?") is False


def test_arm_resilient_to_per_entity_retrieval_failure():
    calls = {"n": 0}

    def flaky(question, k):
        calls["n"] += 1
        if "Knockout" in question:
            raise RuntimeError("dense down for entity A")
        return [Candidate(f"nnf-{i}", question, 1.0) for i in range(k)]

    q = "Are the directors of both The Knockout Kid and One Nine Nine Four from the same country?"
    res = CompareArm(flaky, config=CompareConfig(per_entity_top_k=3, final_top_k=4)).retrieve(q)
    # one entity failed -> still returns the other entity's docs (graceful).
    assert res.fired is True
    assert all("nnf" in pid for pid in res.ranked_passage_ids)


def test_final_top_k_cap_respected():
    arm = CompareArm(_entity_keyed_ann, config=CompareConfig(per_entity_top_k=10, final_top_k=4))
    res = arm.retrieve(COMPARISON_QUERIES[0])
    assert len(res.ranked_passage_ids) == 4


def test_requires_ann_retrieve():
    with pytest.raises(ValueError):
        CompareArm(None)
