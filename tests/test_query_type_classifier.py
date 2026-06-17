# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Validation of mothrag.core.query_type_classifier.

Exercises the rule-based classifier on 20 hand-picked HotpotQA queries and
20 hand-picked 2WikiMultiHopQA queries (questions are public benchmark
samples; gold labels here are author-curated targets).

Calibration target:
  - 2Wiki:    ~60-70% bridge_entity
  - HotpotQA: ~30-40% bridge_entity
"""

from __future__ import annotations

import pytest

from mothrag.core.query_type_classifier import (
    classify_query,
    classify_with_features,
    count_named_entities,
    count_relations,
    has_chain,
    has_relation,
)


# 20 hand-picked HotpotQA-style multi-hop questions (mix of bridge / comparison /
# semantic). Author-curated expected labels for the router.
HOTPOTQA_SAMPLES = [
    # bridge-entity (chain-of-facts between two named entities)
    ("Who directed the film that starred Leonardo DiCaprio in Inception?", "bridge_entity"),
    ("What movie did the wife of Inception's director star in?", "bridge_entity"),
    ("Which company founded by Elon Musk owns Twitter?", "bridge_entity"),
    ("Who composed the soundtrack for the film directed by Christopher Nolan?", "bridge_entity"),
    # comparison / semantic-rich
    ("Were Scott Derrickson and Ed Wood of the same nationality?", "semantic_rich"),
    ("Are both Coronation Street and Casualty long-running British TV series?", "semantic_rich"),
    ("Which is older, the Eiffel Tower or the Statue of Liberty?", "semantic_rich"),
    ("How many storeys does the building where the 2003 Tour de France was announced have?", "semantic_rich"),
    # location / single-entity
    ("In what city is the Nusretiye Clock Tower located?", "semantic_rich"),
    ("What language is spoken in the country where Bytham Castle is located?", "semantic_rich"),
    ("What year was the magazine, that ran the article about the Nightcrawler comic book character, started?", "semantic_rich"),
    ("How many seasons has the show, on which the host of the show Pokemon Black & White is the announcer, run?", "semantic_rich"),
    # bridge with named-entity verbs (short)
    ("Who wrote the novel that was adapted into Blade Runner?", "bridge_entity"),
    ("Who married the actor who starred in Casablanca?", "bridge_entity"),
    # softer bridge but no clear relational verb
    ("What is the elevation of the city where Karen Joy Fowler was born?", "semantic_rich"),
    # purely semantic
    ("What is the largest mammal in the world?", "semantic_rich"),
    ("What does the phrase 'sui generis' mean?", "semantic_rich"),
    # bridge short
    ("Who directed Parasite?", "semantic_rich"),  # only 1 entity
    ("Which composer wrote the Ninth Symphony?", "semantic_rich"),  # 1 entity
    ("Who founded Microsoft?", "semantic_rich"),  # 1 entity
]

# 20 hand-picked 2WikiMultiHopQA-style questions. The dataset is heavy on
# Wikidata-style relational chains (X's spouse's profession / Y's director's
# nationality), which is exactly the bridge_entity regime.
TWOWIKI_SAMPLES = [
    # Classic 2Wiki bridge-entity templates
    ("Who is the spouse of the director of film The Dark Knight?", "bridge_entity"),
    ("What is the nationality of the director of Parasite?", "semantic_rich"),  # 1 entity, no rel-verb in list
    ("Who is the father of the director of The Godfather?", "bridge_entity"),
    ("Who is the mother of the actor who starred in Titanic?", "bridge_entity"),
    ("Who directed the film that won Best Picture at the 1995 Oscars?", "bridge_entity"),
    ("Who wrote the book that was adapted into The Shawshank Redemption?", "bridge_entity"),
    ("Who founded the company that produced the iPhone?", "bridge_entity"),
    ("Who composed the score for the film directed by Stanley Kubrick?", "bridge_entity"),
    ("Who is the predecessor of the prime minister who succeeded Tony Blair?", "bridge_entity"),
    ("Who is the spouse of the founder of Microsoft?", "bridge_entity"),
    ("Who is the child of the king who succeeded Henry VIII?", "bridge_entity"),
    ("Who married the daughter of Queen Elizabeth II?", "bridge_entity"),
    ("Who starred in the film directed by Quentin Tarantino in 1994?", "bridge_entity"),
    ("Who is the father of Princess Diana's husband?", "bridge_entity"),
    # Entity + relational, longer-form
    ("Who was the publisher of the book authored by George Orwell?", "bridge_entity"),
    # less bridge-y
    ("What is the capital of the country where Mount Everest is located?", "semantic_rich"),
    ("What university did the founder of Tesla attend?", "semantic_rich"),  # tesla is 1 entity
    ("Who was the first president of the United States?", "semantic_rich"),
    ("In what year did World War II end?", "semantic_rich"),
    ("Who is the current Pope?", "semantic_rich"),
]


def _accuracy_and_rate(samples: list[tuple[str, str]]) -> tuple[float, float]:
    correct = 0
    bridge_count = 0
    for q, gold in samples:
        pred = classify_query(q)
        if pred == gold:
            correct += 1
        if pred == "bridge_entity":
            bridge_count += 1
    return correct / len(samples), bridge_count / len(samples)


def test_hotpotqa_calibration() -> None:
    """HotpotQA-style queries: ~30-40% bridge_entity, accuracy >= 0.75."""
    acc, bridge_rate = _accuracy_and_rate(HOTPOTQA_SAMPLES)
    assert acc >= 0.75, f"HotpotQA accuracy too low: {acc:.2f}"
    assert 0.20 <= bridge_rate <= 0.50, f"HotpotQA bridge_rate off target: {bridge_rate:.2f}"


def test_twowiki_calibration() -> None:
    """2Wiki-style queries: ~55-80% bridge_entity, accuracy >= 0.85."""
    acc, bridge_rate = _accuracy_and_rate(TWOWIKI_SAMPLES)
    assert acc >= 0.85, f"2Wiki accuracy too low: {acc:.2f}"
    assert 0.55 <= bridge_rate <= 0.80, f"2Wiki bridge_rate off target: {bridge_rate:.2f}"


def test_features_basic() -> None:
    feat = classify_with_features("Who directed the film that starred Leonardo DiCaprio in Inception?")
    assert feat["label"] == "bridge_entity"
    assert feat["n_entities"] >= 2
    assert feat["n_relations"] >= 1
    assert feat["n_tokens"] < 15


def test_empty_input() -> None:
    assert classify_query("") == "semantic_rich"
    assert classify_query("   ") == "semantic_rich"


def test_long_query_falls_back_to_semantic() -> None:
    # >= 15 tokens should never be bridge_entity even with entities + relation
    q = ("Who directed the film that was based on the novel written by the "
         "author who married the daughter of the British prime minister?")
    assert classify_query(q) == "semantic_rich"


def test_count_named_entities_basic() -> None:
    assert count_named_entities("Who directed Inception?") == 1
    assert count_named_entities("Who is the spouse of the director of Inception?") >= 1
    n = count_named_entities("Did Leonardo DiCaprio star in Inception directed by Christopher Nolan?")
    assert n >= 2


def test_count_relations_and_chain() -> None:
    assert count_relations("Who directed Inception?") == 1
    assert count_relations("Who married the actor who starred in Casablanca?") >= 2
    assert has_chain("Who is the spouse of the director of Inception?") is True
    assert has_chain("Who is the spouse of Inception?") is False


def test_has_relation_basic() -> None:
    assert has_relation("Who married the actress?") is True
    assert has_relation("Who founded the company?") is True
    assert has_relation("What is the capital?") is False


if __name__ == "__main__":
    # Run as: python tests/test_query_type_classifier.py
    print("\n=== HotpotQA samples ===")
    correct = 0
    bridge = 0
    for q, gold in HOTPOTQA_SAMPLES:
        pred = classify_query(q)
        feat = classify_with_features(q)
        ok = "OK " if pred == gold else "MISS"
        if pred == gold:
            correct += 1
        if pred == "bridge_entity":
            bridge += 1
        print(f"{ok} pred={pred:14s} gold={gold:14s} ne={feat['n_entities']} rel={feat['n_relations']} chain={feat['has_chain']} tok={feat['n_tokens']:2d} | {q}")
    print(f"\nHotpotQA  acc={correct/len(HOTPOTQA_SAMPLES):.2f}  bridge_rate={bridge/len(HOTPOTQA_SAMPLES):.2f}  (target 30-40%)")

    print("\n=== 2WikiMultiHopQA samples ===")
    correct = 0
    bridge = 0
    for q, gold in TWOWIKI_SAMPLES:
        pred = classify_query(q)
        feat = classify_with_features(q)
        ok = "OK " if pred == gold else "MISS"
        if pred == gold:
            correct += 1
        if pred == "bridge_entity":
            bridge += 1
        print(f"{ok} pred={pred:14s} gold={gold:14s} ne={feat['n_entities']} rel={feat['n_relations']} chain={feat['has_chain']} tok={feat['n_tokens']:2d} | {q}")
    print(f"\n2Wiki     acc={correct/len(TWOWIKI_SAMPLES):.2f}  bridge_rate={bridge/len(TWOWIKI_SAMPLES):.2f}  (target 60-70%)")


# ---- sel_v2 chain_deep tests ----

from mothrag.core.query_type_classifier import (
    classify_query_v2,
    count_nested_np_depth,
    is_chain_deep,
)
from mothrag.core.selective_ensemble import route_by_query_type_v2


def test_count_nested_np_depth_simple() -> None:
    assert count_nested_np_depth("Who directed Inception?") == 0
    assert count_nested_np_depth("Who is the director of Inception?") == 1
    assert count_nested_np_depth("Who is the spouse of the director of Inception?") == 2


def test_count_nested_np_depth_3plus() -> None:
    q = "Who is the spouse of the founder of the company that built the bridge of Brooklyn?"
    assert count_nested_np_depth(q) >= 3


def test_is_chain_deep_via_np_depth() -> None:
    q = "Who is the spouse of the director of the film that won the Oscar of 1972?"
    assert is_chain_deep(q) is True


def test_is_chain_deep_via_many_relations() -> None:
    q = "Who founded, directed, produced and wrote that movie?"
    assert is_chain_deep(q) is True


def test_classify_query_v2_chain_deep_priority() -> None:
    q = "Who is the spouse of the director of the film of 1972?"
    assert classify_query_v2(q) == "chain_deep"


def test_classify_query_v2_falls_back_to_v1() -> None:
    # 1 of: bridge_entity territory not chain_deep
    q = "Who directed Inception?"
    assert classify_query_v2(q) in ("bridge_entity", "semantic_rich")


def test_classify_query_v1_unchanged() -> None:
    """Backward compat: sel_v1 still returns only 2 classes."""
    assert classify_query("Who is the spouse of the director of the film of 1972?") in (
        "bridge_entity", "semantic_rich")


def test_classify_with_features_includes_v2() -> None:
    feat = classify_with_features("Who is the spouse of the director of the film of 1972?")
    assert "label" in feat and "label_v2" in feat
    assert feat["label_v2"] == "chain_deep"
    assert feat["np_depth"] >= 3


def test_route_v2_chain_deep_uses_iter() -> None:
    final, reason = route_by_query_type_v2(
        v3bu_pred="x", dec_pred="y", iter_pred="z",
        question="Who is the spouse of the director of the film of 1972?")
    assert final == "z"
    assert reason == "router_v2:chain-deep-use-iter"


def test_route_v2_chain_deep_no_iter_falls_back_to_decompose() -> None:
    final, reason = route_by_query_type_v2(
        v3bu_pred="x", dec_pred="y", iter_pred=None,
        question="Who is the spouse of the director of the film of 1972?")
    assert final == "y"
    assert "chain-deep-no-iter" in reason

