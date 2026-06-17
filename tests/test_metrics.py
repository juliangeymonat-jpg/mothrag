# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for canonical HotpotQA metrics."""

from mothrag.eval.metrics import (
    normalize_answer, em_score, f1_score, recall_at_k, ndcg_at_k,
)


def test_normalize_answer_yang_2018():
    assert normalize_answer("The Boston Red Sox.") == "boston red sox"
    assert normalize_answer("a 1984") == "1984"
    assert normalize_answer("YES") == "yes"


def test_em_score():
    assert em_score("Boston", "the Boston") == 1.0
    assert em_score("yes", "Yes,") == 1.0
    assert em_score("", "boston") == 0.0


def test_f1_score():
    assert f1_score("Pavel Alexandrov", "Pavel Alexandrov") == 1.0
    assert f1_score("", "Pavel") == 0.0
    f = f1_score("Pavel Sergeyevich Alexandrov", "Pavel Alexandrov")
    assert f > 0.6  # 2/3 of pred tokens overlap, 2/2 gold tokens overlap


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["a", "d"], k=2) == 0.5
    assert recall_at_k(["a", "b"], [], k=2) == 0.0
    assert recall_at_k(["b", "c", "a"], ["a"], k=3) == 1.0


def test_ndcg_at_k_sanity():
    score = ndcg_at_k(["a", "b"], ["a"], k=2)
    assert score == 1.0  # perfect ranking
    score_bad = ndcg_at_k(["x", "a"], ["a"], k=2)
    assert 0 < score_bad < 1.0
