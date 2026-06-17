# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""PIT fusion unit tests (pure, deterministic)."""
from __future__ import annotations

import pytest

from mothrag.retrieval.bridge_haiku.pit_fusion import (
    percentile_rank,
    pit_fuse,
    rank_candidates,
)


def test_percentile_rank_monotonic():
    pr = percentile_rank([10.0, 20.0, 30.0])
    assert pr[0] < pr[1] < pr[2]
    assert 0.0 < pr[0] < pr[2] < 1.0
    # symmetric around 0.5 for evenly spaced
    assert pr[1] == pytest.approx(0.5)


def test_percentile_rank_ties_get_mid_rank():
    pr = percentile_rank([5.0, 5.0, 5.0])
    assert pr == [0.5, 0.5, 0.5]
    pr2 = percentile_rank([1.0, 2.0, 2.0, 3.0])
    # the two 2.0s share the average rank -> equal pit
    assert pr2[1] == pr2[2]
    assert pr2[0] < pr2[1] < pr2[3]


def test_percentile_rank_edge_cases():
    assert percentile_rank([]) == []
    assert percentile_rank([42.0]) == [0.5]


def test_pit_fuse_weights_judge_heavier():
    # candidate A: top judge, bottom svo; candidate B: bottom judge, top svo.
    # With alpha=0.1, judge dominates -> A should fuse higher.
    fused = pit_fuse(judge_scores=[10.0, 0.0], svo_scores=[0.0, 10.0], alpha=0.1)
    assert fused[0] > fused[1]


def test_pit_fuse_alpha_zero_is_judge_only():
    fused = pit_fuse([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], alpha=0.0)
    pr = percentile_rank([1.0, 2.0, 3.0])
    assert fused == pytest.approx(pr)


def test_pit_fuse_alpha_one_is_svo_only():
    fused = pit_fuse([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], alpha=1.0)
    pr_svo = percentile_rank([3.0, 2.0, 1.0])
    assert fused == pytest.approx(pr_svo)


def test_pit_fuse_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        pit_fuse([1.0, 2.0], [1.0])


def test_pit_fuse_bad_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        pit_fuse([1.0], [1.0], alpha=1.5)


def test_rank_candidates_orders_by_fused_desc():
    ids = ["a", "b", "c"]
    # judge clearly prefers c > b > a; svo flat -> ranking follows judge.
    ranked = rank_candidates(ids, judge_scores=[1.0, 5.0, 9.0],
                             svo_scores=[0.0, 0.0, 0.0], alpha=0.1)
    assert ranked == ["c", "b", "a"]


def test_rank_candidates_top_k_truncates():
    ids = ["a", "b", "c", "d"]
    ranked = rank_candidates(ids, [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0],
                             alpha=0.1, top_k=2)
    assert ranked == ["d", "c"]


def test_rank_candidates_stable_on_ties():
    ids = ["a", "b", "c"]
    # all equal -> original order preserved (stable by index)
    ranked = rank_candidates(ids, [5.0, 5.0, 5.0], [5.0, 5.0, 5.0], alpha=0.1)
    assert ranked == ["a", "b", "c"]


def test_rank_candidates_misaligned_raises():
    with pytest.raises(ValueError):
        rank_candidates(["a", "b"], [1.0], [1.0, 2.0])
