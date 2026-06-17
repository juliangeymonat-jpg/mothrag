# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tripartite judge batching + PIT fusion verification.

These tests pin the Bacellar §3.4-§3.5 contract: a SINGLE batched judge call
scores the whole pool (shared q+b+e1+e2 prefix, candidates appended), and PIT
fusion ranks with alpha=0.1 by default.
"""
from __future__ import annotations

import json

import pytest

from mothrag.retrieval.bridge_haiku.pit_fusion import pit_fuse, rank_candidates
from mothrag.retrieval.bridge_haiku.tripartite_judge import TripartiteJudge
from mothrag.retrieval.bridge_haiku.types import BridgeConfig, BridgeStats


class _CapturingJudge(TripartiteJudge):
    """Records the single prompt + call count."""

    def __init__(self, scores_json):
        super().__init__(require_backend=False)
        self._scores_json = scores_json
        self.calls = 0
        self.last_user = None

    def _call(self, system, user, *, max_tokens=1024, temperature=0.0):
        self.calls += 1
        self.last_user = user
        return self._scores_json, 500, 40

    def _cost(self, a, b):
        return 0.003


def test_judge_scores_20_candidates_in_single_call():
    pool = [f"candidate passage number {i}" for i in range(20)]
    scores = list(range(20))  # 0..19, clamped to [0,10] by parser
    j = _CapturingJudge(json.dumps(scores))
    stats = BridgeStats()
    out, n_in, n_out, cost = j.score("q", "bridge b", "e1", "e2", pool,
                                     stats=stats)
    assert j.calls == 1                      # ONE batched call for all 20
    assert len(out) == 20                    # aligned with the pool
    assert stats.n_judge_calls == 1
    assert all(0.0 <= s <= 10.0 for s in out)


def test_judge_prompt_shares_prefix_and_appends_all_candidates():
    pool = [f"passage_{i}" for i in range(20)]
    j = _CapturingJudge(json.dumps([5] * 20))
    j.score("the question", "the bridge", "Curie", "Nobel", pool)
    u = j.last_user
    # prefix (question/bridge/entities) appears ONCE...
    assert u.count("the question") == 1
    assert u.count("the bridge") == 1
    assert "Curie" in u and "Nobel" in u
    # ...and every candidate is appended, enumerated [1]..[20]
    for i in range(20):
        assert f"passage_{i}" in u
    assert "[1]" in u and "[20]" in u


def test_pit_fusion_default_alpha_is_0_1():
    assert BridgeConfig().alpha == 0.1
    # judge-dominant: A tops judge, bottoms svo; with alpha=0.1 A fuses higher.
    fused = pit_fuse([10.0, 0.0], [0.0, 10.0], alpha=BridgeConfig().alpha)
    assert fused[0] > fused[1]


def test_pit_ranking_judge_dominates_at_alpha_0_1():
    ids = ["a", "b", "c"]
    # svo order is c>b>a; judge order is a>b>c. alpha=0.1 -> judge wins -> a first.
    ranked = rank_candidates(ids, judge_scores=[9.0, 5.0, 1.0],
                             svo_scores=[1.0, 5.0, 9.0], alpha=0.1)
    assert ranked == ["a", "b", "c"]
