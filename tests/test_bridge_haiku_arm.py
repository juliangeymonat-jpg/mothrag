# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""BridgeArm orchestrator (stub ANN + injected mock LLM stages)."""
from __future__ import annotations

import pytest

from mothrag.retrieval.bridge_haiku.bridge_arm import BridgeArm
from mothrag.retrieval.bridge_haiku.types import BridgeConfig, Candidate


# ---- fakes ---------------------------------------------------------------

def _make_ann(table):
    """table: dict[query_substring -> list[(pid, score)]]. Default empty."""
    def ann(query, top_k):
        for key, rows in table.items():
            if key in query:
                return [Candidate(pid, f"text of {pid}", score)
                        for pid, score in rows[:top_k]]
        return []
    return ann


class _FakeSVO:
    def __init__(self, queries):
        self.queries = queries

    def generate(self, q, bridge, *, n=3, stats=None):
        if stats is not None:
            stats.add_call("svo", 100, 20, 0.0005)
        return self.queries, 100, 20, 0.0005


class _FakeEntities:
    def __init__(self, e1, e2):
        self.e = (e1, e2)

    def extract(self, q, bridge, stats=None):
        if stats is not None:
            stats.add_call("entity", 80, 15, 0.0005)
        return self.e, 80, 15, 0.0005


class _FakeJudge:
    def __init__(self, score_by_pid):
        self.score_by_pid = score_by_pid

    def score(self, q, b, e1, e2, texts, *, lo=0.0, hi=10.0,
              max_tokens=1024, stats=None):
        if stats is not None:
            stats.add_call("judge", 300, 40, 0.002)
        # texts are "text of <pid>"; map back to pid for deterministic scores
        scores = []
        for t in texts:
            pid = t.replace("text of ", "")
            scores.append(float(self.score_by_pid.get(pid, 5.0)))
        return scores, 300, 40, 0.002


def _arm(ann_table, svo_queries, entities, judge_scores, **cfg_kw):
    cfg = BridgeConfig(**cfg_kw)
    return BridgeArm(
        _make_ann(ann_table),
        config=cfg,
        svo_generator=_FakeSVO(svo_queries),
        entity_extractor=_FakeEntities(*entities),
        judge=_FakeJudge(judge_scores),
        require_backend=False,
    )


# ---- tests ---------------------------------------------------------------

def test_empty_question_returns_empty():
    arm = _arm({}, [], ("", ""), {})
    res = arm.retrieve("")
    assert res.ranked_passage_ids == []
    assert res.bridge_passage_id is None


def test_no_hop1_returns_empty():
    arm = _arm({"zzz": [("p", 1.0)]}, [], ("", ""), {})
    res = arm.retrieve("question with no ann hit")
    assert res.bridge_passage_id is None
    assert res.ranked_passage_ids == []


def test_full_pipeline_ranks_judge_favoured_gold_top():
    # hop-1 on the question; SVO + entity expansions surface a gold passage
    # 'g' that hop-1 ranked low. Judge scores g highest -> g ranks #1.
    ann = {
        "Marie Curie": [("b", 0.9), ("x1", 0.5), ("x2", 0.4)],   # bridge=b
        "discovered":  [("x3", 0.6), ("g", 0.30)],                # SVO query
        "France":      [("g", 0.30), ("x4", 0.2)],                # entity e2
    }
    arm = _arm(
        ann,
        svo_queries=["Curie discovered radium"],
        entities=("Curie", "France"),
        judge_scores={"g": 10.0, "b": 4.0, "x1": 1.0, "x2": 1.0,
                      "x3": 2.0, "x4": 1.0},
        hop1_top_k=3, final_top_k=5,
    )
    res = arm.retrieve("When did Marie Curie work in France?")
    assert res.bridge_passage_id == "b"
    assert res.entities == ("Curie", "France")
    assert res.ranked_passage_ids[0] == "g"   # judge-favoured gold wins
    assert "g" in res.ranked_passage_ids
    # cost accumulated across the 3 LLM stages
    assert res.stats.n_judge_calls == 1
    assert res.stats.estimated_cost_usd == pytest.approx(0.0005 + 0.0005 + 0.002)


def test_pool_dedup_and_cap():
    # every stage returns overlapping pids; pool must dedup + respect cap.
    ann = {
        "q": [("a", 0.9), ("b", 0.8)],
        "svo": [("b", 0.7), ("c", 0.6)],
        "ent": [("c", 0.5), ("d", 0.4)],
    }
    arm = _arm(ann, ["svo query"], ("ent", ""), {},
               hop1_top_k=2, pool_cap=3, final_top_k=5)
    res = arm.retrieve("q")
    # union of {a,b} {b,c} {c,d} capped at 3 -> a,b,c (dedup, order-preserving)
    assert res.pool_size == 3
    assert set(res.ranked_passage_ids) <= {"a", "b", "c"}


def test_budget_guard_skips_judge_and_uses_dense_order():
    # max_cost_usd below the SVO+entity spend so the judge stage is skipped;
    # ranking then follows dense ann_score (PIT_judge == PIT_svo).
    ann = {"q": [("hi", 0.9), ("lo", 0.1)]}
    arm = _arm(ann, ["svo"], ("e", ""), {"hi": 0.0, "lo": 10.0},
               hop1_top_k=2, max_cost_usd=0.0009, final_top_k=2)
    res = arm.retrieve("q")
    # judge NOT called (over budget after svo 0.0005 + entity 0.0005 = 0.001)
    assert res.stats.n_judge_calls == 0
    # dense order: hi (0.9) before lo (0.1) despite judge_scores favouring lo
    assert res.ranked_passage_ids == ["hi", "lo"]
