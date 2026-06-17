# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""SVO / entity / tripartite-judge LLM modules (mocked Haiku)."""
from __future__ import annotations

import json

import pytest

from mothrag.retrieval.bridge_haiku.entity_extractor import (
    DualEntityExtractor,
    parse_entity_response,
)
from mothrag.retrieval.bridge_haiku.svo_generator import (
    SVOQueryGenerator,
    parse_svo_response,
)
from mothrag.retrieval.bridge_haiku.tripartite_judge import (
    TripartiteJudge,
    parse_judge_scores,
)
from mothrag.retrieval.bridge_haiku.types import BridgeStats


# ---- helpers: subclasses that stub the network seam -----------------------

def _svo(response: str, n_in=100, n_out=20):
    class _M(SVOQueryGenerator):
        def _call(self, system, user, *, max_tokens=1024, temperature=0.0):
            return response, n_in, n_out

        def _cost(self, n_in_, n_out_):
            return 0.0005
    return _M(require_backend=False)


def _entity(response):
    class _M(DualEntityExtractor):
        def _call(self, system, user, *, max_tokens=1024, temperature=0.0):
            return response, 80, 15

        def _cost(self, a, b):
            return 0.0005
    return _M(require_backend=False)


def _judge(response):
    class _M(TripartiteJudge):
        def _call(self, system, user, *, max_tokens=1024, temperature=0.0):
            return response, 300, 40

        def _cost(self, a, b):
            return 0.002
    return _M(require_backend=False)


# ---- parsers --------------------------------------------------------------

def test_parse_svo_dedup_and_cap():
    txt = json.dumps(["A is B", "a is b", "C is D", "E is F"])
    out = parse_svo_response(txt, max_n=2)
    assert out == ["A is B", "C is D"]  # dedup case-insensitive + cap 2


def test_parse_svo_handles_fences_and_garbage():
    assert parse_svo_response("```json\n[\"x is y\"]\n```", max_n=3) == ["x is y"]
    assert parse_svo_response("not json", max_n=3) == []
    assert parse_svo_response("", max_n=3) == []


def test_parse_entity_object():
    assert parse_entity_response('{"e1": "Curie", "e2": "France"}') == ("Curie", "France")
    assert parse_entity_response('```json\n{"e1":"A","e2":"B"}\n```') == ("A", "B")
    assert parse_entity_response("garbage") == ("", "")
    assert parse_entity_response('{"e1": "only"}') == ("only", "")


def test_parse_judge_scores_clamp_pad_truncate():
    # too few -> padded with midpoint 5.0; values clamped to [0,10]
    assert parse_judge_scores("[12, -3, 7]", n=4) == [10.0, 0.0, 7.0, 5.0]
    # too many -> truncated
    assert parse_judge_scores("[1,2,3,4,5]", n=2) == [1.0, 2.0]
    # garbage -> all midpoints
    assert parse_judge_scores("nope", n=3) == [5.0, 5.0, 5.0]
    assert parse_judge_scores("[1, \"x\", 3]", n=3) == [1.0, 5.0, 3.0]


# ---- modules --------------------------------------------------------------

def test_svo_generate_returns_queries_and_tracks_cost():
    gen = _svo(json.dumps(["Newton discovered gravity", "gravity affects orbits"]))
    stats = BridgeStats()
    qs, n_in, n_out, cost = gen.generate("q?", "bridge text", n=3, stats=stats)
    assert qs == ["Newton discovered gravity", "gravity affects orbits"]
    assert stats.n_svo_calls == 1
    assert stats.estimated_cost_usd == pytest.approx(0.0005)


def test_svo_generate_fallback_to_question_on_empty():
    gen = _svo("not parseable")
    qs, *_ = gen.generate("the original question", "b")
    assert qs == ["the original question"]


def test_entity_extract():
    ext = _entity('{"e1": "Marie Curie", "e2": "Nobel Prize"}')
    stats = BridgeStats()
    (e1, e2), *_ = ext.extract("q", "bridge", stats=stats)
    assert (e1, e2) == ("Marie Curie", "Nobel Prize")
    assert stats.n_entity_calls == 1


def test_entity_extract_requires_question_and_bridge():
    ext = _entity('{"e1":"x","e2":"y"}')
    assert ext.extract("", "bridge")[0] == ("", "")
    assert ext.extract("q", "")[0] == ("", "")


def test_judge_scores_aligned_to_candidates():
    j = _judge("[9, 1, 5]")
    stats = BridgeStats()
    scores, *_ = j.score("q", "b", "e1", "e2",
                         ["cand A", "cand B", "cand C"], stats=stats)
    assert scores == [9.0, 1.0, 5.0]
    assert stats.n_judge_calls == 1
    assert stats.estimated_cost_usd == pytest.approx(0.002)


def test_judge_empty_pool_returns_empty():
    j = _judge("[]")
    scores, n_in, n_out, cost = j.score("q", "b", "e1", "e2", [])
    assert scores == []
    assert (n_in, n_out, cost) == (0, 0, 0.0)


def test_judge_failure_returns_midpoints_not_crash():
    class _Boom(TripartiteJudge):
        def _call(self, *a, **k):
            raise RuntimeError("api down")
    j = _Boom(require_backend=False)
    scores, *_ = j.score("q", "b", "e1", "e2", ["a", "b"])
    assert scores == [5.0, 5.0]  # neutral midpoint, no candidate killed
