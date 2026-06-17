# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""GeminiANNRetriever + end-to-end BridgeArm integration.

Uses a deterministic keyword-bag fake embedder (offline, no API) so the real
ANN cosine search drives the four retrieval stages, with the LLM stages
mocked. Proves the full pipeline wires together against a real ANN.
"""
from __future__ import annotations

import numpy as np
import pytest

from mothrag.retrieval.bridge_haiku.ann import GeminiANNRetriever, build_gemini_ann
from mothrag.retrieval.bridge_haiku.bridge_arm import BridgeArm
from mothrag.retrieval.bridge_haiku.types import BridgeConfig, Candidate


# ---- deterministic offline embedder (bag-of-keywords over a fixed vocab) ---

class _KeywordEmbedder:
    """Embeds text as an L2-normalised count vector over a fixed vocabulary.

    Texts sharing keywords get high cosine similarity, so the ANN behaves
    meaningfully + deterministically without any network call.
    """

    VOCAB = ["curie", "radium", "france", "paris", "nobel", "physics",
             "born", "warsaw", "filler", "unrelated"]

    def embed(self, texts, *, batch_size: int = 1):
        idx = {w: i for i, w in enumerate(self.VOCAB)}
        out = []
        for t in texts:
            v = np.zeros(len(self.VOCAB), dtype=np.float32)
            for w in t.lower().replace(".", " ").replace(",", " ").split():
                if w in idx:
                    v[idx[w]] += 1.0
            n = np.linalg.norm(v) or 1.0
            out.append(v / n)
        return out


def _corpus():
    # gold "g" only matches via radium/nobel (hop-2 / entity), NOT the
    # question's hop-1 surface (curie/france), so hop-1 won't rank it top-5.
    return [
        {"passage_id": "b", "text": "Marie Curie physics born Warsaw"},     # bridge
        {"passage_id": "g", "text": "radium nobel"},                         # gold
        {"passage_id": "f1", "text": "filler unrelated"},
        {"passage_id": "f2", "text": "filler unrelated filler"},
        {"passage_id": "f3", "text": "paris france filler"},
    ]


# ---- ANN unit behaviour ---------------------------------------------------

def test_ann_retrieve_orders_by_cosine():
    ann = build_gemini_ann(_corpus(), embedder=_KeywordEmbedder())
    out = ann.retrieve("radium nobel", top_k=3)
    assert isinstance(out[0], Candidate)
    assert out[0].passage_id == "g"          # exact keyword match ranks #1
    assert out[0].ann_score > out[-1].ann_score


def test_ann_precomputed_doc_vectors_shape_validated():
    with pytest.raises(ValueError, match="doc_vectors rows"):
        GeminiANNRetriever(_corpus(), embedder=_KeywordEmbedder(),
                           doc_vectors=np.zeros((2, 10), dtype=np.float32))


def test_ann_empty_query_and_topk():
    ann = build_gemini_ann(_corpus(), embedder=_KeywordEmbedder())
    assert ann.retrieve("", 5) == []
    assert ann.retrieve("radium", 0) == []
    assert len(ann) == 5


def test_ann_callable_interface():
    ann = build_gemini_ann(_corpus(), embedder=_KeywordEmbedder())
    # BridgeArm calls ann_retrieve(query, top_k); the instance is callable.
    out = ann("curie", 2)
    assert out and out[0].passage_id == "b"


# ---- end-to-end with real ANN + mocked LLM stages -------------------------

class _StubSVO:
    def generate(self, q, bridge, *, n=3, stats=None):
        if stats is not None:
            stats.add_call("svo", 10, 5, 0.0005)
        return ["radium discovery"], 10, 5, 0.0005   # SVO query surfaces gold via 'radium'


class _StubEntities:
    def extract(self, q, bridge, stats=None):
        if stats is not None:
            stats.add_call("entity", 8, 3, 0.0005)
        return ("Curie", "nobel"), 8, 3, 0.0005      # e2='nobel' surfaces gold


class _StubJudge:
    def score(self, q, b, e1, e2, texts, *, lo=0.0, hi=10.0, max_tokens=1024,
              stats=None):
        if stats is not None:
            stats.add_call("judge", 50, 10, 0.002)
        # judge favours the gold passage 'radium nobel'
        return [10.0 if "radium nobel" in t else 1.0 for t in texts], 50, 10, 0.002


def test_full_pipeline_real_ann_surfaces_gold():
    ann = build_gemini_ann(_corpus(), embedder=_KeywordEmbedder())
    arm = BridgeArm(
        ann,                                 # real cosine ANN as the callable
        config=BridgeConfig(hop1_top_k=3, svo_top_k=5, entity_top_k=5,
                            pool_cap=20, final_top_k=5),
        svo_generator=_StubSVO(),
        entity_extractor=_StubEntities(),
        judge=_StubJudge(),
        require_backend=False,
    )
    res = arm.retrieve("Marie Curie physics work")  # hop-1 surface = curie/physics
    assert res.bridge_passage_id == "b"           # hop-1 top-1 (curie/physics)
    assert res.entities == ("Curie", "nobel")
    assert "g" in res.ranked_passage_ids          # gold surfaced via SVO+entity ANN
    assert res.ranked_passage_ids[0] == "g"       # judge-favoured gold ranks #1
    assert res.stats.n_judge_calls == 1
    assert res.pool_size >= 2
