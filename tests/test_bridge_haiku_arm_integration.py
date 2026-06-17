# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""bridge_arm integration into the MOTHRAG arm pool."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import mothrag.core.selective_ensemble as se
from mothrag.core.api import MothRAG, _DEFAULTS


# ---- arbitration: bridge_pred as a 4th ensemble candidate -----------------

@pytest.fixture
def spy_apply_c7(monkeypatch):
    seen = {}

    def _spy(chosen, candidates, **kw):
        seen["chosen"] = chosen
        seen["candidates"] = list(candidates)
        return None

    monkeypatch.setattr(se, "apply_c7", _spy)
    return seen


def _patch_qtype(monkeypatch, label):
    import mothrag.core.query_type_classifier as qtc
    monkeypatch.setattr(qtc, "classify_query_v2", lambda q: label)


def test_bridge_pred_wins_on_bridge_entity(monkeypatch, spy_apply_c7):
    _patch_qtype(monkeypatch, "bridge_entity")
    chosen, reason, _ = se.arbitrate_with_c7(
        "v3bu ans", "dec ans", "q", iter_pred="iter ans",
        use_router_v2=True, bridge_pred="BRIDGE ANSWER",
    )
    assert chosen == "BRIDGE ANSWER"
    assert reason == "bridge_arm:multi-hop-primary"
    # bridge_pred is also added to the C7 candidate set
    assert "BRIDGE ANSWER" in spy_apply_c7["candidates"]


def test_bridge_pred_wins_on_chain_deep(monkeypatch, spy_apply_c7):
    _patch_qtype(monkeypatch, "chain_deep")
    chosen, reason, _ = se.arbitrate_with_c7(
        "v", "d", "q", iter_pred="i", use_router_v2=True,
        bridge_pred="BRIDGE",
    )
    assert chosen == "BRIDGE"


def test_bridge_pred_not_primary_on_semantic_rich(monkeypatch, spy_apply_c7):
    _patch_qtype(monkeypatch, "semantic_rich")
    chosen, reason, _ = se.arbitrate_with_c7(
        "v3bu", "dec", "q", iter_pred="iter", use_router_v2=True,
        bridge_pred="BRIDGE",
    )
    # standard routing wins; bridge is only a C7 candidate, not chosen
    assert chosen != "BRIDGE"
    assert "BRIDGE" in spy_apply_c7["candidates"]


def test_uncertain_bridge_pred_ignored(monkeypatch, spy_apply_c7):
    _patch_qtype(monkeypatch, "bridge_entity")
    chosen, reason, _ = se.arbitrate_with_c7(
        "v", "dec ans", "q", iter_pred="i", use_router_v2=True,
        bridge_pred="Not in passages.",   # uncertain -> not chosen
    )
    assert chosen != "Not in passages."


def test_no_bridge_pred_is_byte_identical(monkeypatch, spy_apply_c7):
    """bridge_pred=None (default) -> behaviour unchanged, no extra candidate."""
    _patch_qtype(monkeypatch, "bridge_entity")
    chosen, reason, _ = se.arbitrate_with_c7(
        "v", "dec ans", "q", iter_pred="i", use_router_v2=True,
    )
    assert chosen == "dec ans"   # router_v2:bridge-force-decompose
    assert len(spy_apply_c7["candidates"]) == 3  # v3bu, dec, iter (no bridge)


# ---- config default + _arm_bridge graceful behaviour ----------------------

def test_use_bridge_arm_default_off():
    assert _DEFAULTS.get("use_bridge_arm") is False


def _stub_rag(config):
    rag = MothRAG.__new__(MothRAG)
    rag.config = {**_DEFAULTS, **config}

    class _StubRetriever:
        def retrieve(self, q, *, top_k):
            return [SimpleNamespace(doc_id=f"p{i}", text=f"passage {i}", score=1.0 - i * 0.1)
                    for i in range(top_k)]

    class _StubReader:
        def read(self, q, passages):
            return f"ANSWER from {len(passages)} passages"

    rag.retriever = _StubRetriever()
    rag.reader = _StubReader()
    return rag


def test_arm_bridge_returns_answer_and_pids():
    rag = _stub_rag({"use_bridge_arm": True})
    ans, pids = rag._arm_bridge("a multi hop question", ["seed"],
                                q_emb=[0.0], top_k=5)
    assert ans.startswith("ANSWER from")     # reader read the bridge passages
    assert isinstance(pids, list)
    assert all(isinstance(p, str) for p in pids)


def test_arm_bridge_survives_retriever_failure():
    rag = _stub_rag({"use_bridge_arm": True})

    class _BoomRetriever:
        def retrieve(self, q, *, top_k):
            raise RuntimeError("index down")

    rag.retriever = _BoomRetriever()
    ans, pids = rag._arm_bridge("q", ["seed"], q_emb=[0.0], top_k=5)
    assert ans == "" and pids == []          # never breaks the main pipeline
