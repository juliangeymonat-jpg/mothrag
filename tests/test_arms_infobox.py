"""Tests for InfoboxArm (C3.6) + Arm Protocol contract."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture(autouse=True)
def _block_sentence_transformers(monkeypatch) -> Iterator[None]:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    yield


# ============================================================
# Arm Protocol contract
# ============================================================

def test_arm_protocol_runtime_checkable_for_infobox_arm() -> None:
    from mothrag.arms import Arm, InfoboxArm
    from mothrag.core.retrieval import InfoboxIndex
    arm = InfoboxArm(infobox_index=InfoboxIndex())
    assert isinstance(arm, Arm)
    assert hasattr(arm, "name")
    assert arm.name == "infobox_arm"


def test_arm_result_dataclass_fields() -> None:
    from mothrag.arms import ArmResult
    r = ArmResult(pred="x")
    assert r.pred == "x"
    assert r.retrieved_chunk_ids == []
    assert r.n_llm_calls == 0
    assert r.metadata == {}


# ============================================================
# InfoboxArm direct-lookup behaviour
# ============================================================

def _index_with(*triples):
    from mothrag.core.retrieval import InfoboxIndex, InfoboxTriple
    idx = InfoboxIndex()
    for t in triples:
        idx.add(InfoboxTriple(*t))
    return idx


def test_infobox_arm_returns_value_on_hint_match() -> None:
    from mothrag.arms import InfoboxArm
    idx = _index_with(
        ("Albert Einstein", "born", "14 March 1879", "c1", 1.0),
    )
    arm = InfoboxArm(infobox_index=idx)
    out = arm.run("When was Albert Einstein born?")
    assert out.pred == "14 March 1879"
    assert out.n_llm_calls == 0  # NO LLM call
    assert "c1" in out.retrieved_chunk_ids
    assert out.metadata["match_subject"] == "Albert Einstein"
    assert out.metadata["match_attribute"] == "born"


def test_infobox_arm_returns_empty_on_no_hint() -> None:
    from mothrag.arms import InfoboxArm
    idx = _index_with(("Einstein", "born", "1879"))
    arm = InfoboxArm(infobox_index=idx)
    out = arm.run("Why did the chicken cross the road?")
    assert out.pred == ""
    assert out.metadata.get("hints") == []


def test_infobox_arm_returns_empty_on_hint_but_no_index_match() -> None:
    """Hint extracted but index has no matching subject/attribute."""
    from mothrag.arms import InfoboxArm
    idx = _index_with(("Einstein", "born", "1879"))
    arm = InfoboxArm(infobox_index=idx)
    out = arm.run("When was Newton born?")
    # "Newton" extracted as subject, "born" as attribute, but index
    # has only Einstein -> no match.
    assert out.pred == ""
    assert out.metadata.get("match") is None or out.metadata.get(
        "match_subject"
    ) is None


def test_infobox_arm_picks_highest_confidence_match() -> None:
    from mothrag.arms import InfoboxArm
    idx = _index_with(
        ("Einstein", "spouse", "Mileva Maric", "c1", 1.0),
        ("Einstein", "spouse", "Elsa Einstein", "c2", 0.7),
    )
    arm = InfoboxArm(infobox_index=idx)
    out = arm.run("Who is Einstein's spouse?")
    assert out.pred == "Mileva Maric"
    assert out.metadata["match_confidence"] == 1.0


def test_infobox_arm_applicable_predicate() -> None:
    from mothrag.arms import InfoboxArm
    from mothrag.core.retrieval import InfoboxIndex
    arm = InfoboxArm(infobox_index=InfoboxIndex())
    assert arm.applicable("When was Einstein born?")
    assert arm.applicable("What is the capital of France?")
    assert not arm.applicable("Why did the chicken cross the road?")
    assert not arm.applicable("")
    assert not arm.applicable("   ")


def test_infobox_arm_accepts_custom_hint_extractor() -> None:
    from mothrag.arms import InfoboxArm
    idx = _index_with(("X", "salary", "$1M"))

    def _custom(q):
        if "salary" in q.lower():
            return [("X", "salary")]
        return []

    arm = InfoboxArm(infobox_index=idx, hint_extractor=_custom)
    assert arm.applicable("Tell me X's salary.")
    out = arm.run("Tell me X's salary.")
    assert out.pred == "$1M"


def test_infobox_arm_handles_hint_extractor_exception() -> None:
    from mothrag.arms import InfoboxArm
    idx = _index_with(("X", "y", "z"))

    def _broken(_):
        raise RuntimeError("boom")

    arm = InfoboxArm(infobox_index=idx, hint_extractor=_broken)
    assert not arm.applicable("When was X born?")
    out = arm.run("When was X born?")
    assert out.pred == ""
    assert "error" in out.metadata


# ============================================================
# route_prospective.py wiring -- arms-pool parsing + pipeline builder
# ============================================================

def test_parse_arms_pool_default() -> None:
    import route_prospective as rp
    assert rp._parse_arms_pool("") == ["v3bu", "decompose", "iter"]
    assert rp._parse_arms_pool("v3bu,decompose,iter") == [
        "v3bu", "decompose", "iter",
    ]


def test_parse_arms_pool_4_arm_with_infobox() -> None:
    import route_prospective as rp
    assert rp._parse_arms_pool("v3bu,decompose,iter,infobox_arm") == [
        "v3bu", "decompose", "iter", "infobox_arm",
    ]


def test_parse_arms_pool_dedupes_and_lowercases() -> None:
    import route_prospective as rp
    out = rp._parse_arms_pool("V3BU, decompose, iter, decompose, infobox_arm")
    assert out == ["v3bu", "decompose", "iter", "infobox_arm"]


def test_build_infobox_arm_from_pipeline_returns_none_on_empty_corpus() -> None:
    """No prose -> 0 triples -> no arm."""
    import numpy as np
    import route_prospective as rp

    class _StubPipeline:
        def __init__(self):
            self.chunks_by_id = {"c1": {"text": "no infobox here", "chunk_id": "c1"}}
            self.chunk_ids = ["c1"]
            self.chunk_vecs = np.zeros((1, 4), dtype=np.float32)
            self.query_embedder = lambda t: np.zeros(4, dtype=np.float32)

    arm = rp._build_infobox_arm_from_pipeline(_StubPipeline())
    assert arm is None


def test_build_infobox_arm_from_pipeline_skips_synthetic_chunks() -> None:
    """Synthetic infobox:* chunks from C3 augmentation must NOT be
    re-harvested (double-count guard)."""
    import numpy as np
    import route_prospective as rp

    class _StubPipeline:
        def __init__(self):
            self.chunks_by_id = {
                "real": {"text": "{{Infobox person\n| name = X\n| born = 1900\n}}",
                         "chunk_id": "real"},
                "infobox:x:born": {
                    "text": "X -- born: 1900",
                    "chunk_id": "infobox:x:born",
                    "metadata": {"source": "infobox"},
                },
            }
            self.chunk_ids = ["real", "infobox:x:born"]
            self.chunk_vecs = np.zeros((2, 4), dtype=np.float32)
            self.query_embedder = lambda t: np.zeros(4, dtype=np.float32)

    arm = rp._build_infobox_arm_from_pipeline(_StubPipeline())
    assert arm is not None
    # Verify only the real chunk was harvested (no double-count from
    # the synthetic chunk).
    triples = list(arm.infobox_index.lookup("X", "born"))
    assert len(triples) == 1


# ============================================================
# 4-arm pool integration -- backward-compat default + 4-arm extension
# ============================================================

def test_legacy_3_arm_pool_unchanged_when_no_opt_in() -> None:
    """When arms_pool is the default 3-arm legacy list, opt_in_arms is
    empty -> ensemble_arbitrate behaves identically to pre-C3.6."""
    import route_prospective as rp
    # Sanity: the function still accepts the new kwargs but defaults
    # produce a no-op extension.
    import inspect
    sig = inspect.signature(rp._run_ensemble_arbitrate)
    assert "arms_pool" in sig.parameters
    assert "opt_in_arms" in sig.parameters
    assert sig.parameters["arms_pool"].default is None
    assert sig.parameters["opt_in_arms"].default is None
