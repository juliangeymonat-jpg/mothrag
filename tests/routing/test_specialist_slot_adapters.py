# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Specialist -> decompose-slot adapters.

Verifies the M8 retrieval-shaper specialists are adapted into decompose-slot
candidate dicts by SUBSTITUTION (swap the retrieval feeding the slot, share the
reader), and that every decline path returns None so the slot router falls back
to the generic decompose arm (pool stays 4).
"""
from __future__ import annotations

from types import SimpleNamespace

from mothrag.routing import SpecialistSlotRouter
from mothrag.routing.specialist_slot_adapters import (
    make_reader_slot_reader,
    make_specialist_slot_runner,
)


# ---- fakes -----------------------------------------------------------------

class _FakeSpecialist:
    """Retrieval-shaper stub: .retrieve(q) -> result(ranked_passage_ids, fired/fallback)."""

    def __init__(self, *, ids=("p1", "p2"), fired=True, fallback=False, raises=False):
        self._ids = list(ids)
        self._fired = fired
        self._fallback = fallback
        self._raises = raises
        self.calls = []

    def retrieve(self, question):
        self.calls.append(question)
        if self._raises:
            raise RuntimeError("boom")
        return SimpleNamespace(
            ranked_passage_ids=self._ids, fired=self._fired, fallback=self._fallback)


def _echo_read_slot(pred="answer"):
    seen = {}

    def read_slot(question, ids):
        seen["question"] = question
        seen["ids"] = list(ids)
        return {"pred": pred, "n_llm_calls": 1, "prompt_tokens": 0,
                "completion_tokens": 0, "latency_s": 0.0}

    read_slot.seen = seen
    return read_slot


# ---- runner: fires -> slot dict + provenance -------------------------------

def test_runner_fires_returns_slot_dict_with_provenance():
    spec = _FakeSpecialist(ids=["pA", "pB"])
    read_slot = _echo_read_slot(pred="Paris")
    runner = make_specialist_slot_runner(specialist=spec, read_slot=read_slot,
                                         name="compare_arm")
    out = runner("Are X and Y the same?")
    assert out["pred"] == "Paris"
    assert out["retrieved_chunk_ids"] == ["pA", "pB"]
    assert out["metadata"]["specialist_slot"] == "compare_arm"
    assert out["metadata"]["specialist_passage_ids"] == ["pA", "pB"]
    assert read_slot.seen["ids"] == ["pA", "pB"]          # read the specialist's ids


# ---- runner: every decline path -> None (generic fallback) -----------------

def test_runner_not_fired_returns_none():
    runner = make_specialist_slot_runner(
        specialist=_FakeSpecialist(fired=False), read_slot=_echo_read_slot(),
        name="compare_arm")
    assert runner("q") is None


def test_runner_fallback_returns_none():
    runner = make_specialist_slot_runner(
        specialist=_FakeSpecialist(fallback=True), read_slot=_echo_read_slot(),
        name="decompose_arm_v2")
    assert runner("q") is None


def test_runner_empty_ids_returns_none():
    runner = make_specialist_slot_runner(
        specialist=_FakeSpecialist(ids=[]), read_slot=_echo_read_slot(),
        name="compare_arm")
    assert runner("q") is None


def test_runner_specialist_raises_returns_none():
    runner = make_specialist_slot_runner(
        specialist=_FakeSpecialist(raises=True), read_slot=_echo_read_slot(),
        name="compare_arm")
    assert runner("q") is None


def test_runner_reader_no_pred_returns_none():
    runner = make_specialist_slot_runner(
        specialist=_FakeSpecialist(), read_slot=lambda q, ids: {"pred": ""},
        name="compare_arm")
    assert runner("q") is None


# ---- make_reader_slot_reader: fetch_texts -> reader.read -------------------

def test_reader_slot_reader_happy_path():
    reader = SimpleNamespace(read=lambda q, texts: f"read[{len(texts)}]")
    read_slot = make_reader_slot_reader(
        reader=reader, fetch_texts=lambda ids: [f"text:{i}" for i in ids])
    out = read_slot("q", ["p1", "p2", "p3"])
    assert out["pred"] == "read[3]"
    assert out["retrieved_chunk_ids"] == ["p1", "p2", "p3"]
    assert out["n_llm_calls"] == 1


def test_reader_slot_reader_empty_texts_and_failures_return_none():
    reader_ok = SimpleNamespace(read=lambda q, texts: "x")
    # empty texts
    assert make_reader_slot_reader(reader=reader_ok, fetch_texts=lambda ids: [])(
        "q", ["p1"]) is None
    # fetch_texts raises
    def boom(ids):
        raise RuntimeError("store down")
    assert make_reader_slot_reader(reader=reader_ok, fetch_texts=boom)("q", ["p1"]) is None
    # reader raises
    reader_bad = SimpleNamespace(read=lambda q, texts: (_ for _ in ()).throw(RuntimeError()))
    assert make_reader_slot_reader(reader=reader_bad, fetch_texts=lambda ids: ["t"])(
        "q", ["p1"]) is None


# ---- end-to-end: adapter injected into SpecialistSlotRouter ----------------

def test_router_substitutes_with_adapter_on_cohort():
    spec = _FakeSpecialist(ids=["c1", "c2"])
    compare_runner = make_specialist_slot_runner(
        specialist=spec, read_slot=_echo_read_slot(pred="compared"),
        name="compare_arm")

    generic_calls = []
    def generic(q, **k):
        generic_calls.append(q)
        return {"pred": "generic"}

    router = SpecialistSlotRouter(
        compare_arm=compare_runner,
        is_comparison=lambda q: True, is_compositional=lambda q: False,
        enabled=True,
    )
    result, decision = router.run_decompose_slot(
        "Are X and Y the same?", generic_runner=generic)
    assert result["pred"] == "compared"               # specialist filled the slot
    assert result["metadata"]["specialist_slot"] == "compare_arm"
    assert decision.is_specialist and decision.arm_name == "compare_arm"
    assert not generic_calls                           # generic NOT run → pool 4


def test_router_falls_back_to_generic_when_specialist_declines():
    # specialist on-cohort but does not fire → router must use generic (pool 4).
    compare_runner = make_specialist_slot_runner(
        specialist=_FakeSpecialist(fired=False), read_slot=_echo_read_slot(),
        name="compare_arm")
    router = SpecialistSlotRouter(
        compare_arm=compare_runner, is_comparison=lambda q: True,
        is_compositional=lambda q: False, enabled=True)
    result, decision = router.run_decompose_slot(
        "Are X and Y the same?", generic_runner=lambda q, **k: {"pred": "generic"})
    assert result["pred"] == "generic"
    assert not decision.is_specialist
