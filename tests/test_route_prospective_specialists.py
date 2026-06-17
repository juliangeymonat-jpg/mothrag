# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""route_prospective M8 specialist slot-router wiring (live).

Default-OFF identity (router None ⇒ byte-identical generic decompose slot) is
already proven by the full suite. This pins the *enabled* wiring offline (no
LLM): `_build_specialist_router` constructs a pool-safe router from the live
pipeline surface, and a specialist that doesn't fire (empty retrieval) degrades
to the generic decompose slot — the pool stays exactly 4.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "route_prospective",
    Path(__file__).resolve().parents[1] / "scripts" / "route_prospective.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


class _StubPipe:
    """Minimal pipeline surface `_build_specialist_router` consumes.

    `retrieve` returns no hits → the specialists never fire → no reader_client
    call happens (so this stays fully offline / LLM-free)."""

    reader_client = object()
    chunk_ids: list = []
    chunks_by_id: dict = {}

    def retrieve(self, query):
        return ([], "dense", 1.0)


def test_build_specialist_router_is_pool_safe_and_enabled():
    router = mod._build_specialist_router(_StubPipe(), "reader-model", 10)
    assert router.enabled
    assert router.pool_keys() == ("v3bu", "decompose", "iter", "iter_dup_a")
    assert len(router.pool_keys()) == 4


def test_comparison_specialist_declines_to_generic_when_no_hits():
    router = mod._build_specialist_router(_StubPipe(), "reader-model", 10)
    # comparison cohort, but empty retrieval → CompareArm can't fire → generic.
    result, decision = router.run_decompose_slot(
        "Are X and Y both located in the same country?",
        generic_runner=lambda q, **k: {"pred": "GENERIC"})
    assert result == {"pred": "GENERIC"}
    assert not decision.is_specialist
    assert router.pool_keys() == ("v3bu", "decompose", "iter", "iter_dup_a")


def test_non_cohort_question_uses_generic_slot():
    router = mod._build_specialist_router(_StubPipe(), "reader-model", 10)
    result, decision = router.run_decompose_slot(
        "What color is the sky?",
        generic_runner=lambda q, **k: {"pred": "G2"})
    assert result == {"pred": "G2"}
    assert not decision.is_specialist
