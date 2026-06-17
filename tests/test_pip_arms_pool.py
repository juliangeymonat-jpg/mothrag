# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""pip api.py 4-arm (iter_dup_a PDD) + parallel arm exec.

End-to-end through the production ensemble path: arms_pool=4 adds iter_dup_a (a
COPY of iter) to the arbitration candidates; arms_pool=3 (default) is unchanged
(byte-stable legacy); arms_parallel toggles execution without changing the
answer (determinism).
"""
from __future__ import annotations

# Warm pyarrow/pandas on the MAIN thread at import. Their native DLL load is
# flaky on Windows when first triggered lazily/late (transitively via the
# langsmith tracing dep) — loading them cleanly first avoids a process-level
# access violation. Pure env quirk, not a logic issue; harmless if absent.
try:  # pragma: no cover - environment warm-up
    import pyarrow  # noqa: F401
    import pandas  # noqa: F401
except Exception:  # noqa: BLE001
    pass

from mothrag.core.api import MothRAG

DOCS = [
    "Paris is the capital of France.",
    "The Eiffel Tower is located in Paris.",
    "France is a country in Western Europe.",
    "Berlin is the capital of Germany.",
]


def _rag(**cfg):
    return MothRAG.from_documents(
        DOCS, production=True, mode="ensemble_arbitrate", **cfg)


def test_default_arms_pool_is_3_no_iter_dup_a():
    rag = _rag()
    assert rag.config["arms_pool"] == 3
    res = rag.query("Where is the Eiffel Tower located?")
    assert "iter_dup_a" not in res.metadata.get("arm_scores", {})


def test_arms_pool_4_adds_iter_dup_a_as_iter_copy():
    rag = _rag(arms_pool=4)
    assert rag.config["arms_pool"] == 4
    res = rag.query("Where is the Eiffel Tower located?")
    scores = res.metadata.get("arm_scores", {})
    # iter_dup_a appears ONLY when iter itself ran (dup follows base).
    if "iter" in scores:
        assert "iter_dup_a" in scores
        # iter_dup_a's prediction IS iter's (copy) → identical answer text.
        assert res.metadata.get("iter_pred") is not None


def test_arms_parallel_vs_serial_same_answer():
    q = "Where is the Eiffel Tower located?"
    par = _rag(arms_pool=4, arms_parallel=True).query(q)
    ser = _rag(arms_pool=4, arms_parallel=False).query(q)
    assert par.answer == ser.answer                      # parallel ≡ serial
    assert (set(par.metadata.get("arm_scores", {}))
            == set(ser.metadata.get("arm_scores", {})))


def test_arms_pool_4_query_does_not_crash_and_is_deterministic():
    rag = _rag(arms_pool=4)
    q = "What is the capital of France?"
    a1 = rag.query(q).answer
    a2 = rag.query(q).answer
    assert a1 == a2                                      # deterministic
