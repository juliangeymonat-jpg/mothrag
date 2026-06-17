# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Unit tests — P4 abstain filter + P8 few-shot + P24 unified
ABSTAIN_MARKERS in pip-install api.py.

Verifies:
1. ``mothrag.core.abstain_markers.ABSTAIN_MARKERS`` is the canonical
   set, importable from both eval-pipeline and pip api.py.
2. ``mothrag.eval.iterative_pipeline.ABSTAIN_MARKERS`` is identical to
   the canonical (re-exported).
3. Pip ``_is_uncertain_answer`` uses the canonical list (catches
   "i don't know", "cannot answer", etc.).
4. ``MothRAG._arm_iter`` honours P4: abstain markers NOT propagated to
   the iterator accumulator.
5. ``MothRAG._arm_iter`` honours P8: when accumulator non-empty + few-shot
   ON (default), the augmented_q contains the few-shot synthesis frame.
6. ``MothRAG._arm_iter`` accepts ``config={"iter_use_few_shot": False}``
   for users opting out.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


def test_canonical_abstain_markers_importable():
    from mothrag.core.abstain_markers import ABSTAIN_MARKERS, is_abstain_marker
    assert isinstance(ABSTAIN_MARKERS, frozenset)
    # Canonical strings the eval pipeline (P24) honours:
    for required in ("not in passages", "i don't know", "i do not know",
                      "unknown", "no answer", "cannot answer"):
        assert required in ABSTAIN_MARKERS
    # Pip-install historical alias "none":
    assert "none" in ABSTAIN_MARKERS
    # Helper sanity:
    assert is_abstain_marker("I DON'T KNOW") is True   # case-insensitive
    assert is_abstain_marker("  not in passages  ") is True  # whitespace-trimmed
    assert is_abstain_marker(None) is True
    assert is_abstain_marker("") is True
    assert is_abstain_marker("   ") is True
    assert is_abstain_marker("Paris") is False


def test_eval_pipeline_reexports_canonical_set():
    """The eval pipeline must use the canonical module so the two
    codepaths can never drift apart (P24 unification)."""
    from mothrag.core.abstain_markers import ABSTAIN_MARKERS as CANONICAL
    from mothrag.eval.iterative_pipeline import (
        ABSTAIN_MARKERS as EVAL_PIPELINE,
    )
    assert EVAL_PIPELINE is CANONICAL, \
        "iterative_pipeline.ABSTAIN_MARKERS must BE the same object as the canonical"


def test_pip_is_uncertain_answer_uses_canonical_set():
    """pip api.py _is_uncertain_answer must catch eval-pipeline markers."""
    from mothrag.core.api import _is_uncertain_answer
    # Previously, the pip helper only knew 5 markers (no "i do not know",
    # no "cannot answer"). Now the full canonical set is honoured.
    assert _is_uncertain_answer("I do not know") is True
    assert _is_uncertain_answer("Cannot answer") is True
    assert _is_uncertain_answer("not in passages") is True
    # Concrete answers must NOT trigger uncertainty.
    assert _is_uncertain_answer("Paris is the capital of France.") is False


def test_is_abstain_marker_accepts_extra_markers():
    from mothrag.core.abstain_markers import is_abstain_marker
    assert is_abstain_marker("Insufficient information",
                              extra_markers={"insufficient information"}) is True
    # Without the extra hint, the same string is NOT canonical:
    assert is_abstain_marker("Insufficient information") is False


def test_arm_iter_drops_abstain_responses_from_accumulator(monkeypatch):
    """P4 abstain filter: abstain markers must NOT be propagated to the
    accumulator so the augmented_q on subsequent iters stays clean."""
    from mothrag.core.api import MothRAG

    # Build a stub MothRAG with a deterministic reader/retriever.
    rag = MothRAG.__new__(MothRAG)
    rag.config = {"max_iter_steps": 3, "iter_use_few_shot": False}

    captured_augmented_qs: list[str] = []

    class _StubReader:
        # First call returns abstain marker; subsequent calls return real answer.
        _seq = ["I don't know", "Paris", "Paris"]

        def __init__(self):
            self._i = 0

        def read(self, q, passages):
            captured_augmented_qs.append(q)
            ans = self._seq[min(self._i, len(self._seq) - 1)]
            self._i += 1
            return ans

    class _StubRetriever:
        def retrieve(self, q, *, top_k):
            from types import SimpleNamespace
            return [SimpleNamespace(text="A relevant passage about France.")]

    rag.reader = _StubReader()
    rag.retriever = _StubRetriever()

    final = rag._arm_iter("What is the capital of France?", ["seed passage"],
                            q_emb=[0.0], top_k=5)
    assert final == "Paris"
    # First call: question only (no accumulator yet).
    assert captured_augmented_qs[0] == "What is the capital of France?"
    # Second call: accumulator should be EMPTY (abstain dropped via P4),
    # so the augmented_q must NOT contain "I don't know".
    assert "I don't know" not in captured_augmented_qs[1], \
        f"P4 violation — abstain leaked into augmented_q: {captured_augmented_qs[1]!r}"


def test_arm_iter_p8_few_shot_default_on(monkeypatch):
    """P8 few-shot ON by default: augmented_q on iter>=2 must contain the
    synthesis-frame language (not the legacy 'Context from prior steps').

    We have to force the iter loop to populate the accumulator + then
    enter at least one more iteration. Easiest: monkey-patch
    ``_is_uncertain_answer`` so 'PARTIAL' answers count as uncertain
    (they're not canonical markers, so P4 won't drop them from the
    accumulator). The second answer then triggers the few-shot frame.
    """
    import mothrag.core.api as api_mod
    from mothrag.core.api import MothRAG

    rag = MothRAG.__new__(MothRAG)
    rag.config = {"max_iter_steps": 3}  # iter_use_few_shot default True

    captured: list[str] = []

    # Force 'PARTIAL' answers to count as uncertain so the iter loop
    # populates the accumulator.
    original = api_mod._is_uncertain_answer
    monkeypatch.setattr(api_mod, "_is_uncertain_answer",
                         lambda p: p == "PARTIAL" or original(p))

    class _StubReader:
        _seq = ["PARTIAL: Marie Curie was a physicist", "FinalAnswer"]
        def __init__(self):
            self._i = 0
        def read(self, q, passages):
            captured.append(q)
            ans = self._seq[min(self._i, len(self._seq) - 1)]
            self._i += 1
            return ans

    class _StubRetriever:
        def retrieve(self, q, *, top_k):
            from types import SimpleNamespace
            return [SimpleNamespace(text="passage")]

    rag.reader = _StubReader()
    rag.retriever = _StubRetriever()
    # Use answer that monkey-patched heuristic flags as uncertain.
    _StubReader._seq = ["PARTIAL", "FinalAnswer"]
    _ = rag._arm_iter("Q?", ["seed"], q_emb=[0.0], top_k=5)
    # Some iter >= 2 augmented_q should use the few-shot synthesis frame.
    assert any("Synthesise an answer" in q for q in captured), \
        f"P8 few-shot not present in any augmented_q: {captured!r}"


def test_arm_iter_p8_few_shot_opt_out(monkeypatch):
    """When ``iter_use_few_shot: False``, the legacy template is used.

    Same monkey-patch trick as the few-shot-default-on test: force
    'PARTIAL' to count as uncertain so the accumulator populates.
    """
    import mothrag.core.api as api_mod
    from mothrag.core.api import MothRAG

    rag = MothRAG.__new__(MothRAG)
    rag.config = {"max_iter_steps": 3, "iter_use_few_shot": False}

    captured: list[str] = []
    original = api_mod._is_uncertain_answer
    monkeypatch.setattr(api_mod, "_is_uncertain_answer",
                         lambda p: p == "PARTIAL" or original(p))

    class _StubReader:
        _seq = ["PARTIAL", "FinalAnswer"]
        def __init__(self):
            self._i = 0
        def read(self, q, passages):
            captured.append(q)
            ans = self._seq[min(self._i, len(self._seq) - 1)]
            self._i += 1
            return ans

    class _StubRetriever:
        def retrieve(self, q, *, top_k):
            from types import SimpleNamespace
            return [SimpleNamespace(text="passage")]

    rag.reader = _StubReader()
    rag.retriever = _StubRetriever()
    _ = rag._arm_iter("Q?", ["seed"], q_emb=[0.0], top_k=5)
    # Opt-out path: no few-shot synthesis frame; legacy "Context from
    # prior steps" template instead.
    assert all("Synthesise an answer" not in q for q in captured)
    assert any("Context from prior steps:" in q for q in captured)
