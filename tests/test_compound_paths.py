"""Compound: ensemble_arbitrate + retry-on-abstain composability smoke tests.

Verifies that the two production paths -- the ensemble path and the retry
path -- compose correctly when activated together via:

    MothRAG(
        production=True,
        mode='ensemble_arbitrate',                  # ensemble path
        retry_strategies=['default_7'],             # retry path
        retry_mode='loop',                          # retry path terminal: SoftFallback
    )

Conflict resolution audit (recorded as test assertions):
  - MothRAG `mode` parameter remains the ensemble-path production-strategy
    selector ("adaptive" | "ensemble_arbitrate"). The retry-on-abstain
    path's original `mode=` ("loop" | "abstention") was renamed to
    `retry_mode=` to disambiguate.
  - Metadata key "mode" continues to echo the ensemble-path mode; new
    metadata key "retry_mode" echoes the cascade terminal mode.
  - "default_7" is a backward-compat alias for the canonical "all"
    7-strategy preset; both expand to the same DEFAULT_PRIORITY tuple.
  - Mixed list expansion: retry_strategies=["default_7", "<extra>"] is
    supported via the new _expand_preset_names helper.
"""

from __future__ import annotations

import sys
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _block_sentence_transformers(monkeypatch) -> Iterator[None]:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    yield


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch) -> Iterator[None]:
    for k in ("VERTEX_AI_PROJECT", "GOOGLE_CLOUD_PROJECT",
              "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    yield


# ============================================================
# 1. Two-axis mode parameter: mode (ensemble) + retry_mode (retry)
# ============================================================

def test_mothrag_two_axis_modes_independent() -> None:
    """mode + retry_mode are validated against disjoint enums."""
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader
    rag = MothRAG(
        embedder=_HashEmbedder(), reader=_EchoReader(),
        mode="ensemble_arbitrate",
        retry_mode="abstention",
    )
    assert rag.mode == "ensemble_arbitrate"
    assert rag.retry_mode == "abstention"


def test_mothrag_mode_validation_unchanged() -> None:
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader
    with pytest.raises(ValueError, match="MothRAG mode must be"):
        MothRAG(embedder=_HashEmbedder(), reader=_EchoReader(), mode="circular")


def test_mothrag_retry_mode_validation() -> None:
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader
    with pytest.raises(ValueError, match="MothRAG retry_mode must be"):
        MothRAG(embedder=_HashEmbedder(), reader=_EchoReader(),
                retry_mode="forwards")


# ============================================================
# 2. "default_7" preset alias
# ============================================================

def test_default_7_preset_alias_equals_all() -> None:
    """build_default_orchestrator('default_7') is byte-identical to ('all')."""
    from mothrag.core.retry import build_default_orchestrator
    o_all = build_default_orchestrator("all")
    o_d7 = build_default_orchestrator("default_7")
    assert [s.name for s in o_all.strategies] == [s.name for s in o_d7.strategies]


def test_mixed_list_expansion_default_7_plus_extra() -> None:
    """retry_strategies=['default_7', 'X'] expands to DEFAULT_PRIORITY + X."""
    from mothrag.core.retry import build_default_orchestrator
    from mothrag.core.retry.orchestrator import DEFAULT_PRIORITY
    # Use an existing strategy name as the "extra" so instantiation succeeds.
    # 'iter_extension' is already in DEFAULT_PRIORITY so its duplicate is
    # de-duped; the test purposefully picks a known strategy to avoid coupling
    # to a not-yet-implemented one.
    orch = build_default_orchestrator(["default_7", "iter_extension"])
    expected = list(DEFAULT_PRIORITY)
    # soft_fallback may be moved to end by build_strategies_by_name but the
    # 7-strategy set must be present in the ordered output.
    assert {s.name for s in orch.strategies} == set(expected)


# ============================================================
# 3. ensemble + retry composability
# ============================================================

def test_ensemble_arbitrate_with_default_7_retry_composes() -> None:
    """The two paths can be enabled simultaneously without runtime error."""
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader
    rag = MothRAG.from_documents(
        ["First doc.", "Second doc."],
        embedder=_HashEmbedder(), reader=_EchoReader(),
        production=True,
        mode="ensemble_arbitrate",
        retry_strategies=["default_7"],
        retry_mode="loop",
    )
    qr = rag.query("Q?")
    # ensemble-path telemetry
    assert qr.metadata["mode"] == "ensemble_arbitrate"
    assert qr.metadata["production_strategy"] == "ensemble_arbitrate"
    assert qr.metadata["selected_arm"] in ("v3bu", "decompose", "iter")
    assert qr.metadata["arbitrate_signal"] in (
        "consensus", "gamma", "faith", "fallback",
    )
    # retry-path telemetry (escalation_* keys merged in from _maybe_escalate)
    assert qr.metadata["retry_mode"] == "loop"
    assert "terminal_abstain" in qr.metadata
    assert "escalation_applied" in qr.metadata
    assert "final_answer_confidence" in qr.metadata


def test_ensemble_arbitrate_abstention_terminal_when_fallback_signal() -> None:
    """Ensemble + retry_mode=abstention: when ensemble emits 'fallback'
    arbitrate_signal AND retry cascade declines, qr.answer should be
    empty and terminal_abstain True."""
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder

    class _AlwaysEmptyReader:
        def read(self, q, p):  # noqa: ARG002
            return ""

    rag = MothRAG.from_documents(
        ["A.", "B."],
        embedder=_HashEmbedder(), reader=_AlwaysEmptyReader(),
        production=True,
        mode="ensemble_arbitrate",
        retry_mode="abstention",
        retry_strategies=["arm_fallback"],  # zero-LLM, no soft_fallback
    )
    qr = rag.query("Q?")
    # Ensemble produces no non-empty arm -> arbitrate_signal=fallback.
    assert qr.metadata["arbitrate_signal"] == "fallback"
    # Retry sees abstention signal, runs arm_fallback (no recovery: no
    # non-empty arm), then terminal_abstain since retry_mode='abstention'.
    assert qr.metadata["retry_mode"] == "abstention"
    assert qr.metadata["terminal_abstain"] is True
    assert qr.answer == ""


# ============================================================
# 4. Backward compat: adaptive + retry remains intact
# ============================================================

def test_adaptive_mode_with_retry_still_works() -> None:
    """Baseline behaviour (mode='adaptive' + retry) unchanged."""
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader
    rag = MothRAG.from_documents(
        ["A.", "B."],
        embedder=_HashEmbedder(), reader=_EchoReader(),
        production=True,
        mode="adaptive",                # default ensemble-path value
        retry_strategies="all",         # default retry-path preset
    )
    qr = rag.query("Q?")
    assert qr.metadata["mode"] == "adaptive"
    assert qr.metadata["production_strategy"] == "adaptive"
    assert "retry_mode" in qr.metadata


def test_default_construction_matches_pre_patch_behaviour() -> None:
    """MothRAG() with no ensemble / retry kwargs defaults cleanly."""
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader
    rag = MothRAG(embedder=_HashEmbedder(), reader=_EchoReader())
    assert rag.mode == "adaptive"
    assert rag.retry_mode == "loop"
