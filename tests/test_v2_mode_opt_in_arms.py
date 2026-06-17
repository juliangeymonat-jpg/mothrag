"""Tests for --mode v2 dispatch extension to opt-in arms.

Covers:
  - scripts/route_prospective.py:_arbitrate_candidates DRY helper
    (extracted from _run_ensemble_arbitrate)
  - The v2 dispatch extension that runs opt-in arms (infobox_arm /
    mothgraph_arm) after the legacy 3-arm path and arbitrates the
    combined candidate set

These are unit-level tests on _arbitrate_candidates + the structural
shape of the v2 dispatch block. The full --mode v2 main loop is
integration-tested via the cloud smoke suite.

All routing rules are deterministic linguistic; no per-dataset tuning, no
test inspection.
"""

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
# Stubs
# ============================================================

class _StubPipeline:
    """Minimal MothRAGPipeline-shaped stub for _arbitrate_candidates."""

    def __init__(self):
        self.embedder_model = None
        # Fake query_embedder: returns a fixed-length unit vector.
        self.chunk_ids: list[str] = []
        self.chunks_by_id: dict = {}

    def query_embedder(self, text):  # noqa: ARG002
        import numpy as np
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


# ============================================================
# _arbitrate_candidates helper (DRY-extracted from _run_ensemble_arbitrate)
# ============================================================

def test_arbitrate_candidates_empty_returns_fallback() -> None:
    """Empty candidates dict -> empty pred + fallback signal."""
    import route_prospective as rp
    out = rp._arbitrate_candidates(_StubPipeline(), candidates={})
    assert out["pred"] == ""
    assert out["arbitrate_signal"] == "fallback"
    assert out["arm_scores"] == {}


def test_arbitrate_candidates_threads_gamma_signal() -> None:
    """γ='invalid' on iter must zero-weight that arm; non-iter arm wins."""
    import route_prospective as rp
    candidates = {
        "iter": {
            "pred": "iter_answer",
            "retrieved_chunk_ids": ["c1"],
            "n_llm_calls": 1, "prompt_tokens": 10, "completion_tokens": 5,
            "latency_s": 0.1,
        },
        "v3bu": {
            "pred": "v3bu_answer",
            "retrieved_chunk_ids": ["c2"],
            "n_llm_calls": 1, "prompt_tokens": 10, "completion_tokens": 5,
            "latency_s": 0.1,
        },
    }
    out = rp._arbitrate_candidates(
        _StubPipeline(),
        candidates=candidates,
        iter_gamma_status="invalid",
    )
    # iter signal zeroed -> v3bu must win.
    assert out["selected_arm"] == "v3bu"
    assert out["pred"] == "v3bu_answer"


def test_arbitrate_candidates_threads_arm_probabilities() -> None:
    """arm_probabilities multiplies signal scores; high-P arm wins."""
    import route_prospective as rp
    candidates = {
        "decompose": {
            "pred": "decompose_answer",
            "retrieved_chunk_ids": [],
            "n_llm_calls": 1, "prompt_tokens": 10, "completion_tokens": 5,
            "latency_s": 0.1,
        },
        "infobox_arm": {
            "pred": "infobox_answer",
            "retrieved_chunk_ids": [],
            "n_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "latency_s": 0.001,
        },
    }
    out = rp._arbitrate_candidates(
        _StubPipeline(),
        candidates=candidates,
        arm_probabilities={"decompose": 0.1, "infobox_arm": 0.95},
    )
    assert out["selected_arm"] == "infobox_arm"


def test_arbitrate_candidates_merges_retrieved_chunk_ids() -> None:
    """Retrieved chunk ids from all candidates union into the output (deduped)."""
    import route_prospective as rp
    candidates = {
        "v3bu": {
            "pred": "x", "retrieved_chunk_ids": ["c1", "c2"],
            "n_llm_calls": 1, "prompt_tokens": 0, "completion_tokens": 0,
            "latency_s": 0.0,
        },
        "decompose": {
            "pred": "y", "retrieved_chunk_ids": ["c2", "c3"],
            "n_llm_calls": 1, "prompt_tokens": 0, "completion_tokens": 0,
            "latency_s": 0.0,
        },
    }
    out = rp._arbitrate_candidates(_StubPipeline(), candidates=candidates)
    assert set(out["retrieved_chunk_ids"]) == {"c1", "c2", "c3"}
    # Order preserved (v3bu first, decompose second; c2 dedup'd).
    assert out["retrieved_chunk_ids"] == ["c1", "c2", "c3"]


# ============================================================
# v2 dispatch extension -- structural shape tests
# ============================================================

def test_v2_dispatch_extension_block_exists_in_main() -> None:
    """The v2 dispatch block must include the opt-in arm composition path.

    Source-scan: look for the specific guard ``if args.mode == "v2"``
    and the legacy_candidate construction. If a future refactor removes
    these, the v2-mode opt-in composition silently breaks; this test
    forces explicit replacement.
    """
    import inspect
    import route_prospective as rp
    main_src = inspect.getsource(rp.main)
    assert 'args.mode == "v2"' in main_src, (
        "v2-mode guard missing from main(); opt-in arms will not "
        "execute under --mode v2."
    )
    assert "_arbitrate_candidates" in main_src, (
        "_arbitrate_candidates call missing from main(); opt-in arm "
        "composition path broken."
    )


def test_v2_mode_legacy_unchanged_when_pool_none() -> None:
    """When arms_pool defaults to legacy 3, opt-in extension is no-op.

    Source-scan invariant: the gating predicate must require BOTH
    args.mode == 'v2' AND opt_in_arms (truthy dict). With pool=None
    -> _parse_arms_pool returns legacy 3 -> opt_in_arms stays empty
    -> the v2 extension block does not fire.
    """
    import inspect
    import route_prospective as rp
    main_src = inspect.getsource(rp.main)
    # The gate must include `and opt_in_arms`, otherwise the block
    # would attempt opt-in composition even on legacy-only runs.
    assert 'args.mode == "v2" and opt_in_arms' in main_src, (
        "v2-mode opt-in gate must require opt_in_arms truthy; "
        "otherwise legacy 3-arm runs incur extra work."
    )


def test_v2_mode_arbitrate_composes_opt_in_with_legacy() -> None:
    """When opt-in arms in subset AND opt_in_arms wired, v2 path calls
    _arbitrate_candidates with the legacy + opt-in composite candidate
    set.

    Structural check: the v2 extension block must construct a
    candidates dict that starts with the legacy result and appends
    opt-in arm results, then dispatch via the shared
    _arbitrate_candidates helper. Pinning the call site protects
    against silent regression to the old "no v2 composition" behavior.
    """
    import inspect
    import route_prospective as rp
    main_src = inspect.getsource(rp.main)
    # The block must build a legacy_candidate before composing opt-in.
    assert "legacy_candidate" in main_src, (
        "v2 opt-in extension missing legacy_candidate construction; "
        "would arbitrate over opt-in-only candidates."
    )
    # Must call the DRY helper, not a duplicated inline arbitration block.
    assert "_arbitrate_candidates(" in main_src
    # Must respect "if len(v2_candidates) >= 2" guard so a single-arm
    # outcome doesn't pointlessly enter arbitration.
    assert "len(v2_candidates) >= 2" in main_src


def test_v2_mode_executes_opt_in_arms_when_in_subset() -> None:
    """End-to-end on the v2 extension block: with a stub opt-in arm
    that returns a non-empty pred, the helper composes candidates and
    runs arbitration.

    Drives _arbitrate_candidates directly with a hand-built candidate
    dict mirroring what the v2 dispatch produces (legacy_v2 pred +
    opt-in pred). Asserts arbitration emits both arms in arm_scores.
    """
    import route_prospective as rp
    candidates = {
        "v3bu": {
            "pred": "Berlin", "retrieved_chunk_ids": ["c1"],
            "n_llm_calls": 1, "prompt_tokens": 10, "completion_tokens": 5,
            "latency_s": 0.05,
        },
        "infobox_arm": {
            "pred": "Berlin", "retrieved_chunk_ids": [],
            "n_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "latency_s": 0.001,
        },
    }
    out = rp._arbitrate_candidates(
        _StubPipeline(), candidates=candidates,
    )
    assert set(out["arm_scores"].keys()) == {"v3bu", "infobox_arm"}
    # Both arms agree -> consensus signal expected; either may win
    # the tie-break (alphabetical name order in DeterministicArbitrator).
    assert out["selected_arm"] in {"v3bu", "infobox_arm"}


# ============================================================
# DRY regression: _run_ensemble_arbitrate still works after extraction
# ============================================================

def test_run_ensemble_arbitrate_uses_shared_helper() -> None:
    """_run_ensemble_arbitrate must call _arbitrate_candidates (no
    duplicated inline arbitration block)."""
    import inspect
    import route_prospective as rp
    src = inspect.getsource(rp._run_ensemble_arbitrate)
    assert "_arbitrate_candidates(" in src, (
        "_run_ensemble_arbitrate no longer calls the shared helper; "
        "DRY refactor regressed."
    )
    # And it must NOT re-inline arbitrator construction (that would
    # mean the refactor left a duplicate path behind).
    # Allow the import statement; check no full instantiation pattern.
    assert "DeterministicArbitrator()" not in src, (
        "_run_ensemble_arbitrate appears to instantiate "
        "DeterministicArbitrator directly; should delegate via "
        "_arbitrate_candidates."
    )
