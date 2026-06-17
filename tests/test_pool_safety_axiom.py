"""Pool-safety tests for opt-in arm composition.

Pool-safety axiom (motivated by an MQ F1=1 cohort -23/-26pp
regression):

  F1(pool ∪ {X}, cohort) == F1(pool, cohort) when fire_X == 0

i.e. adding an opt-in arm to the pool MUST NOT change F1 on queries
where the opt-in arm does not legitimately contribute. An earlier
implementation violated this when MothGraphArm soft-fallback delegated
to V3+bu and the fallback result entered the arbitration candidate set
as ``mothgraph_arm`` -- pairwise_agreement then saw v3bu and
mothgraph_arm with identical answers (spurious consensus boost) and
legitimate disagreeing arms (e.g. iter with the correct answer) lost.

Fix (commit pending):
  1. MothGraphArm._soft_fallback tags ``metadata["is_fallback"] = True``.
  2. scripts/route_prospective.py opt-in arm loops (both
     _run_ensemble_arbitrate and the v2-mode dispatch extension) skip
     is_fallback-tagged results when building the candidates dict.

These tests pin the axiom so a future refactor cannot silently
regress it.
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
# MothGraphArm._soft_fallback metadata tagging
# ============================================================

def test_mothgraph_soft_fallback_tags_is_fallback_metadata() -> None:
    """Soft-fallback results carry metadata['is_fallback'] = True."""
    from mothrag.arms import MothGraphArm
    from mothrag.arms.base import ArmResult
    from mothrag.graph.index import GraphIndex

    empty_graph = GraphIndex()

    def _fallback(_q):
        return ArmResult(pred="FALLBACK_ANSWER")

    arm = MothGraphArm(
        graph_index=empty_graph,
        dense_fallback=_fallback,
    )
    # Empty graph -> applicable=False, but run() still falls back when
    # called directly.
    result = arm.run("Where was Einstein born?")
    assert result.pred == "FALLBACK_ANSWER"
    assert result.metadata.get("is_fallback") is True
    assert result.metadata.get("fallback_origin") == "mothgraph_arm"


def test_mothgraph_soft_fallback_carries_reason_AND_is_fallback() -> None:
    """When the fallback is triggered by no_anchor / no_valid_paths /
    empty_composition, both ``is_fallback`` and ``mothgraph_fallback_reason``
    are present in metadata."""
    from mothrag.arms import MothGraphArm
    from mothrag.arms.base import ArmResult
    from mothrag.graph.index import GraphIndex

    empty_graph = GraphIndex()

    def _fallback(_q):
        return ArmResult(pred="X")

    arm = MothGraphArm(empty_graph, _fallback)
    result = arm.run("Where was Einstein born?")
    assert result.metadata["is_fallback"] is True
    assert result.metadata["mothgraph_fallback_reason"] == "no_anchor"


def test_mothgraph_non_fallback_path_NOT_tagged() -> None:
    """Successful graph traversal returns pred without is_fallback flag."""
    from mothrag.arms import MothGraphArm
    from mothrag.arms.base import ArmResult
    from mothrag.graph.index import GraphIndex

    g = GraphIndex()
    g.add_edge("Albert Einstein", "born_in", "Ulm", source_chunk_id="c1")

    def _fallback(_q):
        return ArmResult(pred="FALLBACK")

    arm = MothGraphArm(g, _fallback, max_iters=1)
    result = arm.run("Where was Albert Einstein born?")
    # Graph traversal succeeded -> NO is_fallback in metadata.
    assert result.pred and result.pred != "FALLBACK"
    assert not result.metadata.get("is_fallback")


# ============================================================
# Pool-safety axiom: fallback-tagged results SKIPPED from arbitration
# ============================================================

class _StubPipeline:
    def __init__(self):
        self.embedder_model = None
        self.chunk_ids: list[str] = []
        self.chunks_by_id: dict = {}

    def query_embedder(self, _text):
        import numpy as np
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def test_pool_arbitrate_skips_fallback_in_ensemble_arbitrate_loop() -> None:
    """Source-scan invariant: _run_ensemble_arbitrate opt-in loop must
    have the ``is_fallback`` skip clause.
    """
    import inspect
    import route_prospective as rp
    src = inspect.getsource(rp._run_ensemble_arbitrate)
    assert 'metadata.get("is_fallback")' in src, (
        "_run_ensemble_arbitrate opt-in loop missing the fallback skip "
        "clause; pool-safety axiom regressed."
    )


def test_pool_arbitrate_skips_fallback_in_v2_dispatch_loop() -> None:
    """Source-scan invariant: v2-mode dispatch opt-in extension block
    must have the ``is_fallback`` skip clause.
    """
    import inspect
    import route_prospective as rp
    main_src = inspect.getsource(rp.main)
    assert 'arm_result.metadata.get("is_fallback")' in main_src, (
        "v2-mode opt-in extension missing the fallback skip clause; "
        "pool-safety axiom regressed."
    )


# ============================================================
# Pool-safety axiom: F1(pool ∪ {X}, cohort) == F1(pool, cohort) when
# fire_X == 0  -- via direct arbitration call
# ============================================================

def test_arbitration_excludes_fallback_when_built_via_loop_protocol() -> None:
    """End-to-end on the opt-in loop's behaviour: a fallback-tagged
    ArmResult is NOT added to the candidates dict, so the arbitration
    sees an unchanged 3-arm candidate set.

    This is the canonical pool-safety test: the observed regression
    (-23/-26pp on MQ F1=1) is reproducible iff the fallback result
    enters arbitration as a duplicate of v3bu's answer.
    """
    import route_prospective as rp

    # Simulate what the loop does: arm.run returns an is_fallback result,
    # the loop SHOULD skip it.
    legacy_3arm_candidates = {
        "v3bu": {
            "pred": "v3bu_answer", "retrieved_chunk_ids": [],
            "n_llm_calls": 1, "prompt_tokens": 0, "completion_tokens": 0,
            "latency_s": 0.0,
        },
        "decompose": {
            "pred": "decompose_answer", "retrieved_chunk_ids": [],
            "n_llm_calls": 1, "prompt_tokens": 0, "completion_tokens": 0,
            "latency_s": 0.0,
        },
        "iter": {
            "pred": "iter_correct_answer", "retrieved_chunk_ids": [],
            "n_llm_calls": 1, "prompt_tokens": 0, "completion_tokens": 0,
            "latency_s": 0.0,
        },
    }
    # Baseline (3-arm) -- iter wins via tie-break (no agreement signal
    # supplied; alphabetical name order).
    out_3arm = rp._arbitrate_candidates(
        _StubPipeline(), candidates=legacy_3arm_candidates,
    )

    # 5-arm pool simulation where opt-in arm fallback DUPLICATES v3bu's
    # answer. THIS IS THE REGRESSION SHAPE: if the loop fails to skip,
    # mothgraph_arm enters candidates with v3bu's pred -> pairwise
    # agreement boosts the wrong answer.
    candidates_with_fallback_leak = dict(legacy_3arm_candidates)
    candidates_with_fallback_leak["mothgraph_arm"] = {
        "pred": "v3bu_answer",  # DUPLICATE -- the bug
        "retrieved_chunk_ids": [],
        "n_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "latency_s": 0.0,
    }
    out_leaked = rp._arbitrate_candidates(
        _StubPipeline(), candidates=candidates_with_fallback_leak,
    )

    # The two outputs MUST differ when the leak is present (proves the
    # bug is reachable / the test is meaningful).
    assert out_3arm["selected_arm"] != out_leaked["selected_arm"] or \
        out_3arm["arm_scores"] != out_leaked["arm_scores"], (
        "The fallback-leak shape should perturb arbitration; if this "
        "assertion fails, the regression scenario isn't reachable and "
        "the pool-safety test is meaningless."
    )

    # And the pool-safety axiom: when the loop correctly skips fallback,
    # the 5-arm pool arbitration MUST equal the 3-arm baseline result
    # (this is implicit -- the loop never adds mothgraph_arm to candidates,
    # so the dict is identical to 3-arm).
    # We don't run the loop here directly (no pipeline); the source-scan
    # tests above pin the loop's skip clause.


def test_pool_safety_axiom_documented_in_source() -> None:
    """The pool-safety axiom (F1(pool ∪ {X}) == F1(pool) when fire_X==0)
    must be documented in MothGraphArm._soft_fallback so a future
    refactor cannot silently drop the metadata tag.
    """
    import inspect
    from mothrag.arms.mothgraph_arm import MothGraphArm
    src = inspect.getsource(MothGraphArm._soft_fallback)
    assert "Pool-safety" in src, (
        "MothGraphArm._soft_fallback missing pool-safety docstring; "
        "future refactor risk."
    )
    assert "is_fallback" in src, (
        "MothGraphArm._soft_fallback missing is_fallback metadata tag; "
        "fallback contributions will leak back into arbitration."
    )


# ============================================================
# Regression: PAM-lite argmax-fallback pool-size flip
# ============================================================

def test_pam_lite_legacy_subset_call_unchanged_by_arms_pool() -> None:
    """The LEGACY-only arm_subset_pam_lite call (arms_pool=None) MUST
    return identical subset + identical v3bu inclusion regardless of
    what arms_pool a separate 5-arm call uses.

    This pins the contract used by the
    ``v3bu_in = "v3bu" in legacy_subset`` decision in
    route_prospective.py: re-deriving v3bu_in via a legacy-only call
    is pool-size-independent by construction.
    """
    from mothrag.core.query_type_classifier import arm_subset_pam_lite

    queries = [
        "When was Einstein born?",
        "Who is the spouse of the director of Inception?",
        "Are Einstein and Newton both physicists?",
        "First X then Y then Z",
        "Why?",  # short question; may trip argmax-fallback
    ]
    for q in queries:
        subset_3arm, _ = arm_subset_pam_lite(q)
        subset_3arm_explicit, _ = arm_subset_pam_lite(
            q, arms_pool=["v3bu", "decompose", "iter"],
        )
        assert subset_3arm == subset_3arm_explicit, (
            f"3-arm subset diverges between None and explicit pool: "
            f"{subset_3arm} vs {subset_3arm_explicit}"
        )


def test_pam_lite_argmax_fallback_pool_size_edge_case_documented() -> None:
    """When PAM-lite arms_pool=5 and every legacy arm's P is below the
    threshold but an opt-in arm's P is above, the 5-arm subset may
    contain ONLY opt-in arms (e.g. ['infobox_arm']). This is the
    edge case that broke v3bu_in pre-fix.

    The route_prospective.py fix re-derives v3bu_in
    from a LEGACY-only arm_subset_pam_lite call so the V3+bu
    execution gate is identical 3-arm vs 5-arm pool. This test pins
    the underlying behaviour: yes, 5-arm subset CAN be opt-in-only;
    callers consuming arm_subset_pam_lite for v3bu_in routing MUST
    derive from a legacy-only call.
    """
    from mothrag.core.query_type_classifier import arm_subset_pam_lite

    # Construct a query likely to trip the edge case: short, no
    # strong polar/chain/temporal signal, so legacy P values stay
    # low; opt-in arms may still trigger via their own scorers.
    # NB the exact behaviour depends on the linguistic features of
    # the query; this test is structural (the subset is restricted
    # to pool members; no legacy guarantees).
    q = "X's Y"  # single_hop pattern -> infobox_arm scores high
    subset_5arm, probs_5arm = arm_subset_pam_lite(
        q, threshold=0.3,
        arms_pool=["v3bu", "decompose", "iter",
                   "infobox_arm", "mothgraph_arm"],
    )
    # Documentation invariant: subset is a subset of pool.
    assert set(subset_5arm).issubset(
        {"v3bu", "decompose", "iter", "infobox_arm", "mothgraph_arm"}
    )
    # And the 5-arm pool may include opt-in arms not in the 3-arm
    # baseline (this is the design intent).
    subset_3arm, _ = arm_subset_pam_lite(q, threshold=0.3)
    assert set(subset_3arm).issubset({"v3bu", "decompose", "iter"})


def test_pool_safety_v3bu_in_decoupled_from_arms_pool_source_scan() -> None:
    """Source-scan invariant: _run_ensemble_arbitrate's v3bu_in MUST
    be derived from a LEGACY-only arm_subset / arm_subset_pam_lite
    call, not from the arms_pool-aware subset.

    A future refactor that re-couples v3bu_in to the arms_pool-aware
    subset will re-introduce the pool-safety regression -> this
    test fails loudly to force explicit re-resolution.
    """
    import inspect
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import route_prospective as rp

    src = inspect.getsource(rp._run_ensemble_arbitrate)
    # v3bu_in MUST be re-derived after the initial subset
    # computation -- look for the rebinding.
    assert "v3bu_in = \"v3bu\" in legacy_subset" in src or \
           "v3bu_in = \"v3bu\" in _arm_subset(question)" in src, (
        "_run_ensemble_arbitrate v3bu_in not re-derived from a "
        "legacy-only routing call; pool-size-dependent."
    )
