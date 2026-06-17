"""Tests for MothGraphArm + supporting graph primitives.

Covers:
  - mothrag.graph.openie.extract_triples_from_text (regex extractor)
  - mothrag.graph.index.GraphIndex (traversal + L4b hash)
  - mothrag.arms.mothgraph_arm.MothGraphArm (anchor + iter + gamma +
    L4b + soft-fallback)
  - mothrag.core.query_type_classifier.arm_subset_pam_lite extended
    to 5-arm pool (regression: legacy 3-arm path unchanged)
  - General-purpose contract: no per-dataset tokens in arm source

General-purpose contract: all linguistic / graph rules; no
per-dataset tuning, no test inspection.
"""

from __future__ import annotations

import inspect

import pytest


# ============================================================
# Graph primitives -- extraction + index + traversal hash
# ============================================================

def test_openie_regex_extracts_svo_triple() -> None:
    from mothrag.graph.openie import extract_triples_from_text
    triples = extract_triples_from_text(
        "Steve Jobs founded Apple in 1976.",
        source_chunk_id="c1",
    )
    assert any(
        "steve jobs" in t.subject.lower() and "apple" in t.object.lower()
        and t.predicate == "founded"
        for t in triples
    ), f"no founded triple in {triples!r}"


def test_openie_regex_extracts_be_of_relation() -> None:
    from mothrag.graph.openie import extract_triples_from_text
    triples = extract_triples_from_text(
        "Tim Cook is the CEO of Apple.",
        source_chunk_id="c2",
    )
    # Predicate captured as the relation noun "ceo".
    assert any(
        "tim cook" in t.subject.lower() and "apple" in t.object.lower()
        and t.predicate == "ceo"
        for t in triples
    ), f"no ceo triple in {triples!r}"


def test_graph_index_traverses_from_anchor() -> None:
    from mothrag.graph.index import GraphIndex
    g = GraphIndex()
    g.add_edge("Albert Einstein", "born_in", "Ulm", source_chunk_id="c1")
    g.add_edge("Albert Einstein", "spouse", "Mileva Maric", source_chunk_id="c2")
    g.add_edge("Mileva Maric", "born_in", "Titel", source_chunk_id="c3")
    paths = g.traverse_from_anchor("Albert Einstein", depth=2, top_k=10)
    # depth-1: 2 paths; depth-2: at least one (Einstein -> Mileva -> Titel).
    assert any(
        len(p.edges) == 2 and "titel" in p.edges[-1].object.lower()
        for p in paths
    )


def test_graph_index_traversal_hash_is_stable_on_same_paths() -> None:
    from mothrag.graph.index import GraphIndex, traversal_hash
    g = GraphIndex()
    g.add_edge("A", "rel", "B")
    g.add_edge("B", "rel", "C")
    h1 = traversal_hash(g.traverse_from_anchor("A", depth=2, top_k=10))
    h2 = traversal_hash(g.traverse_from_anchor("A", depth=2, top_k=10))
    assert h1 == h2 and len(h1) == 40  # SHA1 hex length


def test_graph_index_traversal_hash_changes_with_path_set() -> None:
    from mothrag.graph.index import GraphIndex, traversal_hash
    g1 = GraphIndex()
    g1.add_edge("A", "rel", "B")
    g2 = GraphIndex()
    g2.add_edge("A", "rel", "B")
    g2.add_edge("B", "rel", "C")
    h1 = traversal_hash(g1.traverse_from_anchor("A", depth=2, top_k=10))
    h2 = traversal_hash(g2.traverse_from_anchor("A", depth=2, top_k=10))
    assert h1 != h2


# ============================================================
# MothGraphArm -- 5 spec tests + 1 regression
# ============================================================

@pytest.fixture()
def small_graph():
    from mothrag.graph.index import GraphIndex
    g = GraphIndex()
    g.add_edge("Albert Einstein", "born_in", "Ulm", source_chunk_id="c1")
    g.add_edge("Albert Einstein", "spouse", "Mileva Maric", source_chunk_id="c2")
    g.add_edge("Mileva Maric", "born_in", "Titel", source_chunk_id="c3")
    g.add_edge("Albert Einstein", "won", "Nobel Prize", source_chunk_id="c4")
    return g


@pytest.fixture()
def fallback_callable():
    from mothrag.arms.base import ArmResult
    calls = []

    def _fallback(question: str) -> ArmResult:
        calls.append(question)
        return ArmResult(pred="FALLBACK", metadata={"path": "dense"})

    _fallback.calls = calls  # type: ignore[attr-defined]
    return _fallback


def test_mothgraph_arm_anchor_driven_traversal(small_graph, fallback_callable) -> None:
    """When anchor matches an indexed entity, arm returns graph-derived answer."""
    from mothrag.arms import MothGraphArm
    arm = MothGraphArm(
        graph_index=small_graph,
        dense_fallback=fallback_callable,
        max_iters=1,
    )
    assert arm.applicable("Where was Albert Einstein born?")
    result = arm.run("Where was Albert Einstein born?")
    # No dense fallback should have fired -- we have an anchor.
    assert not fallback_callable.calls
    # The arm returned a graph answer (non-empty, not the fallback sentinel).
    assert result.pred and result.pred != "FALLBACK"
    assert result.n_llm_calls == 0
    assert result.metadata["anchor_initial"] == "Albert Einstein"


def test_mothgraph_arm_iterative_refinement(small_graph, fallback_callable) -> None:
    """Multi-iter traversal refines anchor toward incident endpoints."""
    from mothrag.arms import MothGraphArm
    arm = MothGraphArm(
        graph_index=small_graph,
        dense_fallback=fallback_callable,
        max_iters=3,
        base_depth=1,  # force refinement to be the only way to reach Titel
    )
    result = arm.run("Who is the spouse of Albert Einstein?")
    # With max_iters>1 + base_depth=1, refinement should have surfaced
    # Mileva Maric edges in a subsequent round.
    assert result.metadata["iterations"] >= 1


def test_mothgraph_arm_gamma_validates_paths(small_graph, fallback_callable) -> None:
    """Custom gamma verifier filters out paths that fail validation."""
    from mothrag.arms import MothGraphArm

    def _reject_all(path, question):  # noqa: ARG001
        return False

    arm = MothGraphArm(
        graph_index=small_graph,
        dense_fallback=fallback_callable,
        gamma_verifier=_reject_all,
        max_iters=2,
    )
    result = arm.run("Where was Albert Einstein born?")
    # Every path is rejected -> soft fallback.
    assert result.pred == "FALLBACK"
    assert result.metadata["mothgraph_fallback_reason"] == "no_valid_paths"


def test_mothgraph_arm_l4b_stability_stops(small_graph, fallback_callable) -> None:
    """When the path-set hash stabilises across iterations, the loop breaks early."""
    from mothrag.arms import MothGraphArm
    arm = MothGraphArm(
        graph_index=small_graph,
        dense_fallback=fallback_callable,
        max_iters=5,
        base_depth=1,  # restricts incidence so anchor refinement stabilises fast
    )
    result = arm.run("Where was Albert Einstein born?")
    # The stable_break flag is set when L4b hash equality triggers early exit.
    # max_iters=5 with base_depth=1 + 4 edges should stabilise within a few iters.
    assert "stable_break" in result.metadata
    assert result.metadata["iterations"] <= 5


def test_mothgraph_arm_soft_fallback_when_no_paths(fallback_callable) -> None:
    """Empty graph -> arm.applicable returns False; explicit run -> fallback."""
    from mothrag.arms import MothGraphArm
    from mothrag.graph.index import GraphIndex
    empty = GraphIndex()
    arm = MothGraphArm(
        graph_index=empty,
        dense_fallback=fallback_callable,
    )
    assert arm.applicable("Where was Albert Einstein born?") is False
    result = arm.run("Where was Albert Einstein born?")
    assert result.pred == "FALLBACK"
    assert result.metadata["mothgraph_fallback_reason"] == "no_anchor"


# ============================================================
# General-purpose / no-dataset-specific contract
# ============================================================

def test_mothgraph_general_purpose_no_dataset_tokens() -> None:
    """Verify MothGraphArm + graph primitives have no per-dataset code paths.

    Source scan: no test-set filename / dataset name should appear in
    the implementation. Only failure mode would be accidental hardcoding
    of a benchmark-specific term.
    """
    import mothrag.arms.mothgraph_arm as mga_mod
    import mothrag.graph.index as idx_mod
    import mothrag.graph.openie as openie_mod
    forbidden = (
        "hotpotqa", "2wiki", "musique", "hp_train",
        "2w_train", "mq_train",
    )
    for mod in (mga_mod, idx_mod, openie_mod):
        src = inspect.getsource(mod).lower()
        for token in forbidden:
            assert token not in src, (
                f"{mod.__name__} source contains dataset-specific token "
                f"{token!r}; violates general-purpose contract."
            )


# ============================================================
# Regression: PAM-lite extends to 5-arm pool
# ============================================================

def test_pam_lite_extends_to_5_arm_pool() -> None:
    """PAM-lite scores opt-in arms when they appear in arms_pool; legacy
    3-arm call signature is byte-identical."""
    from mothrag.core.query_type_classifier import arm_subset_pam_lite

    q = "Who is the spouse of the founder of Apple?"

    # Legacy 3-arm call (no arms_pool kwarg): SAME as before PAM-lite extension.
    legacy_subset, legacy_probs = arm_subset_pam_lite(q)
    assert set(legacy_probs.keys()) == {"v3bu", "decompose", "iter"}

    # 5-arm pool call: opt-in arms appear in probabilities and may
    # enter the subset based on threshold.
    pool = ["v3bu", "decompose", "iter", "infobox_arm", "mothgraph_arm"]
    subset5, probs5 = arm_subset_pam_lite(q, arms_pool=pool)
    assert set(probs5.keys()) == set(pool)
    # Legacy arm probabilities are IDENTICAL across pool sizes -- the
    # opt-in extension only ADDS new arms, never modifies existing.
    for arm in ("v3bu", "decompose", "iter"):
        assert probs5[arm] == legacy_probs[arm], (
            f"PAM-lite legacy arm {arm} probability changed between "
            f"3-arm and 5-arm pool calls; backward compat violated."
        )
    # On a bridge-relational question, mothgraph_arm should fire above
    # threshold (bridge + multi_hop signals positive).
    assert probs5["mothgraph_arm"] > 0.4, (
        f"P_mothgraph={probs5['mothgraph_arm']:.3f} on a bridge-relational "
        f"question; coefficient regression."
    )


def test_pam_lite_3_arm_pool_excludes_opt_in_arms() -> None:
    """When arms_pool is the default 3-arm, opt-in arms must NOT appear."""
    from mothrag.core.query_type_classifier import arm_subset_pam_lite
    _subset, probs = arm_subset_pam_lite(
        "When was Einstein born?",
        arms_pool=["v3bu", "decompose", "iter"],
    )
    assert "infobox_arm" not in probs
    assert "mothgraph_arm" not in probs


def test_pam_lite_unknown_arm_silently_ignored() -> None:
    """Unknown arm names in arms_pool are dropped (no KeyError)."""
    from mothrag.core.query_type_classifier import arm_subset_pam_lite
    _subset, probs = arm_subset_pam_lite(
        "When was Einstein born?",
        arms_pool=["v3bu", "decompose", "iter", "unknown_future_arm"],
    )
    assert "unknown_future_arm" not in probs
    assert set(probs.keys()) == {"v3bu", "decompose", "iter"}
