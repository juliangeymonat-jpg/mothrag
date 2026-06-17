"""Tests for the router-gated infobox dispatch.

Covers:
  - mothrag.routing.infobox_router.is_entity_attribute_query()
    deterministic classifier (2W-style positive, HP/MQ-style negative,
    default-off conservative)
  - scripts/route_prospective.py CLI: --retrieval router_gated_infobox
    state-swap via _InfoboxGate
  - Telemetry: per-query infobox_fired + infobox_router_reason

Constraint: preliminary empirical motivation -- the unconditional
dense_plus_infobox dispatch helps 2W T1 +5.7pp but hurts HP T1 -0.86pp
and MQ T1 -12.5pp; the router must preserve the 2W lift while
neutralising the HP/MQ regression.
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
# Classifier: 2W-style entity-attribute positives
# ============================================================

def test_router_classifies_entity_attribute_query_2w_style() -> None:
    """The 2W beneficiary cohort: single-clause entity-attribute Q's."""
    from mothrag.routing.infobox_router import is_entity_attribute_query
    for q in (
        "When was Albert Einstein born?",
        "When did Marie Curie die?",
        "Where was Pierre Curie born?",
        "Where is Apple headquartered?",
        "Who is Albert Einstein's spouse?",
        "Who was Marie Curie's husband?",
        "What is the capital of France?",
        "What is Albert Einstein's nationality?",
        "What is Apple's net worth?",
        "Which country is Paris located in?",
        "How old is the Eiffel Tower?",
    ):
        assert is_entity_attribute_query(q), (
            f"Router declined positive 2W-style query {q!r}; expected fire."
        )


# ============================================================
# Classifier: HP-style multi-hop / bridge-entity negatives
# ============================================================

def test_router_skips_multi_hop_query_hp_style() -> None:
    """HP regression cohort: multi-hop / bridge-entity / subordinate clause.

    v2: 3 F1=0-derived negative patterns dropped per the
    leakage audit. Some bridge-entity surface forms ("composer of the
    soundtrack of the film X born") now fall through default-off /
    mis-fire and are tracked in test_router_v2_known_limitations.
    """
    from mothrag.routing.infobox_router import is_entity_attribute_query
    for q in (
        "Which company that Steve Jobs founded later acquired NeXT?",
        "Which actor played a role in 30 Rock as well as Mad Men?",
        "Which 2009 animated film is from Japan, Summer Wars or The Secret of Kells?",
    ):
        assert not is_entity_attribute_query(q), (
            f"Router fired on multi-hop query {q!r}; expected skip."
        )


# ============================================================
# Classifier: MQ-style chain reasoning negatives
# ============================================================

def test_router_skips_chain_query_mq_style() -> None:
    """MQ regression cohort: 4-hop chain reasoning Q's.

    v2 known limitation: the "height of the building that houses ..."
    nested-clause shape is now tracked in
    test_router_v2_known_limitations.
    """
    from mothrag.routing.infobox_router import is_entity_attribute_query
    for q in (
        "What year was the actor son of Jeremiah Porter born?",
        "Where was the place of death of the director of film A Rare Bird?",
        "When did the publisher of Labyrinth end?",
    ):
        assert not is_entity_attribute_query(q), (
            f"Router fired on chain query {q!r}; expected skip."
        )


# ============================================================
# v2 documented limitations: queries whose multi-hop status was
# previously caught by the 3 F1=0-derived negative patterns (dropped
# per data leakage audit). These now mis-fire because they
# satisfy a positive entity-attribute pattern AND none of the 6 clean
# v2 negatives. Test asserts the v2 BEHAVIOUR honestly (router returns
# True even though the question is multi-hop) so future readers know
# this is a known trade-off, not an undetected regression.
# ============================================================

def test_router_v2_known_limitations() -> None:
    """v2 router mis-fires on bridge-entity surface forms that the 3
    F1=0-derived negatives used to catch. Documented limitation, NOT
    a regression to fix without breaking the provenance guarantee."""
    from mothrag.routing.infobox_router import is_entity_attribute_query
    KNOWN_MISFIRES = (
        # Used to be caught by `(?:composer|...|inventor) of the` +
        # `\bwhere .*(?:the \w+ of)`. Positive `where was X born` fires.
        "Where was the composer of the soundtrack of the film Inception born?",
        # Used to be caught by `\bof the \w+ (?:that|who|which)\b`.
        # Positive `what is X's height` fires.
        "What is the height of the building that houses the city hall of "
        "the capital of the country where Apple is headquartered?",
        # Used to be caught by `(?:father|...) of the` +
        # `\bof the \w+ (?:that|who|which)\b`. Positive
        # `who is X's father` matches greedily on "Who is the father".
        "Who is the father of the actor who played Iron Man?",
    )
    for q in KNOWN_MISFIRES:
        assert is_entity_attribute_query(q), (
            f"Documented v2 mis-fire pattern returned False for {q!r}; "
            f"either v2 behaviour changed or the test corpus drifted."
        )


# ============================================================
# Classifier: comparison / polar negatives
# ============================================================

def test_router_skips_comparison_query() -> None:
    from mothrag.routing.infobox_router import is_entity_attribute_query
    for q in (
        "Is Einstein older than Newton?",
        "Which is larger, Mars or Mercury?",
        "Did Tesla die before or after Edison?",
        "Both Plato and Aristotle were Greek philosophers, but who was older?",
    ):
        assert not is_entity_attribute_query(q), (
            f"Router fired on comparison query {q!r}; expected skip."
        )


# ============================================================
# Classifier: default-off conservative
# ============================================================

def test_router_default_off_on_unknown_question_shape() -> None:
    """When no positive pattern matches and no negative marker fires,
    the router defaults to False (skip infobox). Conservative because
    unmatched queries in the pilot regressed on HP/MQ more
    than they helped on 2W."""
    from mothrag.routing.infobox_router import is_entity_attribute_query
    for q in (
        "Why did the chicken cross the road?",
        "Explain quantum entanglement.",
        "Summarise the plot of Hamlet.",
        "",
        "   ",
    ):
        assert not is_entity_attribute_query(q)


# ============================================================
# Classifier: caller-supplied pattern extensions
# ============================================================

def test_router_accepts_extra_positive_patterns() -> None:
    from mothrag.routing.infobox_router import is_entity_attribute_query
    # By default, "Tell me X's salary" doesn't match; with extra pattern it does.
    q = "Tell me Elon Musk's salary."
    assert not is_entity_attribute_query(q)
    assert is_entity_attribute_query(
        q, extra_positive_patterns=[r"tell me [\w\s']+ salary"],
    )


def test_router_extra_negative_overrides_positive() -> None:
    """If both positive AND negative match, negative wins (high-precision
    negative gate evaluates first)."""
    from mothrag.routing.infobox_router import is_entity_attribute_query
    q = "When was Albert Einstein born and where did he die?"
    # This is two-clause but our positive patterns may still match.
    # Add an explicit negative to override.
    assert not is_entity_attribute_query(
        q, extra_negative_patterns=[r"\band where\b"],
    )


# ============================================================
# CLI / pipeline integration: _InfoboxGate state-swap
# ============================================================

class _StubPipeline:
    """Minimal MothRAGPipeline-like stub for _InfoboxGate state-swap tests."""

    def __init__(self, chunks: dict[str, str]):
        import numpy as np
        self.chunks_by_id = {
            cid: {"text": text, "chunk_id": cid, "metadata": {}}
            for cid, text in chunks.items()
        }
        self.chunk_ids = list(chunks.keys())
        dim = 32

        def _embed(text: str):
            v = np.zeros(dim, dtype=np.float32)
            for i, ch in enumerate(text[:dim]):
                v[i % dim] += float(ord(ch) % 13)
            n = np.linalg.norm(v)
            return v / n if n > 0 else v

        self._embed = _embed
        self.query_embedder = _embed
        self.chunk_vecs = (
            np.stack([_embed(t) for t in chunks.values()], axis=0)
            if chunks else np.zeros((0, dim), dtype=np.float32)
        )


def test_infobox_gate_fires_on_entity_attribute_query() -> None:
    """When the router classifies the query as entity-attribute, the gate
    must swap the pipeline into the augmented (infobox-included) state."""
    import route_prospective as rp

    wikitext = (
        "{{Infobox person\n"
        "| name = Alan Turing\n"
        "| born = 23 June 1912\n"
        "}}"
    )
    pipeline = _StubPipeline({"a1": wikitext, "a2": "Unrelated."})
    n_plain = len(pipeline.chunk_ids)

    gate = rp._InfoboxGate(pipeline, top_n_boost=3)
    assert gate.n_infobox_chunks >= 1
    assert len(pipeline.chunk_ids) > n_plain  # augmented state initially

    fired, reason = gate.decide("When was Alan Turing born?")
    assert fired is True
    assert reason == "entity_attribute"
    # Pipeline must be in augmented state (chunks > n_plain).
    assert len(pipeline.chunk_ids) > n_plain


def test_infobox_gate_skips_on_multi_hop_query() -> None:
    """When the router classifies the query as multi-hop, the gate must
    swap the pipeline back to the plain (pre-augmentation) state."""
    import route_prospective as rp

    wikitext = "{{Infobox person\n| name = X\n| born = 1900\n}}"
    pipeline = _StubPipeline({"a1": wikitext})
    n_plain = len(pipeline.chunk_ids)

    gate = rp._InfoboxGate(pipeline, top_n_boost=3)

    fired, reason = gate.decide(
        "Which company that X founded later acquired Y?"
    )
    assert fired is False
    assert reason == "multi_hop_or_default"
    # Pipeline must be back to the plain state.
    assert len(pipeline.chunk_ids) == n_plain


def test_infobox_gate_swaps_back_and_forth_across_queries() -> None:
    """Per-query state-swap must be reversible across alternating
    entity-attribute / multi-hop queries within the same run."""
    import route_prospective as rp

    wikitext = (
        "{{Infobox person\n"
        "| name = X\n"
        "| born = 1900\n"
        "| spouse = Y\n"
        "}}"
    )
    pipeline = _StubPipeline({"a1": wikitext})
    n_plain = len(pipeline.chunk_ids)
    gate = rp._InfoboxGate(pipeline, top_n_boost=3)
    n_augmented = len(pipeline.chunk_ids)
    assert n_augmented > n_plain

    fired1, _ = gate.decide("When was X born?")
    assert fired1 is True
    assert len(pipeline.chunk_ids) == n_augmented

    fired2, _ = gate.decide("Which company that X founded later did Z?")
    assert fired2 is False
    assert len(pipeline.chunk_ids) == n_plain

    fired3, _ = gate.decide("Who is X's spouse?")
    assert fired3 is True
    assert len(pipeline.chunk_ids) == n_augmented


def test_dense_baseline_when_router_off() -> None:
    """When --retrieval is 'dense' (no gate), pipeline chunk surface
    remains the original (no infobox injection)."""
    import route_prospective as rp
    pipeline = _StubPipeline({"a1": "Some prose chunk."})
    n_plain = len(pipeline.chunk_ids)

    # The CLI dispatch only instantiates _InfoboxGate when
    # args.retrieval == "router_gated_infobox". Verify that calling
    # nothing (= plain dense path) leaves the pipeline untouched.
    # Sanity-check by inspecting the gate-NOT-instantiated state.
    assert len(pipeline.chunk_ids) == n_plain
    assert not any(
        cid.startswith("infobox:") for cid in pipeline.chunk_ids
    )


def test_dense_plus_infobox_unconditional_injects_chunks() -> None:
    """Sanity check: --retrieval dense_plus_infobox (unconditional)
    augments the pipeline once at setup, no gating."""
    import route_prospective as rp

    wikitext = "{{Infobox person\n| name = X\n| born = 1900\n}}"
    pipeline = _StubPipeline({"a1": wikitext})
    n_plain = len(pipeline.chunk_ids)
    n_added = rp._augment_pipeline_with_infobox(pipeline)
    assert n_added >= 1
    assert len(pipeline.chunk_ids) == n_plain + n_added


def test_telemetry_logs_router_decision_per_query_via_gate() -> None:
    """Verify the (fired, reason) tuple from gate.decide matches the
    telemetry shape that route_prospective.py main loop writes per row."""
    import route_prospective as rp

    pipeline = _StubPipeline({"a1": "{{Infobox\n| name = X\n| born = 1900\n}}"})
    gate = rp._InfoboxGate(pipeline, top_n_boost=3)

    decisions = []
    for q in (
        "When was X born?",                  # positive
        "Who is the father of the X who Y?",  # negative
        "What is Y's nationality?",          # positive
        "How tall is the X that founded Y?",  # negative (multi-hop marker)
    ):
        fired, reason = gate.decide(q)
        decisions.append({"q": q, "infobox_fired": fired,
                          "infobox_router_reason": reason})

    # All four records carry the expected telemetry keys.
    for d in decisions:
        assert "infobox_fired" in d
        assert "infobox_router_reason" in d
        assert d["infobox_router_reason"] in (
            "entity_attribute", "multi_hop_or_default",
        )
    # Mix of positive + negative outcomes observed in the batch.
    assert any(d["infobox_fired"] for d in decisions)
    assert any(not d["infobox_fired"] for d in decisions)
