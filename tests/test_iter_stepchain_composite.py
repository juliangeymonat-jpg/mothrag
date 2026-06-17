# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for the StepChain parity composite (P6 + P7 + P8 + P9).

Gated under a single flag `use_stepchain_parity_composite`.
- P6: composite_max_iterations / composite_gamma_max_retrigger override
- P7: _entity_seeded_next_query helper builds question + extracted entities
- P8: INTERMEDIATE_SYSTEM_FEW_SHOT prompt variant
- P9: composite_top_k_total + _rerank_accum_passages helper

Anti-leak: NO StepChain text / exemplar copied. Few-shot exemplars are hand-
written synthetic non-MQ chains (capital-of-country, comparison-of-dates).
Module-level constants asserted via direct import. No live LLM calls in any
test — helpers exercised in isolation.
"""
from __future__ import annotations

import inspect

import numpy as np

from mothrag.eval.iterative_pipeline import (
    INTERMEDIATE_SYSTEM,
    INTERMEDIATE_SYSTEM_FEW_SHOT,
    IterativeConfig,
    _entity_seeded_next_query,
    _extract_entities_from_text,
    _rerank_accum_passages,
)


# ============================================================
# Composite flag default + override
# ============================================================

def test_composite_flag_default_false():
    cfg = IterativeConfig()
    assert cfg.use_stepchain_parity_composite is False


def test_composite_flag_overridable():
    cfg = IterativeConfig(use_stepchain_parity_composite=True)
    assert cfg.use_stepchain_parity_composite is True


# ============================================================
# P6 — iter cap defaults
# ============================================================

def test_p6_composite_caps_present():
    cfg = IterativeConfig()
    assert cfg.composite_max_iterations == 5
    assert cfg.composite_gamma_max_retrigger == 3


def test_p6_composite_max_iter_strictly_greater_than_baseline():
    cfg = IterativeConfig()
    assert cfg.composite_max_iterations > cfg.max_iterations
    assert cfg.composite_gamma_max_retrigger > cfg.gamma_max_retrigger


def test_p6_overridable():
    cfg = IterativeConfig(composite_max_iterations=7, composite_gamma_max_retrigger=4)
    assert cfg.composite_max_iterations == 7
    assert cfg.composite_gamma_max_retrigger == 4


# ============================================================
# P7 — entity-seeded next query helper
# ============================================================

def test_entity_seeded_extracts_proper_nouns():
    entities = _extract_entities_from_text("Bill Gates founded Microsoft in 1975.")
    assert "Bill Gates" in entities
    assert "Microsoft" in entities


def test_entity_seeded_skips_wh_words():
    entities = _extract_entities_from_text("Who is Bill Gates?")
    assert "Who" not in entities
    assert "Bill Gates" in entities


def test_entity_seeded_dedups():
    entities = _extract_entities_from_text("Apple Apple Microsoft")
    # "Apple Apple" appears as one span, dedup
    assert len(entities) <= 2


def test_entity_seeded_caps_at_max():
    # Use lowercase separators so each capitalized noun is its own span
    entities = _extract_entities_from_text(
        "Apple is here. Microsoft is also here. Google was founded. "
        "Amazon launched. Tesla grew. Nvidia rose.",
        max_entities=3,
    )
    assert len(entities) == 3


def test_entity_seeded_next_query_uses_extracted_entities():
    q = "Who founded the company?"
    cue = "The company is Microsoft, founded by Bill Gates in Albuquerque."
    out = _entity_seeded_next_query(q, cue)
    # Question preserved + at least one entity appended
    assert q in out
    assert any(e in out for e in ("Microsoft", "Bill Gates", "Albuquerque"))


def test_entity_seeded_falls_back_to_question_when_no_entities():
    q = "What is the answer?"
    cue = "no proper nouns here just lowercase text"
    out = _entity_seeded_next_query(q, cue)
    assert out == q


def test_entity_seeded_uses_pipe_link_entities_when_provided():
    def fake_link(text):
        return ["FakeEntity"]

    q = "Q?"
    out = _entity_seeded_next_query(q, "Some claim", pipe_link_entities=fake_link)
    assert "FakeEntity" in out


def test_entity_seeded_falls_back_when_link_raises():
    def crashy(text):
        raise RuntimeError("link broken")

    q = "Q?"
    cue = "Bill Gates founded Microsoft."
    out = _entity_seeded_next_query(q, cue, pipe_link_entities=crashy)
    # Falls back to cheap extractor → "Bill Gates" or "Microsoft" appears
    assert any(e in out for e in ("Bill Gates", "Microsoft"))


# ============================================================
# P8 — few-shot system prompt
# ============================================================

def test_p8_few_shot_prompt_is_distinct():
    assert INTERMEDIATE_SYSTEM != INTERMEDIATE_SYSTEM_FEW_SHOT


def test_p8_few_shot_prompt_contains_exemplar_markers():
    """Exemplars block must be present and labeled."""
    assert "EXAMPLE 1" in INTERMEDIATE_SYSTEM_FEW_SHOT
    assert "EXAMPLE 2" in INTERMEDIATE_SYSTEM_FEW_SHOT


def test_p8_few_shot_includes_next_entity_field():
    """P7 NEXT_ENTITY: schema documented in few-shot prompt."""
    assert "NEXT_ENTITY:" in INTERMEDIATE_SYSTEM_FEW_SHOT


def test_p8_few_shot_anti_leak_no_mq_exemplars():
    """Exemplars must be synthetic / public (non-MQ)."""
    # Synthetic exemplars use well-known public facts: Bell/Edinburgh/UK +
    # Brooklyn Bridge/Eiffel Tower. No MuSiQue test entities like the
    # decomposed multi-hop question patterns that appear in MQ gold.
    body = INTERMEDIATE_SYSTEM_FEW_SHOT.lower()
    # MQ-specific query phrasings shouldn't appear verbatim
    assert "musique" not in body
    assert "2wiki" not in body


def test_p8_few_shot_preserves_scaffold():
    """4-stage scaffold (EXTRACT/INTEGRATE/ASSESS/OUTPUT) still present."""
    for stage in ("EXTRACT", "INTEGRATE", "ASSESS", "OUTPUT"):
        assert stage in INTERMEDIATE_SYSTEM_FEW_SHOT


# ============================================================
# P9 — context accumulation cap + rerank helper
# ============================================================

def test_p9_composite_top_k_default():
    cfg = IterativeConfig()
    assert cfg.composite_top_k_total == 20


def test_p9_composite_top_k_greater_than_baseline():
    cfg = IterativeConfig()
    assert cfg.composite_top_k_total > cfg.top_k_total


def test_p9_rerank_no_embedder_returns_fifo_truncation():
    ids = ["a", "b", "c", "d", "e"]
    texts = ["A", "B", "C", "D", "E"]
    out_ids, out_texts = _rerank_accum_passages(
        ids, texts, "query", embed_fn=None, top_k=3,
    )
    assert out_ids == ["a", "b", "c"]
    assert out_texts == ["A", "B", "C"]


def test_p9_rerank_empty_returns_empty():
    out_ids, out_texts = _rerank_accum_passages(
        [], [], "query", embed_fn=lambda xs: np.zeros((len(xs), 4)), top_k=5,
    )
    assert out_ids == []
    assert out_texts == []


def test_p9_rerank_reorders_by_cosine():
    """Mock embedder makes "C" highest-cosine to "query"."""
    def mock_embed(texts: list[str]):
        # Hand-tune: query embed aligned with passage "C" only
        emb = np.zeros((len(texts), 3), dtype=np.float32)
        for i, t in enumerate(texts):
            if t == "query":
                emb[i] = np.array([1.0, 0.0, 0.0])
            elif t == "C":
                emb[i] = np.array([1.0, 0.0, 0.0])  # max sim to query
            elif t == "A":
                emb[i] = np.array([0.5, 0.5, 0.0])  # medium
            elif t == "B":
                emb[i] = np.array([0.0, 1.0, 0.0])  # orthogonal
            else:
                emb[i] = np.array([0.0, 0.0, 1.0])  # orthogonal
        return emb

    ids = ["id_a", "id_b", "id_c"]
    texts = ["A", "B", "C"]
    out_ids, out_texts = _rerank_accum_passages(
        ids, texts, "query", embed_fn=mock_embed, top_k=3,
    )
    # C should be first (highest cosine), then A, then B
    assert out_texts[0] == "C"


def test_p9_rerank_embedder_failure_falls_back_to_fifo():
    def crashy(texts):
        raise RuntimeError("embed broken")

    ids = ["a", "b"]
    texts = ["A", "B"]
    out_ids, _ = _rerank_accum_passages(
        ids, texts, "query", embed_fn=crashy, top_k=5,
    )
    assert out_ids == ["a", "b"]  # FIFO preserved


# ============================================================
# Composite flag side-effect (sanity — does NOT mutate baseline config fields)
# ============================================================

def test_composite_flag_does_not_mutate_baseline_caps():
    """Flag is additive — base fields unchanged for downstream non-composite consumers."""
    cfg = IterativeConfig(use_stepchain_parity_composite=True)
    assert cfg.max_iterations == 4  # baseline default unchanged
    assert cfg.gamma_max_retrigger == 2
    assert cfg.top_k_total == 15


# ============================================================
# Anti-leak signatures (CRITICAL)
# ============================================================

_FORBIDDEN = {"gold", "f1", "em", "dataset", "ds", "ds_label",
              "corpus", "benchmark", "label", "answer_label", "gold_doc_ids"}


def test_iterative_config_signature_clean():
    sig = inspect.signature(IterativeConfig)
    leaked = set(sig.parameters.keys()) & _FORBIDDEN
    assert not leaked, f"IterativeConfig leaked: {leaked}"


def test_entity_seeded_signature_clean():
    sig = inspect.signature(_entity_seeded_next_query)
    leaked = set(sig.parameters.keys()) & _FORBIDDEN
    assert not leaked


def test_rerank_signature_clean():
    sig = inspect.signature(_rerank_accum_passages)
    leaked = set(sig.parameters.keys()) & _FORBIDDEN
    assert not leaked


def test_composite_rerank_embed_fn_default_none():
    cfg = IterativeConfig()
    assert cfg.composite_rerank_embed_fn is None
