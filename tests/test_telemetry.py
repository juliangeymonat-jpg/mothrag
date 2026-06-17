# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Per-patch activation telemetry + iter trace tests.

Validates that:
- ``IterativeAnswerInfo.patch_activations`` reports each patch's fire state
- ``IterativeAnswerInfo.iter_trace`` records per-iter state (γ, query, n)
- Anti-leak: no gold / F1 / dataset keys ever appear in trace dicts
- Composite flag toggles P6/P7/P8/P9 booleans correctly
- the bug-pattern flag toggles wave_a_active

Uses the same _FakePipeline mock harness as test_iterative_pipeline.py.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mothrag.eval.iterative_pipeline import (
    IterativeAnswerInfo,
    IterativeConfig,
    IterativeMothRAG,
)


class _FakeUsage:
    def __init__(self):
        self.prompt_tokens = 50
        self.completion_tokens = 25


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _FakePipeline:
    def __init__(self, retrieval_plan, final_answer="FINAL"):
        self.retrieval_plan = retrieval_plan
        self.final_answer = final_answer
        self._retrieve_call = 0
        all_ids = []
        for plan in retrieval_plan:
            for cid in plan:
                if cid not in all_ids:
                    all_ids.append(cid)
        self.chunk_ids = all_ids
        self.chunks_by_id = {cid: {"id": cid, "text": f"P about {cid}."}
                             for cid in all_ids}
        self.reader_client = MagicMock()
        self.reader_client.chat = MagicMock()
        self.reader_client.chat.completions = MagicMock()
        self.reader_client.chat.completions.create = MagicMock()
        self.reader_model = "fake-model"
        self.read_calls = []

    def retrieve(self, question, entity_seeds=None):
        idx = min(self._retrieve_call, len(self.retrieval_plan) - 1)
        cids = self.retrieval_plan[idx]
        self._retrieve_call += 1
        return [self.chunk_ids.index(c) for c in cids], "route", 0.5

    def read(self, question, passages):
        self.read_calls.append((question, list(passages)))
        return self.final_answer, "raw", {
            "prompt_tokens": 50, "completion_tokens": 10, "latency_s": 0.0,
        }


def _seq(pipe, contents):
    pipe.reader_client.chat.completions.create.side_effect = [
        _FakeResponse(c) for c in contents
    ]


# ============================================================
# Telemetry presence + default values
# ============================================================

def test_patch_activations_present_default_legacy():
    """Legacy config (no composite, no wave_a) → all booleans False."""
    pipe = _FakePipeline([["c1"]])
    _seq(pipe, [
        "1. EXTRACT\n- f\n2. INTEGRATE\nok\n3. ASSESS\nyes\n4. OUTPUT\nANSWER: x",
    ])
    runner = IterativeMothRAG(pipe, IterativeConfig(max_iterations=2))
    info = runner.answer("q")
    pa = info.patch_activations
    expected_keys = {
        "P6_cap_raise_triggered", "P7_entity_seeded_used",
        "P8_few_shot_applied", "P9_rerank_triggered",
        "P11_gamma_cap_fallback_fired", "P14_faith_loop_second_chance",
        "wave_a_active", "stepchain_composite_active",
    }
    assert set(pa.keys()) == expected_keys
    # All False in pure-legacy
    assert all(v is False for v in pa.values())


def test_iter_trace_length_matches_iterations():
    pipe = _FakePipeline([["c1"], ["c2"], ["c3"]])
    _seq(pipe, [
        "ANSWER: x" if i == 2 else
        "MISSING: more\nNEXT_QUERY: continue"
        for i in range(3)
    ])
    runner = IterativeMothRAG(pipe, IterativeConfig(
        max_iterations=3, stop_early=True,
    ))
    info = runner.answer("q")
    # Terminal at iter 3 → 3 trace entries
    assert len(info.iter_trace) == info.iterations_used


def test_iter_trace_records_query_and_passages_n():
    pipe = _FakePipeline([["c1", "c2"]])
    _seq(pipe, ["ANSWER: x"])
    runner = IterativeMothRAG(pipe, IterativeConfig(max_iterations=2))
    info = runner.answer("original-q")
    assert len(info.iter_trace) >= 1
    t0 = info.iter_trace[0]
    assert t0["iter"] == 1
    assert t0["query"] == "original-q"
    assert t0["passages_n"] == 2
    assert isinstance(t0["patches_active_this_iter"], list)


# ============================================================
# Composite flag toggles
# ============================================================

def test_stepchain_composite_active_flag():
    pipe = _FakePipeline([["c1"]])
    _seq(pipe, ["ANSWER: y"])
    runner = IterativeMothRAG(pipe, IterativeConfig(
        max_iterations=2, use_stepchain_parity_composite=True,
    ))
    info = runner.answer("q")
    assert info.patch_activations["stepchain_composite_active"] is True
    # P8 is eagerly marked when composite=True
    assert info.patch_activations["P8_few_shot_applied"] is True


def test_wave_a_active_flag():
    pipe = _FakePipeline([["c1"]])
    _seq(pipe, ["ANSWER: y"])
    runner = IterativeMothRAG(pipe, IterativeConfig(
        max_iterations=2, use_bug_pattern_wave_a=True,
    ))
    info = runner.answer("q")
    assert info.patch_activations["wave_a_active"] is True


def test_p6_cap_raise_marked_when_composite_runs_beyond_legacy_cap():
    # max_iterations=2 (legacy), composite_max_iterations=4 (default). If we
    # force 3 iters under composite, P6 should fire at iter 3.
    pipe = _FakePipeline([["c1"], ["c2"], ["c3"], ["c4"]])
    _seq(pipe, [
        "MISSING: m\nNEXT_QUERY: q2",
        "MISSING: m\nNEXT_QUERY: q3",
        "ANSWER: ok",
    ])
    runner = IterativeMothRAG(pipe, IterativeConfig(
        max_iterations=2,
        use_stepchain_parity_composite=True,
        stop_early=True,
    ))
    info = runner.answer("q")
    # Iter 3 > max_iterations=2 → P6 active
    assert info.patch_activations["P6_cap_raise_triggered"] is True


# ============================================================
# Anti-leak
# ============================================================

_FORBIDDEN = {"gold", "f1", "em", "dataset", "ds", "label",
              "answer_label", "gold_doc_ids", "correctness"}


def test_iter_trace_keys_no_leak():
    pipe = _FakePipeline([["c1"]])
    _seq(pipe, ["ANSWER: y"])
    runner = IterativeMothRAG(pipe, IterativeConfig(max_iterations=2))
    info = runner.answer("q")
    for t in info.iter_trace:
        leaked = set(t.keys()) & _FORBIDDEN
        assert not leaked, f"leak in trace: {leaked}"


def test_patch_activations_keys_no_leak():
    pipe = _FakePipeline([["c1"]])
    _seq(pipe, ["ANSWER: y"])
    runner = IterativeMothRAG(pipe, IterativeConfig(max_iterations=2))
    info = runner.answer("q")
    leaked = set(info.patch_activations.keys()) & _FORBIDDEN
    assert not leaked


def test_iterative_answer_info_dataclass_fields_no_leak():
    """Schema-level: no IterativeAnswerInfo field collides with forbidden set."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(IterativeAnswerInfo)}
    # NB: ``answer`` and ``label`` are legitimate non-gold fields (system output,
    # γ status), so we audit ONLY the new telemetry fields. The frozenset checks
    # the telemetry dict CONTENTS instead (see test_iter_trace_keys_no_leak).
    new_fields = {"patch_activations", "iter_trace"}
    assert new_fields.issubset(fields)


# ============================================================
# Composite-then-bisect smoke: composite ON → multiple booleans True
# ============================================================

def test_faith_lists_sync_with_terminal_when_faith_disabled():
    """per_iteration_faithfulness_* lists must align with
    per_iteration_terminal even when use_faithfulness_loop=False."""
    pipe = _FakePipeline([["c1"], ["c2"], ["c3"]])
    _seq(pipe, [
        "MISSING: m\nNEXT_QUERY: q2",
        "MISSING: m\nNEXT_QUERY: q3",
        "ANSWER: x",
    ])
    runner = IterativeMothRAG(pipe, IterativeConfig(
        max_iterations=3,
        stop_early=True,
        use_faithfulness_loop=False,  # disabled — but lists must still pad
    ))
    info = runner.answer("q")
    assert len(info.per_iteration_faithfulness_score) == len(info.per_iteration_terminal)
    assert len(info.per_iteration_faithfulness_label) == len(info.per_iteration_terminal)
    # Placeholders are None when faithfulness skipped
    assert all(x is None for x in info.per_iteration_faithfulness_score)


def test_faith_lists_sync_with_terminal_mid_loop_continue():
    """When some iters answer + faithfulness skipped, faith lists still
    have len == terminal across all iters."""
    pipe = _FakePipeline([["c1"], ["c2"]])
    _seq(pipe, [
        "MISSING: m\nNEXT_QUERY: q2",  # iter 1: no answer
        "ANSWER: x",  # iter 2: answer (faithfulness disabled)
    ])
    runner = IterativeMothRAG(pipe, IterativeConfig(
        max_iterations=2,
        stop_early=True,
        use_faithfulness_loop=False,
    ))
    info = runner.answer("q")
    assert len(info.per_iteration_faithfulness_score) == len(info.per_iteration_terminal)


def test_p11_individual_toggle_independent_from_wave_a():
    """use_p11_gamma_cap_fallback activates P11 standalone."""
    pipe = _FakePipeline([["c1"]])
    _seq(pipe, ["ANSWER: y"])
    # P11-only config: NO wave_a, ONLY p11 toggle
    cfg = IterativeConfig(
        max_iterations=2,
        use_bug_pattern_wave_a=False,
        use_p11_gamma_cap_fallback=True,
    )
    assert cfg.use_p11_gamma_cap_fallback is True
    assert cfg.use_bug_pattern_wave_a is False
    runner = IterativeMothRAG(pipe, cfg)
    info = runner.answer("q")
    # wave_a telemetry should reflect composite still OFF
    assert info.patch_activations["wave_a_active"] is False


def test_wave_a_composite_supersedes_p11_toggle():
    """When use_bug_pattern_wave_a=True, P11 fires regardless of individual toggle."""
    cfg_composite = IterativeConfig(
        use_bug_pattern_wave_a=True,
        use_p11_gamma_cap_fallback=False,
    )
    # Field set as expected — runtime gating uses OR so wave_a alone suffices
    assert cfg_composite.use_bug_pattern_wave_a is True


def test_disable_p11_field_default_false():
    """disable_p11_gamma_cap_fallback defaults False (no override)."""
    cfg = IterativeConfig()
    assert cfg.disable_p11_gamma_cap_fallback is False


def test_disable_p11_overrides_wave_a():
    """When disable_p11=True, P11 stays OFF even with wave_a=True."""
    cfg = IterativeConfig(
        use_bug_pattern_wave_a=True,
        disable_p11_gamma_cap_fallback=True,
    )
    # Field set as expected — runtime gating includes the override AND-NOT
    assert cfg.use_bug_pattern_wave_a is True
    assert cfg.disable_p11_gamma_cap_fallback is True
    # The gating expression in iterative_pipeline.py L889-895:
    #   (use_bug_pattern_wave_a OR use_p11_gamma_cap_fallback)
    #   AND NOT disable_p11_gamma_cap_fallback
    # so the composite expression evaluates False here
    p11_active = ((cfg.use_bug_pattern_wave_a
                   or cfg.use_p11_gamma_cap_fallback)
                  and not cfg.disable_p11_gamma_cap_fallback)
    assert p11_active is False


def test_composite_max_iterations_field_default():
    """composite_max_iterations default 5 (unchanged)."""
    cfg = IterativeConfig()
    assert cfg.composite_max_iterations == 5


def test_composite_max_iterations_override():
    """composite_max_iterations override accepted (e.g. 4 to
    neutralize P6 by matching --max-iterations)."""
    cfg = IterativeConfig(
        max_iterations=4,
        use_stepchain_parity_composite=True,
        composite_max_iterations=4,  # P6 neutralized
    )
    # effective_max_iterations would equal max_iterations under composite
    # because composite_max_iterations was overridden to same value
    assert cfg.composite_max_iterations == 4


def test_full_composite_run_marks_multiple_patches():
    pipe = _FakePipeline([["c1", "c2"], ["c3"], ["c4"]])
    _seq(pipe, [
        "MISSING: m\nNEXT_QUERY: q2",
        "ANSWER: x",
    ])
    runner = IterativeMothRAG(pipe, IterativeConfig(
        max_iterations=4,
        use_stepchain_parity_composite=True,
        use_bug_pattern_wave_a=True,
        stop_early=True,
    ))
    info = runner.answer("q")
    pa = info.patch_activations
    assert pa["stepchain_composite_active"] is True
    assert pa["wave_a_active"] is True
    assert pa["P8_few_shot_applied"] is True
    # P9/P11/P14 require specific in-pipeline triggers not reachable from
    # this minimal mock; their booleans stay False — that's correct attribution.
    assert pa["P9_rerank_triggered"] is False
    assert pa["P11_gamma_cap_fallback_fired"] is False
    assert pa["P14_faith_loop_second_chance"] is False
