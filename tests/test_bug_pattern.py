# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for the bug-pattern composite (P11+P12+P13+P14+P15+P24)."""
from __future__ import annotations

import inspect
import os

from mothrag.core.selective_ensemble import (
    _EXTENDED_UNCERTAIN,
    _LEGACY_UNCERTAIN,
    is_uncertain,
)
from mothrag.eval.iterative_pipeline import IterativeConfig


# ============================================================
# bug-pattern flag (config-side)
# ============================================================

def test_wave_a_flag_default_false():
    cfg = IterativeConfig()
    assert cfg.use_bug_pattern_wave_a is False


def test_wave_a_flag_overridable():
    cfg = IterativeConfig(use_bug_pattern_wave_a=True)
    assert cfg.use_bug_pattern_wave_a is True


def test_wave_a_flag_orthogonal_to_other_levers():
    cfg = IterativeConfig(
        use_bug_pattern_wave_a=True,
        use_gamma_router=False,
        cap_branch_softfallback=False,
    )
    assert cfg.use_bug_pattern_wave_a is True
    assert cfg.use_gamma_router is False
    assert cfg.cap_branch_softfallback is False


# ============================================================
# P24 — extended ABSTAIN_MARKERS via is_uncertain
# ============================================================

def test_p24_extended_marker_set_strictly_includes_legacy():
    for m in _LEGACY_UNCERTAIN:
        assert m in _EXTENDED_UNCERTAIN


def test_p24_extended_adds_idk_variants():
    # Stored as post-normalized (punctuation stripped)
    assert "i dont know" in _EXTENDED_UNCERTAIN
    assert "i do not know" in _EXTENDED_UNCERTAIN
    assert "cannot answer" in _EXTENDED_UNCERTAIN
    # Legacy did NOT cover these
    assert "i dont know" not in _LEGACY_UNCERTAIN
    assert "cannot answer" not in _LEGACY_UNCERTAIN


def test_p24_legacy_default_is_uncertain_behavior(monkeypatch):
    """Without env var, behaves as legacy (4 markers)."""
    monkeypatch.delenv("MOTHRAG_BUG_PATTERN_WAVE_A", raising=False)
    assert is_uncertain("not in passages") is True
    assert is_uncertain("I don't know") is False  # legacy misses this
    assert is_uncertain("Paris") is False


def test_p24_wave_a_extended_behavior(monkeypatch):
    monkeypatch.setenv("MOTHRAG_BUG_PATTERN_WAVE_A", "1")
    assert is_uncertain("not in passages") is True
    assert is_uncertain("I don't know") is True  # P24 catches this
    assert is_uncertain("I do not know") is True
    assert is_uncertain("Cannot answer") is True
    assert is_uncertain("Paris") is False
    # Legacy markers still in set
    assert is_uncertain("unknown") is True
    assert is_uncertain("no answer") is True


def test_p24_env_var_off_string_falls_back_to_legacy(monkeypatch):
    monkeypatch.setenv("MOTHRAG_BUG_PATTERN_WAVE_A", "0")
    assert is_uncertain("I don't know") is False  # legacy behavior


def test_p24_empty_string_always_uncertain():
    assert is_uncertain("") is True
    assert is_uncertain("   ") is True


# ============================================================
# P12+P13 sentinel: env var gating semantics + module-level constants
# ============================================================

def test_p12_p13_env_var_default_unset(monkeypatch):
    """Default env var unset → legacy behavior preserved (decompose collapses
    on >6 sub_qs; prior_facts include all sub_qa).
    The actual decompose logic is exercised in integration tests; here we
    assert that the env-var-check pattern fires correctly.
    """
    monkeypatch.delenv("MOTHRAG_BUG_PATTERN_WAVE_A", raising=False)
    wave_a = os.environ.get("MOTHRAG_BUG_PATTERN_WAVE_A") == "1"
    assert wave_a is False


def test_p12_p13_env_var_set(monkeypatch):
    monkeypatch.setenv("MOTHRAG_BUG_PATTERN_WAVE_A", "1")
    wave_a = os.environ.get("MOTHRAG_BUG_PATTERN_WAVE_A") == "1"
    assert wave_a is True


# ============================================================
# P11+P14+P15 — code presence (smoke-level, full behavior via integration)
# ============================================================

def test_iterative_pipeline_module_references_wave_a():
    """The bug-pattern flag must be referenced at the documented patch sites."""
    src = open(
        os.path.join(
            os.path.dirname(__file__), "..", "mothrag", "eval", "iterative_pipeline.py"
        ),
        encoding="utf-8",
    ).read()
    # bug-pattern flag declared
    assert "use_bug_pattern_wave_a: bool = False" in src
    # P11 wiring: free-text reader fallback when flag True (or per-patch toggle
    # since later changes extended the original composite-only gate to a
    # composite-OR-individual-AND-NOT-disabled form).
    assert "P11" in src
    assert "cfg.use_bug_pattern_wave_a" in src
    assert "accum_texts" in src
    assert "P11_gamma_cap_fallback_fired" in src
    # P14 wiring: faith-loop cap-exhausted ungrounded drop
    assert "P14" in src
    assert "early_answer = None" in src


def test_route_prospective_module_references_wave_a():
    src = open(
        os.path.join(
            os.path.dirname(__file__), "..", "scripts", "route_prospective.py"
        ),
        encoding="utf-8",
    ).read()
    assert "P12" in src
    assert "P13" in src
    assert "MOTHRAG_BUG_PATTERN_WAVE_A" in src
    # P12: truncate-not-collapse path present
    assert "sub_qs = sub_qs[:6]" in src
    # P13: is_uncertain filter on prior_facts
    assert "is_uncertain(psa)" in src


def test_selective_ensemble_module_references_wave_a():
    src = open(
        os.path.join(
            os.path.dirname(__file__), "..", "mothrag", "core", "selective_ensemble.py"
        ),
        encoding="utf-8",
    ).read()
    assert "P24" in src
    assert "_EXTENDED_UNCERTAIN" in src
    assert "MOTHRAG_BUG_PATTERN_WAVE_A" in src


# ============================================================
# Anti-leak signatures
# ============================================================

_FORBIDDEN = {"gold", "f1", "em", "dataset", "ds", "ds_label",
              "corpus", "benchmark", "label", "answer_label", "gold_doc_ids"}


def test_iterative_config_signature_clean():
    sig = inspect.signature(IterativeConfig)
    leaked = set(sig.parameters.keys()) & _FORBIDDEN
    assert not leaked


def test_is_uncertain_signature_clean():
    sig = inspect.signature(is_uncertain)
    leaked = set(sig.parameters.keys()) & _FORBIDDEN
    assert not leaked
