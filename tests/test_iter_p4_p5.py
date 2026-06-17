# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for the P4 abstain-marker filter + P5 cap-branch SoftFallback flag.

P4: filter `next_query = "Find passages that ground the claim: <abstain>"`
when answer is a known abstain marker — restores intent of an earlier audit fix.

P5: new `cap_branch_softfallback` flag in IterativeConfig that fires
free-text reader on accumulated passages at the cap branch when γ tree has no
naturalized_answer. Default False — flag toggle test only here; smoke validation
shipped separately.
"""
from __future__ import annotations

import inspect

from mothrag.eval.iterative_pipeline import (
    ABSTAIN_MARKERS,
    IterativeConfig,
    _is_abstain_marker,
)


# ============================================================
# P4 — abstain-marker filter
# ============================================================

def test_abstain_markers_constant_is_frozen():
    """Module-level ABSTAIN_MARKERS is frozen to prevent runtime mutation."""
    assert isinstance(ABSTAIN_MARKERS, frozenset)


def test_abstain_markers_covers_canonical_set():
    """Spec set: 'not in passages' is the headline marker per the audit."""
    assert "not in passages" in ABSTAIN_MARKERS
    assert "i don't know" in ABSTAIN_MARKERS
    assert "unknown" in ABSTAIN_MARKERS


def test_is_abstain_marker_canonical_string():
    assert _is_abstain_marker("Not in passages") is True
    assert _is_abstain_marker("NOT IN PASSAGES") is True
    assert _is_abstain_marker("  not in passages  ") is True


def test_is_abstain_marker_idk_variants():
    assert _is_abstain_marker("I don't know") is True
    assert _is_abstain_marker("i do not know") is True


def test_is_abstain_marker_empty_or_none():
    assert _is_abstain_marker(None) is True
    assert _is_abstain_marker("") is True
    assert _is_abstain_marker("   ") is True


def test_is_abstain_marker_real_answer_falsy():
    assert _is_abstain_marker("Bill Gates") is False
    assert _is_abstain_marker("Paris") is False
    assert _is_abstain_marker("The capital of France is Paris.") is False


def test_is_abstain_marker_does_not_substring_match():
    """'no' or 'unknown rocker' should NOT match — exact lower match only."""
    assert _is_abstain_marker("No, that is wrong") is False
    assert _is_abstain_marker("the unknown soldier was buried") is False


# ============================================================
# P5 — cap_branch_softfallback flag toggle
# ============================================================

def test_cap_branch_softfallback_field_exists():
    cfg = IterativeConfig()
    assert hasattr(cfg, "cap_branch_softfallback")


def test_cap_branch_softfallback_default_false():
    """Default False per spec — smoke validation gates default flip."""
    cfg = IterativeConfig()
    assert cfg.cap_branch_softfallback is False


def test_cap_branch_softfallback_overridable_true():
    cfg = IterativeConfig(cap_branch_softfallback=True)
    assert cfg.cap_branch_softfallback is True


def test_cap_branch_softfallback_is_orthogonal_to_use_gamma_router():
    """P5 and use_gamma_router are independent levers."""
    cfg = IterativeConfig(
        cap_branch_softfallback=True, use_gamma_router=False,
    )
    assert cfg.cap_branch_softfallback is True
    assert cfg.use_gamma_router is False


def test_cap_branch_softfallback_default_preserves_legacy_behavior():
    """Default False = backward-compat (legacy baseline behavior)."""
    cfg = IterativeConfig(use_gamma_refuse_loop=True)
    # P1 default flipped True, P5 default False — confirm
    assert cfg.use_gamma_refuse_loop is True
    assert cfg.cap_branch_softfallback is False


# ============================================================
# Anti-leak signatures — both flags
# ============================================================

_FORBIDDEN = {"gold", "f1", "em", "dataset", "ds", "ds_label",
              "corpus", "benchmark", "label", "answer_label", "gold_doc_ids"}


def test_iterative_config_signature_no_gold_args():
    sig = inspect.signature(IterativeConfig)
    leaked = set(sig.parameters.keys()) & _FORBIDDEN
    assert not leaked, f"IterativeConfig leaked: {leaked}"


def test_is_abstain_marker_signature_clean():
    sig = inspect.signature(_is_abstain_marker)
    leaked = set(sig.parameters.keys()) & _FORBIDDEN
    assert not leaked
