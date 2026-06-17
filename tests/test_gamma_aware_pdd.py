# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Reversed γ-aware PDD cohort gate.

The dup (PDD) arm ``iter_dup_a`` is a COPY of iter's prediction; its effect on
arbitration is to DOUBLE-COUNT iter in pairwise agreement (signal-dup). "Zeroing
its weight" therefore means DROPPING it from the arbitration pool. This variant
reverses the earlier cohort gate: the dup is dropped ONLY on ``chain_deep``
queries that are γ=valid (the ensemble vote is noise once iter is confident
there); on every other cohort — and on the unclassified default
(``qtype=None``, pip path) — the dup is PRESERVED (the semantic_rich bulk
genuinely benefits from PDD). When iter is γ=partial/invalid the dup is always
kept. Flag OFF ⇒ legacy 4-arm behaviour, always.

These tests assert the contract at the shared core (``arbitrate_pool`` +
``gamma_aware_pdd_should_skip``) by capturing the exact ``answers`` the
``DeterministicArbitrator`` is asked to score.
"""
from __future__ import annotations

import pytest

from mothrag.core.arms_runner import arbitrate_pool, gamma_aware_pdd_should_skip


# --------------------------------------------------------------------------- #
# Test doubles: capture what the arbitrator actually scores, deterministically.
# --------------------------------------------------------------------------- #
class _StubEmbedder:
    """Unused once pairwise_agreement is stubbed; arbitrate_pool only forwards it."""


class _FakeArbResult:
    def __init__(self, answers):
        # pick any arm deterministically; the tests assert on captured pool, not this.
        self.answer = next(iter(answers.values()), "")
        self.selected_arm = next(iter(answers), "")
        self.arbitrate_signal = "stub"
        self.arm_scores = {k: 0.0 for k in answers}


def _patch_core(monkeypatch):
    """Patch the lazily-imported arbitrate symbols; record the pool passed in."""
    import mothrag.core.arbitrate as arb_mod
    captured: dict = {}

    class _FakeArbitrator:
        def __init__(self, **_kw):
            pass

        def arbitrate(self, *, answers, gamma_signals, agreement_signals,
                      arm_probabilities=None):
            captured["answers"] = dict(answers)
            captured["gamma_signals"] = dict(gamma_signals)
            captured["agreement_signals"] = dict(agreement_signals)
            return _FakeArbResult(answers)

    monkeypatch.setattr(arb_mod, "DeterministicArbitrator", _FakeArbitrator)
    monkeypatch.setattr(arb_mod, "pairwise_agreement",
                        lambda answers, **_kw: {k: 0.0 for k in answers})
    return captured


def _pool():
    # iter + its dup (PDD) + a distinct arm. iter_dup_a copies iter's pred.
    return {
        "iter": {"pred": "Frank Herbert"},
        "iter_dup_a": {"pred": "Frank Herbert"},
        "decompose": {"pred": "Brian Herbert"},
    }


def _arb(results, gamma, flag, captured, qtype=None):
    arbitrate_pool(
        results, pred_of=lambda r: r.get("pred", ""), embedder=_StubEmbedder(),
        iter_gamma_status=gamma, use_gamma_aware_pdd=flag, qtype=qtype)
    return captured["answers"]


# --------------------------------------------------------------------------- #
# 1. iter γ=valid + chain_deep + flag ON → dup dropped (effective 3-arm pool)
# --------------------------------------------------------------------------- #
def test_gamma_aware_pdd_drops_dup_when_chain_deep_valid(monkeypatch):
    captured = _patch_core(monkeypatch)
    answers = _arb(_pool(), "valid", True, captured, qtype="chain_deep")
    assert "iter_dup_a" not in answers                 # PDD signal-dup removed
    assert set(answers) == {"iter", "decompose"}       # 3-arm effective (no dup)
    # and the predicate agrees
    assert gamma_aware_pdd_should_skip(
        _pool(), "valid", "chain_deep", enabled=True) is True


# --------------------------------------------------------------------------- #
# 2. iter γ=valid + semantic_rich / default → dup PRESERVED (v2 bulk-safe)
# --------------------------------------------------------------------------- #
def test_gamma_aware_pdd_preserves_dup_on_bulk_and_default(monkeypatch):
    captured = _patch_core(monkeypatch)
    # semantic_rich bulk: preserve even when iter is γ=valid + flag ON
    answers = _arb(_pool(), "valid", True, captured, qtype="semantic_rich")
    assert "iter_dup_a" in answers
    assert gamma_aware_pdd_should_skip(
        _pool(), "valid", "semantic_rich", enabled=True) is False
    # unclassified default (qtype=None, pip path) is treated as non-chain_deep ⇒ keep
    answers2 = _arb(_pool(), "valid", True, captured, qtype=None)
    assert "iter_dup_a" in answers2
    assert gamma_aware_pdd_should_skip(_pool(), "valid", enabled=True) is False


# --------------------------------------------------------------------------- #
# 3. iter γ=invalid/partial + chain_deep + flag ON → dup KEPT (only valid drops)
# --------------------------------------------------------------------------- #
def test_gamma_aware_pdd_active_when_iter_not_valid(monkeypatch):
    captured = _patch_core(monkeypatch)
    answers = _arb(_pool(), "invalid", True, captured, qtype="chain_deep")
    assert "iter_dup_a" in answers                     # standard signal-dup active
    assert set(answers) == {"iter", "iter_dup_a", "decompose"}
    assert gamma_aware_pdd_should_skip(
        _pool(), "invalid", "chain_deep", enabled=True) is False
    assert gamma_aware_pdd_should_skip(
        _pool(), "partial", "chain_deep", enabled=True) is False


# --------------------------------------------------------------------------- #
# 4. flag OFF → legacy 4-arm always, even when chain_deep + iter γ=valid
# --------------------------------------------------------------------------- #
def test_gamma_aware_pdd_legacy_when_flag_off(monkeypatch):
    captured = _patch_core(monkeypatch)
    answers = _arb(_pool(), "valid", False, captured, qtype="chain_deep")
    assert "iter_dup_a" in answers                     # untouched legacy pool
    assert set(answers) == {"iter", "iter_dup_a", "decompose"}
    assert gamma_aware_pdd_should_skip(
        _pool(), "valid", "chain_deep", enabled=False) is False
    # no dup present → nothing to skip even when chain_deep + valid + flag on
    assert gamma_aware_pdd_should_skip(
        {"iter": {"pred": "x"}, "decompose": {"pred": "y"}},
        "valid", "chain_deep", enabled=True) is False
