"""Tests for CLI flag wiring on route_prospective.py.

Source-scan tests for route_prospective.py. The full functional tests
require corpus + APIs; we verify here that the flags are defined,
default-preserved, and threaded into the arbitrate / dispatch path.
"""

from __future__ import annotations

from pathlib import Path


_RP_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "route_prospective.py"
)


def _rp() -> str:
    return _RP_PATH.read_text(encoding="utf-8")


# ============================================================
# Flag presence
# ============================================================

def test_route_prospective_has_tie_break_flag() -> None:
    src = _rp()
    assert '"--tie-break"' in src
    assert '"priority"' in src and '"lexicographic"' in src
    assert '"first"' in src and '"random"' in src


def test_route_prospective_has_disable_fallback_flag() -> None:
    src = _rp()
    assert '"--disable-fallback"' in src


def test_route_prospective_has_w_weight_flags() -> None:
    src = _rp()
    assert '"--w-agree"' in src
    assert '"--w-gamma"' in src
    assert '"--w-faith"' in src


def test_route_prospective_has_dup_random_answer_flag() -> None:
    src = _rp()
    assert '"--dup-random-answer"' in src


def test_route_prospective_has_simulate_n_cap_flag() -> None:
    src = _rp()
    assert '"--simulate-n-cap"' in src


# ============================================================
# Default behavior preservation
# ============================================================

def test_route_prospective_tie_break_default_priority() -> None:
    src = _rp()
    assert '"--tie-break", default="priority"' in src


def test_route_prospective_w_defaults_match_DeterministicArbitrator() -> None:
    """Defaults must match DEFAULT_WEIGHTS so byte-compat with the
    legacy arbitrate path is preserved when no flag is passed."""
    src = _rp()
    assert '"--w-agree", type=float, default=0.5' in src
    assert '"--w-gamma", type=float, default=1.0' in src
    assert '"--w-faith", type=float, default=0.3' in src


# ============================================================
# Pass-through wiring
# ============================================================

def test_route_prospective_threads_weights_to_arbitrate_candidates() -> None:
    src = _rp()
    assert "w_gamma=args.w_gamma" in src
    assert "w_agree=args.w_agree" in src
    assert "w_faith=args.w_faith" in src


def test_route_prospective_threads_dup_random_answer() -> None:
    src = _rp()
    assert "dup_random_answer=args.dup_random_answer" in src


def test_route_prospective_threads_simulate_n_cap() -> None:
    src = _rp()
    assert "simulate_n_cap=args.simulate_n_cap" in src


# ============================================================
# DeterministicArbitrator instantiated with weights
# ============================================================

def test_arbitrate_candidates_instantiates_with_w_weights() -> None:
    """The weights must be threaded into the arbitrator. The
    DeterministicArbitrator construction lives in the unified
    ``mothrag.core.arms_runner.arbitrate_pool``; ``_arbitrate_candidates``
    THREADS w_gamma/w_agree/w_faith into ``arbitrate_pool``, which instantiates
    the arbitrator with them (functionally identical, single source of truth)."""
    src = _rp()
    assert "arbitrate_pool(" in src
    assert "w_gamma=w_gamma, w_agree=w_agree, w_faith=w_faith" in src
    # arms_runner.arbitrate_pool constructs the arbitrator WITH the threaded weights.
    ar_src = (Path(__file__).resolve().parent.parent
              / "mothrag" / "core" / "arms_runner.py").read_text(encoding="utf-8")
    assert "DeterministicArbitrator(" in ar_src
    assert "w_gamma=w_gamma, w_agree=w_agree, w_faith=w_faith" in ar_src


# ============================================================
# Dup-random + n-cap implementation source-scan
# ============================================================

def test_dup_random_answer_uses_is_dup_arm() -> None:
    src = _rp()
    assert "from mothrag.routing.dup_arm import is_dup_arm" in src
    # Confirm the loop replaces dup preds via random.choice over other preds
    assert "is_dup_arm" in src and "_rng047.choice" in src


def test_simulate_n_cap_rescales_agreement_denominator() -> None:
    # _arbitrate_candidates threads simulate_n_cap into the unified scoring core.
    src = _rp()
    assert "simulate_n_cap" in src
    assert "simulate_n_cap=simulate_n_cap" in src
    # The agreement-denominator re-scaling lives in
    # arms_runner.arbitrate_pool (single source of truth), functionally identical.
    ar_src = (Path(__file__).resolve().parent.parent
              / "mothrag" / "core" / "arms_runner.py").read_text(encoding="utf-8")
    assert "scale = n_others / float(capped_others)" in ar_src


# ============================================================
# Anti-leak audit on the flags
# ============================================================

def test_flags_signature_is_generic_no_per_dataset() -> None:
    """No per-dataset / gold args introduced by the flag block."""
    src = _rp()
    # Confirm no per-DS hook added near the flag block
    forbidden_substrings = (
        "--dataset-target", "--gold-hint", "--f1-hint", "--em-hint",
    )
    for s in forbidden_substrings:
        assert s not in src, f"flag block must not introduce {s}"
