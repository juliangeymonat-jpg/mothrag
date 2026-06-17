"""Tests for the gamma + L4b AND-gate primitive.

Anti-leak contract verified: the rule reads only system telemetry
(gamma_status string + l4b_stability_score float). No per-dataset
arguments, no F1 inspection, no gold-derived thresholds.
"""

from __future__ import annotations

import pytest


# ============================================================
# AND-gate truth table (10 cells)
# ============================================================

def test_valid_high_stability_keeps() -> None:
    from mothrag.core.arbitrate.gamma_l4b_andgate import (
        gamma_l4b_andgate_decision,
    )

    assert gamma_l4b_andgate_decision("valid", 0.9) == "keep"
    assert gamma_l4b_andgate_decision("valid", 1.0) == "keep"


def test_valid_low_stability_defers() -> None:
    from mothrag.core.arbitrate.gamma_l4b_andgate import (
        gamma_l4b_andgate_decision,
    )

    assert gamma_l4b_andgate_decision("valid", 0.1) == "defer"
    assert gamma_l4b_andgate_decision("valid", 0.0) == "defer"


def test_invalid_any_stability_defers() -> None:
    """gamma=invalid forces defer regardless of stability."""
    from mothrag.core.arbitrate.gamma_l4b_andgate import (
        gamma_l4b_andgate_decision,
    )

    for stab in (0.0, 0.3, 0.5, 0.7, 1.0):
        assert gamma_l4b_andgate_decision("invalid", stab) == "defer"


def test_partial_any_stability_defers() -> None:
    from mothrag.core.arbitrate.gamma_l4b_andgate import (
        gamma_l4b_andgate_decision,
    )

    for stab in (0.0, 0.5, 1.0):
        assert gamma_l4b_andgate_decision("partial", stab) == "defer"


def test_refuse_any_stability_defers() -> None:
    from mothrag.core.arbitrate.gamma_l4b_andgate import (
        gamma_l4b_andgate_decision,
    )

    for stab in (0.0, 0.5, 1.0):
        assert gamma_l4b_andgate_decision("refuse", stab) == "defer"


def test_none_status_defers() -> None:
    from mothrag.core.arbitrate.gamma_l4b_andgate import (
        gamma_l4b_andgate_decision,
    )

    assert gamma_l4b_andgate_decision(None, 0.9) == "defer"


def test_unknown_status_defers() -> None:
    """Unknown status strings fail safe to defer (no aliasing to valid)."""
    from mothrag.core.arbitrate.gamma_l4b_andgate import (
        gamma_l4b_andgate_decision,
    )

    assert gamma_l4b_andgate_decision("approved", 0.9) == "defer"
    assert gamma_l4b_andgate_decision("", 0.9) == "defer"


# ============================================================
# Threshold semantics
# ============================================================

def test_default_threshold_is_half() -> None:
    """Default threshold = 0.5 (theory-derived midpoint, no F1 fitting)."""
    from mothrag.core.arbitrate.gamma_l4b_andgate import (
        gamma_l4b_andgate_decision,
    )

    # Stability exactly at default threshold is INCLUSIVE (>= semantics)
    assert gamma_l4b_andgate_decision("valid", 0.5) == "keep"
    # Just below default threshold defers
    assert gamma_l4b_andgate_decision("valid", 0.49) == "defer"


def test_custom_threshold_overrides_default() -> None:
    from mothrag.core.arbitrate.gamma_l4b_andgate import (
        gamma_l4b_andgate_decision,
    )

    # Higher threshold: 0.5 stability now defers under 0.7 cutoff
    assert gamma_l4b_andgate_decision("valid", 0.5, threshold=0.7) == "defer"
    assert gamma_l4b_andgate_decision("valid", 0.71, threshold=0.7) == "keep"
    # Lower threshold: 0.3 stability now keeps under 0.2 cutoff
    assert gamma_l4b_andgate_decision("valid", 0.3, threshold=0.2) == "keep"


def test_threshold_boundary_inclusive() -> None:
    """Stability == threshold is INCLUSIVE (keep)."""
    from mothrag.core.arbitrate.gamma_l4b_andgate import (
        gamma_l4b_andgate_decision,
    )

    assert gamma_l4b_andgate_decision("valid", 0.6, threshold=0.6) == "keep"


# ============================================================
# Input validation (anti-leak: bounded inputs only)
# ============================================================

def test_rejects_out_of_range_stability() -> None:
    from mothrag.core.arbitrate.gamma_l4b_andgate import (
        gamma_l4b_andgate_decision,
    )

    with pytest.raises(ValueError, match="stability_score"):
        gamma_l4b_andgate_decision("valid", -0.01)
    with pytest.raises(ValueError, match="stability_score"):
        gamma_l4b_andgate_decision("valid", 1.01)


def test_rejects_out_of_range_threshold() -> None:
    from mothrag.core.arbitrate.gamma_l4b_andgate import (
        gamma_l4b_andgate_decision,
    )

    with pytest.raises(ValueError, match="threshold"):
        gamma_l4b_andgate_decision("valid", 0.5, threshold=-0.01)
    with pytest.raises(ValueError, match="threshold"):
        gamma_l4b_andgate_decision("valid", 0.5, threshold=1.01)


# ============================================================
# Anti-leak: signature has no per-dataset args
# ============================================================

def test_signature_is_generic_no_per_dataset_args() -> None:
    """The function signature MUST NOT accept dataset hints. The
    AND-gate rule is generic system telemetry only.

    Source-scan assertion: any (dataset, ds, ds_label, ds_hint,
    corpus, benchmark, ...) parameter would be a leak surface.
    """
    import inspect
    from mothrag.core.arbitrate import gamma_l4b_andgate

    sig = inspect.signature(gamma_l4b_andgate.gamma_l4b_andgate_decision)
    params = set(sig.parameters.keys())
    forbidden = {
        "dataset", "ds", "ds_label", "ds_hint", "ds_family",
        "corpus", "benchmark", "gold", "f1", "em",
    }
    leaked = params & forbidden
    assert not leaked, (
        f"AND-gate signature must not accept dataset / leak args; "
        f"found: {leaked}"
    )


# ============================================================
# Re-export from arbitrate __init__
# ============================================================

def test_andgate_reexported_from_arbitrate_package() -> None:
    from mothrag.core.arbitrate import (
        gamma_l4b_andgate_decision,
        gamma_l4b_andgate_diagnostic,
    )

    assert gamma_l4b_andgate_decision("valid", 0.9) == "keep"
    d = gamma_l4b_andgate_diagnostic("valid", 0.9)
    assert d.decision == "keep"


# ============================================================
# Diagnostic variant (telemetry-rich)
# ============================================================

def test_diagnostic_returns_keep_with_stability_reason() -> None:
    from mothrag.core.arbitrate.gamma_l4b_andgate import (
        gamma_l4b_andgate_diagnostic,
    )

    d = gamma_l4b_andgate_diagnostic("valid", 0.85)
    assert d.decision == "keep"
    assert "gamma_valid_and_stable" in d.reason
    assert "0.850" in d.reason


def test_diagnostic_returns_defer_with_gamma_reason() -> None:
    from mothrag.core.arbitrate.gamma_l4b_andgate import (
        gamma_l4b_andgate_diagnostic,
    )

    d = gamma_l4b_andgate_diagnostic("invalid", 0.9)
    assert d.decision == "defer"
    assert "gamma_not_valid" in d.reason
    assert "invalid" in d.reason


def test_diagnostic_returns_defer_with_stability_reason() -> None:
    from mothrag.core.arbitrate.gamma_l4b_andgate import (
        gamma_l4b_andgate_diagnostic,
    )

    d = gamma_l4b_andgate_diagnostic("valid", 0.3, threshold=0.5)
    assert d.decision == "defer"
    assert "stability_below_threshold" in d.reason
    assert "0.300" in d.reason
