"""Tests for the generic refuse abstention rule.

Anti-leak verified: rule depends only on gamma_status string +
operator-selected pipeline_mode. No per-DS args, no F1 inspection,
no gold-derived thresholds.
"""

from __future__ import annotations

import pytest


# ============================================================
# Trigger truth (gamma_status binary classification)
# ============================================================

def test_refuse_triggers_true() -> None:
    from mothrag.core.arbitrate.refuse_abstention import (
        refuse_abstention_trigger,
    )

    assert refuse_abstention_trigger("refuse") is True


def test_valid_does_not_trigger() -> None:
    from mothrag.core.arbitrate.refuse_abstention import (
        refuse_abstention_trigger,
    )

    assert refuse_abstention_trigger("valid") is False


def test_invalid_does_not_trigger() -> None:
    from mothrag.core.arbitrate.refuse_abstention import (
        refuse_abstention_trigger,
    )

    assert refuse_abstention_trigger("invalid") is False


def test_partial_does_not_trigger() -> None:
    from mothrag.core.arbitrate.refuse_abstention import (
        refuse_abstention_trigger,
    )

    assert refuse_abstention_trigger("partial") is False


def test_none_does_not_trigger() -> None:
    from mothrag.core.arbitrate.refuse_abstention import (
        refuse_abstention_trigger,
    )

    assert refuse_abstention_trigger(None) is False


def test_unknown_status_does_not_trigger() -> None:
    """Fail-safe: any non-'refuse' string returns False (no aliasing)."""
    from mothrag.core.arbitrate.refuse_abstention import (
        refuse_abstention_trigger,
    )

    assert refuse_abstention_trigger("Refuse") is False
    assert refuse_abstention_trigger("REFUSE") is False
    assert refuse_abstention_trigger("abstain") is False
    assert refuse_abstention_trigger("") is False


# ============================================================
# Dual-mode dispatch (loop vs abstention)
# ============================================================

def test_dispatch_abstention_mode_emits_marker_when_triggered() -> None:
    from mothrag.core.arbitrate.refuse_abstention import (
        refuse_abstention_dispatch,
    )

    d = refuse_abstention_dispatch("refuse", "abstention")
    assert d.triggered is True
    assert d.emit_abstain_marker is True
    assert d.pipeline_mode == "abstention"
    assert d.gamma_status == "refuse"


def test_dispatch_loop_mode_never_emits_marker() -> None:
    """Loop mode ALWAYS produces an answer via Stage 6 soft fallback;
    refuse trigger should NOT emit the abstain marker in loop mode
    (only telemetry propagates)."""
    from mothrag.core.arbitrate.refuse_abstention import (
        refuse_abstention_dispatch,
    )

    d = refuse_abstention_dispatch("refuse", "loop")
    assert d.triggered is True
    assert d.emit_abstain_marker is False
    assert d.pipeline_mode == "loop"


def test_dispatch_non_refuse_status_does_not_emit() -> None:
    from mothrag.core.arbitrate.refuse_abstention import (
        refuse_abstention_dispatch,
    )

    for status in ("valid", "invalid", "partial", None):
        for mode in ("loop", "abstention"):
            d = refuse_abstention_dispatch(status, mode)
            assert d.triggered is False
            assert d.emit_abstain_marker is False


def test_dispatch_rejects_unknown_pipeline_mode() -> None:
    from mothrag.core.arbitrate.refuse_abstention import (
        refuse_abstention_dispatch,
    )

    with pytest.raises(ValueError, match="pipeline_mode"):
        refuse_abstention_dispatch("refuse", "bogus")  # type: ignore[arg-type]


def test_dispatch_returns_frozen_dataclass() -> None:
    from mothrag.core.arbitrate.refuse_abstention import (
        refuse_abstention_dispatch,
    )
    from dataclasses import FrozenInstanceError

    d = refuse_abstention_dispatch("refuse", "abstention")
    with pytest.raises(FrozenInstanceError):
        d.triggered = False  # type: ignore[misc]


def test_dispatch_trigger_name_constant_for_telemetry() -> None:
    from mothrag.core.arbitrate.refuse_abstention import (
        refuse_abstention_dispatch, RefuseTrigger,
    )

    d = refuse_abstention_dispatch("refuse", "abstention")
    assert d.trigger_name == RefuseTrigger
    assert RefuseTrigger == "refuse_abstention"


# ============================================================
# Anti-leak: signature has no per-dataset args
# ============================================================

def test_trigger_signature_is_generic() -> None:
    import inspect
    from mothrag.core.arbitrate import refuse_abstention

    sig = inspect.signature(refuse_abstention.refuse_abstention_trigger)
    params = set(sig.parameters.keys())
    forbidden = {
        "dataset", "ds", "ds_label", "ds_hint", "ds_family",
        "corpus", "benchmark", "gold", "f1", "em", "threshold",
    }
    leaked = params & forbidden
    assert not leaked, f"Trigger signature must not accept leak args; found: {leaked}"


def test_dispatch_signature_is_generic() -> None:
    import inspect
    from mothrag.core.arbitrate import refuse_abstention

    sig = inspect.signature(refuse_abstention.refuse_abstention_dispatch)
    params = set(sig.parameters.keys())
    forbidden = {
        "dataset", "ds", "ds_label", "ds_hint", "ds_family",
        "corpus", "benchmark", "gold", "f1", "em", "threshold",
    }
    leaked = params & forbidden
    assert not leaked, f"Dispatch signature must not accept leak args; found: {leaked}"


# ============================================================
# Re-export from arbitrate package
# ============================================================

def test_refuse_helpers_reexported() -> None:
    from mothrag.core.arbitrate import (
        refuse_abstention_trigger,
        refuse_abstention_dispatch,
        RefuseDispatch,
        RefuseTrigger,
    )

    assert refuse_abstention_trigger("refuse") is True
    d = refuse_abstention_dispatch("refuse", "abstention")
    assert isinstance(d, RefuseDispatch)
    assert RefuseTrigger == "refuse_abstention"
