# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Generic refuse abstention rule.

Pure deterministic rule:

    IF gamma_final_status == "refuse" THEN trigger abstain pathway.

No per-dataset conditioning, no F1 threshold tuning, no gold inspection.
The rule reads only the gamma verifier output and produces a single
boolean trigger for downstream dual-mode dispatch.

Origin: an empirical gamma calibration sweep observed that the
``refuse`` gamma status fires 12.9% of MQ queries and is followed
by an empirical F1 mean of 0.048 -- a high-precision wrong-answer
*observation*. This module encodes the corresponding deterministic
rule. Whether the rule's dispatch is right for any specific
pipeline / dataset is a downstream operator decision; this
module supplies the primitive only.

Anti-leak contract:
* Trigger depends ONLY on the gamma_status string.
* No per-dataset parameters.
* No F1 thresholds, no gold-derived constants.
* Output is a boolean + (in the dispatch helper) the pipeline mode
  the operator already chose (loop / abstention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional


__all__ = [
    "RefuseTrigger",
    "RefuseDispatchMode",
    "RefuseDispatch",
    "refuse_abstention_trigger",
    "refuse_abstention_dispatch",
]


# Type alias for the pipeline mode the dispatch helper accepts.
RefuseDispatchMode = Literal["loop", "abstention"]


# Constant identifier for use in telemetry / structured logs.
RefuseTrigger = "refuse_abstention"


@dataclass(frozen=True)
class RefuseDispatch:
    """Structured dispatch outcome for downstream wiring.

    Attributes:
        triggered: True iff the rule fired (gamma_status == "refuse").
        emit_abstain_marker: True iff the operator-selected pipeline
            mode is ``"abstention"`` AND the trigger fired. In
            ``"loop"`` mode, downstream stages still receive the
            trigger telemetry (via ``triggered``) but should NOT emit
            the abstain marker -- they soft-fallback per their own
            Stage 6 contract.
        gamma_status: echo of the input, for telemetry log parity.
        pipeline_mode: echo of the input.
        trigger_name: constant string identifier for log aggregation.
    """

    triggered: bool
    emit_abstain_marker: bool
    gamma_status: Optional[str]
    pipeline_mode: RefuseDispatchMode
    trigger_name: str = RefuseTrigger


def refuse_abstention_trigger(gamma_status: Optional[str]) -> bool:
    """Return True iff ``gamma_status == "refuse"``.

    The single-line decision rule. Generic system-telemetry only.

    Args:
        gamma_status: gamma verifier output. Any value other than the
            literal string ``"refuse"`` returns False (including
            ``None`` and unknown statuses -- fail-safe to no-trigger).

    Returns:
        bool: True only when the gamma verifier explicitly emitted
        the refuse status.
    """
    return gamma_status == "refuse"


def refuse_abstention_dispatch(
    gamma_status: Optional[str],
    pipeline_mode: RefuseDispatchMode,
) -> RefuseDispatch:
    """Produce a structured :class:`RefuseDispatch` for dual-mode wiring.

    Args:
        gamma_status: gamma verifier output.
        pipeline_mode: operator-selected pipeline mode (``"loop"`` or
            ``"abstention"``).

    Returns:
        :class:`RefuseDispatch` with the trigger boolean + a
        downstream-actionable ``emit_abstain_marker`` flag. In
        ``"loop"`` mode the marker is never emitted (loop ALWAYS
        produces an answer via Stage 6 soft fallback). In
        ``"abstention"`` mode the marker fires iff the trigger fired.

    Raises:
        ValueError: when ``pipeline_mode`` is not one of
            ``"loop"`` / ``"abstention"``.
    """
    if pipeline_mode not in ("loop", "abstention"):
        raise ValueError(
            f"pipeline_mode must be 'loop' or 'abstention', "
            f"got {pipeline_mode!r}"
        )
    triggered = refuse_abstention_trigger(gamma_status)
    emit = triggered and pipeline_mode == "abstention"
    return RefuseDispatch(
        triggered=triggered,
        emit_abstain_marker=emit,
        gamma_status=gamma_status,
        pipeline_mode=pipeline_mode,
    )
