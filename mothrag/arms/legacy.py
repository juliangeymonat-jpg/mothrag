# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Thin Arm-Protocol wrappers around the legacy function-based runners.

The production stack executes V3+bu / decompose / iter
as function-based runners (``_run_v3bu`` / ``_run_decompose`` /
``_run_iter`` in ``scripts/route_prospective.py``). These wrappers
expose the same arms behind the :class:`Arm` Protocol so MOTHRAG
component-autonomy code paths (e.g.
:meth:`mothrag.core.arbitrate.ArbitratorV2.arbitrate_pool`) can consult
``Arm.applicable(question)`` uniformly across legacy and opt-in arms.

Design intent
-------------

The wrappers DO NOT replace the function-based runners. They thin-wrap
a caller-supplied runner callable + an applicability predicate from
:mod:`mothrag.routing.arm_applicability`:

  - :class:`V3buArmWrapper`        uses :func:`is_v3bu_applicable`
  - :class:`DecomposeArmWrapper`   uses :func:`is_decompose_applicable`
  - :class:`IterArmWrapper`        uses :func:`is_iter_applicable`

The legacy production main-loop is UNCHANGED -- it still calls
``_run_v3bu(pipeline, question)`` directly. The wrappers are an
ADDITIVE surface for MOTHRAG consumers that want to drive the
arm pool through the Arm Protocol abstraction.

The architecture is HYBRID (PAM-orchestrator for
routing-by-winner-prediction + component-autonomy on
``Arm.applicable``). These wrappers are the component-autonomy half.

Design contract: no per-dataset tuning; predicates are deterministic
linguistic regex.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from mothrag.arms.base import ArmResult
from mothrag.routing.arm_applicability import (
    is_decompose_applicable,
    is_iter_applicable,
    is_v3bu_applicable,
)


def _adapt_runner_result(raw: Mapping[str, Any] | ArmResult) -> ArmResult:
    """Adapt legacy ``_run_*`` dict shape to :class:`ArmResult`.

    The legacy runners return dicts with keys ``pred``,
    ``retrieved_chunk_ids``, ``n_llm_calls``, ``prompt_tokens``,
    ``completion_tokens``, ``latency_s`` (+ arm-specific extras like
    ``gamma_final_status`` for iter). This adapter packs the standard
    fields into :class:`ArmResult` and stashes the rest in
    ``metadata``.
    """
    if isinstance(raw, ArmResult):
        return raw
    extras = {k: v for k, v in raw.items() if k not in (
        "pred", "retrieved_chunk_ids", "n_llm_calls", "prompt_tokens",
        "completion_tokens", "latency_s",
    )}
    return ArmResult(
        pred=str(raw.get("pred", "") or ""),
        retrieved_chunk_ids=list(raw.get("retrieved_chunk_ids", []) or []),
        n_llm_calls=int(raw.get("n_llm_calls", 0) or 0),
        prompt_tokens=int(raw.get("prompt_tokens", 0) or 0),
        completion_tokens=int(raw.get("completion_tokens", 0) or 0),
        latency_s=float(raw.get("latency_s", 0.0) or 0.0),
        metadata=extras,
    )


class _LegacyArmWrapper:
    """Shared base: applicability via a module-level predicate; run via
    a caller-supplied runner callable returning the legacy dict shape.
    """

    name: str = ""
    _predicate: Callable[..., bool]

    def __init__(self, runner: Callable[[str], Mapping[str, Any] | ArmResult]) -> None:
        if runner is None:
            raise ValueError(f"{type(self).__name__}: runner callable required")
        self._runner = runner

    def applicable(self, question: str) -> bool:
        if not question or not question.strip():
            return False
        return bool(type(self)._predicate(question))

    def run(self, question: str, **ctx: Any) -> ArmResult:  # noqa: ARG002
        try:
            raw = self._runner(question)
        except Exception as exc:  # noqa: BLE001
            return ArmResult(
                pred="",
                metadata={"error": f"{type(exc).__name__}: {exc}"},
            )
        return _adapt_runner_result(raw)


class V3buArmWrapper(_LegacyArmWrapper):
    """V3+bu arm wrapper. Applicable on polar-comparison questions."""

    name = "v3bu"
    _predicate = staticmethod(is_v3bu_applicable)


class DecomposeArmWrapper(_LegacyArmWrapper):
    """decompose arm wrapper. Applicable on two-entity bridge questions."""

    name = "decompose"
    _predicate = staticmethod(is_decompose_applicable)


class IterArmWrapper(_LegacyArmWrapper):
    """iter arm wrapper. Applicable on explicit chain / temporal-order questions."""

    name = "iter"
    _predicate = staticmethod(is_iter_applicable)


__all__ = [
    "DecomposeArmWrapper",
    "IterArmWrapper",
    "V3buArmWrapper",
]
