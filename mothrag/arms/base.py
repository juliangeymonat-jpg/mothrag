# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Arm Protocol + ArmResult dataclass shared by the opt-in arm pool.

Per the production-stack contract, the legacy three
arms (V3+bu, decompose, iter) stay function-based; this Protocol
governs the opt-in CLASS-based arms (:class:`InfoboxArm`,
:class:`BM25Arm`, etc.) that compose alongside via the
``--arms-pool`` CLI flag in ``scripts/route_prospective.py``.

The Protocol is intentionally narrow: a ``name``, a deterministic
``applicable(question)`` predicate (used by sel_v2 to decide whether
to include this arm in the per-query subset), and a ``run(question,
**ctx)`` method returning an :class:`ArmResult`.

Cost telemetry mirrors the function-based arm dict shape so the
existing route_prospective.py main loop can aggregate prompt_tokens /
completion_tokens / latency_s identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ArmResult:
    """Per-arm output. Mirrors the dict shape returned by the legacy
    function-based arm runners (_run_v3bu / _run_decompose / _run_iter
    in route_prospective.py) for telemetry aggregation parity."""

    pred: str
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    n_llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Arm(Protocol):
    """Protocol every opt-in arm implements.

    Attributes
    ----------
    name
        Identifier consumed by ``--arms-pool`` CLI parsing and by
        sel_v2 router extension. Examples: ``"infobox_arm"``,
        ``"bm25_arm"``. Reserved names (legacy function-based arms):
        ``"v3bu"``, ``"decompose"``, ``"iter"`` -- new class-based
        arms must use distinct names.
    """

    name: str

    def applicable(self, question: str) -> bool:
        """Deterministic predicate: True iff this arm is structurally
        able to contribute to ``question``. Used by sel_v2 to decide
        per-query arm-subset inclusion. Must NOT perform expensive
        work (no LLM call, no embedding, no retrieval)."""
        ...

    def run(self, question: str, **ctx: Any) -> ArmResult:
        """Execute the arm. ``ctx`` may carry optional dependencies
        (e.g. ``reader_client``, ``reader_model``) the arm needs but
        does not own internally. The arm MUST be robust to missing
        ``ctx`` keys; when a required dependency is absent the arm
        returns ``ArmResult(pred="")`` (other arms in the pool then
        cover via arbitration)."""
        ...


__all__ = ["Arm", "ArmResult"]
