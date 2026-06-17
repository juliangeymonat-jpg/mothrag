# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Shared types for the Iterative Ragnatela γ-feedback loop.

A closed loop that
runs ALL four arms (v3bu / decompose / iter / iter_dup_a PDD), pools their
answers by a per-answer γ (confidence/faithfulness) score, generates
context-aware sub-questions from the UNCERTAIN parts, re-retrieves, and repeats
until the arbitrator's γ converges.

γ bands (the "ragnatela" weighting):
  * HIGH γ → facts            (well-supported; carry forward, dominate the pool)
  * MID  γ → uncertain        (drive sub-question generation to resolve)
  * LOW  γ → anti-context     (contradicted; excluded from the pool, may be
                               refuted by a verification sub-question)

Anti-leak: operates on question text + arm answers + γ scores only. No gold /
F1 / dataset fields, no per-dataset branching. γ is an INPUT-feature-grade
signal (faithfulness of an answer to its retrieved context), never gold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class GammaBand(str, Enum):
    """The three γ regimes that drive ragnatela weighting."""

    HIGH = "high"   # facts
    MID = "mid"     # uncertain
    LOW = "low"     # anti-context


@dataclass(frozen=True)
class ArmAnswer:
    """One arm's answer for the current iteration, with its γ confidence.

    ``gamma`` ∈ [0, 1]: a continuous faithfulness/confidence score (the
    continuous analogue of the production γ verifier's valid/partial/invalid).
    """

    arm: str
    answer: str
    gamma: float = 0.0

    def clamped_gamma(self) -> float:
        return 0.0 if self.gamma < 0.0 else 1.0 if self.gamma > 1.0 else self.gamma


@dataclass
class RagnatelaConfig:
    """Knobs for the γ-feedback loop. All keyword-defaulted; general-purpose."""

    max_iterations: int = 5
    # γ band thresholds: gamma >= gamma_high → HIGH; gamma < gamma_low → LOW;
    # in between → MID.
    gamma_high: float = 0.66
    gamma_low: float = 0.33
    # Convergence: the pooled answer's γ must reach this AND be stable for
    # ``convergence_stability_iters`` consecutive iterations.
    convergence_gamma: float = 0.70
    convergence_stability_iters: int = 2
    # How many context-aware sub-questions to generate per iteration.
    n_sub_questions: int = 2
    # Stop early once no MID/LOW (uncertain/anti-context) answers remain — the
    # loop has nothing left to resolve.
    stop_when_no_uncertainty: bool = True


@dataclass
class PoolOutcome:
    """Result of γ-weighted answer pooling for one iteration."""

    answer: str
    pooled_gamma: float
    bands: dict[str, GammaBand] = field(default_factory=dict)
    high: list[ArmAnswer] = field(default_factory=list)
    mid: list[ArmAnswer] = field(default_factory=list)
    low: list[ArmAnswer] = field(default_factory=list)

    @property
    def has_uncertainty(self) -> bool:
        return bool(self.mid or self.low)


@dataclass
class IterationTrace:
    """Per-iteration telemetry (the γ trajectory + what the loop did)."""

    iteration: int
    pooled_answer: str
    pooled_gamma: float
    n_high: int
    n_mid: int
    n_low: int
    sub_questions: list[str] = field(default_factory=list)
    converged: bool = False
    stop_reason: Optional[str] = None


@dataclass
class RagnatelaResult:
    """Output of the full γ-feedback loop."""

    answer: str
    iterations_used: int
    converged: bool
    final_gamma: float
    stop_reason: str
    gamma_trace: list[float] = field(default_factory=list)
    traces: list[IterationTrace] = field(default_factory=list)


__all__ = [
    "GammaBand",
    "ArmAnswer",
    "RagnatelaConfig",
    "PoolOutcome",
    "IterationTrace",
    "RagnatelaResult",
]
