# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Iterative Ragnatela — γ-feedback multi-hop retrieval loop.

A closed loop over the 4-arm pool
(v3bu / decompose / iter / iter_dup_a PDD) where per-answer γ pools the arms
(high=facts / mid=uncertain / low=anti-context), the uncertain part drives
context-aware sub-question generation + re-retrieval, and the arbitrator stops
on γ-convergence. The iterative γ-feedback is the differentiator vs one-shot
bridge conditioning and training-free graph-free baselines.

Public surface::

    from mothrag.iterative_ragnatela import (
        RagnatelaOrchestrator, RagnatelaConfig, ArmAnswer,
    )
"""
from __future__ import annotations

from mothrag.iterative_ragnatela.convergence import ConvergenceDetector
from mothrag.iterative_ragnatela.gamma_pooling import (
    classify_band,
    normalize_answer,
    pool_answers,
)
from mothrag.iterative_ragnatela.ragnatela_orchestrator import (
    ArmRunner,
    RagnatelaOrchestrator,
    Retriever,
)
from mothrag.iterative_ragnatela.subq_generation import (
    SubQLLM,
    generate_sub_questions,
)
from mothrag.iterative_ragnatela.types import (
    ArmAnswer,
    GammaBand,
    IterationTrace,
    PoolOutcome,
    RagnatelaConfig,
    RagnatelaResult,
)

__all__ = [
    "RagnatelaOrchestrator",
    "ArmRunner",
    "Retriever",
    "RagnatelaConfig",
    "RagnatelaResult",
    "ArmAnswer",
    "GammaBand",
    "PoolOutcome",
    "IterationTrace",
    "ConvergenceDetector",
    "pool_answers",
    "classify_band",
    "normalize_answer",
    "generate_sub_questions",
    "SubQLLM",
]
