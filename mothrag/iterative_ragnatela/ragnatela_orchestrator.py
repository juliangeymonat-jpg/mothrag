# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Iterative Ragnatela γ-feedback orchestrator.

Closes the feedback loop::

    retrieve → ALL 4 arms (v3bu/decompose/iter/iter_dup_a PDD)
            → γ-weighted answer pooling (high=facts / mid=uncertain / low=anti)
            → context-aware sub-question generation (from the uncertain part)
            → re-retrieve (augment the shared context)
            → arbitrator γ-convergence  ──► stop, else loop

The orchestrator is backend-agnostic: the arm pool and the retriever are
injected callables (``arm_runner`` / ``retriever``), so the whole loop is
unit-testable offline without an index or an LLM (the same seam pattern as
``BridgeArm.ann_retrieve``). A live wiring (real 4-arm pool + dense/bridge
retriever) is a thin adapter built on top — not part of this POC module.

Differentiator vs one-shot bridge conditioning and training-free
graph-free baselines (HippoRAG2 / PropRAG): the γ-feedback is ITERATIVE and the
uncertainty (MID/LOW band) drives what gets re-retrieved next.

Pool-safety: this loop is an
INTERNAL upgrade of the *iter* machinery — it re-runs the SAME locked 4-arm pool
(v3bu / decompose / iter / iter_dup_a PDD) each iteration over a growing shared
context. It is NOT a fifth arm, and the bridge substrate it layers on is a
retrieval layer upstream of the pool, never an arm. The invariant — pool size
== 4 every iteration, the candidate/context set is monotone-additive, and the
arbitrator's arm_scores never gain a ``bridge`` key — is pinned by
``tests/iterative_ragnatela/test_pool_safety_invariant.py``.

Anti-leak: question text + arm answers + γ + retrieved context only. No gold,
no per-dataset branching.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from mothrag.iterative_ragnatela.convergence import ConvergenceDetector
from mothrag.iterative_ragnatela.gamma_pooling import pool_answers
from mothrag.iterative_ragnatela.subq_generation import (
    SubQLLM,
    generate_sub_questions,
)
from mothrag.iterative_ragnatela.types import (
    ArmAnswer,
    IterationTrace,
    RagnatelaConfig,
    RagnatelaResult,
)

# arm_runner(question, context) -> sequence[ArmAnswer]
ArmRunner = Callable[[str, list], Sequence[ArmAnswer]]
# retriever(sub_questions, context) -> sequence[str]  (new context items)
Retriever = Callable[[list, list], Sequence[str]]


class RagnatelaOrchestrator:
    """Runs the γ-feedback loop. ``arm_runner`` is required."""

    def __init__(
        self,
        arm_runner: ArmRunner,
        *,
        retriever: Optional[Retriever] = None,
        subq_llm: Optional[SubQLLM] = None,
        config: Optional[RagnatelaConfig] = None,
    ) -> None:
        if arm_runner is None:
            raise ValueError("RagnatelaOrchestrator requires an arm_runner")
        self.arm_runner = arm_runner
        self.retriever = retriever
        self.subq_llm = subq_llm
        self.cfg = config or RagnatelaConfig()

    def run(self, question: str) -> RagnatelaResult:
        cfg = self.cfg
        detector = ConvergenceDetector(cfg)
        context: list[str] = []
        traces: list[IterationTrace] = []
        gamma_trace: list[float] = []
        last_answer = ""
        last_gamma = 0.0
        converged = False
        stop_reason = "max_iterations"

        for it in range(1, cfg.max_iterations + 1):
            answers = list(self.arm_runner(question, list(context)))
            outcome = pool_answers(answers, cfg)
            last_answer, last_gamma = outcome.answer, outcome.pooled_gamma
            gamma_trace.append(round(outcome.pooled_gamma, 4))

            trace = IterationTrace(
                iteration=it,
                pooled_answer=outcome.answer,
                pooled_gamma=round(outcome.pooled_gamma, 4),
                n_high=len(outcome.high),
                n_mid=len(outcome.mid),
                n_low=len(outcome.low),
            )

            if detector.update(outcome):
                converged = True
                stop_reason = "gamma_converged"
                trace.converged = True
                trace.stop_reason = stop_reason
                traces.append(trace)
                break

            if cfg.stop_when_no_uncertainty and not outcome.has_uncertainty:
                stop_reason = "no_uncertainty"
                trace.stop_reason = stop_reason
                traces.append(trace)
                break

            if it >= cfg.max_iterations:
                stop_reason = "max_iterations"
                trace.stop_reason = stop_reason
                traces.append(trace)
                break

            # --- γ-feedback: context-aware sub-questions → re-retrieve. ---
            sub_qs = generate_sub_questions(
                question, outcome, cfg, llm=self.subq_llm)
            trace.sub_questions = sub_qs
            trace.stop_reason = None
            traces.append(trace)
            self._augment_context(context, sub_qs)

        return RagnatelaResult(
            answer=last_answer,
            iterations_used=len(traces),
            converged=converged,
            final_gamma=round(last_gamma, 4),
            stop_reason=stop_reason,
            gamma_trace=gamma_trace,
            traces=traces,
        )

    def _augment_context(self, context: list, sub_qs: list) -> None:
        """Re-retrieve on the sub-questions and fold new items into context.

        With no retriever wired, the sub-questions themselves seed the context
        (so a no-index dry-run still exercises the feedback path)."""
        if not sub_qs:
            return
        new_items: Sequence[str]
        if self.retriever is not None:
            try:
                new_items = self.retriever(list(sub_qs), list(context)) or []
            except Exception:  # noqa: BLE001 — retrieval failure must not kill loop
                new_items = []
        else:
            new_items = list(sub_qs)
        for item in new_items:
            if item and item not in context:
                context.append(item)


__all__ = ["RagnatelaOrchestrator", "ArmRunner", "Retriever"]
