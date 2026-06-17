# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""RetryContext + RetryStrategy + RetryOutcome — public types for the
retry-on-abstain escalation cascade."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence


# ============================================================
# Trigger taxonomy
# ============================================================

# Recognised abstention signal strings carried in RetryContext.abstention_signal.
# Strategies use these to decide :meth:`RetryStrategy.applicable`.
ABSTENTION_SIGNALS = (
    "gamma_refuse",         # γ verifier flagged the answer invalid
    "h4_refuse",            # pre-registered H4 selective rule fired
    "h12_refuse",           # pre-registered H12 MuSiQue how_many rule fired
    "iter_abstain",         # iterative arm exhausted budget without answer
    "cross_arm_disagree",   # arms disagree and arbiter could not pick a winner
    "empty_answer",         # chosen answer is empty / uncertainty template
)


# ============================================================
# Context
# ============================================================

@dataclass
class RetryContext:
    """Read-only state passed to each :class:`RetryStrategy`.

    Strategies receive everything they need to attempt recovery without
    pulling in the :class:`mothrag.MothRAG` instance directly: handles for
    embedder / reader / vector_db (plus the running config + already-
    computed arm outputs) are surfaced as attributes here.
    """

    # Query + retrieval
    question: str
    passages: list[str]
    q_emb: list[float]
    top_k: int

    # Arm outputs already computed by `_query_production`
    arm_subset: list[str]
    v3bu_pred: str | None
    dec_pred: str | None
    iter_pred: str | None

    # Arbitration outcome that triggered escalation
    chosen: str
    arbitrate_reason: str
    c7_info: Any = None
    abstention_signal: str = "empty_answer"

    # Budget tracking (cascades stop when budget_used reaches budget_limit).
    budget_used: int = 0
    budget_limit: int = 8

    # Backends accessible by strategies (e.g. for re-retrieval, re-read,
    # entity extraction). Type-hinted loosely to avoid a cyclic import on
    # :class:`mothrag.core.api.Embedder` / :class:`Reader` / :class:`VectorStore`.
    embedder: Any = None       # has .embed_batch(texts) -> list[list[float]]
    reader: Any = None         # has .read(question, passages) -> str
    vector_db: Any = None      # has .retrieve(q_emb, top_k) -> list[Chunk]
    config: dict[str, Any] = field(default_factory=dict)

    # Optional callables used by escalation strategies that need to invoke
    # the running arm implementations on revised configurations (e.g.
    # IterBudgetExtension re-runs the iter arm with a wider budget).
    run_arm_iter: Callable[..., str] | None = None  # (question, passages, q_emb, top_k, max_steps) -> str
    run_arm_v3bu: Callable[..., str] | None = None
    run_arm_decompose: Callable[..., str] | None = None

    # Recursive escalation guard. Strategies that re-enter the pipeline
    # (e.g. #3 QueryReformulation) MUST increment this and refuse to
    # recurse past depth 1.
    escalation_depth: int = 0

    # True when the iterative arm exhausted its iteration cap without a
    # confident answer (set from MothRAG._arm_iter's sidecar). Marks the
    # "iter==cap AND gamma_refuse" cohort. Defaults False so existing
    # constructions are unaffected.
    iter_hit_cap: bool = False

    def spend(self, llm_calls: int) -> bool:
        """Try to charge ``llm_calls`` against the cascade budget.

        Returns True if charged successfully, False if the budget would
        be exhausted (the calling strategy should then return None).
        """
        if self.budget_used + llm_calls > self.budget_limit:
            return False
        self.budget_used += llm_calls
        return True


# ============================================================
# Outcome
# ============================================================

@dataclass
class RetryOutcome:
    """What :meth:`EscalationOrchestrator.try_escalate` returns.

    In ``mode="loop"`` (default) ``answer`` is guaranteed non-empty thanks
    to the terminal SoftFallback; callers should still check ``recovered_by``
    to know whether the answer came from a recovery strategy or from the
    original arbitration outcome.

    In ``mode="abstention"`` the outcome may carry ``answer=""`` and
    ``terminal_abstain=True`` to surface a KB-audit / gap-discovery signal.
    """

    answer: str
    recovered_by: str = ""              # strategy name or "" if no recovery fired
    strategies_tried: list[str] = field(default_factory=list)
    original_signal: str = ""           # echoed RetryContext.abstention_signal
    # 'high' | 'medium_recovered' | 'low_soft_fallback' | 'terminal_abstain'
    final_confidence: str = "high"
    budget_used: int = 0                # cumulative LLM calls during cascade
    mode: str = "loop"                  # 'loop' | 'abstention'
    terminal_abstain: bool = False      # True when abstention-mode cascade exhausted


# ============================================================
# Strategy protocol
# ============================================================

class RetryStrategy(Protocol):
    """Protocol implemented by every recovery strategy.

    Strategies are pure-function-ish: they may mutate ``ctx.budget_used`` via
    :meth:`RetryContext.spend` but should otherwise treat ``ctx`` as read-only.

    Attributes
    ----------
    name
        Identifier used in metadata + ablation flags
        (e.g. ``"iter_extension"``, ``"cross_arm_consensus"``).
    cost_estimate
        Worst-case LLM-call cost. ``0`` = no LLM call (may still hit the
        embedder / vector store / NER), ``1+`` = additional reader calls.
    """

    name: str
    cost_estimate: int

    def applicable(self, ctx: RetryContext) -> bool:
        """Return True if the strategy is structurally able to recover.

        Pure predicate over ``ctx``; must not perform expensive work.
        """
        ...

    def try_recover(self, ctx: RetryContext) -> str | None:
        """Attempt recovery; return the new answer or ``None`` if it
        cannot recover within budget (the cascade then continues)."""
        ...


__all__ = [
    "ABSTENTION_SIGNALS",
    "RetryContext",
    "RetryOutcome",
    "RetryStrategy",
]
