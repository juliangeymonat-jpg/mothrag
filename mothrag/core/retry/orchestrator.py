# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""EscalationOrchestrator — runs a priority-ordered list of
:class:`RetryStrategy` instances over a :class:`RetryContext`."""

from __future__ import annotations

import logging
from typing import Iterable

from mothrag.core.retry.protocol import (
    RetryContext,
    RetryOutcome,
    RetryStrategy,
)

logger = logging.getLogger(__name__)


# Strategy execution order. SoftFallback MUST stay terminal.
DEFAULT_PRIORITY = (
    "iter_extension",           # #1 zero-LLM, cheap
    "arm_fallback",             # #2 zero-LLM, cheap
    "cross_arm_consensus",      # #4 zero-LLM, semantic
    "bottom_up_boost",          # #5 zero-LLM, NER + re-retrieve
    "l4b_anchor_retry",         # #6 zero-LLM, anchor swap
    "query_reformulation",      # #3 one LLM call (last resort)
    "soft_fallback",            # #7 always-final guarantee, zero-LLM
)

SWEET_SPOT_PRIORITY = (
    "iter_extension",
    "arm_fallback",
    "cross_arm_consensus",
    "soft_fallback",
)

# Preset: canonical 7 + active-learning extensions #8 + #9.
# Naming convention: "all_8" expands to the canonical 7 strategies PLUS
# the two opt-in active-learning strategies (#8 ActiveGapQuery, #9
# SubQuestionRerouteCascade). #6 l4b_anchor_retry is deferred to v0.5.1
# per design constraint (MothRAG._arm_iter does not yet expose the L4b
# anchor handle), so it remains in the canonical 7 but is NOT counted
# separately in the alias name.
ALL_8_PRIORITY = (
    "iter_extension",
    "arm_fallback",
    "cross_arm_consensus",
    "bottom_up_boost",
    "l4b_anchor_retry",
    "query_reformulation",
    "active_gap_query",
    "sub_question_reroute",
    "soft_fallback",
)

class EscalationOrchestrator:
    """Walk an ordered list of :class:`RetryStrategy` instances.

    The first strategy whose :meth:`applicable` is True and whose
    :meth:`try_recover` returns a non-None answer wins.

    Dual-mode terminal behaviour:

    - ``mode="loop"`` (default, production): if every non-terminal strategy
      returns None, the terminal SoftFallback fires and guarantees a
      non-empty answer. The strategy list MUST end with
      :class:`SoftFallbackStrategy`.
    - ``mode="abstention"`` (KB-audit / Aurora-v2 gap-discovery): if every
      strategy returns None, surface a terminal abstain
      (``answer=""``, ``terminal_abstain=True``) so downstream KB-audit
      consumers can act on the gap signal. No terminal-strategy
      requirement; soft_fallback is allowed but optional.
    """

    def __init__(
        self,
        strategies: list[RetryStrategy],
        *,
        mode: str = "loop",
    ) -> None:
        if not strategies:
            raise ValueError("EscalationOrchestrator requires at least one strategy.")
        if mode not in ("loop", "abstention"):
            raise ValueError(
                f"EscalationOrchestrator mode must be 'loop' or 'abstention', "
                f"got {mode!r}."
            )
        if mode == "loop" and strategies[-1].name != "soft_fallback":
            raise ValueError(
                "EscalationOrchestrator(mode='loop'): the terminal strategy "
                f"must be SoftFallbackStrategy (got {strategies[-1].name!r}). "
                "Use build_strategies_by_name() to assemble a valid stack, "
                "or pass mode='abstention' to allow terminal abstain."
            )
        self.strategies = strategies
        self.mode = mode

    def try_escalate(self, ctx: RetryContext) -> RetryOutcome:
        """Run the cascade; always return a :class:`RetryOutcome`.

        In ``mode="loop"`` the outcome's ``answer`` is guaranteed non-empty.
        In ``mode="abstention"`` the outcome's ``answer`` may be the empty
        string with ``RetryOutcome.terminal_abstain == True``.
        """
        original_signal = ctx.abstention_signal
        tried: list[str] = []

        # In loop mode the terminal SoftFallback runs separately at the end.
        # In abstention mode there is no special terminal — every strategy
        # is a potential recoverer; if none fire, we surface a terminal
        # abstain.
        if self.mode == "loop":
            iterable = self.strategies[:-1]
            terminal: RetryStrategy | None = self.strategies[-1]
        else:
            iterable = self.strategies
            terminal = None

        for strat in iterable:
            if not strat.applicable(ctx):
                continue
            tried.append(strat.name)
            try:
                result = strat.try_recover(ctx)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Retry strategy %s crashed; continuing cascade.",
                                 strat.name)
                continue
            if result:
                return RetryOutcome(
                    answer=result,
                    recovered_by=strat.name,
                    strategies_tried=tried,
                    original_signal=original_signal,
                    final_confidence="medium_recovered",
                    budget_used=ctx.budget_used,
                    mode=self.mode,
                )

        # Terminal handling diverges by mode.
        if terminal is not None:
            tried.append(terminal.name)
            fallback = terminal.try_recover(ctx) or ctx.chosen or "[no answer recovered]"
            return RetryOutcome(
                answer=fallback,
                recovered_by=terminal.name,
                strategies_tried=tried,
                original_signal=original_signal,
                final_confidence="low_soft_fallback",
                budget_used=ctx.budget_used,
                mode=self.mode,
            )
        # abstention mode: terminal abstain surfaces gap-discovery signal.
        return RetryOutcome(
            answer="",
            recovered_by="terminal_abstain",
            strategies_tried=tried,
            original_signal=original_signal,
            final_confidence="terminal_abstain",
            budget_used=ctx.budget_used,
            mode="abstention",
            terminal_abstain=True,
        )


# ============================================================
# Strategy factory
# ============================================================

def _instantiate(name: str) -> RetryStrategy:
    """Lazy strategy instantiation by name (avoids importing all 7 always)."""
    if name == "iter_extension":
        from mothrag.core.retry.strategies.iter_extension import IterBudgetExtensionStrategy
        return IterBudgetExtensionStrategy()
    if name == "arm_fallback":
        from mothrag.core.retry.strategies.arm_fallback import ArmFallbackStrategy
        return ArmFallbackStrategy()
    if name == "cross_arm_consensus":
        from mothrag.core.retry.strategies.cross_arm_consensus import CrossArmConsensusStrategy
        return CrossArmConsensusStrategy()
    if name == "bottom_up_boost":
        from mothrag.core.retry.strategies.bottom_up_boost import BottomUpBoostStrategy
        return BottomUpBoostStrategy()
    if name == "l4b_anchor_retry":
        from mothrag.core.retry.strategies.l4b_anchor_retry import L4bAnchorRetryStrategy
        return L4bAnchorRetryStrategy()
    if name == "query_reformulation":
        from mothrag.core.retry.strategies.query_reformulation import QueryReformulationStrategy
        return QueryReformulationStrategy()
    if name == "soft_fallback":
        from mothrag.core.retry.strategies.soft_fallback import SoftFallbackStrategy
        return SoftFallbackStrategy()
    # Active-learning extensions (#8, #9) -- opt-in, NOT in DEFAULT_PRIORITY
    # so existing deployments keep the 7-strategy default behaviour.
    if name == "active_gap_query":
        from mothrag.core.retry.strategies.active_gap_query import ActiveGapQueryStrategy
        return ActiveGapQueryStrategy()
    if name == "sub_question_reroute":
        from mothrag.core.retry.strategies.sub_question_reroute import (
            DEFAULT_LAYERS,
            SubQuestionRerouteCascadeStrategy,
        )
        return SubQuestionRerouteCascadeStrategy(layers=DEFAULT_LAYERS)
    # Opt-in primitives, NOT in DEFAULT_PRIORITY so existing
    # deployments stay byte-identical. SuppressGate is intended to be placed
    # FIRST in any cascade that opts in (pre-cascade short-circuit for
    # gamma false-positives).
    if name == "suppress_gate":
        from mothrag.core.retry.strategies.suppress_gate import SuppressGateStrategy
        return SuppressGateStrategy()
    if name == "reroute_iter_with_boost":
        from mothrag.core.retry.strategies.reroute_iter_with_boost import (
            RerouteIterWithBoostStrategy,
        )
        return RerouteIterWithBoostStrategy()
    raise ValueError(f"Unknown retry strategy: {name!r}")


def build_strategies_by_name(names: Iterable[str]) -> list[RetryStrategy]:
    """Build an ordered strategy list from names.

    Auto-appends ``"soft_fallback"`` if missing, since SoftFallback must
    stay terminal. Duplicate names are silently de-duped (first occurrence
    wins).
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        ordered.append(n)
    if "soft_fallback" in ordered:
        # Move soft_fallback to the end.
        ordered.remove("soft_fallback")
    ordered.append("soft_fallback")
    return [_instantiate(n) for n in ordered]


def build_default_orchestrator(
    preset: str | Iterable[str] = "all",
    *,
    mode: str = "loop",
) -> EscalationOrchestrator:
    """Convenience constructor.

    Parameters
    ----------
    preset
        - ``"all"`` (default): full 7-strategy cascade.
        - ``"sweet_spot"``: bundle #1 + #2 + #4 + #7 (zero LLM cost).
        - ``"soft_fallback_only"``: only #7.
        - iterable of names: explicit selection (soft_fallback auto-appended
          only in ``mode="loop"``; in ``mode="abstention"`` the list is used
          verbatim).
    mode
        - ``"loop"`` (default): SoftFallback always-final, non-empty guarantee.
        - ``"abstention"``: terminal abstain allowed (KB-audit /
          gap-discovery deployments).
    """
    if preset == "all" or preset == "default_7":
        # "default_7" is a backward-compat alias for the canonical
        # 7-strategy cascade ("all"). They are byte-identical.
        names: Iterable[str] = DEFAULT_PRIORITY
    elif preset == "sweet_spot":
        names = SWEET_SPOT_PRIORITY
    elif preset == "all_8":
        # canonical 7 + #8 + #9 (with terminal soft_fallback).
        names = ALL_8_PRIORITY
    elif preset == "soft_fallback_only":
        names = ("soft_fallback",)
    elif isinstance(preset, str):
        # Single named strategy. Honour it as a one-element list.
        names = (preset,)
    else:
        # Iterable of names. May contain preset aliases ("default_7",
        # "sweet_spot") mixed with individual strategy names; expand
        # those before forwarding to the by-name builder.
        names = _expand_preset_names(preset)
    if mode == "loop":
        strategies = build_strategies_by_name(names)
    else:
        # Abstention mode: do NOT auto-append soft_fallback; honour caller's list.
        strategies = [_instantiate(n) for n in _dedup(names)]
        if not strategies:
            raise ValueError(
                "build_default_orchestrator(mode='abstention'): empty strategy list."
            )
    return EscalationOrchestrator(strategies, mode=mode)


def _expand_preset_names(items: Iterable[str]) -> list[str]:
    """Expand any preset-alias tokens in an iterable of names.

    ``"default_7"`` and ``"all"`` expand to :data:`DEFAULT_PRIORITY`;
    ``"sweet_spot"`` expands to :data:`SWEET_SPOT_PRIORITY`; everything
    else is passed through verbatim. Duplicates are de-duped (first
    occurrence wins) -- e.g. ``["default_7", "active_gap_query"]``
    expands to the canonical 7 strategies followed by
    ``"active_gap_query"`` (which lands just before any auto-appended
    SoftFallback terminal).
    """
    out: list[str] = []
    for item in items:
        if item in ("all", "default_7"):
            out.extend(DEFAULT_PRIORITY)
        elif item == "sweet_spot":
            out.extend(SWEET_SPOT_PRIORITY)
        elif item == "all_8":
            out.extend(ALL_8_PRIORITY)
        elif item == "soft_fallback_only":
            out.append("soft_fallback")
        else:
            out.append(item)
    return _dedup(out)


def _dedup(names: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        ordered.append(n)
    return ordered


__all__ = [
    "DEFAULT_PRIORITY",
    "SWEET_SPOT_PRIORITY",
    "ALL_8_PRIORITY",
    "EscalationOrchestrator",
    "build_default_orchestrator",
    "build_strategies_by_name",
]
