# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Retry-on-abstain escalation cascade.

Reframes abstention from a terminal skip into a *trigger for deeper
investment*: when a query fails the standard arbitration (γ-invalid,
H4 / H12 fires, iter abstain, or cross-arm disagreement), the system
walks an ordered list of recovery strategies. Loop mode falls through
to a soft fallback that guarantees a non-empty answer; abstention mode
surfaces a terminal-abstain signal for downstream KB-audit consumers.

Public surface:

- :class:`RetryContext` — the read-only state object each strategy receives.
- :class:`RetryStrategy` — the Protocol every strategy implements.
- :class:`EscalationOrchestrator` — runs strategies in priority order,
  returns the first successful recovery (or the SoftFallback terminal).
- :func:`build_default_orchestrator` — convenience constructor returning
  the sweet-spot bundle (#1 + #2 + #4 + #7) or the full 7-strategy stack.

The orchestrator and strategies are independent of the embedder / reader /
vector-store backend; they receive callables / handles through the
:class:`RetryContext`.
"""

from __future__ import annotations

from mothrag.core.retry.protocol import (
    RetryContext,
    RetryOutcome,
    RetryStrategy,
)
from mothrag.core.retry.orchestrator import (
    EscalationOrchestrator,
    build_default_orchestrator,
    build_strategies_by_name,
)

__all__ = [
    "RetryContext",
    "RetryOutcome",
    "RetryStrategy",
    "EscalationOrchestrator",
    "build_default_orchestrator",
    "build_strategies_by_name",
]
