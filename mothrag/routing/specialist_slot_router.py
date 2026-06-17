# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Pool-safe specialist slot router (M8 specialist live-wiring core).

The N=4 pool-safety axiom locks the arm pool at
exactly four arms: ``v3bu / decompose / iter / iter_dup_a`` (PDD). The M8
specialists (CompareArm for comparison cohorts, DecomposeArm 2.0 for
compositional/chain cohorts) must be wired in **without** growing the pool to 5.

The only axiom-consistent way to add specialist capability is **substitution,
not addition**: make the ``decompose`` slot *polymorphic*. On a specialist's
cohort, the specialist FILLS the decompose slot; otherwise the generic decompose
arm fills it. Both specialists are decompose-*family* (smarter query splitters),
so the substitution is **architecturally** motivated, never gold- or
dataset-derived. The pool is therefore provably ``{v3bu, decompose, iter,
iter_dup_a}`` — size 4 — for every query (``pool_keys`` is the invariant).

Design properties (mirrors the bridge-substrate / ragnatela seams):
  * **Backend-agnostic** — the specialists, the cohort detectors, and the PDD
    router are all injected callables; the module is fully offline-testable and
    has no hard dependency on the (separately-branched) specialist / PDD code.
  * **Opt-in / default-OFF** — with ``enabled=False`` (default) or no specialist
    injected, every question resolves to the generic decompose slot, i.e. the
    behaviour is byte-identical to the current pool (zero regression). The live
    specialists become fire-ready by injection + flipping ``enabled``.
  * **Anti-leak** — cohort routing keys on question-text input features
    (comparison / chain structure) only; never a corpus / dataset signal.
  * **PDD** — reader-inverse cardinality tunes the ``iter_dup_a`` arm
    *internally* (a within-arm dial); it never adds an arm.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger(__name__)

# The LOCKED 4-arm pool. The decompose slot is polymorphic; the *keys* are fixed.
CANONICAL_POOL: tuple[str, ...] = ("v3bu", "decompose", "iter", "iter_dup_a")
DECOMPOSE_SLOT = "decompose"

# Cohort labels (input-feature derived, never dataset-derived).
QT_COMPARISON = "comparison"
QT_COMPOSITIONAL = "compositional"
QT_GENERIC = "generic"

# Predicate(question) -> bool
QTypePredicate = Callable[[str], bool]
# runner(question, **kw) -> result  (or an object exposing .run / .applicable)
ArmRunner = Callable[..., Any]


@dataclass(frozen=True)
class SlotDecision:
    """Which arm fills the (always-named ``decompose``) slot for one question."""

    slot: str               # always DECOMPOSE_SLOT — the pool key never changes
    arm_name: str           # actual arm: "decompose" | "compare_arm" | "decompose_arm_v2"
    qtype: str              # cohort that drove the choice
    is_specialist: bool     # True iff a specialist (not the generic arm) fired


def _default_is_comparison(question: str) -> bool:
    """Default comparison-cohort detector (main query_type_classifier)."""
    try:
        from mothrag.core.query_type_classifier import (
            is_comparative_selection,
            is_polar_comparison,
        )
    except Exception:  # noqa: BLE001 — keep the router importable in isolation
        return False
    q = question or ""
    return bool(is_polar_comparison(q) or is_comparative_selection(q))


def _default_is_compositional(question: str) -> bool:
    """Default compositional/chain-cohort detector (main query_type_classifier)."""
    try:
        from mothrag.core.query_type_classifier import has_chain, is_chain_deep
    except Exception:  # noqa: BLE001
        return False
    q = question or ""
    return bool(is_chain_deep(q) or has_chain(q))


class SpecialistSlotRouter:
    """Resolves the polymorphic ``decompose`` slot, preserving the N=4 pool.

    All collaborators are injected; with the defaults (no specialists,
    ``enabled=False``) the router is a pass-through to the generic decompose arm.
    """

    def __init__(
        self,
        *,
        compare_arm: Optional[ArmRunner] = None,
        decompose_arm_v2: Optional[ArmRunner] = None,
        is_comparison: Optional[QTypePredicate] = None,
        is_compositional: Optional[QTypePredicate] = None,
        pdd_router: Optional[Callable[..., int]] = None,
        base_pdd_cardinality: int = 1,
        enabled: bool = False,
    ) -> None:
        self.compare_arm = compare_arm
        self.decompose_arm_v2 = decompose_arm_v2
        self.is_comparison = is_comparison or _default_is_comparison
        self.is_compositional = is_compositional or _default_is_compositional
        self.pdd_router = pdd_router
        self.base_pdd_cardinality = max(1, int(base_pdd_cardinality))
        self.enabled = bool(enabled)

    # ---- pool-safety invariant -------------------------------------------
    @staticmethod
    def pool_keys(question: str | None = None) -> tuple[str, ...]:
        """The arbitration pool keys — ALWAYS the 4 canonical arms.

        Independent of routing / config / question: the decompose slot is
        polymorphic but its *key* never changes, so the pool is provably 4.
        """
        return CANONICAL_POOL

    # ---- cohort classification (input-feature only) ----------------------
    def classify(self, question: str) -> str:
        """Return the cohort label for ``question`` (comparison wins ties).

        Comparison is checked first: a boolean comparison question (e.g. "are X
        and Y in the same country?") has a more specific structure than a plain
        chain, so CompareArm's both-entities-covered guarantee takes precedence.
        """
        q = question or ""
        try:
            if self.is_comparison(q):
                return QT_COMPARISON
            if self.is_compositional(q):
                return QT_COMPOSITIONAL
        except Exception:  # noqa: BLE001 — a flaky detector must not break routing
            logger.debug("cohort detector raised; defaulting to generic", exc_info=True)
        return QT_GENERIC

    def _specialist_for(self, qtype: str) -> tuple[str, Optional[ArmRunner]]:
        if qtype == QT_COMPARISON:
            return "compare_arm", self.compare_arm
        if qtype == QT_COMPOSITIONAL:
            return "decompose_arm_v2", self.decompose_arm_v2
        return DECOMPOSE_SLOT, None

    @staticmethod
    def _applicable(arm: ArmRunner, question: str) -> bool:
        """A specialist may decline its own cohort via an optional .applicable."""
        probe = getattr(arm, "applicable", None)
        if probe is None:
            return True
        try:
            return bool(probe(question))
        except Exception:  # noqa: BLE001
            return False

    # ---- the routing decision --------------------------------------------
    def decide(self, question: str) -> SlotDecision:
        """Decide which arm fills the decompose slot (no execution)."""
        if not self.enabled:
            return SlotDecision(DECOMPOSE_SLOT, DECOMPOSE_SLOT, QT_GENERIC, False)
        qtype = self.classify(question)
        arm_name, arm = self._specialist_for(qtype)
        if arm is None or not self._applicable(arm, question):
            # cohort had no available/applicable specialist → generic slot.
            return SlotDecision(DECOMPOSE_SLOT, DECOMPOSE_SLOT, qtype, False)
        return SlotDecision(DECOMPOSE_SLOT, arm_name, qtype, True)

    # ---- execute the slot (substitution; pool stays 4) -------------------
    def run_decompose_slot(
        self,
        question: str,
        *,
        generic_runner: ArmRunner,
        **run_kwargs: Any,
    ) -> tuple[Any, SlotDecision]:
        """Run whichever arm fills the decompose slot; return (result, decision).

        The caller stores ``result`` under the ``"decompose"`` pool key, so
        arbitration always sees exactly the 4 canonical arms. A specialist that
        raises degrades gracefully to the generic decompose arm (never drops the
        slot — that would shrink the pool).
        """
        decision = self.decide(question)
        if not decision.is_specialist:
            return generic_runner(question, **run_kwargs), decision

        arm = (self.compare_arm if decision.arm_name == "compare_arm"
               else self.decompose_arm_v2)
        runner = getattr(arm, "run", arm)   # object-with-.run or bare callable
        try:
            result = runner(question, **run_kwargs)
        except Exception:  # noqa: BLE001 — specialist failure must not drop the slot
            logger.warning("specialist %s failed; falling back to generic decompose",
                           decision.arm_name, exc_info=True)
            generic = generic_runner(question, **run_kwargs)
            return generic, SlotDecision(
                DECOMPOSE_SLOT, DECOMPOSE_SLOT, decision.qtype, False)
        if result is None:
            generic = generic_runner(question, **run_kwargs)
            return generic, SlotDecision(
                DECOMPOSE_SLOT, DECOMPOSE_SLOT, decision.qtype, False)
        return result, decision

    # ---- PDD reader-inverse cardinality (within iter_dup_a) ----------
    def pdd_cardinality(self, question: str, **kw: Any) -> int:
        """PDD dup cardinality for the iter_dup_a arm (never a new arm).

        Delegates to the injected ``pdd_router`` when present; otherwise
        returns the locked base cardinality. Guaranteed ``>= base`` so PDD only
        ever dials the EXISTING 4th arm up, never below its validated baseline.
        """
        if self.pdd_router is None:
            return self.base_pdd_cardinality
        try:
            card = int(self.pdd_router(question, **kw))
        except Exception:  # noqa: BLE001
            return self.base_pdd_cardinality
        return max(self.base_pdd_cardinality, card)

    # ---- telemetry helper -------------------------------------------------
    def telemetry(self, decisions: Sequence[SlotDecision]) -> dict:
        """Aggregate slot-routing telemetry for a run (provenance, not gold)."""
        total = len(decisions)
        spec = sum(1 for d in decisions if d.is_specialist)
        by_arm: dict[str, int] = {}
        for d in decisions:
            by_arm[d.arm_name] = by_arm.get(d.arm_name, 0) + 1
        return {
            "enabled": self.enabled,
            "n_questions": total,
            "n_specialist_substitutions": spec,
            "decompose_slot_by_arm": by_arm,
            "pool_size": len(CANONICAL_POOL),   # always 4
        }


__all__ = [
    "CANONICAL_POOL",
    "DECOMPOSE_SLOT",
    "SlotDecision",
    "SpecialistSlotRouter",
    "QT_COMPARISON",
    "QT_COMPOSITIONAL",
    "QT_GENERIC",
]
