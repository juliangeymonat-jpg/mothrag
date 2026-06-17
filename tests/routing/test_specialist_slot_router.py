# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Pool-safe specialist slot router (M8 + PDD live-wiring core).

Proves the polymorphic ``decompose`` slot wires the M8 specialists (CompareArm /
DecomposeArm 2.0) + PDD routing into the pool by SUBSTITUTION, never
addition: the pool is provably the 4 canonical arms for every question, the
default (opt-OFF) path is byte-identical to the current generic decompose, and
routing keys only on question-text input features (anti-leak).
"""
from __future__ import annotations

from mothrag.routing import (
    CANONICAL_POOL,
    DECOMPOSE_SLOT,
    SlotDecision,
    SpecialistSlotRouter,
)
from mothrag.routing.specialist_slot_router import (
    QT_COMPARISON,
    QT_COMPOSITIONAL,
    QT_GENERIC,
)


# ---- fakes -----------------------------------------------------------------

class _FakeArm:
    """Specialist stub: records calls; optional .applicable gate."""

    def __init__(self, tag, *, applicable=True, raises=False, returns="ok"):
        self.tag = tag
        self._applicable = applicable
        self._raises = raises
        self._returns = returns
        self.calls = []

    def applicable(self, question):
        return self._applicable

    def run(self, question, **kw):
        self.calls.append((question, kw))
        if self._raises:
            raise RuntimeError(f"{self.tag} boom")
        return {"arm": self.tag, "pred": self._returns}


def _generic_runner_factory():
    calls = []

    def runner(question, **kw):
        calls.append((question, kw))
        return {"arm": "decompose", "pred": "generic"}

    runner.calls = calls
    return runner


YES = lambda q: True       # noqa: E731
NO = lambda q: False       # noqa: E731


# ---- 1. default-OFF is a byte-identical pass-through -----------------------

def test_default_off_is_generic_identity():
    router = SpecialistSlotRouter(
        compare_arm=_FakeArm("compare_arm"),
        decompose_arm_v2=_FakeArm("decompose_arm_v2"),
        is_comparison=YES, is_compositional=YES,   # would route, but disabled
        enabled=False,
    )
    for q in ["Are X and Y in the same country?", "chain question", ""]:
        d = router.decide(q)
        assert d == SlotDecision(DECOMPOSE_SLOT, DECOMPOSE_SLOT, QT_GENERIC, False)
        assert not d.is_specialist


# ---- 2. pool is ALWAYS the 4 canonical arms (the invariant) ----------------

def test_pool_keys_always_four():
    configs = [
        SpecialistSlotRouter(),
        SpecialistSlotRouter(enabled=True),
        SpecialistSlotRouter(enabled=True, compare_arm=_FakeArm("c"),
                             decompose_arm_v2=_FakeArm("d"),
                             is_comparison=YES, is_compositional=YES),
    ]
    for router in configs:
        for q in ["", "Are X and Y the same?", "Who directed the film whose star died?"]:
            assert router.pool_keys(q) == CANONICAL_POOL
            assert len(router.pool_keys(q)) == 4
    assert CANONICAL_POOL == ("v3bu", "decompose", "iter", "iter_dup_a")


# ---- 3/4. cohort → specialist fills the decompose slot ---------------------

def test_comparison_routes_to_compare_arm():
    router = SpecialistSlotRouter(
        compare_arm=_FakeArm("compare_arm"),
        is_comparison=YES, is_compositional=NO, enabled=True,
    )
    d = router.decide("Are X and Y both located in the same country?")
    assert d.slot == DECOMPOSE_SLOT          # pool key unchanged
    assert d.arm_name == "compare_arm"
    assert d.qtype == QT_COMPARISON
    assert d.is_specialist


def test_compositional_routes_to_decompose_v2():
    router = SpecialistSlotRouter(
        decompose_arm_v2=_FakeArm("decompose_arm_v2"),
        is_comparison=NO, is_compositional=YES, enabled=True,
    )
    d = router.decide("Who directed the film whose composer was born in Paris?")
    assert d.slot == DECOMPOSE_SLOT
    assert d.arm_name == "decompose_arm_v2"
    assert d.qtype == QT_COMPOSITIONAL
    assert d.is_specialist


# ---- 5. comparison wins when a question matches both cohorts ----------------

def test_comparison_precedence_over_compositional():
    router = SpecialistSlotRouter(
        compare_arm=_FakeArm("compare_arm"),
        decompose_arm_v2=_FakeArm("decompose_arm_v2"),
        is_comparison=YES, is_compositional=YES, enabled=True,
    )
    assert router.decide("ambiguous q").arm_name == "compare_arm"


# ---- 6/7. graceful fallback to generic when no specialist available --------

def test_cohort_without_injected_specialist_falls_back():
    router = SpecialistSlotRouter(   # comparison cohort, but compare_arm not injected
        compare_arm=None, is_comparison=YES, is_compositional=NO, enabled=True,
    )
    d = router.decide("Are X and Y in the same country?")
    assert not d.is_specialist
    assert d.arm_name == DECOMPOSE_SLOT
    assert d.qtype == QT_COMPARISON           # cohort still reported


def test_specialist_applicable_false_falls_back():
    router = SpecialistSlotRouter(
        compare_arm=_FakeArm("compare_arm", applicable=False),
        is_comparison=YES, is_compositional=NO, enabled=True,
    )
    assert not router.decide("Are X and Y the same?").is_specialist


# ---- 8. run_decompose_slot SUBSTITUTES (generic not run when specialist fires)

def test_run_decompose_slot_substitutes():
    compare = _FakeArm("compare_arm", returns="compared")
    generic = _generic_runner_factory()
    router = SpecialistSlotRouter(
        compare_arm=compare, is_comparison=YES, is_compositional=NO, enabled=True,
    )
    result, decision = router.run_decompose_slot(
        "Are X and Y the same?", generic_runner=generic, reader_model="m")
    assert result == {"arm": "compare_arm", "pred": "compared"}
    assert decision.is_specialist
    assert compare.calls and not generic.calls   # generic NOT run → pool still 4


def test_run_decompose_slot_generic_when_disabled():
    compare = _FakeArm("compare_arm")
    generic = _generic_runner_factory()
    router = SpecialistSlotRouter(compare_arm=compare, is_comparison=YES,
                                  enabled=False)
    result, decision = router.run_decompose_slot(
        "Are X and Y the same?", generic_runner=generic)
    assert result == {"arm": "decompose", "pred": "generic"}
    assert not decision.is_specialist
    assert generic.calls and not compare.calls


# ---- 9/10. specialist failure / None never drops the slot ------------------

def test_specialist_exception_graceful_fallback():
    compare = _FakeArm("compare_arm", raises=True)
    generic = _generic_runner_factory()
    router = SpecialistSlotRouter(compare_arm=compare, is_comparison=YES,
                                  is_compositional=NO, enabled=True)
    result, decision = router.run_decompose_slot(
        "Are X and Y the same?", generic_runner=generic)
    assert result == {"arm": "decompose", "pred": "generic"}   # slot preserved
    assert not decision.is_specialist
    assert generic.calls                                       # fell back, pool 4


def test_specialist_none_result_falls_back():
    compare = _FakeArm("compare_arm", returns=None)
    # _FakeArm returns a dict; emulate None by a bare callable returning None.
    router = SpecialistSlotRouter(
        compare_arm=lambda q, **k: None, is_comparison=YES, is_compositional=NO,
        enabled=True,
    )
    generic = _generic_runner_factory()
    result, decision = router.run_decompose_slot("q", generic_runner=generic)
    assert result == {"arm": "decompose", "pred": "generic"}
    assert not decision.is_specialist


# ---- 11. PDD cardinality never below the locked base -----------------------

def test_pdd_cardinality_never_below_base():
    base = SpecialistSlotRouter(base_pdd_cardinality=1)
    assert base.pdd_cardinality("q") == 1                      # no router → base

    dial_up = SpecialistSlotRouter(pdd_router=lambda q, **k: 3,
                                   base_pdd_cardinality=1)
    assert dial_up.pdd_cardinality("q") == 3                   # weak reader → up

    clamp = SpecialistSlotRouter(pdd_router=lambda q, **k: 0,
                                 base_pdd_cardinality=1)
    assert clamp.pdd_cardinality("q") == 1                     # never below base

    flaky = SpecialistSlotRouter(pdd_router=lambda q, **k: (_ for _ in ()).throw(ValueError()),
                                 base_pdd_cardinality=2)
    assert flaky.pdd_cardinality("q") == 2                     # error → base


# ---- 12. default detectors wire to the main classifier ---------------------

def test_default_detectors_wire_to_classifier():
    router = SpecialistSlotRouter(
        compare_arm=_FakeArm("compare_arm"), enabled=True)   # default detectors
    # a clear polar-comparison question → comparison cohort via real classifier.
    d = router.decide("Are Imperial River and Amaradia both located in the same country?")
    assert d.qtype == QT_COMPARISON
    assert d.arm_name == "compare_arm"


# ---- 13. anti-leak: routing sees ONLY question text ------------------------

def test_routing_keys_on_question_text_only():
    seen = []

    def spy_is_comparison(q):
        seen.append(q)
        return True

    router = SpecialistSlotRouter(
        compare_arm=_FakeArm("compare_arm"),
        is_comparison=spy_is_comparison, is_compositional=NO, enabled=True,
    )
    router.decide("Are X and Y the same?")
    assert seen == ["Are X and Y the same?"]   # only the question string, no DS/gold


# ---- 14. telemetry reports pool_size 4 + substitution counts ---------------

def test_telemetry_pool_size_four():
    router = SpecialistSlotRouter(
        compare_arm=_FakeArm("compare_arm"),
        decompose_arm_v2=_FakeArm("decompose_arm_v2"),
        enabled=True,
    )
    decisions = [
        SlotDecision(DECOMPOSE_SLOT, "compare_arm", QT_COMPARISON, True),
        SlotDecision(DECOMPOSE_SLOT, "decompose_arm_v2", QT_COMPOSITIONAL, True),
        SlotDecision(DECOMPOSE_SLOT, DECOMPOSE_SLOT, QT_GENERIC, False),
    ]
    t = router.telemetry(decisions)
    assert t["pool_size"] == 4
    assert t["n_questions"] == 3
    assert t["n_specialist_substitutions"] == 2
    assert t["decompose_slot_by_arm"]["compare_arm"] == 1
