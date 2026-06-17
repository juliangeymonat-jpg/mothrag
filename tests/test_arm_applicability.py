"""Tests for per-arm applicability predicates + legacy arm wrappers.

Under the hybrid architecture, component-autonomy on Arm.applicable() is safe
for the solid features (single_hop, is_polar_comparison, two_entity,
chain_marker; F1 >= 0.85 in a small audit). These tests pin the predicate
behaviour + the wrapper-class delegation.

The predicates are deterministic linguistic regex; no per-dataset tuning, no
test inspection.
"""

from __future__ import annotations


# ============================================================
# Predicate behaviour (mothrag.routing.arm_applicability)
# ============================================================

def test_v3bu_applicable_fires_on_polar_comparison() -> None:
    from mothrag.routing.arm_applicability import is_v3bu_applicable
    assert is_v3bu_applicable("Are Einstein and Newton both physicists?") is True
    assert is_v3bu_applicable("Were Mozart and Beethoven from the same country?") is True


def test_v3bu_applicable_FALSE_on_non_polar() -> None:
    from mothrag.routing.arm_applicability import is_v3bu_applicable
    assert is_v3bu_applicable("When was Einstein born?") is False
    assert is_v3bu_applicable("Who is the founder of Microsoft?") is False


def test_decompose_applicable_fires_on_two_entities() -> None:
    from mothrag.routing.arm_applicability import is_decompose_applicable
    # Two distinct cap NPs.
    assert is_decompose_applicable("Did Apple and Microsoft both IPO?") is True


def test_decompose_applicable_FALSE_on_single_entity() -> None:
    from mothrag.routing.arm_applicability import is_decompose_applicable
    assert is_decompose_applicable("When was Einstein born?") is False


def test_iter_applicable_fires_on_chain_marker() -> None:
    from mothrag.routing.arm_applicability import is_iter_applicable
    assert is_iter_applicable(
        "First X happened, then Y was founded later, and subsequently Z."
    ) is True


def test_iter_applicable_FALSE_on_no_chain_lexicon() -> None:
    from mothrag.routing.arm_applicability import is_iter_applicable
    assert is_iter_applicable("When was Einstein born?") is False


def test_infobox_arm_applicable_fires_on_single_hop_possessive() -> None:
    from mothrag.routing.arm_applicability import is_infobox_arm_applicable
    assert is_infobox_arm_applicable("Who is Einstein's mother?") is True


def test_infobox_arm_applicable_FALSE_on_no_possessive() -> None:
    from mothrag.routing.arm_applicability import is_infobox_arm_applicable
    assert is_infobox_arm_applicable("When was Einstein born?") is False


def test_all_predicates_handle_empty_question() -> None:
    from mothrag.routing.arm_applicability import (
        is_decompose_applicable, is_infobox_arm_applicable,
        is_iter_applicable, is_v3bu_applicable,
    )
    for pred in (is_v3bu_applicable, is_decompose_applicable,
                 is_iter_applicable, is_infobox_arm_applicable):
        assert pred("") is False
        assert pred("   ") is False


# ============================================================
# applicability_snapshot (helper for ArbitratorV2.arbitrate_pool)
# ============================================================

def test_applicability_snapshot_includes_all_pool_arms() -> None:
    from mothrag.routing.arm_applicability import applicability_snapshot
    pool = ["v3bu", "decompose", "iter", "infobox_arm"]
    snap = applicability_snapshot("Who is Einstein's mother?", pool)
    assert set(snap.keys()) == set(pool)
    # single_hop -> infobox_arm True; non-polar/no-chain/no-two-ent ->
    # other legacy arms False.
    assert snap["infobox_arm"] is True
    assert snap["v3bu"] is False
    assert snap["decompose"] is False
    assert snap["iter"] is False


def test_applicability_snapshot_unknown_arm_defaults_true() -> None:
    """Arms without a registered predicate default to True (assume
    applicable; downstream filters decide)."""
    from mothrag.routing.arm_applicability import applicability_snapshot
    snap = applicability_snapshot("hello", ["v3bu", "future_arm"])
    assert snap["future_arm"] is True


# ============================================================
# Legacy arm wrappers (mothrag.arms.legacy)
# ============================================================

def test_v3bu_wrapper_applicable_matches_predicate() -> None:
    from mothrag.arms import V3buArmWrapper
    arm = V3buArmWrapper(runner=lambda q: {"pred": "stub"})
    assert arm.applicable("Are X and Y both Z?") is True
    assert arm.applicable("When was X born?") is False


def test_decompose_wrapper_applicable_matches_predicate() -> None:
    from mothrag.arms import DecomposeArmWrapper
    arm = DecomposeArmWrapper(runner=lambda q: {"pred": "stub"})
    assert arm.applicable("Did Apple and Microsoft both IPO?") is True
    assert arm.applicable("When was Einstein born?") is False


def test_iter_wrapper_applicable_matches_predicate() -> None:
    from mothrag.arms import IterArmWrapper
    arm = IterArmWrapper(runner=lambda q: {"pred": "stub"})
    assert arm.applicable(
        "First X happened, then Y was founded later, and subsequently Z."
    ) is True
    assert arm.applicable("When was Einstein born?") is False


def test_legacy_wrapper_run_adapts_dict_to_ArmResult() -> None:
    """Wrapper's run() converts the legacy dict shape to ArmResult."""
    from mothrag.arms import V3buArmWrapper

    def _runner(_q):
        return {
            "pred": "Berlin",
            "retrieved_chunk_ids": ["c1", "c2"],
            "n_llm_calls": 1,
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "latency_s": 0.42,
            "gamma_final_status": "valid",  # extra -> metadata
        }

    arm = V3buArmWrapper(runner=_runner)
    result = arm.run("any question")
    assert result.pred == "Berlin"
    assert result.retrieved_chunk_ids == ["c1", "c2"]
    assert result.n_llm_calls == 1
    assert result.metadata["gamma_final_status"] == "valid"


def test_legacy_wrapper_run_catches_exception() -> None:
    """Wrapper returns ArmResult(pred='', metadata={error}) on runner failure."""
    from mothrag.arms import IterArmWrapper

    def _failing(_q):
        raise RuntimeError("simulated runner failure")

    arm = IterArmWrapper(runner=_failing)
    result = arm.run("any")
    assert result.pred == ""
    assert "RuntimeError" in result.metadata.get("error", "")


def test_legacy_wrapper_passes_through_ArmResult() -> None:
    """If runner returns ArmResult directly (not dict), wrapper passes through."""
    from mothrag.arms import DecomposeArmWrapper
    from mothrag.arms.base import ArmResult

    def _runner(_q):
        return ArmResult(pred="X", n_llm_calls=2)

    arm = DecomposeArmWrapper(runner=_runner)
    result = arm.run("any")
    assert result.pred == "X"
    assert result.n_llm_calls == 2


# ============================================================
# Arm Protocol compliance
# ============================================================

def test_legacy_wrappers_satisfy_Arm_protocol() -> None:
    from mothrag.arms import (
        Arm, DecomposeArmWrapper, IterArmWrapper, V3buArmWrapper,
    )
    for cls in (V3buArmWrapper, DecomposeArmWrapper, IterArmWrapper):
        arm = cls(runner=lambda q: {"pred": "stub"})
        # runtime_checkable Protocol
        assert isinstance(arm, Arm), f"{cls.__name__} does not satisfy Arm Protocol"
