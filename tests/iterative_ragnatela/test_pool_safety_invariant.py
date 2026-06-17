# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Pool-safety formal verification for the Iterative Ragnatela.

The pool-safety axiom: the arm pool is LOCKED at N=4 (v3bu / decompose / iter /
iter_dup_a PDD). The γ-feedback loop is an *internal upgrade of the iter
machinery* — it iterates over those same four arms, growing the shared
retrieval context — it is NOT a fifth arm, and the bridge is a retrieval
SUBSTRATE upstream of the pool, never an arm either.

These tests pin that invariant programmatically at every iteration, so a
regression that smuggled a 5th arm (or counted the bridge as one, or shrank the
pool when the γ-pool muted a distractor) fails CI before any eval run.

The orchestrator is backend-agnostic: the arm pool and retriever are injected
callables, so every check here runs offline — no index, no LLM, no eval run.
"""
from __future__ import annotations

from mothrag.iterative_ragnatela import (
    ArmAnswer,
    RagnatelaConfig,
    RagnatelaOrchestrator,
)
from mothrag.iterative_ragnatela.gamma_pooling import normalize_answer, pool_answers

# The LOCKED 4-arm pool. iter_dup_a == the PDD 4th arm (dup_arm mechanism).
CANONICAL_ARMS = ("v3bu", "decompose", "iter", "iter_dup_a")


def _recording(arm_runner):
    """Wrap an arm_runner, recording the context + answers it saw each call."""

    def wrapped(question, context):
        answers = list(arm_runner(question, context))
        wrapped.calls.append(
            {
                "context": list(context),
                "arm_names": [a.arm for a in answers],
                "answers": answers,
            }
        )
        return answers

    wrapped.calls = []
    return wrapped


def _synthetic_retriever(sub_questions, context):
    """One fresh evidence item per sub-question per round (monotone growth)."""
    round_id = len(context)
    return [
        f"evidence-r{round_id}-{i}::{normalize_answer(s)[:30]}"
        for i, s in enumerate(sub_questions)
    ]


# --- the four canonical arms, three agreeing on the answer + one distractor ---

def _rising_four_arms(question, context):
    """3 arms converge on 'Paris' as context arrives; iter is a LOW distractor."""
    n = len(context)
    g = min(0.40 + 0.12 * n, 0.95)
    return [
        ArmAnswer("v3bu", "Paris", g + 0.02),
        ArmAnswer("decompose", "Paris", g),
        ArmAnswer("iter_dup_a", "Paris", g - 0.02),   # PDD
        ArmAnswer("iter", "Lyon", 0.20),              # anti-context distractor
    ]


def _stuck_mid_four_arms(question, context):
    """All four stay in the MID band → never converges (drives kmax bound)."""
    return [
        ArmAnswer("v3bu", "Paris", 0.50),
        ArmAnswer("decompose", "Paris", 0.48),
        ArmAnswer("iter_dup_a", "Paris", 0.46),
        ArmAnswer("iter", "Lyon", 0.40),
    ]


# ----------------------------------------------------------------------------
# 1. pool size is strictly 4 at every iteration (bridge NOT a 5th arm)
# ----------------------------------------------------------------------------

def test_iter_loop_preserves_arm_count():
    arm = _recording(_rising_four_arms)
    orch = RagnatelaOrchestrator(
        arm, retriever=_synthetic_retriever,
        config=RagnatelaConfig(max_iterations=3),
    )
    result = orch.run("In which city is the Eiffel Tower located?")

    assert arm.calls, "the loop must have run the arm pool at least once"
    # Every arm_runner invocation returned EXACTLY the four canonical arms.
    for call in arm.calls:
        assert sorted(call["arm_names"]) == sorted(CANONICAL_ARMS)
        assert len(call["arm_names"]) == 4
    # And every iteration partitioned exactly 4 arms into γ bands (the pool
    # cardinality the arbitrator sees — never 5, never 3).
    for trace in result.traces:
        assert trace.n_high + trace.n_mid + trace.n_low == 4


# ----------------------------------------------------------------------------
# 2. candidate (context) set is monotone-additive across iterations
# ----------------------------------------------------------------------------

def test_candidate_set_monotone_additive():
    arm = _recording(_stuck_mid_four_arms)   # never converges → runs to kmax
    orch = RagnatelaOrchestrator(
        arm, retriever=_synthetic_retriever,
        config=RagnatelaConfig(max_iterations=4),
    )
    orch.run("q with persistent uncertainty")

    contexts = [call["context"] for call in arm.calls]
    assert len(contexts) >= 2, "need multiple iterations to test growth"
    for prev, curr in zip(contexts, contexts[1:]):
        # ADDS candidates via γ-feedback, NEVER removes (set superset + len).
        assert set(prev).issubset(set(curr))
        assert len(curr) >= len(prev)
    assert len(contexts[-1]) > len(contexts[0]), "context must genuinely grow"


# ----------------------------------------------------------------------------
# 3. arbitrator arm_scores keys are the 4 arms — NO 'bridge' key
# ----------------------------------------------------------------------------

def test_arm_scores_no_bridge_key():
    def _bridge_reshaped_retriever(sub_questions, context):
        """Bridge SUBSTRATE seam: reshapes context, never adds an arm."""
        rid = len(context)
        return [
            f"bridge::evidence-r{rid}-{i}::{normalize_answer(s)[:24]}"
            for i, s in enumerate(sub_questions)
        ]

    arm = _recording(_rising_four_arms)
    orch = RagnatelaOrchestrator(
        arm, retriever=_bridge_reshaped_retriever,
        config=RagnatelaConfig(max_iterations=3),
    )
    orch.run("bridge-substrate-fed question")

    for call in arm.calls:
        arm_scores = {a.arm: a.gamma for a in call["answers"]}
        bands = pool_answers(call["answers"], RagnatelaConfig()).bands
        assert set(arm_scores) == set(CANONICAL_ARMS)
        assert "bridge" not in arm_scores
        assert "bridge" not in bands
    # The bridge-reshaped context DID flow into the loop (substrate active)...
    assert any(
        item.startswith("bridge::")
        for call in arm.calls[1:]
        for item in call["context"]
    )
    # ...yet it never surfaced as an arm. Substrate stays substrate.


# ----------------------------------------------------------------------------
# 4. γ-convergence terminates within K_max — no infinite loop
# ----------------------------------------------------------------------------

def test_gamma_convergence_bounded():
    # (a) a pool that NEVER reaches convergence_gamma must stop at kmax.
    never = _recording(_stuck_mid_four_arms)
    res_bound = RagnatelaOrchestrator(
        never, retriever=_synthetic_retriever,
        config=RagnatelaConfig(max_iterations=3),
    ).run("never converges")
    assert res_bound.iterations_used == 3
    assert res_bound.iterations_used <= 3
    assert res_bound.stop_reason == "max_iterations"
    assert not res_bound.converged

    # (b) a pool that DOES converge does so within the kmax=3 budget.
    rising = _recording(_rising_four_arms)
    res_conv = RagnatelaOrchestrator(
        rising, retriever=_synthetic_retriever,
        config=RagnatelaConfig(max_iterations=3),
    ).run("does converge")
    assert res_conv.converged
    assert res_conv.iterations_used <= 3
    assert res_conv.answer == "Paris"


# ----------------------------------------------------------------------------
# 5. muting a γ-LOW distractor leaves the pool at 4 (muted, not removed)
# ----------------------------------------------------------------------------

def test_distractor_arm_drop_safe():
    def _strong_with_distractor(question, context):
        return [
            ArmAnswer("v3bu", "Paris", 0.90),
            ArmAnswer("decompose", "Paris", 0.88),
            ArmAnswer("iter_dup_a", "Paris", 0.86),
            ArmAnswer("iter", "Lyon", 0.10),   # γ_LOW anti-context distractor
        ]

    arm = _recording(_strong_with_distractor)
    result = RagnatelaOrchestrator(
        arm, config=RagnatelaConfig(max_iterations=2),
    ).run("q with a confident majority + one distractor")

    # The distractor's wrong answer is EXCLUDED from the pooled consensus...
    assert result.answer == "Paris"
    # ...but the architecture still carries all four arms each iteration: the
    # LOW arm is MUTED (in the low band), not removed from the pool.
    for trace in result.traces:
        assert trace.n_high + trace.n_mid + trace.n_low == 4
        assert trace.n_low >= 1

    # Direct check on the pool outcome: distractor present in .low, total == 4.
    outcome = pool_answers(arm.calls[0]["answers"], RagnatelaConfig())
    assert len(outcome.high) + len(outcome.mid) + len(outcome.low) == 4
    assert any(a.arm == "iter" and a.answer == "Lyon" for a in outcome.low)
    assert outcome.answer == "Paris"
