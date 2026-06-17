"""Tests for the PAM-lite mechanism walkthrough trace.

Verifies the trace_pam_lite_mechanism reconstruction matches the
arbitrator's actual scoring, plus exercises the PDD mechanism
predictions on a synthetic 4-arm analog (v3bu / decompose / iter
+ v3bu_dup_a).
"""

from __future__ import annotations

import pytest


# ============================================================
# Trace reconstruction parity
# ============================================================

def test_trace_3arm_basic() -> None:
    from mothrag.core.arbitrate.pam_lite_mechanism import (
        trace_pam_lite_mechanism,
    )

    arms = ("v3bu", "decompose", "iter")
    raw = {"v3bu": 0.6, "decompose": 0.4, "iter": 0.2}
    answers = {"v3bu": "A", "decompose": "B", "iter": "C"}
    agree = {"v3bu": 0.0, "decompose": 0.0, "iter": 0.0}

    t = trace_pam_lite_mechanism(arms, raw, answers, agree)

    assert t.arms == arms
    assert t.subset == ("v3bu", "decompose")  # default threshold 0.3
    # winner = highest combined score (P_arm * (1.0*1.0 + 0.5*0 + 0.3*1.0)
    # = P_arm * 1.3). max P_arm = v3bu 0.6 -> v3bu wins.
    assert t.winner == "v3bu"


def test_trace_subset_empty_fallback() -> None:
    from mothrag.core.arbitrate.pam_lite_mechanism import (
        trace_pam_lite_mechanism,
    )

    arms = ("v3bu", "decompose", "iter")
    # All below default 0.3 threshold; argmax fallback should fire
    raw = {"v3bu": 0.1, "decompose": 0.2, "iter": 0.05}
    answers = {"v3bu": "A", "decompose": "B", "iter": "C"}
    agree = {"v3bu": 0.0, "decompose": 0.0, "iter": 0.0}

    t = trace_pam_lite_mechanism(arms, raw, answers, agree)
    # Argmax fallback picks decompose (max raw)
    assert t.subset == ("decompose",)


# ============================================================
# PDD mechanism predictions (the math from the module docstring)
# ============================================================

def test_pdd_dup_boost_v3bu_agreement_step5() -> None:
    """Mechanism step 5: dup-v3bu adds a guaranteed +1 to v3bu's
    agreement numerator. With no other-arm agreement, v3bu's
    agreement jumps from 0/2 to 1/3."""
    from mothrag.core.arbitrate.pam_lite_mechanism import (
        trace_pam_lite_mechanism,
    )

    # 3-arm pool (baseline). v3bu disagrees with decompose / iter.
    arms3 = ("v3bu", "decompose", "iter")
    raw3 = {"v3bu": 0.6, "decompose": 0.55, "iter": 0.5}
    ans3 = {"v3bu": "Paris", "decompose": "London", "iter": "Berlin"}
    agree3 = {"v3bu": 0.0, "decompose": 0.0, "iter": 0.0}

    t3 = trace_pam_lite_mechanism(arms3, raw3, ans3, agree3)
    s3_v3bu = t3.combined_score["v3bu"]

    # 4-arm pool with dup-v3bu (same answer as v3bu)
    arms4 = ("v3bu", "v3bu_dup_a", "decompose", "iter")
    raw4 = {**raw3, "v3bu_dup_a": 0.6}
    ans4 = {**ans3, "v3bu_dup_a": "Paris"}  # identical to v3bu
    # Mechanism: agreement(v3bu) = 1/3 (matches v3bu_dup_a only)
    agree4 = {
        "v3bu": 1.0 / 3.0,
        "v3bu_dup_a": 1.0 / 3.0,
        "decompose": 0.0,
        "iter": 0.0,
    }

    t4 = trace_pam_lite_mechanism(arms4, raw4, ans4, agree4)
    s4_v3bu = t4.combined_score["v3bu"]

    # Hypothesis: v3bu score with dup STRICTLY exceeds v3bu score
    # without dup, because agreement signal contributes positively.
    # Magnitude: w_agree(0.5) * 1/3 * P_arm(0.6) = 0.1
    assert s4_v3bu > s3_v3bu, (
        f"v3bu score must lift with dup-v3bu added: "
        f"3-arm={s3_v3bu:.4f}, 4-arm={s4_v3bu:.4f}"
    )
    delta = s4_v3bu - s3_v3bu
    expected_delta = 0.5 * (1.0 / 3.0) * 0.6
    assert abs(delta - expected_delta) < 1e-4, (
        f"v3bu score delta {delta:.4f} should match the agreement "
        f"contribution {expected_delta:.4f}"
    )


def test_pdd_decompose_agreement_penalty_step5() -> None:
    """Mechanism step 5 corollary: decompose / iter's agreement
    denominator grows from N-2=2 to N-1=3, demoting them when their
    numerator stays constant or grows slower."""
    from mothrag.core.arbitrate.pam_lite_mechanism import (
        trace_pam_lite_mechanism,
    )

    # 3-arm: decompose and iter agree with each other (semantic match)
    arms3 = ("v3bu", "decompose", "iter")
    raw3 = {"v3bu": 0.6, "decompose": 0.6, "iter": 0.6}
    ans3 = {"v3bu": "Paris", "decompose": "Lyon", "iter": "Lyon"}
    agree3 = {"v3bu": 0.0, "decompose": 0.5, "iter": 0.5}  # 1/2 each

    t3 = trace_pam_lite_mechanism(arms3, raw3, ans3, agree3)
    s3_dec = t3.combined_score["decompose"]

    # 4-arm pool with dup-v3bu added. decompose still agrees with iter
    # only (1 match), but denominator grows to N-1=3 (others). So
    # agreement(decompose) goes from 1/2 = 0.5 to 1/3 = 0.333.
    arms4 = ("v3bu", "v3bu_dup_a", "decompose", "iter")
    raw4 = {**raw3, "v3bu_dup_a": 0.6}
    ans4 = {**ans3, "v3bu_dup_a": "Paris"}
    agree4 = {
        "v3bu": 1.0 / 3.0,         # matches v3bu_dup_a only
        "v3bu_dup_a": 1.0 / 3.0,
        "decompose": 1.0 / 3.0,    # matches iter only, but denom up
        "iter": 1.0 / 3.0,
    }

    t4 = trace_pam_lite_mechanism(arms4, raw4, ans4, agree4)
    s4_dec = t4.combined_score["decompose"]

    assert s4_dec < s3_dec, (
        f"decompose score must decrease with dup-v3bu added: "
        f"3-arm={s3_dec:.4f}, 4-arm={s4_dec:.4f}"
    )


def test_pdd_relative_ranking_score_gap_narrows() -> None:
    """End-to-end: with dup-v3bu added, the score gap between v3bu and
    its competitor narrows (or flips) -- the mechanism shifts relative
    scores even when raw P_arm is unchanged. F1=0 cohort signal: the
    cohort where v3bu has the CORRECT but unpopular answer.
    """
    from mothrag.core.arbitrate.pam_lite_mechanism import (
        trace_pam_lite_mechanism,
    )

    arms3 = ("v3bu", "decompose", "iter")
    raw3 = {"v3bu": 0.55, "decompose": 0.6, "iter": 0.6}
    ans3 = {"v3bu": "Paris", "decompose": "Lyon", "iter": "Lyon"}
    agree3 = {"v3bu": 0.0, "decompose": 0.5, "iter": 0.5}
    t3 = trace_pam_lite_mechanism(arms3, raw3, ans3, agree3)
    gap_3arm = t3.combined_score["decompose"] - t3.combined_score["v3bu"]

    arms4 = ("v3bu", "v3bu_dup_a", "decompose", "iter")
    raw4 = {**raw3, "v3bu_dup_a": 0.55}
    ans4 = {**ans3, "v3bu_dup_a": "Paris"}
    agree4 = {
        "v3bu": 1.0 / 3.0,
        "v3bu_dup_a": 1.0 / 3.0,
        "decompose": 1.0 / 3.0,
        "iter": 1.0 / 3.0,
    }
    t4 = trace_pam_lite_mechanism(arms4, raw4, ans4, agree4)
    gap_4arm = t4.combined_score["decompose"] - t4.combined_score["v3bu"]

    # The score gap (decompose - v3bu) must NARROW with dup-v3bu added.
    # In F1=0 cohorts where the gap is small enough, this
    # narrowing flips the winner.
    assert gap_4arm < gap_3arm, (
        f"score gap (decompose - v3bu) must narrow with dup: "
        f"3-arm gap={gap_3arm:.4f}, 4-arm gap={gap_4arm:.4f}"
    )


# ============================================================
# N-dependency assertions per step (source-scan)
# ============================================================

def test_module_documents_n_dependency_table() -> None:
    from pathlib import Path

    src = (
        Path(__file__).resolve().parent.parent
        / "mothrag" / "core" / "arbitrate" / "pam_lite_mechanism.py"
    ).read_text(encoding="utf-8")
    # Step-by-step dependency table must be present
    for step_label in (
        "Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6", "Step 7",
    ):
        assert step_label in src, f"missing analysis section: {step_label}"
    # The N-dependency summary table
    assert "N-dependency" in src
    assert "PDD contrib" in src or "PDD contribution" in src


def test_module_documents_ablation_suggestions() -> None:
    from pathlib import Path

    src = (
        Path(__file__).resolve().parent.parent
        / "mothrag" / "core" / "arbitrate" / "pam_lite_mechanism.py"
    ).read_text(encoding="utf-8")
    # 4 suggested ablations A-D
    for tag in ("A.", "B.", "C.", "D."):
        assert tag in src, f"missing ablation suggestion {tag}"
    # Anti-leak section present
    assert "Anti-leak contract" in src or "anti-leak" in src.lower()


# ============================================================
# Defaults
# ============================================================

def test_agreement_threshold_default_exported() -> None:
    from mothrag.core.arbitrate.pam_lite_mechanism import (
        AGREEMENT_THRESHOLD_DEFAULT,
    )

    # Mirrors signals.pairwise_agreement default tau=0.70
    assert AGREEMENT_THRESHOLD_DEFAULT == 0.70


def test_mechanism_trace_dataclass_frozen() -> None:
    from mothrag.core.arbitrate.pam_lite_mechanism import (
        MechanismTrace,
    )
    from dataclasses import FrozenInstanceError

    t = MechanismTrace(
        arms=(), raw_scores={}, p_arm={}, subset=(),
        answers={}, agreement={}, gamma={}, faith={},
        combined_score={}, winner="", winner_reason="",
    )
    with pytest.raises(FrozenInstanceError):
        t.winner = "x"  # type: ignore[misc]
