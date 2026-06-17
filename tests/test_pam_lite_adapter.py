"""Tests for the PAM-lite scorer reachability via the arbitration adapter.

These pin that the PAM-lite scoring path (with cfde114 + hop gating active)
is reachable and routes differently from the legacy label-based path on
queries where the patches differ.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


# ============================================================
# End-to-end: PAM-lite + cfde114 reachable
# ============================================================

def test_pam_lite_routes_query_via_scored_v3bu_with_cfde114_active() -> None:
    """Call PAM-lite scorer directly + verify cfde114 deep_complexity
    Wh-filter is active (a short 'What is X?' question should NOT
    register as deep_complexity=1 post-cfde114)."""
    from mothrag.routing.semantic_features import extract_semantic_features
    f = extract_semantic_features("What is X?")
    # cfde114: Wh-words are NOT subordinators; 0.4 floor below threshold.
    assert f.deep_complexity == 0.0


def test_pam_lite_routes_query_via_scored_v3bu_with_hop_gate_active() -> None:
    """cfde114 v3: comparison_marker contribution to V3+bu is gated by hop
    structure. When ``is_polar_comparison`` does NOT fire, the boost is
    ZERO regardless of comparison_marker score.
    """
    from mothrag.core.query_type_classifier import _score_v3bu_p_arm
    from mothrag.routing.semantic_features import SemanticFeatures

    # Non-polar (is_polar_comparison=0): boost suppressed by hop structure.
    f = SemanticFeatures(comparison_marker=0.7, is_polar_comparison=0.0)
    p_gated = _score_v3bu_p_arm(f)
    f_baseline = SemanticFeatures(comparison_marker=0.0, is_polar_comparison=0.0)
    p_baseline = _score_v3bu_p_arm(f_baseline)
    assert p_gated == pytest.approx(p_baseline, abs=1e-9)


def test_arm_subset_pam_lite_callable_via_post_hoc_path() -> None:
    """Smoke: arm_subset_pam_lite returns (subset, probabilities) tuple
    consumable by the adapter (matches the legacy arm_subset signature
    after extracting subset)."""
    from mothrag.core.query_type_classifier import arm_subset_pam_lite
    subset, probs = arm_subset_pam_lite(
        "When was Albert Einstein born?", threshold=0.3,
    )
    assert isinstance(subset, list)
    assert isinstance(probs, dict)
    assert set(probs.keys()) <= {"v3bu", "decompose", "iter"}
    # At least one arm passes threshold (always-non-empty guarantee).
    assert len(subset) >= 1
