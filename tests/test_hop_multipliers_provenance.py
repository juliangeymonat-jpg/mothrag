"""Tests for _HOP_MULTIPLIERS weight provenance.

Source-scan tests verify the docstring contains the audit verdict +
the symmetric theory-derived pattern (2.0 / 0.1 / 1.0 with documented
specialist / anti-specialist / neutral assignments).
"""

from __future__ import annotations

from pathlib import Path


_QTC_PATH = (
    Path(__file__).resolve().parent.parent
    / "mothrag" / "core" / "query_type_classifier.py"
)


def _src() -> str:
    return _QTC_PATH.read_text(encoding="utf-8")


# ============================================================
# Provenance docstring presence
# ============================================================

def test_provenance_audit_marker_present() -> None:
    src = _src()
    assert "WEIGHT PROVENANCE AUDIT" in src


def test_audit_verdict_is_theory_derived() -> None:
    """The audit verdict line MUST explicitly say theory-derived, NOT
    F1-tuned, so future readers can confirm the anti-leak status."""
    src = _src()
    assert "theory-derived, NOT F1-tuned" in src or (
        "theory-derived" in src and "NOT F1-tuned" in src
    )


def test_symmetric_pattern_documented() -> None:
    """The 2.0 / 0.1 / 1.0 symmetric pattern is explicitly described
    as specialist / anti-specialist / neutral."""
    src = _src()
    assert "specialist arm on its cohort" in src
    assert "non-specialist arm on its cohort" in src
    assert "default neutral" in src or "neutral" in src


def test_per_arm_specialty_documented() -> None:
    """Each arm's specialty assignment is documented in the audit."""
    src = _src()
    # v3bu: polar + entity_attr specialist
    assert "v3bu" in src and "single-shot" in src
    # decompose: bridge + chain specialist
    assert "decompose" in src and "sub-question decomposer" in src
    # iter: chain + general specialist
    assert "iter" in src and "iterative refiner" in src


# ============================================================
# Numeric pattern preserved (no regression in weight values)
# ============================================================

def test_specialist_weight_is_two() -> None:
    """The specialist boost stays 2.0 (theory-derived)."""
    from mothrag.core.query_type_classifier import _HOP_MULTIPLIERS

    assert _HOP_MULTIPLIERS["v3bu"]["is_1hop_polar"] == 2.0
    assert _HOP_MULTIPLIERS["v3bu"]["is_1hop_entity_attr"] == 2.0
    assert _HOP_MULTIPLIERS["decompose"]["is_2hop_bridge"] == 2.0
    assert _HOP_MULTIPLIERS["decompose"]["is_3hop_chain"] == 2.0
    assert _HOP_MULTIPLIERS["iter"]["is_3hop_chain"] == 2.0
    assert _HOP_MULTIPLIERS["iter"]["is_general_multihop"] == 2.0


def test_anti_specialist_weight_is_point_one() -> None:
    """Anti-specialist penalty stays 0.1 (theory-derived)."""
    from mothrag.core.query_type_classifier import _HOP_MULTIPLIERS

    assert _HOP_MULTIPLIERS["v3bu"]["is_2hop_bridge"] == 0.1
    assert _HOP_MULTIPLIERS["v3bu"]["is_3hop_chain"] == 0.1
    assert _HOP_MULTIPLIERS["decompose"]["is_1hop_polar"] == 0.1
    assert _HOP_MULTIPLIERS["iter"]["is_1hop_polar"] == 0.1


def test_default_weight_is_one() -> None:
    """Default (no matching cohort) stays 1.0 (neutral)."""
    from mothrag.core.query_type_classifier import _HOP_MULTIPLIERS

    for arm in ("v3bu", "decompose", "iter"):
        assert _HOP_MULTIPLIERS[arm]["default"] == 1.0


def test_only_three_distinct_weight_values() -> None:
    """The whole table uses ONLY {2.0, 0.1, 1.0} -- the symmetric
    theoretical pattern. Any other value would indicate F1 tuning."""
    from mothrag.core.query_type_classifier import _HOP_MULTIPLIERS

    used_values = set()
    for arm_table in _HOP_MULTIPLIERS.values():
        for w in arm_table.values():
            used_values.add(w)
    assert used_values == {0.1, 1.0, 2.0}, (
        f"Unexpected weight values in _HOP_MULTIPLIERS: {used_values}. "
        f"Theory pattern allows only {{2.0, 0.1, 1.0}}. If a tuning "
        f"change is intentional, update the provenance docstring."
    )
