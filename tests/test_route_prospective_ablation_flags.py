"""Tests for the route_prospective.py ablation flag extension.

Source-scan tests (the script is a LIVE-eval driver requiring corpus +
APIs; full functional tests would need real fixtures). These verify
the flag wiring + monkey-patch pattern + telemetry counter scaffold
are present and mirror the arbitrate_post.py implementation.
"""

from __future__ import annotations

from pathlib import Path


_RP_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "route_prospective.py"
)


def _src() -> str:
    return _RP_PATH.read_text(encoding="utf-8")


# ============================================================
# Flag wiring
# ============================================================

def test_arbitrator_flag_added() -> None:
    src = _src()
    assert '"--arbitrator"' in src
    assert '"legacy", "pam_lite"' in src or '"legacy","pam_lite"' in src


def test_arbitrator_mode_flag_added() -> None:
    src = _src()
    assert '"--arbitrator-mode"' in src
    # Three modes mirror arbitrate_post / arbitrate_pam_lite
    assert '"argmax"' in src
    assert '"weighted_mix"' in src
    assert '"subset"' in src


def test_disable_cfde114_boost_flag_added() -> None:
    src = _src()
    assert '"--disable-cfde114-boost"' in src


def test_disable_hop_multipliers_flag_added() -> None:
    src = _src()
    assert '"--disable-hop-multipliers"' in src


# ============================================================
# Setup: PAM-lite arbitrator validation + monkey-patches
# ============================================================

def test_arbitrator_pam_lite_requires_router_pam_lite() -> None:
    """The arbitrator=pam_lite path must validate router=pam_lite."""
    src = _src()
    assert "use_pam_lite_arb" in src
    assert (
        '--arbitrator=pam_lite requires --router=pam_lite' in src
        or "arbitrate_pam_lite consumes" in src
    )


def test_disable_cfde114_boost_monkey_patches_score_v3bu() -> None:
    src = _src()
    # The monkey-patch replaces _score_v3bu_p_arm with a boost-zeroed
    # variant on the query_type_classifier module.
    assert "_qtc._score_v3bu_p_arm = _v3bu_no_boost" in src or (
        "_score_v3bu_p_arm" in src and "_v3bu_no_boost" in src
    )


def test_disable_hop_multipliers_monkey_patches_get_hop_weight() -> None:
    src = _src()
    # The monkey-patch replaces get_hop_weight to always return 1.0.
    assert "_qtc.get_hop_weight = lambda arm, hop: 1.0" in src


def test_setup_announces_ablation_flags() -> None:
    """The setup block must print announcement lines when ablations
    are enabled (visibility for downstream eval audits)."""
    src = _src()
    assert "[setup] ablation: --disable-cfde114-boost" in src
    assert "[setup] ablation: --disable-hop-multipliers" in src


# ============================================================
# Telemetry counters
# ============================================================

def test_telemetry_counters_initialized() -> None:
    src = _src()
    assert "cfde114_fire_count = 0" in src
    assert "hop_multiplier_active_count = 0" in src
    assert "non_unitary_p_arm_count = 0" in src


def test_telemetry_counters_in_summary_output() -> None:
    """Summary JSON must include the per-run telemetry counters."""
    src = _src()
    assert '"cfde114_fire_count": cfde114_fire_count' in src
    assert '"hop_multiplier_active_count": hop_multiplier_active_count' in src
    assert '"non_unitary_p_arm_count": non_unitary_p_arm_count' in src


def test_telemetry_block_defensive_against_feature_extraction_errors() -> None:
    """Telemetry MUST NOT block the main routing path. Source-scan
    that the per-query counter increment block is wrapped in
    try/except (mirrors arbitrate_post pattern)."""
    src = _src()
    # The per-query counter block sits in the main loop next to
    # `feat = classify_with_features(question)` and uses
    # extract_semantic_features. Locate it via the call.
    marker = "from mothrag.routing.semantic_features import ("
    assert marker in src, "per-query telemetry block marker missing"
    # There may be many imports; find the one inside the per-query loop
    # by anchoring to the inner _qtc_feats alias.
    assert "_qtc_feats = " not in src  # we use `as _qtc_feats` form
    idx = src.index(marker)
    nearby = src[idx : idx + 1200]
    # The block sits inside try / except Exception
    assert "try:" in src[max(0, idx - 200): idx]
    assert "except Exception" in nearby or "except:" in nearby


# ============================================================
# Default behavior preserved (no regression)
# ============================================================

def test_defaults_preserve_legacy_arbitrator() -> None:
    """--arbitrator default MUST be 'legacy' (no behavior change unless
    explicitly opted in)."""
    src = _src()
    assert '"--arbitrator", default="legacy"' in src


def test_defaults_preserve_disabled_ablations() -> None:
    """Ablation flags default to False (existing behavior)."""
    src = _src()
    assert (
        '"--disable-cfde114-boost", action="store_true", default=False' in src
    )
    assert (
        '"--disable-hop-multipliers", action="store_true", default=False' in src
    )


def test_arbitrator_mode_default_is_argmax() -> None:
    src = _src()
    assert '"--arbitrator-mode", default="argmax"' in src


# ============================================================
# Cross-script parity with arbitrate_post.py
# ============================================================

def test_same_four_flags_as_arbitrate_post() -> None:
    """The four flag names introduced here mirror arbitrate_post.py
    exactly so one set of CLI args works across both scripts.
    """
    rp = _src()
    ap_path = _RP_PATH.parent / "arbitrate_post.py"
    if not ap_path.exists():
        return  # arbitrate_post not present in this checkout (skip)
    ap = ap_path.read_text(encoding="utf-8")
    for flag in (
        "--arbitrator",
        "--arbitrator-mode",
        "--disable-cfde114-boost",
        "--disable-hop-multipliers",
    ):
        assert flag in rp, f"flag missing from route_prospective: {flag}"
        assert flag in ap, f"flag missing from arbitrate_post: {flag}"
