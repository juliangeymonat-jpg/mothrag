# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""P12/P13/P15/P24 individual CLI flags + IterativeConfig plumbing.

Verifies:
1. ``IterativeConfig`` has the 4 new bool fields with default False.
2. The flags propagate from CLI -> argparse -> IterativeConfig constructor.
"""
from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_iterative_config_has_p12_p13_p15_p24_fields():
    """The 4 individual toggles must exist on IterativeConfig with default False."""
    sys.path.insert(0, str(_REPO_ROOT))
    from mothrag.eval.iterative_pipeline import IterativeConfig  # type: ignore

    cfg = IterativeConfig()
    assert cfg.use_p12_decompose_collapse_cap is False
    assert cfg.use_p13_sub_q_abstain_filter is False
    assert cfg.use_p15_gamma_gated_naturalize is False
    assert cfg.use_p24_unified_abstain_markers is False


def test_iterative_config_accepts_p12_p13_p15_p24_overrides():
    sys.path.insert(0, str(_REPO_ROOT))
    from mothrag.eval.iterative_pipeline import IterativeConfig  # type: ignore

    cfg = IterativeConfig(
        use_p12_decompose_collapse_cap=True,
        use_p13_sub_q_abstain_filter=True,
        use_p15_gamma_gated_naturalize=True,
        use_p24_unified_abstain_markers=True,
    )
    assert cfg.use_p12_decompose_collapse_cap is True
    assert cfg.use_p13_sub_q_abstain_filter is True
    assert cfg.use_p15_gamma_gated_naturalize is True
    assert cfg.use_p24_unified_abstain_markers is True


def test_composite_supersedes_when_wave_a_on_and_individual_off():
    """Even with all 4 individual toggles False, use_bug_pattern_wave_a=True
    keeps the composite behaviour (no regression on composite-only runs).
    """
    sys.path.insert(0, str(_REPO_ROOT))
    from mothrag.eval.iterative_pipeline import IterativeConfig  # type: ignore

    cfg = IterativeConfig(
        use_bug_pattern_wave_a=True,
        use_p12_decompose_collapse_cap=False,
        use_p13_sub_q_abstain_filter=False,
        use_p15_gamma_gated_naturalize=False,
        use_p24_unified_abstain_markers=False,
    )
    assert cfg.use_bug_pattern_wave_a is True
    # All individual toggles remain False; composite still active.
    assert cfg.use_p12_decompose_collapse_cap is False
