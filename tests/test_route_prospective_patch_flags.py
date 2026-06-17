# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Patch CLI parity in route_prospective.

An earlier run crashed on ``unrecognized arguments: --use-gamma-refuse-loop``
because route_prospective did not expose the patch toggles. This verifies the
flags now parse with the grounded defaults and that every flag dest maps 1:1
onto an ``IterativeConfig`` field (so the ``iter_cfg`` wiring cannot silently
drift).

Grounding (iterative_pipeline.py dataclass):
  * ``use_gamma_refuse_loop`` dataclass-defaults TRUE and is locked ON
    → route_prospective defaults it ON (no regression); --disable-* ablates.
  * all other patch fields dataclass-default False → default OFF.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "route_prospective",
    Path(__file__).resolve().parents[1] / "scripts" / "route_prospective.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)

from mothrag.eval.iterative_pipeline import IterativeConfig

# Flag dest == IterativeConfig field name for every patch toggle.
PATCH_FIELDS = [
    "use_gamma_refuse_loop",
    "use_stepchain_parity_composite",
    "use_bug_pattern_wave_a",
    "use_p11_gamma_cap_fallback",
    "disable_p11_gamma_cap_fallback",
    "use_p12_decompose_collapse_cap",
    "use_p13_sub_q_abstain_filter",
    "use_p15_gamma_gated_naturalize",
    "use_p24_unified_abstain_markers",
    "composite_max_iterations",
]


class _CapturedParse(Exception):
    pass


def _grab_parser() -> argparse.ArgumentParser:
    """Capture route_prospective's argparse parser before any heavy work."""
    orig = argparse.ArgumentParser.parse_args
    captured: dict = {}

    def _cap(self, *a, **k):
        captured["parser"] = self
        raise _CapturedParse()

    argparse.ArgumentParser.parse_args = _cap
    try:
        mod.main()
    except _CapturedParse:
        pass
    finally:
        argparse.ArgumentParser.parse_args = orig
    return captured["parser"], orig


def _parse(argv: list[str]) -> argparse.Namespace:
    parser, orig = _grab_parser()
    return orig(parser, argv)


# ---- defaults (no-regression: refuse-loop ON, the rest OFF/None) ----------

def test_default_refuse_loop_on_others_off():
    ns = _parse([])
    assert ns.use_gamma_refuse_loop is True           # ON (no regression)
    assert ns.use_stepchain_parity_composite is False
    assert ns.use_bug_pattern_wave_a is False
    assert ns.use_p11_gamma_cap_fallback is False
    assert ns.disable_p11_gamma_cap_fallback is False
    assert ns.use_p12_decompose_collapse_cap is False
    assert ns.use_p13_sub_q_abstain_filter is False
    assert ns.use_p15_gamma_gated_naturalize is False
    assert ns.use_p24_unified_abstain_markers is False
    assert ns.composite_max_iterations is None        # → 5 at iter_cfg


def test_disable_gamma_refuse_loop_ablation():
    assert _parse(["--disable-gamma-refuse-loop"]).use_gamma_refuse_loop is False
    # explicit ON spelling still accepted.
    assert _parse(["--use-gamma-refuse-loop"]).use_gamma_refuse_loop is True


def test_full_patch_combo_parses():
    ns = _parse([
        "--use-bug-pattern-wave-a", "--use-stepchain-parity-composite",
        "--use-p11-gamma-cap-fallback", "--use-p12-decompose-collapse-cap",
        "--use-p13-sub-q-abstain-filter", "--use-p15-gamma-gated-naturalize",
        "--use-p24-unified-abstain-markers", "--composite-max-iterations", "6",
    ])
    assert ns.use_bug_pattern_wave_a is True
    assert ns.use_stepchain_parity_composite is True
    assert ns.use_p11_gamma_cap_fallback is True
    assert ns.use_p12_decompose_collapse_cap is True
    assert ns.use_p13_sub_q_abstain_filter is True
    assert ns.use_p15_gamma_gated_naturalize is True
    assert ns.use_p24_unified_abstain_markers is True
    assert ns.composite_max_iterations == 6


# ---- the specific flag that previously crashed must now be recognized -----

def test_blocker_flag_recognized():
    # Previously: "unrecognized arguments: --use-gamma-refuse-loop".
    ns = _parse(["--use-gamma-refuse-loop"])
    assert ns.use_gamma_refuse_loop is True


# ---- dest <-> IterativeConfig field 1:1 (wiring cannot drift) -------------

def test_every_flag_maps_to_an_iterative_config_field():
    ns = _parse([])
    cfg_fields = {f.name for f in dataclasses.fields(IterativeConfig)}
    for name in PATCH_FIELDS:
        assert hasattr(ns, name), f"arg dest missing: {name}"
        assert name in cfg_fields, f"IterativeConfig field missing: {name}"


def test_iterative_config_accepts_the_wired_kwargs():
    # Mirror the exact route_prospective iter_cfg mapping (None -> 5).
    ns = _parse([])
    cfg = IterativeConfig(
        use_gamma_refuse_loop=ns.use_gamma_refuse_loop,
        use_stepchain_parity_composite=ns.use_stepchain_parity_composite,
        use_bug_pattern_wave_a=ns.use_bug_pattern_wave_a,
        use_p11_gamma_cap_fallback=ns.use_p11_gamma_cap_fallback,
        disable_p11_gamma_cap_fallback=ns.disable_p11_gamma_cap_fallback,
        composite_max_iterations=(ns.composite_max_iterations
                                  if ns.composite_max_iterations is not None
                                  else 5),
        use_p12_decompose_collapse_cap=ns.use_p12_decompose_collapse_cap,
        use_p13_sub_q_abstain_filter=ns.use_p13_sub_q_abstain_filter,
        use_p15_gamma_gated_naturalize=ns.use_p15_gamma_gated_naturalize,
        use_p24_unified_abstain_markers=ns.use_p24_unified_abstain_markers,
    )
    assert cfg.use_gamma_refuse_loop is True           # default ON
    assert cfg.composite_max_iterations == 5           # None -> 5
    assert cfg.use_bug_pattern_wave_a is False
