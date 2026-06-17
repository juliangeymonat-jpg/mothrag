# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""route_prospective ChainFilter CLI pre-wire (builder)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_SPEC = importlib.util.spec_from_file_location(
    "route_prospective",
    Path(__file__).resolve().parents[1] / "scripts" / "route_prospective.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def _args(**kw):
    base = dict(use_chainfilter=False, chainfilter_hop_min=2,
               chainfilter_gamma_min=None, chainfilter_top_out=5)
    base.update(kw)
    return SimpleNamespace(**base)


def test_chainfilter_off_builds_nothing():
    assert mod._build_chain_filter(_args(use_chainfilter=False)) is None


def test_chainfilter_on_builds_enabled_filter_from_flags():
    cf = mod._build_chain_filter(_args(
        use_chainfilter=True, chainfilter_hop_min=3, chainfilter_top_out=7,
        chainfilter_gamma_min=0.4))
    assert cf is not None
    assert cf.cfg.enabled
    assert cf.cfg.hop_gate_min == 3
    assert cf.cfg.top_k_out == 7
    assert cf.cfg.gamma_cfg.gamma_low == 0.4


def test_chainfilter_gamma_min_defaults_to_ragnatela_low():
    cf = mod._build_chain_filter(_args(use_chainfilter=True))
    # gamma_min unset → keeps the RagnatelaConfig default (0.33).
    assert cf.cfg.gamma_cfg.gamma_low == 0.33
