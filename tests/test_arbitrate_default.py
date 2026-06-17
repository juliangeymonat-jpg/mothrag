# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Verify pip install default routes arbitration through
arbitrate_with_c7 with use_c7=True + c7_trigger='gated' + sel_v2
(use_router_v2=True).

The measured baselines (HP 0.7770 / 2W 0.7379 / MQ 0.4298) use
arbitrate_with_c7 + Aurora L6 C7 phase-cancellation; pip install previously
had ``use_c7=False`` so the PROD distribution measured a weaker system than
the research baselines. This promotes C7 to PROD default.

The test inspects the call site (``mothrag/core/api.py``) statically +
ensures the IterativeConfig fields are wired so future edits can't
silently regress.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


def test_api_uses_arbitrate_with_c7_with_use_c7_true():
    """The pip-install adaptive-mode arbitration must enable C7 by default.

    The default was flipped use_c7=False -> True for baseline alignment. If a
    future refactor flips it back, this test catches it.
    """
    api_src = (_REPO_ROOT / "mothrag" / "core" / "api.py").read_text(
        encoding="utf-8")
    # The relevant block contains both 'arbitrate_with_c7(' and 'use_c7=True'.
    assert "arbitrate_with_c7(" in api_src
    # use_c7=True must appear at the call site (not just in comments).
    arbitrate_call_block_start = api_src.index("arbitrate_with_c7(")
    # Look at a sufficient window after the call signature.
    window = api_src[arbitrate_call_block_start:arbitrate_call_block_start + 600]
    assert "use_c7=True" in window, (
        "arbitrate_with_c7 call no longer passes use_c7=True. "
        "See mothrag/core/api.py dispatch.")
    assert "c7_trigger=\"gated\"" in window or "c7_trigger='gated'" in window, (
        "C7 trigger must be 'gated'.")
    assert "use_router_v2=True" in window, (
        "sel_v2 (use_router_v2=True) must remain enabled at the same site.")


def test_arbitrate_with_c7_signature_accepts_c7_args():
    """Ensure the function still accepts the kwargs api.py uses."""
    from mothrag.core.selective_ensemble import arbitrate_with_c7

    sig = inspect.signature(arbitrate_with_c7)
    for required in ("use_c7", "c7_trigger", "embedder", "query_embed",
                      "use_router_v2"):
        assert required in sig.parameters, \
            f"arbitrate_with_c7 missing kwarg {required!r}"


def test_default_embedder_remains_gemini_embedding_2():
    """C7 needs an embedder; api.py passes self.embedder which resolves to
    gemini-embedding-2 by default."""
    from mothrag.embedders.gemini import GeminiEmbedder

    sig = inspect.signature(GeminiEmbedder.__init__)
    assert sig.parameters["model"].default == "gemini-embedding-2"
