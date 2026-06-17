# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Pluggable reading-arm Protocol for the MothRag query pipeline.

The legacy production stack ships three arms as
function-based per-arm helpers in :mod:`mothrag.eval.pipeline` /
``scripts/route_prospective.py``: V3+bu, decompose, iter. They stay
function-based for byte-identical production behaviour.

This sub-package adds CLASS-based optional arms that compose alongside
the legacy three via the ``--arms-pool`` CLI flag in
``scripts/route_prospective.py``. Each arm implements the :class:`Arm`
Protocol:

- :class:`InfoboxArm` (C3.6) -- direct structured-fact lookup over
  :class:`mothrag.core.retrieval.InfoboxIndex` triples. No LLM call;
  the value of a matching ``(subject, attribute)`` triple IS the
  answer. Useful for entity-attribute questions where reading a
  free-text passage would just paraphrase a structured fact already
  in the corpus's infoboxes.

- :class:`MothGraphArm` -- anchor-driven iterative graph traversal
  over :class:`mothrag.graph.GraphIndex` (OpenIE-extracted triples).
  Wraps MothRag-native principles around the bare graph walk: sel_v2-
  aligned anchor extraction, iterative anchor refinement,
  gamma-validated paths, L4b temporal-stability hashing, and a
  caller-supplied soft fallback. No LLM call inside the arm.

- :class:`BM25Arm` (C3.7) -- sparse keyword retrieval over the corpus
  via :mod:`rank_bm25`. Complements the dense semantic arms on
  queries with rare entities, quoted strings, exact dates, lexical-
  match needs. Requires ``pip install mothrag[retrieval]``
  (``rank-bm25`` is in that extra).

All arms in this sub-package are opt-in. Default ``--arms-pool``
remains ``v3bu,decompose,iter`` (3-arm production baseline). Setting
``--arms-pool v3bu,decompose,iter,infobox_arm`` activates the 4-arm
pool; adding ``mothgraph_arm`` or ``bm25_arm`` to that list activates
a 5-arm MothRag pool.
"""

from __future__ import annotations

from mothrag.arms.base import Arm, ArmResult
from mothrag.arms.infobox_arm import InfoboxArm
from mothrag.arms.legacy import (
    DecomposeArmWrapper,
    IterArmWrapper,
    V3buArmWrapper,
)
from mothrag.arms.mothgraph_arm import (
    GammaProtocol,
    MothGraphArm,
    StructuralGammaVerifier,
)

__all__ = [
    "Arm",
    "ArmResult",
    "DecomposeArmWrapper",
    "GammaProtocol",
    "InfoboxArm",
    "IterArmWrapper",
    "MothGraphArm",
    "StructuralGammaVerifier",
    "V3buArmWrapper",
]
