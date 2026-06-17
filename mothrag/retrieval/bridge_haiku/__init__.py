# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""BridgeRAG-Haiku — bridge-conditioned multi-hop retrieval single arm.

POC implementation of Bacellar's tripartite-judge bridge-conditioned
retrieval (arXiv 2604.03384v2) with Claude Haiku as the cost-grade judge.
Opt-in single arm for the MOTHRAG pool; stepping stone toward the iterative
γ-feedback ragnatela.

Public surface::

    from mothrag.retrieval.bridge_haiku import BridgeArm, BridgeConfig, Candidate
"""
from __future__ import annotations

from mothrag.retrieval.bridge_haiku.ann import (
    GeminiANNRetriever,
    build_gemini_ann,
)
from mothrag.retrieval.bridge_haiku.bridge_arm import (
    AnnRetrieve,
    BridgeArm,
    BridgeArmDegraded,
)
from mothrag.retrieval.bridge_haiku._haiku_base import is_transient_api_error
from mothrag.retrieval.bridge_haiku.entity_extractor import DualEntityExtractor
from mothrag.retrieval.bridge_haiku.pit_fusion import (
    percentile_rank,
    pit_fuse,
    rank_candidates,
)
from mothrag.retrieval.bridge_haiku.svo_generator import SVOQueryGenerator
from mothrag.retrieval.bridge_haiku.tripartite_judge import TripartiteJudge
from mothrag.retrieval.bridge_haiku.types import (
    BridgeConfig,
    BridgeResult,
    BridgeStats,
    Candidate,
)

__all__ = [
    "BridgeArm",
    "BridgeArmDegraded",
    "is_transient_api_error",
    "AnnRetrieve",
    "GeminiANNRetriever",
    "build_gemini_ann",
    "BridgeConfig",
    "BridgeResult",
    "BridgeStats",
    "Candidate",
    "SVOQueryGenerator",
    "DualEntityExtractor",
    "TripartiteJudge",
    "percentile_rank",
    "pit_fuse",
    "rank_candidates",
]
