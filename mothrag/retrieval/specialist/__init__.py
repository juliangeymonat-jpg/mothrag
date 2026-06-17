# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""M8 cluster-specialist retrieval primitives.

Specialist arms for structural query clusters the generic M5-M7 stack
under-serves (per cross-DS gap-cohort tagging). Each is conditional (fires only
on its cluster via an input-feature detector) and general-purpose / anti-leak.
"""
from __future__ import annotations

from mothrag.retrieval.specialist.compare_arm import (
    Candidate,
    CompareArm,
    CompareConfig,
    CompareResult,
    extract_comparison_attributes,
    extract_compared_entities,
    is_comparison_query,
)
from mothrag.retrieval.specialist.decompose_arm_v2 import (
    ChainCoherence,
    DecomposeArmV2,
    DecomposeConfig,
    DecomposeResult,
    HopResult,
    chain_key_terms,
    contains_compositional_markers,
    needs_decomposition,
    validate_chain_coherence,
)

__all__ = [
    "CompareArm",
    "CompareConfig",
    "CompareResult",
    "Candidate",
    "is_comparison_query",
    "extract_compared_entities",
    "extract_comparison_attributes",
    "DecomposeArmV2",
    "DecomposeConfig",
    "DecomposeResult",
    "HopResult",
    "ChainCoherence",
    "needs_decomposition",
    "contains_compositional_markers",
    "validate_chain_coherence",
    "chain_key_terms",
]
