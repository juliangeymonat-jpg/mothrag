# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Pluggable pre-retrieval routers + arm-pool routing primitives for MothRag.

A "router" in this sub-package is a deterministic classifier that inspects the
question text BEFORE retrieval and returns a routing decision. Routers carry no
LLM dependency and add zero per-query latency beyond a handful of regex matches.

Current routers / primitives:

- :func:`infobox_router.is_entity_attribute_query` -- gates the
  ``dense_plus_infobox`` retrieval mode so the infobox modality fires ONLY on
  single-clause entity-attribute questions.
- :mod:`pdd_inverse_scaling` -- reader-inverse PDD dup cardinality /
  weight scaling for the ``iter_dup_a`` arm (a within-arm dial, never a new arm).
- :class:`specialist_slot_router.SpecialistSlotRouter` -- pool-safe
  polymorphic ``decompose`` slot dispatching the M8 specialists by SUBSTITUTION
  (the pool stays N=4; the bridge / specialists are never a 5th arm).
"""

from __future__ import annotations

from mothrag.routing.infobox_router import is_entity_attribute_query
from mothrag.routing.pdd_inverse_scaling import (
    PDDRouter,
    PDDRoutingDecision,
    PDDScalingConfig,
    estimate_reader_capability,
    expected_pdd_lift_pp,
    pdd_cardinality,
    pdd_intensity,
    pdd_weight,
    route_pdd,
)
from mothrag.routing.specialist_slot_router import (
    CANONICAL_POOL,
    DECOMPOSE_SLOT,
    SlotDecision,
    SpecialistSlotRouter,
)

__all__ = [
    "is_entity_attribute_query",
    "PDDRouter",
    "PDDRoutingDecision",
    "PDDScalingConfig",
    "estimate_reader_capability",
    "expected_pdd_lift_pp",
    "pdd_cardinality",
    "pdd_intensity",
    "pdd_weight",
    "route_pdd",
    "CANONICAL_POOL",
    "DECOMPOSE_SLOT",
    "SlotDecision",
    "SpecialistSlotRouter",
]
