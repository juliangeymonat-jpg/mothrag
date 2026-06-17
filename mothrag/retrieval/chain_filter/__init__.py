# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""ChainFilter v0.1 — γ-weighted, hop-gated post-retrieval fact-coverage filter."""
from mothrag.retrieval.chain_filter.chain_filter import (
    ChainFilter,
    ChainFilterConfig,
    default_gamma_scorer,
)

__all__ = ["ChainFilter", "ChainFilterConfig", "default_gamma_scorer"]
