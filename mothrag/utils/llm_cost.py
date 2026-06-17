# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Per-1M-token public pricing for common reader / judge models.

Updated 2026-04. ``COST_PER_1M`` keys are model identifiers as the providers
expose them; pricing is in USD per 1M tokens.
"""

from mothrag.eval.latency import COST_PER_1M, estimate_cost  # noqa: F401  (re-export)

__all__ = ["COST_PER_1M", "estimate_cost"]
