# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""γ-convergence detection for the Iterative Ragnatela loop.

The arbitrator stops the loop when the pooled answer is BOTH confident and
stable: its pooled γ has reached ``convergence_gamma`` AND the (normalised)
pooled answer has not changed for ``convergence_stability_iters`` consecutive
iterations. Tracking stability prevents stopping on a one-off high-γ blip that
the next iteration would overturn.
"""
from __future__ import annotations

from mothrag.iterative_ragnatela.gamma_pooling import normalize_answer
from mothrag.iterative_ragnatela.types import PoolOutcome, RagnatelaConfig


class ConvergenceDetector:
    """Stateful γ + answer-stability convergence detector (one per loop run)."""

    def __init__(self, cfg: RagnatelaConfig) -> None:
        self.cfg = cfg
        self._last_norm: str | None = None
        # number of CONSECUTIVE iterations the pooled answer has repeated
        # (0 on the first observation / after any change).
        self._repeat_count = 0

    def update(self, outcome: PoolOutcome) -> bool:
        """Feed one iteration's pool outcome; return True iff converged."""
        norm = normalize_answer(outcome.answer)
        if norm and self._last_norm is not None and norm == self._last_norm:
            self._repeat_count += 1
        else:
            self._repeat_count = 0
        self._last_norm = norm

        gamma_ok = outcome.pooled_gamma >= self.cfg.convergence_gamma
        # ``convergence_stability_iters`` identical answers in a row means the
        # answer repeated (stability_iters - 1) times after its first sighting.
        needed = max(0, self.cfg.convergence_stability_iters - 1)
        stable_ok = self._repeat_count >= needed
        # An empty answer never counts as converged.
        return bool(norm) and gamma_ok and stable_ok

    @property
    def repeat_count(self) -> int:
        return self._repeat_count


__all__ = ["ConvergenceDetector"]
