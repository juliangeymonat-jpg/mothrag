# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""CrossArmConsensusStrategy (#4) — embed all arm answers, pick the majority.

Cost: 0 LLM calls (uses :attr:`RetryContext.embedder`). When ≥2 arms agree
under a cosine threshold, return the majority answer; else return None and
let the cascade continue.
"""

from __future__ import annotations

import logging

from mothrag.core.retry.protocol import RetryContext

logger = logging.getLogger(__name__)


# Cosine threshold for "semantic agreement". Calibration left to a
# follow-up; 0.7 is a defensible default for normalised Gemini-Embedding-2
# vectors.
DEFAULT_AGREEMENT_THRESHOLD = 0.70


class CrossArmConsensusStrategy:
    """Embed v3bu / dec / iter answers; if 2/3 agree, override the chosen."""

    name = "cross_arm_consensus"
    cost_estimate = 0

    def __init__(self, threshold: float = DEFAULT_AGREEMENT_THRESHOLD) -> None:
        self.threshold = threshold

    def applicable(self, ctx: RetryContext) -> bool:
        if ctx.embedder is None:
            return False
        non_empty = sum(1 for p in (ctx.v3bu_pred, ctx.dec_pred, ctx.iter_pred) if p)
        return non_empty >= 2

    def try_recover(self, ctx: RetryContext) -> str | None:
        import numpy as np

        labelled = [
            (ctx.v3bu_pred or "", "v3bu"),
            (ctx.dec_pred or "", "dec"),
            (ctx.iter_pred or "", "iter"),
        ]
        labelled = [(p, n) for p, n in labelled if p]
        if len(labelled) < 2:
            return None

        try:
            vecs = ctx.embedder.embed_batch([p for p, _ in labelled])
        except Exception as exc:  # noqa: BLE001
            logger.warning("cross_arm_consensus embed_batch failed: %s", exc)
            return None
        arr = np.asarray(vecs, dtype=np.float32)
        # Re-normalise (defensive — most embedders already do).
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms

        n = len(labelled)
        sims = arr @ arr.T
        # For each pair (i,j) with i<j, if sim>=threshold count as agree.
        clusters: list[list[int]] = []
        assigned = [-1] * n
        for i in range(n):
            if assigned[i] != -1:
                continue
            members = [i]
            assigned[i] = len(clusters)
            for j in range(i + 1, n):
                if assigned[j] == -1 and float(sims[i, j]) >= self.threshold:
                    members.append(j)
                    assigned[j] = len(clusters)
            clusters.append(members)

        # Pick the largest cluster (ties: prefer the cluster containing iter,
        # then dec, then v3bu — same hierarchy as SoftFallback).
        clusters.sort(key=lambda m: (-len(m), -_arm_priority(labelled, m)))
        winner = clusters[0]
        if len(winner) < 2:
            return None
        # Return the longest answer within the winning cluster (mild
        # information-richness proxy).
        members = [labelled[i] for i in winner]
        members.sort(key=lambda pair: len(pair[0]), reverse=True)
        return members[0][0]


def _arm_priority(labelled: list[tuple[str, str]], indices: list[int]) -> int:
    """Tie-breaker priority for cluster picking: iter > dec > v3bu."""
    rank = {"iter": 3, "dec": 2, "v3bu": 1}
    return max(rank.get(labelled[i][1], 0) for i in indices)


__all__ = ["CrossArmConsensusStrategy"]
