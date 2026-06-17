# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""PIT (Probability Integral Transform) score fusion for BridgeRAG-Haiku.

Bacellar's PIT fusion (arXiv 2604.03384v2) converts each score channel to its
empirical percentile rank (the PIT of the score under its own within-query
distribution), then linearly combines:

    f(c) = (1 - alpha) * PIT_judge(c) + alpha * PIT_svo(c)

with ``alpha = 0.1`` (paper default). PIT makes the two channels
scale-free and comparable: a candidate in the 90th percentile of judge
scores and the 40th percentile of SVO similarity gets ``0.9*0.9 +
0.1*0.4``. Higher is better.

Pure + deterministic — no LLM, no numpy dependency. Ties are handled with
average (mid-) ranks so the transform is order-invariant within a tie
group. General-purpose; anti-leak (operates on scores only).
"""
from __future__ import annotations

from typing import Iterable, Sequence


def percentile_rank(scores: Sequence[float]) -> list[float]:
    """Empirical percentile rank (PIT) of each score within ``scores``.

    Returns values in ``[0, 1]``. Uses average ranks for ties:
    ``pit_i = (mean_rank_i - 0.5) / n`` where ranks are 1-based. So the
    minimum distinct score maps near ``0`` and the maximum near ``1``;
    an all-equal list maps everything to ``0.5`` (no signal).

    Examples
    --------
    >>> percentile_rank([10, 20, 30])
    [0.16666666666666666, 0.5, 0.8333333333333333]
    >>> percentile_rank([5, 5, 5])
    [0.5, 0.5, 0.5]
    """
    n = len(scores)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    # rank by value; average ranks for ties.
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        # positions i..j (0-based) share a tie -> average 1-based rank
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return [(r - 0.5) / n for r in ranks]


def pit_fuse(
    judge_scores: Sequence[float],
    svo_scores: Sequence[float],
    *,
    alpha: float = 0.1,
) -> list[float]:
    """Fuse judge + SVO channels into one score per candidate.

    ``judge_scores`` and ``svo_scores`` must be aligned (same length, same
    candidate order). Returns the fused score per candidate (higher better).
    """
    if len(judge_scores) != len(svo_scores):
        raise ValueError(
            f"channel length mismatch: judge={len(judge_scores)} "
            f"svo={len(svo_scores)}"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    pit_judge = percentile_rank(judge_scores)
    pit_svo = percentile_rank(svo_scores)
    return [
        (1.0 - alpha) * j + alpha * s
        for j, s in zip(pit_judge, pit_svo)
    ]


def rank_candidates(
    candidate_ids: Sequence[str],
    judge_scores: Sequence[float],
    svo_scores: Sequence[float],
    *,
    alpha: float = 0.1,
    top_k: int | None = None,
) -> list[str]:
    """Return ``candidate_ids`` ranked by fused PIT score (desc).

    Stable on ties: original order breaks ties (so a deterministic input
    yields a deterministic ranking). ``top_k`` truncates the result.
    """
    if not (len(candidate_ids) == len(judge_scores) == len(svo_scores)):
        raise ValueError("candidate_ids / judge_scores / svo_scores misaligned")
    fused = pit_fuse(judge_scores, svo_scores, alpha=alpha)
    order = sorted(
        range(len(candidate_ids)),
        key=lambda i: (-fused[i], i),  # desc fused, stable by index
    )
    ranked = [candidate_ids[i] for i in order]
    return ranked[:top_k] if top_k is not None else ranked


__all__ = ["percentile_rank", "pit_fuse", "rank_candidates"]
