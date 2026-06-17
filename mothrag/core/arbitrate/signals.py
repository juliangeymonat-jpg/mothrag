# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Helper signal computations for the :class:`DeterministicArbitrator`.

All helpers return values clamped to ``[0, 1]`` so they compose cleanly
under the weighted-sum scoring formula in
:class:`mothrag.core.arbitrate.arbitrator.DeterministicArbitrator`.
"""

from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)


def pairwise_agreement(
    answers: dict[str, str],
    *,
    embedder,
    threshold: float = 0.70,
) -> dict[str, float]:
    """Cross-arm semantic agreement per arm.

    For each arm A, agreement(A) is the fraction of *other* arms whose
    answer has cosine similarity >= ``threshold`` against A's answer.
    Returns a dict keyed by the input ``answers`` dict's keys with
    values in ``[0, 1]``.

    Arms with empty answers contribute 0 agreement; arms whose other
    arms are all empty get 0.

    Parameters
    ----------
    answers
        Mapping ``{arm_name: answer_text}``. Empty / whitespace-only
        answers are treated as non-agreeing.
    embedder
        Anything with ``.embed_batch(list[str]) -> list[list[float]]``
        (matches the :class:`mothrag.core.api.Embedder` Protocol).
    threshold
        Cosine threshold for "agreement". 0.70 is a sensible default
        on normalised Gemini-Embedding-001 vectors.
    """
    import numpy as np

    names: list[str] = list(answers.keys())
    if len(names) < 2:
        return {n: 0.0 for n in names}

    texts = [(answers.get(n) or "").strip() for n in names]
    non_empty_mask = [bool(t) for t in texts]

    # Replace empty texts with a sentinel so embed_batch doesn't choke;
    # we'll zero out their contribution below.
    safe_texts = [t if t else "<<empty>>" for t in texts]
    try:
        vecs = embedder.embed_batch(safe_texts)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pairwise_agreement embed_batch failed: %s", exc)
        return {n: 0.0 for n in names}
    arr = np.asarray(vecs, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != len(names):
        return {n: 0.0 for n in names}

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms

    sims = arr @ arr.T  # (n, n)

    out: dict[str, float] = {}
    n = len(names)
    for i, name_i in enumerate(names):
        if not non_empty_mask[i]:
            out[name_i] = 0.0
            continue
        # Compare arm i against all OTHER non-empty arms.
        agree_count = 0
        compare_count = 0
        for j in range(n):
            if j == i or not non_empty_mask[j]:
                continue
            compare_count += 1
            if float(sims[i, j]) >= threshold:
                agree_count += 1
        out[name_i] = (agree_count / compare_count) if compare_count else 0.0
    return out


__all__ = ["pairwise_agreement"]
