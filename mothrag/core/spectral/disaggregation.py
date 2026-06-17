# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Per-aspect disaggregation of γ / L4b / cross-arm-agreement primitives.

Each disaggregator returns ``dict[aspect: str] -> score: float`` with
scores clamped to ``[0, 1]``. Missing-signal defaults follow the same
convention as :class:`mothrag.core.arbitrate.DeterministicArbitrator`:

- ``gamma``       -> 1.0   (do not artificially down-weight)
- ``l4b``         -> 1.0   (temporal stable until proven otherwise)
- ``agreement``   -> 0.0   (no agreement until evidence)

At the v0.5.0 alpha pipeline these helpers operate primarily by
*broadcasting* the whole-answer signal across the aspect list. The
agreement disaggregator additionally computes per-aspect cosine
similarity against the corresponding aspect surface in other arm
answers, which is the only signal channel that's truly per-aspect at
alpha (the embedder + arm outputs are already available, no extra
SDK / verifier needed).

When the full per-aspect γ + L4b pipeline ports from
``mothrag.eval.iterative_pipeline`` in v0.5.1, the broadcast defaults
swap out for the granular scores; the public API is stable so no caller
refactor is needed.
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

from mothrag.core.spectral.aspects import (
    DEFAULT_MAX_ASPECTS,
    extract_aspects,
)

logger = logging.getLogger(__name__)


def _clamp(x: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _normalize_aspects(
    answer: str,
    aspects: Sequence[str] | None,
    max_aspects: int,
) -> list[str]:
    if aspects is not None:
        return list(aspects)[:max_aspects]
    return extract_aspects(answer, max_aspects=max_aspects)


def gamma_per_aspect(
    answer: str,
    *,
    gamma_status: str | float | None = None,
    aspects: Sequence[str] | None = None,
    max_aspects: int = DEFAULT_MAX_ASPECTS,
    per_aspect_overrides: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Disaggregate the γ verifier signal across answer aspects.

    Parameters
    ----------
    answer
        Free-text answer.
    gamma_status
        Whole-answer γ signal. Accepted shapes:

        - string  ``"valid"`` (-> 1.0) / ``"partial"`` (-> 0.5) /
                  ``"invalid"`` (-> 0.0) -- mirrors
                  :mod:`mothrag.core.selective_ensemble` conventions.
        - float in ``[0, 1]`` -- explicit numeric override.
        - ``None`` -- default to 1.0 (no down-weight).
    aspects
        Optional pre-computed aspect list. If omitted,
        :func:`extract_aspects` is called on ``answer``.
    max_aspects
        Hard cap on aspect list size.
    per_aspect_overrides
        Optional ``{aspect: float}`` partial overrides (used when a
        granular per-aspect γ verifier IS available in the caller --
        the alpha pipeline's broadcast becomes precise once per-aspect γ
        is wired in v0.5.1).
    """
    asp_list = _normalize_aspects(answer, aspects, max_aspects)
    base = _gamma_status_to_score(gamma_status)
    overrides = dict(per_aspect_overrides or {})
    return {a: _clamp(overrides.get(a, base)) for a in asp_list}


def _gamma_status_to_score(status: str | float | None) -> float:
    if status is None:
        return 1.0
    if isinstance(status, (int, float)):
        return _clamp(float(status))
    s = str(status).strip().lower()
    if s == "valid":
        return 1.0
    if s == "partial":
        return 0.5
    if s == "invalid":
        return 0.0
    return 1.0


def l4b_per_aspect(
    answer: str,
    *,
    l4b_cancelled: bool | None = None,
    l4b_score: float | None = None,
    aspects: Sequence[str] | None = None,
    max_aspects: int = DEFAULT_MAX_ASPECTS,
    per_aspect_overrides: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Disaggregate the L4b temporal-stability signal across aspects.

    Parameters
    ----------
    l4b_cancelled
        Whole-answer L4b cancellation flag. ``True`` -> base 0.0
        (whole answer's temporal stability failed); ``False`` -> 1.0;
        ``None`` -> 1.0 (no signal).
    l4b_score
        Explicit numeric override in ``[0, 1]`` (when the caller has
        already computed a continuous stability score). Overrides
        ``l4b_cancelled`` when both are supplied.
    per_aspect_overrides
        Optional ``{aspect: float}`` partial overrides.
    """
    asp_list = _normalize_aspects(answer, aspects, max_aspects)
    if l4b_score is not None:
        base = _clamp(l4b_score)
    elif l4b_cancelled is None:
        base = 1.0
    else:
        base = 0.0 if l4b_cancelled else 1.0
    overrides = dict(per_aspect_overrides or {})
    return {a: _clamp(overrides.get(a, base)) for a in asp_list}


def agreement_per_aspect(
    answer: str,
    *,
    arm_answers: Mapping[str, str] | None = None,
    embedder=None,
    aspects: Sequence[str] | None = None,
    max_aspects: int = DEFAULT_MAX_ASPECTS,
    threshold: float = 0.70,
    per_aspect_overrides: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Cross-arm agreement per aspect.

    For each aspect ``a`` in ``answer``, compute the fraction of
    *other* arms whose answer contains an aspect cosine-similar
    (>= ``threshold``) to ``a``. Returns ``{aspect: agreement_in_[0,1]}``.

    When ``embedder`` is None or ``arm_answers`` has < 2 entries,
    returns 0.0 for every aspect (consistent with the global
    cross-arm agreement default in
    :class:`mothrag.core.arbitrate.DeterministicArbitrator`).

    Empty / whitespace-only aspects score 0.0.

    Parameters
    ----------
    embedder
        Anything with ``.embed_batch(list[str]) -> list[list[float]]``.
    threshold
        Cosine threshold for aspect-level "agreement"; 0.70 default
        matches :func:`mothrag.core.arbitrate.pairwise_agreement`.
    """
    asp_list = _normalize_aspects(answer, aspects, max_aspects)
    overrides = dict(per_aspect_overrides or {})

    if embedder is None or not arm_answers or len(arm_answers) < 2:
        return {a: _clamp(overrides.get(a, 0.0)) for a in asp_list}

    # Build cross-arm aspect lists. We include EVERY entry in arm_answers
    # (including the one whose text may equal `answer`) because the
    # spectral semantics measure "how broadly is this aspect supported
    # across the configured arms" -- if the chosen happens to equal one
    # arm's output, that arm legitimately corroborates the aspect.
    # Callers who need strict "other-arms-only" semantics should pass a
    # pre-filtered arm_answers dict.
    other_arm_aspects: list[list[str]] = []
    for _other_name, other_text in arm_answers.items():
        other_asp = extract_aspects(other_text or "", max_aspects=max_aspects)
        if other_asp:
            other_arm_aspects.append(other_asp)

    if not other_arm_aspects:
        return {a: _clamp(overrides.get(a, 0.0)) for a in asp_list}

    # Pre-embed everything in one batch for efficiency.
    all_texts = list(asp_list)
    arm_offsets: list[tuple[int, int]] = []
    for other in other_arm_aspects:
        start = len(all_texts)
        all_texts.extend(other)
        arm_offsets.append((start, start + len(other)))

    try:
        vecs = embedder.embed_batch(all_texts)
    except Exception as exc:  # noqa: BLE001
        logger.warning("agreement_per_aspect embed_batch failed: %s", exc)
        return {a: _clamp(overrides.get(a, 0.0)) for a in asp_list}

    import numpy as np

    arr = np.asarray(vecs, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != len(all_texts):
        return {a: _clamp(overrides.get(a, 0.0)) for a in asp_list}

    # L2 normalise defensively.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms

    out: dict[str, float] = {}
    for i, aspect in enumerate(asp_list):
        if not aspect.strip():
            out[aspect] = _clamp(overrides.get(aspect, 0.0))
            continue
        agreeing_arms = 0
        for start, end in arm_offsets:
            if end <= start:
                continue
            other_vecs = arr[start:end]
            sims = other_vecs @ arr[i]
            if float(sims.max()) >= threshold:
                agreeing_arms += 1
        score = agreeing_arms / len(arm_offsets) if arm_offsets else 0.0
        out[aspect] = _clamp(overrides.get(aspect, score))
    return out


__all__ = [
    "gamma_per_aspect",
    "l4b_per_aspect",
    "agreement_per_aspect",
]
