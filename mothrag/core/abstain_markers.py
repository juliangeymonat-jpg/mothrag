# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Canonical abstain-marker recognition (P24 unification).

The same set of "answer = no information" strings appears in three
places: the research-grade iter pipeline
(``mothrag/eval/iterative_pipeline.py``), the pip-install adaptive iter
(``mothrag/core/api.py`` ``_arm_iter`` + ``_is_uncertain_answer``), and
historically the selective-ensemble retry signal.

The PROD pip-install must honour the same canonical list as the eval
pipeline (P24 "unified ABSTAIN_MARKERS" patch). This module is the
single source of truth that both codepaths import.

The list is conservative (whitespace-trimmed, case-insensitive,
canonical phrasings only). External callers can extend ``ABSTAIN_MARKERS``
by subclassing or by passing custom markers to :func:`is_abstain_marker`
via ``extra_markers``.

Anti-leak: tests on raw answer text only — no gold inspection, no DS
label.
"""
from __future__ import annotations

from typing import Iterable


# Canonical abstain markers (P24 unified set). Whitespace-trimmed,
# case-insensitive. Order is irrelevant; the set is consumed via
# membership check.
ABSTAIN_MARKERS: frozenset[str] = frozenset({
    "not in passages",
    "i don't know",
    "i do not know",
    "unknown",
    "no answer",
    "cannot answer",
    "none",                          # historical pip alias (api.py:_is_uncertain_answer)
})


def is_abstain_marker(
    answer: str | None,
    *,
    extra_markers: Iterable[str] | None = None,
) -> bool:
    """Return True when ``answer`` is a canonical abstain marker.

    Empty / None / whitespace-only counts as abstain (no information to
    ground). ``extra_markers`` lets callers add domain-specific phrases
    without mutating the global ``ABSTAIN_MARKERS`` set.
    """
    if not answer:
        return True
    stripped = answer.strip().lower()
    if not stripped:
        return True
    if stripped in ABSTAIN_MARKERS:
        return True
    if extra_markers:
        for em in extra_markers:
            if stripped == em.strip().lower():
                return True
    return False


__all__ = ["ABSTAIN_MARKERS", "is_abstain_marker"]
