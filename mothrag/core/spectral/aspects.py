# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Aspect extraction over an answer string.

An "aspect" is a noun-phrase-level claim segment within an answer that
can be independently verified. For example, the answer "Paris is the
capital of France and has 2.1M people" yields aspects approximately
{"Paris", "the capital of France", "2.1M people"}.

Two extractors:

- :func:`extract_aspects_naive` -- Capitalised-NP regex + numeric-quantity
  regex. Zero deps, deterministic, fast. Default.
- :func:`extract_aspects_spacy` -- spaCy ``en_core_web_sm`` dep-parse over
  ``noun_chunks``. Higher recall on lowercase + multi-token NPs.
  Available when ``mothrag[active-learning]`` is installed.

The public :func:`extract_aspects` function picks the strongest available
extractor automatically, falling back gracefully when spaCy is missing.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

logger = logging.getLogger(__name__)


# Hard cap on aspects per answer. Strategies #9 cascade calls fan out
# over the aspect list so an unbounded extractor could cause runaway
# LLM cost.
DEFAULT_MAX_ASPECTS: int = 8


# Patterns for the naive extractor:
#   - Capitalised noun phrases (1-4 tokens, each Title-Case): "Paris",
#     "Eiffel Tower", "Bank of England".
#   - Numeric quantity tokens: "2.1M", "300 km", "1942", "%-style
#     percentages".
_NAIVE_NP_RE = re.compile(r"\b(?:[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+){0,3})\b")
_NAIVE_NUM_RE = re.compile(
    r"\b\d+(?:[,.]\d+)?(?:[KMB])?(?:\s*(?:%|km|miles?|years?|kg|m|cm))?\b",
    re.IGNORECASE,
)


def extract_aspects_naive(
    answer: str,
    *,
    max_aspects: int = DEFAULT_MAX_ASPECTS,
) -> list[str]:
    """Capitalised-NP + numeric-quantity regex extractor.

    Zero deps. Returns up to ``max_aspects`` aspects in the order they
    appear in ``answer``, de-duped (first occurrence wins).
    """
    if not answer:
        return []
    raw = list(_NAIVE_NP_RE.findall(answer)) + list(_NAIVE_NUM_RE.findall(answer))
    out: list[str] = []
    seen: set[str] = set()
    for asp in raw:
        a = asp.strip()
        if not a or a in seen:
            continue
        seen.add(a)
        out.append(a)
        if len(out) >= max_aspects:
            break
    return out


def extract_aspects_spacy(
    answer: str,
    *,
    max_aspects: int = DEFAULT_MAX_ASPECTS,
    nlp=None,
) -> list[str]:
    """spaCy ``noun_chunks``-based extractor.

    Requires ``mothrag[active-learning]`` (spacy>=3.7) + the
    ``en_core_web_sm`` model. ``nlp`` may be a pre-loaded spaCy pipeline
    so callers running batches don't repeat the model load.
    """
    if not answer:
        return []
    try:
        import spacy  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "extract_aspects_spacy requires spacy. "
            "Install via `pip install mothrag[active-learning]` or "
            "`pip install spacy` + `python -m spacy download en_core_web_sm`."
        ) from e

    if nlp is None:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. Run "
                "`python -m spacy download en_core_web_sm`."
            ) from e

    doc = nlp(answer)
    out: list[str] = []
    seen: set[str] = set()
    for chunk in doc.noun_chunks:
        text = chunk.text.strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        out.append(text)
        if len(out) >= max_aspects:
            break
    return out


def extract_aspects(
    answer: str,
    *,
    max_aspects: int = DEFAULT_MAX_ASPECTS,
    prefer_spacy: bool = False,
    nlp=None,
) -> list[str]:
    """Top-level extractor that picks the strongest available backend.

    Parameters
    ----------
    answer
        Free-text answer string.
    max_aspects
        Hard cap on extracted aspects (default 8).
    prefer_spacy
        If True, attempt spaCy first and fall back to naive only on
        ImportError / model-missing. If False (default), use naive
        only -- spaCy is opt-in via the kwarg, NOT eager-loaded.
    nlp
        Optional pre-loaded spaCy pipeline (forwarded to
        :func:`extract_aspects_spacy`).

    Returns
    -------
    list[str]
        Up to ``max_aspects`` aspect strings in extraction order.
    """
    if not answer:
        return []
    if prefer_spacy:
        try:
            return extract_aspects_spacy(
                answer, max_aspects=max_aspects, nlp=nlp,
            )
        except (ImportError, RuntimeError) as exc:
            logger.debug(
                "extract_aspects: spaCy unavailable (%s); falling back to naive.",
                exc,
            )
    return extract_aspects_naive(answer, max_aspects=max_aspects)


def union_aspects(*aspect_lists: Iterable[str]) -> list[str]:
    """De-dupe a union of aspect lists preserving first-occurrence order."""
    out: list[str] = []
    seen: set[str] = set()
    for lst in aspect_lists:
        for asp in lst:
            key = asp.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(asp)
    return out


__all__ = [
    "DEFAULT_MAX_ASPECTS",
    "extract_aspects",
    "extract_aspects_naive",
    "extract_aspects_spacy",
    "union_aspects",
]
