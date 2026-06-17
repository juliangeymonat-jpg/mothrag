# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
r"""Deterministic per-query router for the dense+infobox retrieval mode.

Empirical motivation (preliminary, n=10-70 per cell):

  - HP T1 ``dense_plus_infobox``: -0.86pp vs dense baseline (multi-hop;
    infobox chunks displace prose chunks compositional readers need)
  - 2W T1 ``dense_plus_infobox``: +5.70pp (entity-attribute regime;
    structured fact-lookup is exactly the right modality)
  - MQ T1 ``dense_plus_infobox``: -12.50pp (4-hop chain reasoning;
    infobox chunks crowd out the chain bridge passages)

The unconditional dispatch is therefore NOT production-deployable.
This router gates the modality per-query: fire infobox augmentation
ONLY on single-clause entity-attribute questions (the 2W-style
beneficiary cohort); skip it on multi-hop / chain / comparison
queries (the HP / MQ regression cohort).

Classification is purely regex-based -- no LLM call, no NER, no
training data. Two pattern groups govern the decision:

1. :data:`MULTI_HOP_MARKERS` -- if ANY matches, the query is treated
   as multi-hop / chain / comparison. The router decides "skip
   infobox" regardless of any entity-attribute pattern matches.
   This is a high-precision negative gate.

2. :data:`ENTITY_ATTRIBUTE_PATTERNS` -- if any of these match AND no
   multi-hop marker fired, the router decides "fire infobox". This is
   the high-precision positive gate.

3. Default (no positive pattern matched, no negative marker) -- "skip
   infobox" (conservative: regressions in HP/MQ outweighed gains on
   the unmatched cohort in the pilot, so a "default-off" stance
   minimises harm).

The pattern lists are extensible without code changes by the caller
via :func:`is_entity_attribute_query(question, extra_positive_patterns=,
extra_negative_patterns=)`.

PROVENANCE DISCLOSURE -- v2 clean:

Two-step audit was performed; v2 ships with the leaked subset DROPPED.

Step 1 (initial disclosure):
- :data:`ENTITY_ATTRIBUTE_PATTERNS` (positives, 7): derived from general
  NL question-answering linguistic categories + seed example
  patterns ("When was X born?", "Who is Y's spouse?",
  "What is the capital of Z?"). NO inspection
  of HP / 2W / MQ TEST or TRAIN queries during pattern dev. Low
  leakage risk. RETAINED.

- :data:`MULTI_HOP_MARKERS` (negatives, originally 9):
  6 patterns ("that ... was/founded/...", "which ... also/then/later",
  "before/after ... than", "both/either/neither", comparative-than,
  polar A-or-B-filler) derived purely from the same seed brief +
  general subordinate-clause / chain / comparative-question linguistic
  categories. Low leakage risk. RETAINED.

  3 patterns extended after inspecting sample failure queries
  drawn from prior eval test queries (abstention-pool subset):

    "\\bof the \\w+ (?:that|who|which)\\b"
    "(?:composer|father|...|inventor) of the"
    "\\bwhere .*(?:the \\w+ of)"

Step 2 (follow-up review):
  The failure-sample display was filtered by ``F1 == 0`` and
  showed ``gold=... pred=...`` alongside the question text. Even
  though the 3 regexes match only question STRUCTURE (not gold
  content), F1-filtered sample selection IS test-set tuning, so
  these 3 are full-leakage and must be removed.

Decision (Option A clean):
  The 3 F1=0-derived patterns DROPPED. Router v2 ships with 6/9 clean
  negatives + 7/7 clean positives. Trade-off: v2 has lower multi-hop
  recall on bridge-entity surface forms like "composer of the film",
  "of the X that ...", "where ... the Y of" (these now fall through
  to default-off rather than negative-match short-circuit). End
  behaviour is identical (skip infobox) when no positive also matches;
  v2 may mis-fire on queries with BOTH a positive form AND a
  now-dropped negative form. Acceptable risk because positives target
  high-precision single-clause forms.

Validation guidance (requires TRAIN data):

- Train-split fire-rate audit: run :func:`is_entity_attribute_query`
  on HP/2W/MQ TRAIN n=200 stratified; per-dataset fire rate should
  match a domain-derived prior. NB:
  2W is 2-hop bridge reasoning (NOT Wikidata entity-attribute);
  only ~14.5% TRAIN is single-hop, so the originally-stated 2W prior
  of 60-80% was wrong -- expect ~10-20% on 2W TRAIN instead.
- ML proxy comparison: logistic regression on TRAIN embeddings vs
  router F1 on TEST. Router << ML -> under-fit (good).
  Router ~ ML -> valid generalization. Router >> ML -> overfitting.

A prior fire-rate audit on the abstention pool
(HP 2.0% / 2W 8.4% / MQ 11.0% on 3368 entries) is NOT a held-out
validation (same-provenance as the now-dropped patterns); preserved
here only as historical traceability. With v2 patterns the abstention-
pool rates will be lower (the 3 dropped negatives no longer suppress
positive-also-matching queries).
"""

from __future__ import annotations

import re
from typing import Iterable


# Positive patterns: single-clause entity-attribute question shapes
# where a structured "subject -- attribute: value" infobox triple is
# DIRECTLY the answer the reader wants to find.
ENTITY_ATTRIBUTE_PATTERNS: tuple[str, ...] = (
    # "When was X born/died/founded/established/created/published"
    r"\bwhen (?:was|did|is) [\w\s\-'\.]+ (?:born|die|died|founded|"
    r"established|created|invented|discovered|published|released|"
    r"signed|crowned|elected)\b",
    # "Who is/was X's spouse/parent/sibling/etc."
    r"\bwho (?:is|was) [\w\s'\-\.]+ (?:spouse|wife|husband|partner|"
    r"parent|father|mother|sibling|brother|sister|child|son|daughter|"
    r"founder|ceo|chairman|president)\b",
    # "What is X's birthplace/nationality/profession/etc."
    r"\bwhat (?:is|was) [\w\s'\-\.]+ (?:date of birth|birthplace|"
    r"birth date|date of death|deathplace|death date|nationality|"
    r"height|profession|occupation|age|net worth|salary|capital|"
    r"population|area|religion|spouse|alma mater)\b",
    # "Where was/is X born/located/headquartered"
    r"\bwhere (?:was|is) [\w\s'\-\.]+ (?:born|die|died|located|"
    r"headquartered|based|situated|founded|established)\b",
    # "Which country/city/state/year ... X born/founded/etc."
    r"\bwhich (?:country|city|state|town|year|month|day|continent|"
    r"region|county) (?:is|was|does|did) [\w\s'\-\.]+ (?:born|located|"
    r"based|founded|established|headquartered|situated|set)\b",
    # "How old/tall/much is X"
    r"\bhow (?:old|tall|much|many) (?:is|was|does) [\w\s'\-\.]+\b",
    # "What is the capital of X" (geographic attribute lookup)
    r"\bwhat (?:is|was) the capital of [\w\s'\-\.]+\b",
)

# Negative markers: clausal / temporal / compositional / comparison
# indicators. If ANY fires, the router treats the query as multi-hop
# and skips infobox even when a positive pattern also matches.
#
# v2: the 3 F1=0-derived patterns dropped per the data
# leakage audit. Remaining 6 patterns derive from the seed brief +
# general linguistic categories (NL question-answering literature on
# subordinate-clause / chain / comparative-question shapes). No
# abstention-pool inspection. See module docstring for full
# provenance disclosure.
MULTI_HOP_MARKERS: tuple[str, ...] = (
    # Subordinate-clause bridge entity ("the [entity] that X did Y")
    r"\bthat (?:is|was|has|had|did|does|founded|wrote|directed|"
    r"composed|invented|discovered|owns|owned)\b",
    # Chain marker: "which ... also/then/later/subsequently"
    r"\bwhich .*\b(?:also|then|later|subsequently|after|before)\b",
    # Explicit temporal-order comparison
    r"\b(?:before|after|earlier|later|prior to|following) (?:than|did|"
    r"the|when)\b",
    # Polar / set-comparison ("both A and B", "either X or Y", ...)
    r"\b(?:both|either|neither) \w+ (?:and|or|nor)\b",
    # Comparative attribute
    r"\b(?:older|younger|larger|smaller|taller|shorter|earlier|later|"
    r"longer|wider|deeper|richer|poorer|faster|slower|more|less)"
    r" than\b",
    # Polar-comparison filler "or" with entities ("A or B?")
    r"\b(?:\w+ ){1,5}or (?:\w+ ){0,3}\?",
)


# Pre-compiled for hot-path performance (the router fires once per query).
_POSITIVE_RE = tuple(re.compile(p, re.IGNORECASE) for p in ENTITY_ATTRIBUTE_PATTERNS)
_NEGATIVE_RE = tuple(re.compile(p, re.IGNORECASE) for p in MULTI_HOP_MARKERS)


def is_entity_attribute_query(
    question: str,
    *,
    extra_positive_patterns: Iterable[str] | None = None,
    extra_negative_patterns: Iterable[str] | None = None,
) -> bool:
    """Deterministic router: True iff the question is single-clause
    entity-attribute (= safe to fire infobox augmentation).

    Decision algorithm:

      1. If any multi-hop marker matches  -> return False (skip infobox)
      2. Elif any entity-attribute pattern matches  -> return True (fire)
      3. Else  -> return False (default-off, conservative)

    Parameters
    ----------
    question
        The user's question text.
    extra_positive_patterns
        Optional caller-supplied additional regex strings to extend
        :data:`ENTITY_ATTRIBUTE_PATTERNS` for domain-specific tuning.
    extra_negative_patterns
        Optional caller-supplied additional regex strings to extend
        :data:`MULTI_HOP_MARKERS`.

    Returns
    -------
    bool
        True iff the router classifies the query as safe to fire
        infobox augmentation.
    """
    if not question or not question.strip():
        return False

    q = question.strip()

    # Step 1: negative markers (high-precision negative gate).
    if any(p.search(q) for p in _NEGATIVE_RE):
        return False
    if extra_negative_patterns:
        for p in extra_negative_patterns:
            if re.search(p, q, re.IGNORECASE):
                return False

    # Step 2: positive patterns (high-precision positive gate).
    if any(p.search(q) for p in _POSITIVE_RE):
        return True
    if extra_positive_patterns:
        for p in extra_positive_patterns:
            if re.search(p, q, re.IGNORECASE):
                return True

    # Step 3: default-off (conservative).
    return False


__all__ = [
    "is_entity_attribute_query",
    "ENTITY_ATTRIBUTE_PATTERNS",
    "MULTI_HOP_MARKERS",
]
