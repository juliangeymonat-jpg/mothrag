# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
r"""Deterministic linguistic-feature extractor for PAM-lite routing.

PAM-lite ("Probabilistic Arm Mixture, lite") extends the binary
:func:`mothrag.core.query_type_classifier.arm_subset` to a continuous
``P_arm`` probability per arm. The probabilities are computed via a
deterministic sigmoid over the per-arm semantic features defined here.

ALL features are pure linguistic rules over the question text. No
training, no per-dataset tuning, no test-set inspection. Each feature
returns a float in ``[0, 1]`` that subsequent linear combinations
project into ``P_arm`` via :func:`math.tanh` / sigmoid.

Feature catalogue:

- ``single_entity``       : single proper-noun-phrase signature
- ``attribute_marker``    : "when was X born", "capital of", "born",
                            "died", "founded", "spouse", "nationality",
                            "occupation", ... (entity-attribute lookup)
- ``multi_hop_marker``    : subordinate-clause bridge ("that X did Y",
                            "of the [thing] that")
- ``bridge_entity_marker``: "X of Y" pattern, two named entities
                            connected by an attribute
- ``chain_marker``        : explicit chain ("first ... then", "later",
                            "subsequently", "after that")
- ``temporal_marker``     : year tokens (19xx / 20xx), date words
                            (month names), temporal adverbs
- ``comparison_marker``   : "or", "vs", "either", "both", comparative
                            adjective + "than"
- ``deep_complexity``     : sentence length + nested-clause depth
                            (more nested -> higher score)
- ``two_entity``          : exactly two capitalized noun phrases

Feature outputs are deterministic + side-effect free; suitable for
unit testing in isolation without any LLM / corpus dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticFeatures:
    """Per-question semantic-feature vector.

    All fields are floats in ``[0, 1]``. Defaults are 0.0 (absence).
    Used by :func:`mothrag.core.query_type_classifier.arm_subset_pam_lite`
    to compute continuous per-arm probabilities.
    """

    single_entity: float = 0.0
    attribute_marker: float = 0.0
    multi_hop_marker: float = 0.0
    bridge_entity_marker: float = 0.0
    chain_marker: float = 0.0
    temporal_marker: float = 0.0
    comparison_marker: float = 0.0
    deep_complexity: float = 0.0
    two_entity: float = 0.0
    single_hop: float = 0.0
    is_polar_comparison: float = 0.0


# ============================================================
# Linguistic patterns (general-purpose, no test inspection)
# ============================================================

_ATTRIBUTE_MARKERS = (
    # birth / death
    r"\bborn\b", r"\bdied\b", r"\bdeath\b", r"\bbirth\b",
    r"\bbirthplace\b", r"\bbirthday\b",
    # founding / creation
    r"\bfounded\b", r"\bestablished\b", r"\bcreated\b",
    # marital
    r"\bspouse\b", r"\bwife\b", r"\bhusband\b", r"\bpartner\b",
    # biographic attributes
    r"\bnationality\b", r"\boccupation\b", r"\bprofession\b",
    r"\bage\b", r"\breligion\b", r"\balma mater\b",
    # geo / org
    r"\bcapital\b", r"\bpopulation\b", r"\barea\b", r"\bheight\b",
    r"\bheadquartered\b", r"\bbased\b", r"\blocated\b",
)
_ATTRIBUTE_RE = re.compile("|".join(_ATTRIBUTE_MARKERS), re.IGNORECASE)

_MULTI_HOP_BRIDGE_RE = re.compile(
    # "that" introducing a subordinate clause: action verb appears within
    # the next ~6 tokens (covers both "the king that started" and
    # "the company that Steve Jobs founded").
    r"\bthat\b(?=(?:\s+\w+){0,6}\s+(?:is|was|has|had|did|does|founded|"
    r"wrote|directed|composed|invented|discovered|owns|owned|acquired|"
    r"started|published|released|established|created|signed|elected|"
    r"played|wrote|hosted|formed|developed))",
    re.IGNORECASE,
)

_CHAIN_RE = re.compile(
    r"\b(?:first|then|later|subsequently|after that|"
    r"prior to|following|next)\b",
    re.IGNORECASE,
)

_TEMPORAL_RE = re.compile(
    r"\b(?:19|20)\d{2}\b"
    r"|\b(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b"
    r"|\b(?:yesterday|today|tomorrow|recently|currently|"
    r"previously|formerly|earlier|later)\b",
    re.IGNORECASE,
)

_COMPARISON_RE = re.compile(
    r"\b(?:both|either|neither|vs|versus|or)\b"
    r"|\b(?:older|younger|larger|smaller|taller|shorter|"
    r"earlier|later|longer|wider|deeper|richer|poorer|"
    r"faster|slower|more|less|better|worse) than\b",
    re.IGNORECASE,
)

# Capitalized noun phrase (one or more consecutive Capitalized words).
# Matches "Albert Einstein", "Apple", "United States of America", etc.
# Filters out sentence-initial single capitalized words (would over-fire
# on every question's first word) by requiring either multi-word OR
# non-initial.
_CAP_NP_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

# Single-hop markers ("X's Y" possessive)
_SINGLE_HOP_RE = re.compile(r"\b\w+'s\s+\w+\b")

# Polar-comparison lexicon expansion. v1 captured yes/no
# auxiliary + (both|same|either|neither) -- recall 0.377 vs silver labels.
# v2 adds:
#   (b) comparative + "than" (e.g. "Is X older than Y?")
#   (c) superlative head (e.g. "Which was released first, X or Y?")
#   (d) choice-between-two (X or Y?) gated by Wh-head + title-cased pair
#
# Precision target preserved at 1.000 against silver-LLM ground truth;
# each new alternation was validated empirically before landing.
#
# Mirrors :func:`mothrag.core.query_type_classifier.is_polar_comparison` but
# duplicated here to avoid a circular import (query_type_classifier already
# imports extract_semantic_features). Keep the two patterns in sync if you
# touch either.
_IS_POLAR_COMPARISON_RE = re.compile(
    # (a) v1 set-comparison: aux + both/same/either/neither
    r"(?:^\s*(?i:are|is|was|were|did|do|does|have|has|can|could|would|will)\b"
    r"[^?]*\b(?i:both|same|either|neither)\b)"
    # (b) v2 comparative: aux/wh + comparative-adj + than (TIGHT distance
    # to avoid chained "Which film has director who is older than..." FPs).
    r"|(?:^\s*(?i:are|is|was|were|did|do|does|which|who|what)\b"
    r"(?:(?!\b(?i:who|whose|which|that|whom)\b)[^?]){0,30}?"
    r"\b(?i:older|younger|larger|smaller|taller|shorter|"
    r"earlier|later|longer|wider|deeper|richer|poorer|"
    r"faster|slower|more|less|better|worse|higher|lower|greater)\s+than\b)"
    # (c) v2 bare two-entity choice ending with "X or Y?". Requires
    # title-cased entity phrases on each side of "or" (case-sensitive
    # entity detection). Excludes questions starting with Wh-aux to
    # avoid chained comparisons; the bare "X or Y?" form is POLAR per
    # silver label. Connectors (and/of/the/on/in/at/...) accepted in
    # either case via (?i:); single-letter titles ("A Crooked Ship")
    # supported via [A-Z]\w*.
    r"|(?:^\s*[A-Z]\w*(?:\s+(?:(?i:and|or|of|on|in|at|the|to|for|"
    r"by|with|from|de|la|le|du|von|van)|&|[A-Z]\w*))*\s+(?i:or)\s+"
    r"[A-Z]\w*(?:\s+(?:(?i:and|of|on|in|at|the|to|for|"
    r"by|with|from|de|la|le|du|von|van)|&|[A-Z]\w*))*\s*\??\s*$)"
    # (d) v2 wh-headed two-entity choice: "Which/Who/Between [...]
    # X or Y?" with NO subordinator (who/whose/which/that) and NO
    # "has/have/had" indirection between head and choice — these
    # patterns mark chained comparison via a bridge noun or aggregation
    # (silver GENERAL_MULTIHOP / HOP2_BRIDGE, not POLAR).
    r"|(?:^\s*(?i:which|who|what|between)\b"
    r"(?:(?!\b(?i:who|whose|which|that|whom|has|have|had)\b)[^?])*?"
    r",\s+[A-Z]\w*(?:\s+(?:(?i:and|of|on|in|at|the|to|for|"
    r"by|with|from|de|la|le|du|von|van)|&|[A-Z]\w*))*\s+(?i:or)\s+"
    r"[A-Z]\w*(?:\s+(?:(?i:and|of|on|in|at|the|to|for|"
    r"by|with|from|de|la|le|du|von|van)|&|[A-Z]\w*))*\s*\??\s*$)",
    # No global IGNORECASE — case-sensitive matching for (c) entity caps.
    # Per-alternation (?i:) used in (a) and (b) for keyword case-insensitivity.
)

# Bridge entity "X of Y" or "the X of Y" (two-entity connector)
_BRIDGE_OF_RE = re.compile(
    r"\b(?:of the|of [A-Z])\b",
    re.IGNORECASE,
)


# ============================================================
# Per-feature scorers (each returns float in [0, 1])
# ============================================================

def _clamp(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def score_attribute_marker(question: str) -> float:
    """Higher when the question asks about a named attribute of an entity."""
    if not question:
        return 0.0
    matches = _ATTRIBUTE_RE.findall(question)
    return _clamp(0.4 + 0.3 * min(len(matches), 2)) if matches else 0.0


def score_multi_hop(question: str) -> float:
    """Higher on subordinate-clause bridges ("that X did Y")."""
    if not question:
        return 0.0
    if _MULTI_HOP_BRIDGE_RE.search(question):
        return 0.9
    if " of the " in question.lower() or " of which " in question.lower():
        return 0.6
    return 0.0


def score_chain(question: str) -> float:
    """Higher on explicit chain markers."""
    if not question:
        return 0.0
    matches = _CHAIN_RE.findall(question)
    return _clamp(0.5 + 0.2 * min(len(matches), 3)) if matches else 0.0


def score_temporal(question: str) -> float:
    """Higher when temporal tokens are present (years, months, adverbs)."""
    if not question:
        return 0.0
    matches = _TEMPORAL_RE.findall(question)
    return _clamp(0.4 + 0.2 * min(len(matches), 3)) if matches else 0.0


def score_comparison(question: str) -> float:
    """Higher on polar / set-comparison / comparative-attribute forms."""
    if not question:
        return 0.0
    return 0.7 if _COMPARISON_RE.search(question) else 0.0


def score_single_entity(question: str) -> float:
    """Higher when exactly one capitalized noun phrase appears (excluding
    sentence-initial single word)."""
    if not question:
        return 0.0
    cap_nps = [m.group(0) for m in _CAP_NP_RE.finditer(question)]
    # Filter out sentence-initial single-word capitalizations
    # ("What", "When", "Who", etc.).
    filtered = [
        np for np in cap_nps
        if " " in np or not question.lstrip().startswith(np)
    ]
    if len(filtered) == 1:
        return 0.8
    if len(filtered) == 0 and len(cap_nps) >= 1:
        return 0.4  # ambiguous (sentence-initial)
    return 0.0


def score_two_entities(question: str) -> float:
    """Higher when exactly two distinct capitalized noun phrases appear."""
    if not question:
        return 0.0
    cap_nps = [m.group(0) for m in _CAP_NP_RE.finditer(question)]
    filtered = [
        np for np in cap_nps
        if " " in np or not question.lstrip().startswith(np)
    ]
    distinct = set(filtered)
    if len(distinct) == 2:
        return 0.8
    return 0.0


def score_bridge_entity(question: str) -> float:
    """Higher on "the X of Y" bridge-entity surface forms."""
    if not question:
        return 0.0
    if _BRIDGE_OF_RE.search(question):
        # If also a two-entity signature -> stronger bridge confidence.
        if score_two_entities(question) > 0:
            return 0.8
        return 0.5
    return 0.0


def score_complexity(question: str) -> float:
    """Higher on long / nested questions (proxy: word count + clause splits).

    Sentence-initial Wh-words (``Who``, ``Which``, ``What``, ``When``,
    ``Where``, ``Why``, ``How``) are NOT subordinators -- they are
    question heads. Counting them as subordinators caused
    ``deep_complexity`` to fire on every simple question (P=0.38 in the
    feature-audit n=50; over-fires globally). The check
    below filters out sentence-initial occurrences.
    """
    if not question:
        return 0.0
    n_tokens = len(question.split())
    n_commas = question.count(",")
    # Count subordinator occurrences EXCLUDING the sentence-initial position.
    q_stripped = question.lstrip()
    n_subord = 0
    for w in ("that", "which", "who", "whom", "whose"):
        matches = list(re.finditer(rf"\b{w}\b", question, re.IGNORECASE))
        for m in matches:
            # Skip the match iff it is the very first non-whitespace token.
            offset_in_stripped = m.start() - (len(question) - len(q_stripped))
            if offset_in_stripped == 0:
                continue
            n_subord += 1
    raw = (n_tokens / 25.0) + 0.2 * n_commas + 0.3 * n_subord
    # Floor: short / simple questions (raw < 0.4 = no subordinators + <10
    # tokens + no commas) should NOT register as "complex". This makes the
    # binary "score > 0" thresholding semantics consistent with the other
    # lexicon-based features (which return strictly 0 on absence).
    if raw < 0.4:
        return 0.0
    return _clamp(raw)


def score_single_hop(question: str) -> float:
    """Higher on single-hop possessive forms ("X's Y")."""
    if not question:
        return 0.0
    if _SINGLE_HOP_RE.search(question):
        # Single-hop possessive AND no multi-hop bridge -> stronger.
        if score_multi_hop(question) < 0.3:
            return 0.7
        return 0.4
    return 0.0


def score_is_polar_comparison(question: str) -> float:
    """Binary indicator: yes/no polar comparison (Phase 6 audit F1=1.00).

    Returns 1.0 if the question opens with a yes/no auxiliary AND contains a
    set-comparison marker (both/same/either/neither) before the question mark;
    0.0 otherwise. Used by ``_score_v3bu_p_arm`` cfde114 v3 hop-aware gating.
    """
    if not question:
        return 0.0
    return 1.0 if _IS_POLAR_COMPARISON_RE.search(question) else 0.0


# ============================================================
# Aggregator
# ============================================================

def extract_semantic_features(question: str) -> SemanticFeatures:
    """Compute the full :class:`SemanticFeatures` vector for ``question``.

    All scorers are deterministic linguistic rules; no LLM, no
    embedding, no test inspection. Output is suitable for direct
    consumption by :func:`arm_subset_pam_lite` via a sigmoid linear
    combination.
    """
    return SemanticFeatures(
        single_entity=score_single_entity(question),
        attribute_marker=score_attribute_marker(question),
        multi_hop_marker=score_multi_hop(question),
        bridge_entity_marker=score_bridge_entity(question),
        chain_marker=score_chain(question),
        temporal_marker=score_temporal(question),
        comparison_marker=score_comparison(question),
        deep_complexity=score_complexity(question),
        two_entity=score_two_entities(question),
        single_hop=score_single_hop(question),
        is_polar_comparison=score_is_polar_comparison(question),
    )


__all__ = [
    "SemanticFeatures",
    "extract_semantic_features",
    "score_single_entity",
    "score_attribute_marker",
    "score_multi_hop",
    "score_bridge_entity",
    "score_chain",
    "score_temporal",
    "score_comparison",
    "score_complexity",
    "score_two_entities",
    "score_single_hop",
    "score_is_polar_comparison",
]
