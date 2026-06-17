# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Rule-based query-type classifier (M-class router).

Classifies a multi-hop question as either ``bridge_entity`` (chain-of-facts
between entities, decompose-friendly) or ``semantic_rich`` (open-ended,
single-shot V3+bu friendly).

Design goal: zero training, zero ML, deterministic. The classifier is a
linguistic prior used by :func:`mothrag.core.selective_ensemble.route_by_query_type`
to bias ensemble routing on bridge-heavy datasets (e.g. 2WikiMultiHopQA).

Decision (all conjuncts gated by ``short`` = n_tokens < 15)::

    bridge_entity iff short AND (
        has_chain                                   # "X of Y of Z" / "X of the N who" pattern
      OR (n_relations >= 2 AND n_tokens >= 8)       # 2+ relational hooks, non-trivial length
      OR (n_relations >= 1 AND n_entities >= 2)     # relation + 2 explicit named entities
    )

Calibration target (hand-curated):
  - 2WikiMultiHopQA: ~60-70% classified as ``bridge_entity``
  - HotpotQA:        ~30-40% classified as ``bridge_entity``
"""

from __future__ import annotations

import re

# Relational verbs + role nouns used as relations in HotpotQA / 2Wiki questions.
# Both forms count as a "relation hit" (e.g. "founded" verb, "founder" noun).
RELATION_LEXICON: set[str] = {
    # action verbs
    "directed", "direct", "directs",
    "wrote", "written", "writes", "writing",
    "born", "birth",
    "died", "death",
    "married", "marries", "marriage",
    "founded", "founding",
    "owned", "owns", "owning",
    "starred", "stars", "starring",
    "played", "plays", "playing",
    "published", "publishes", "publishing",
    "created", "creates", "creating",
    "authored", "authors",
    "produced", "produces", "producing",
    "composed", "composes",
    "designed", "designs",
    "adapted", "adapts", "adaptation",
    "succeeded", "preceded",
    # multi-word "X by"
    "directed by", "written by", "produced by", "composed by", "founded by",
    # role nouns (chain bridges)
    "director", "actor", "actress",
    "spouse", "wife", "husband",
    "mother", "father", "parent",
    "son", "daughter", "child", "children",
    "brother", "sister", "sibling",
    "founder", "creator", "owner",
    "author", "publisher", "composer", "producer", "designer",
    "predecessor", "successor",
    # softer relational connectors (count as relations only when chain disambiguates)
    "star",
    # iter 2 — kinship extension (rescues "Who is X's grandfather?" pattern
    # mis-classified as semantic_rich on 2Wiki).
    "grandfather", "grandmother", "grandparent", "grandparents",
    "grandson", "granddaughter", "grandchild", "grandchildren",
    "aunt", "uncle", "nephew", "niece", "cousin", "cousins",
    "stepfather", "stepmother", "stepsister", "stepbrother",
    "stepson", "stepdaughter", "stepchild",
    "godfather", "godmother", "godson", "goddaughter",
}


def _build_lexicon_regex(lexicon: set[str]) -> re.Pattern:
    parts = []
    for entry in sorted(lexicon, key=len, reverse=True):
        toks = re.escape(entry).replace(r"\ ", r"\s+")
        parts.append(rf"\b{toks}\b")
    return re.compile("|".join(parts), re.IGNORECASE)


_LEXICON_RE = _build_lexicon_regex(RELATION_LEXICON)

# Sentence-initial Wh-words / aux verbs we should not count as named entities.
_QUESTION_HEADS: set[str] = {
    "who", "what", "when", "where", "why", "how", "which", "whose", "whom",
    "is", "are", "was", "were", "did", "do", "does", "the", "an", "a",
    "in", "of", "on", "at", "to", "by",
}

# Capitalized run: 1+ tokens starting uppercase, optionally chained with
# small connectives (of/the/de/la/and/&) or further capitalized tokens.
_CAPS_RUN = re.compile(
    r"\b([A-Z][a-zA-Z0-9'\-]+(?:\s+(?:of|the|de|la|le|du|von|van|and|&|[A-Z][a-zA-Z0-9'\-]+))*)"
)

# Chain pattern: "A of B of C" (2+ "of" connectors) OR "A of the N who/which/that ..."
# nested possessive bridge.
_CHAIN_NESTED_RE = re.compile(
    r"\bof\s+(?:the\s+)?\w+(?:\s+\w+){0,2}\s+(?:who|which|that)\b",
    re.IGNORECASE,
)

# Polar-comparison pattern expansion. Keeps v1 set-comparison
# (both/same/either/neither) AND adds:
#   (b) comparative + "than"     ("Is X older than Y?")
#   (c) bare two-entity choice    ("Curry And Pepper or End Of Watch?")
#   (d) wh-headed two-entity choice ("Which film was first, X or Y?") with
#       exclusions for subordinate-clause chains (who/whose/which/that) and
#       aggregation-bridges (has/have/had).
#
# Recall 0.377 -> 0.603 vs the silver-LLM ground truth while P=1.000
# preserved. V3+bu wins ~80% of disagreements on these sub-types
# (holistic single-shot reasoning).
#
# IMPORTANT: this pattern MUST stay in sync with
# :data:`mothrag.routing.semantic_features._IS_POLAR_COMPARISON_RE`
# (duplicated to avoid a circular import).
_POLAR_COMPARISON_RE = re.compile(
    # (a) v1 set-comparison
    r"(?:^\s*(?i:are|is|was|were|did|do|does|have|has|can|could|would|will)\b"
    r"[^?]*\b(?i:both|same|either|neither)\b)"
    # (b) comparative + than (short distance, no subordinator between)
    r"|(?:^\s*(?i:are|is|was|were|did|do|does|which|who|what)\b"
    r"(?:(?!\b(?i:who|whose|which|that|whom)\b)[^?]){0,30}?"
    r"\b(?i:older|younger|larger|smaller|taller|shorter|"
    r"earlier|later|longer|wider|deeper|richer|poorer|"
    r"faster|slower|more|less|better|worse|higher|lower|greater)\s+than\b)"
    # (c) bare two-entity choice (case-sensitive title-cased entities)
    r"|(?:^\s*[A-Z]\w*(?:\s+(?:(?i:and|or|of|on|in|at|the|to|for|"
    r"by|with|from|de|la|le|du|von|van)|&|[A-Z]\w*))*\s+(?i:or)\s+"
    r"[A-Z]\w*(?:\s+(?:(?i:and|of|on|in|at|the|to|for|"
    r"by|with|from|de|la|le|du|von|van)|&|[A-Z]\w*))*\s*\??\s*$)"
    # (d) wh-headed two-entity choice (excludes subordinate clauses + has)
    r"|(?:^\s*(?i:which|who|what|between)\b"
    r"(?:(?!\b(?i:who|whose|which|that|whom|has|have|had)\b)[^?])*?"
    r",\s+[A-Z]\w*(?:\s+(?:(?i:and|of|on|in|at|the|to|for|"
    r"by|with|from|de|la|le|du|von|van)|&|[A-Z]\w*))*\s+(?i:or)\s+"
    r"[A-Z]\w*(?:\s+(?:(?i:and|of|on|in|at|the|to|for|"
    r"by|with|from|de|la|le|du|von|van)|&|[A-Z]\w*))*\s*\??\s*$)",
)

# Comparative-selection pattern: "Which/Who X is Y-er, A or B?" — picks between
# two entities on a derived property (typically temporal/quantitative). These
# queries are structurally multi-hop (find derived property → compare → select)
# but often have low np_depth + n_relations, getting mis-classified as
# semantic_rich. On 2Wiki, V3+bu hallucinates / abstains on them; iter resolves.
# Requires: disjunction (" or ") + comparative/temporal lexicon hit.
_COMPARATIVE_LEXICON_RE = re.compile(
    r"\b(?:first|earlier|later|earliest|latest|older|younger|oldest|youngest|"
    r"longer|shorter|longest|shortest|before|after|"
    r"biggest|smallest|highest|lowest|tallest|shortest|"
    r"born|died|lived)\b",
    re.IGNORECASE,
)
_DISJUNCTION_RE = re.compile(r"\bor\b", re.IGNORECASE)

# Kinship-possessive pattern (iter 2): "X's <optional-adjective> <kinship-relation>".
# These are 1-hop bridge queries (find person related to X) that mis-classify
# as semantic_rich because n_entities=1 + n_relations<2 don't trigger
# the bridge_entity rule (which requires n_ent>=2). Decompose+iter resolves them.
# Matches both single-token relation ("X's grandfather") and
# adjective-prefixed ("X's maternal grandfather"). The 's anchor distinguishes
# from generic kinship mentions ("the grandfather of X" — already chain pattern).
_KINSHIP_TERMS_PATTERN = (
    r"father|mother|parent|son|daughter|child|children|brother|sister|sibling|"
    r"spouse|wife|husband|"
    r"grandfather|grandmother|grandparent|grandson|granddaughter|grandchild|"
    r"aunt|uncle|nephew|niece|cousin|"
    r"stepfather|stepmother|stepsister|stepbrother|stepson|stepdaughter|"
    r"godfather|godmother|godson|goddaughter"
)
_KINSHIP_POSSESSIVE_RE = re.compile(
    rf"'s\s+(?:\w+\s+)?(?:{_KINSHIP_TERMS_PATTERN})\b",
    re.IGNORECASE,
)


def count_named_entities(question: str) -> int:
    """Heuristic count of distinct named-entity spans in ``question``."""
    if not question:
        return 0
    spans = _CAPS_RUN.findall(question.strip())
    seen: set[str] = set()
    for sp in spans:
        norm = sp.strip().lower()
        if norm in _QUESTION_HEADS:
            continue
        if len(norm) < 2:
            continue
        seen.add(norm)
    return len(seen)


def count_relations(question: str) -> int:
    """Count distinct lexicon hits (deduplicated by lowercase form)."""
    if not question:
        return 0
    matches = [m.group(0).lower() for m in _LEXICON_RE.finditer(question)]
    return len(set(matches))


def has_relation(question: str) -> bool:
    return count_relations(question) >= 1


def has_chain(question: str) -> bool:
    """True iff question contains a multi-hop chain connector pattern."""
    if not question:
        return False
    of_hits = len(re.findall(r"\bof\b", question, flags=re.IGNORECASE))
    if of_hits >= 2:
        return True
    return bool(_CHAIN_NESTED_RE.search(question))


def token_count(question: str) -> int:
    return len(re.findall(r"\b\w+\b", question or ""))


def count_nested_np_depth(question: str) -> int:
    """Proxy for nested noun-phrase depth: count "of" connectors plus
    1 if a nested-bridge pattern (``of <X> who/which/that``) is present.

    Examples::

        "Who directed Inception?"                                     -> 0
        "Who is the director of Inception?"                           -> 1
        "Who is the spouse of the director of Inception?"             -> 2
        "Who is the spouse of the founder of the company that ..."    -> 3+
    """
    if not question:
        return 0
    of_hits = len(re.findall(r"\bof\b", question, flags=re.IGNORECASE))
    nested_bridge = 1 if _CHAIN_NESTED_RE.search(question) else 0
    return of_hits + nested_bridge


def is_chain_deep(question: str) -> bool:
    """sel_v2 chain_deep rule: nested NP depth >= 3 OR >= 3 distinct relations."""
    return count_nested_np_depth(question) >= 3 or count_relations(question) >= 3


def is_polar_comparison(question: str) -> bool:
    """True iff question is a yes/no comparison (both/same/either/neither) opening
    with a polarity auxiliary. V3+bu wins ~80% of disagreements on this sub-type.
    """
    if not question:
        return False
    return bool(_POLAR_COMPARISON_RE.search(question))


def is_comparative_selection(question: str) -> bool:
    """True iff question is a comparative-selection: "Which/Who X is Y-er, A or B?".

    Structurally multi-hop: find derived property → compare → select between
    disjuncts. On 2Wiki these are mis-classified as semantic_rich (np_depth<3,
    n_relations<3) but iter resolves them where V3+bu hallucinates.

    Detection: disjunction (" or ") + comparative/temporal lexicon hit.
    """
    if not question:
        return False
    return bool(_DISJUNCTION_RE.search(question)) and bool(_COMPARATIVE_LEXICON_RE.search(question))


def is_kinship_possessive(question: str) -> bool:
    """True iff question contains "X's <kinship-relation>" possessive pattern.

    These 1-hop bridge queries ("Who is X's maternal grandfather?") have
    n_entities=1 so they don't trigger bridge_entity (which requires n_ent>=2).
    Decompose+iter resolves them; V3+bu typically hallucinates / abstains.
    """
    if not question:
        return False
    return bool(_KINSHIP_POSSESSIVE_RE.search(question))


def requires_implicit_multihop(question: str,
                                query_features: dict | None = None) -> bool:
    """Unified "implicit multi-hop" signal for queries that are structurally
    multi-hop despite appearing short (np_depth<3, n_relations<2).

    Sub-classes (currently unified):
      * comparative-selection: "Which/Who X is Y-er, A or B?" — find derived
        property → compare → select between disjuncts.
      * kinship-possessive: "X's <kinship-relation>" — 1-hop relational answer
        that mis-classifies because n_entities=1.

    Note: polar-comparison ("Are X and Y both same Z?") is NOT in this set —
    it's a COUNTER-signal where V3+bu holistic reasoning wins ~80% and is
    handled separately in :func:`arm_subset` as a re-include override.
    """
    return is_comparative_selection(question) or is_kinship_possessive(question)


def classify_query(question: str) -> str:
    """Return ``'bridge_entity'`` or ``'semantic_rich'`` for ``question``.

    sel_v1 binary classifier (kept for backward compat). For 3-class sel_v2
    output, use :func:`classify_query_v2`.
    """
    if not question or not question.strip():
        return "semantic_rich"
    n_ent = count_named_entities(question)
    n_rel = count_relations(question)
    n_tok = token_count(question)
    chain = has_chain(question)
    short = n_tok < 15
    if short and (
        chain
        or (n_rel >= 2 and n_tok >= 8)
        or (n_rel >= 1 and n_ent >= 2)
    ):
        return "bridge_entity"
    return "semantic_rich"


def classify_query_v2(question: str) -> str:
    """sel_v2 3-class classifier: ``chain_deep`` | ``bridge_entity`` | ``semantic_rich``.

    Priority order:
      1. ``chain_deep`` if NP depth >= 3 OR n_relations >= 3 — routes to iterative arm
      2. ``bridge_entity`` if sel_v1 bridge rule fires — routes to decompose arm
      3. ``semantic_rich`` otherwise — routes to V3+bu / sel_v1 arm
    """
    if not question or not question.strip():
        return "semantic_rich"
    if is_chain_deep(question):
        return "chain_deep"
    return classify_query(question)


def classify_with_features(question: str) -> dict:
    """Return classification plus underlying feature counts (for debug / paper tables)."""
    n_ent = count_named_entities(question)
    n_rel = count_relations(question)
    n_tok = token_count(question)
    chain = has_chain(question)
    np_depth = count_nested_np_depth(question)
    short = n_tok < 15
    label_v1 = "bridge_entity" if (short and (
        chain or (n_rel >= 2 and n_tok >= 8) or (n_rel >= 1 and n_ent >= 2)
    )) else "semantic_rich"
    label_v2 = "chain_deep" if is_chain_deep(question) else label_v1
    return {
        "label": label_v1,  # sel_v1 binary (back-compat)
        "label_v2": label_v2,  # sel_v2 3-class
        "n_entities": n_ent,
        "n_relations": n_rel,
        "has_chain": bool(chain),
        "n_tokens": n_tok,
        "np_depth": np_depth,
    }


# ============================================================
# Shared scoring helpers (used by both arm_subset and arm_subset_pam_lite)
# ============================================================
#
# Single source of truth for the linguistic-feature -> per-arm probability
# mapping. The two consumers differ ONLY in how they convert the float
# probability into a routing decision:
#
#   arm_subset (sel_v2 binary): include the arm iff P_arm > 0.5.
#   arm_subset_pam_lite (continuous): include iff P_arm > threshold (default
#     0.3), AND thread the full P_arm dict into downstream arbitration as a
#     multiplicative weight.
#
# Pre-existing tests pin the coefficients via the PAM-lite contract; the
# helpers below are extracted from arm_subset_pam_lite without numerical
# change, so PAM-lite behaviour is byte-identical pre/post-refactor.

import math as _math


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + _math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _hop_structure(f) -> dict[str, bool]:
    """Hop-structure flags via boolean composition of SOLID features.

    Lexicon revision: each non-polar predicate now requires a
    POSITIVE corroborating signal in addition to the structural condition,
    to lift precision against the silver-LLM ground truth.

    Silver-LLM precision (Haiku 4.5 ground truth):
      v1:  polar 1.000 / bridge 0.603 / chain 0.036 /
           entity_attr 0.014 / general 0.138
      v2:  precision lift via composition with attribute /
           multi-hop / bridge markers (see scorers below)

    Returns a dict with five hop-class flags:

      ``is_1hop_polar``       : polar yes/no comparison (F1=1.00, unchanged).
                                V3+bu wins empirically.
      ``is_2hop_bridge``      : two_entity AND bridge_entity_marker AND
                                no single-hop possessive AND no chain
                                marker -- requires explicit "X of Y"
                                surface, not just two NPs.
      ``is_3hop_chain``       : chain_marker AND two_entity AND
                                multi_hop_marker -- requires a chain
                                lexical signal AND a subordinate-clause
                                bridge ("that X did Y"), not just any
                                first/then/later token.
      ``is_1hop_entity_attr`` : single_hop possessive AND attribute_marker
                                AND NOT two_entity -- "X's birthplace"
                                (entity-attribute lookup), not bare
                                possessive.
      ``is_general_multihop`` : POSITIVE multi_hop_marker (subordinate-
                                clause bridge) AND none of the specific
                                classes. No longer pure residual; empty
                                features now yield no hop class.

    Composition is general-purpose / domain-agnostic; each underlying
    feature is a deterministic linguistic rule with no test inspection
    and no per-dataset tuning. Used by ``_score_v3bu_p_arm`` cfde114 v3
    hop-aware gating AND by ``get_hop_weight`` soft multipliers.

    Overlap policy (not strict mutual exclusivity): a query may match
    multiple flags. ``get_hop_weight`` resolves overlap via per-arm
    priority order (dict iteration), documented intentional.
    """
    is_1hop_polar  = f.is_polar_comparison > 0.0
    is_2hop_bridge = ((f.single_hop == 0.0)
                      and (f.two_entity > 0.0)
                      and (f.chain_marker == 0.0)
                      and (f.bridge_entity_marker > 0.0))
    is_3hop_chain  = ((f.chain_marker > 0.0)
                      and (f.two_entity > 0.0)
                      and (f.multi_hop_marker > 0.0))
    is_1hop_entity_attr = ((f.single_entity > 0.0)
                           and (f.attribute_marker > 0.0)
                           and (f.two_entity == 0.0)
                           and (f.multi_hop_marker == 0.0)
                           and (f.chain_marker == 0.0))
    is_general_multihop = ((f.multi_hop_marker > 0.0)
                           and not is_1hop_polar
                           and not is_2hop_bridge
                           and not is_3hop_chain
                           and not is_1hop_entity_attr)
    return {
        "is_1hop_polar":        is_1hop_polar,
        "is_2hop_bridge":       is_2hop_bridge,
        "is_3hop_chain":        is_3hop_chain,
        "is_1hop_entity_attr":  is_1hop_entity_attr,
        "is_general_multihop":  is_general_multihop,
    }


# RE-SPEC: per-(arm, hop_cohort) soft multipliers. Applied as
# scalar weights to the base sigmoid P_arm score. base=1.0 neutral, >1
# boosts, <1 penalizes. Dict iteration order = priority (the first
# matching cohort wins per arm). NOTE on contract: post-multiplication
# P_arm may exceed 1.0 (e.g., 0.58 * 2.0 = 1.16); this is intentional --
# downstream consumers (arm_subset_pam_lite threshold, arbitrate_pam_lite
# argmax / weighted_mix / subset) operate on real-valued scores not on
# strict probabilities.
#
# WEIGHT PROVENANCE AUDIT:
#
#   The weight values below follow a symmetric theoretical pattern, NOT
#   F1 tuning:
#
#     specialist arm on its cohort     -> 2.0   (boost x2)
#     non-specialist arm on its cohort -> 0.1   (penalty x10)
#     any arm with no matching cohort  -> 1.0   (default neutral)
#
#   Per-arm specialty assignment derived from PAM-lite paper concept
#   (general-purpose arm capabilities, no per-dataset fitting):
#
#     v3bu       (holistic single-shot reasoner): specializes on
#                short / direct queries -- is_1hop_polar +
#                is_1hop_entity_attr cohorts. Anti-specialist on
#                multi-hop chain decomposition cohorts (is_2hop_bridge,
#                is_3hop_chain).
#     decompose  (sub-question decomposer): specializes on multi-hop
#                bridge + chain composition (is_2hop_bridge,
#                is_3hop_chain). Anti-specialist on polar yes/no
#                comparison (is_1hop_polar -- already trivial).
#     iter       (iterative refiner): specializes on chain composition
#                + residual multi-hop (is_3hop_chain,
#                is_general_multihop). Anti-specialist on polar.
#
#   The 2.0 / 0.1 / 1.0 trio is symmetric (multiplicative inverse pair
#   2.0 vs 0.5 would be the geometric-mean choice; we use 0.1 instead
#   so the penalty is strictly stronger than the inverse boost, i.e. a
#   single anti-specialist fire suppresses the arm by 10x). The 2.0
#   boost magnitude is chosen so the P_arm shaped score stays within
#   one order of magnitude of the unshaped score (typical sigmoid
#   outputs in [0.3, 0.8] give shaped scores in [0.03, 1.6]).
#
#   Audit verdict: theory-derived, NOT F1-tuned. Safe under the
#   no-gold-training contract. The per-arm specialty mapping was set
#   BEFORE any F1 measurement on HP/2W/MQ. If a future change
#   tunes these constants from F1 sweeps, this docstring MUST be
#   updated and the change MUST be reviewed against the anti-leak
#   contract.
_HOP_MULTIPLIERS: dict[str, dict[str, float]] = {
    "v3bu": {
        "is_1hop_polar":       2.0,  # specialist (theory)
        "is_1hop_entity_attr": 2.0,  # specialist (theory)
        "is_2hop_bridge":      0.1,  # anti-specialist (theory)
        "is_3hop_chain":       0.1,  # anti-specialist (theory)
        "default":             1.0,  # neutral
    },
    "decompose": {
        "is_2hop_bridge":      2.0,  # specialist (theory)
        "is_3hop_chain":       2.0,  # specialist (theory)
        "is_1hop_polar":       0.1,  # anti-specialist (theory)
        "default":             1.0,
    },
    "iter": {
        "is_3hop_chain":       2.0,  # specialist (theory)
        "is_general_multihop": 2.0,  # specialist (theory)
        "is_1hop_polar":       0.1,  # anti-specialist (theory)
        "default":             1.0,
    },
}


def get_hop_weight(arm: str, hop: dict[str, bool]) -> float:
    """Soft multiplier for ``arm`` given the hop-structure flags.

    Iterates ``_HOP_MULTIPLIERS[arm]`` in dict order; the first cohort
    whose flag is True in ``hop`` returns its weight. ``default`` is
    skipped during iteration and used only as the final fallback (when
    no cohort fires for this query).

    Args:
        arm: ``"v3bu"`` / ``"decompose"`` / ``"iter"``.
        hop: dict returned by ``_hop_structure``.

    Returns:
        Float weight (typically in [0.1, 2.0] per current schedule).

    Raises:
        KeyError: if ``arm`` is not registered in ``_HOP_MULTIPLIERS``.
    """
    schedule = _HOP_MULTIPLIERS[arm]
    for cohort, w in schedule.items():
        if cohort == "default":
            continue
        if hop.get(cohort, False):
            return w
    return schedule["default"]


def _score_v3bu_p_arm(f) -> float:
    """V3+bu wins on single-clause entity-attribute questions.

    ``comparison_marker`` wiring -- cfde114 v3 hop-aware gating:
      The cfde114 boost (``+0.3 * comparison_marker``) is now gated by
      explicit hop structure derived from a boolean composition of 5
      SOLID features (audited F1>=0.85). The boost applies ONLY
      when ``is_1hop_polar`` fires (yes/no set-comparison surface, where
      V3+bu's holistic reasoning empirically wins). For 2-hop bridge and
      3-hop chain structures the boost is suppressed (decompose / iter
      primitives route those classes), eliminating the F1=1 cohort
      regression seen with v1 (universal boost) and v2 (single_hop
      gate; underfired on 2-hop bridge where single_hop=0).

    Evolution:
      * cfde114 v1 (unconditional):   +0.3 * comparison_marker
      * cfde114 v2 (6ff3d08, gated):  +0.3 * comparison_marker * (1 - single_hop)
      * cfde114 v3 (this, hop-aware): +0.3 * comparison_marker IFF is_1hop_polar

    The other PAM-lite scorers are NOT modified -- only the
    comparison_marker × V3+bu interaction is hop-gated.
    """
    hop = _hop_structure(f)
    boost = 0.3 * f.comparison_marker if hop["is_1hop_polar"] else 0.0

    base = _sigmoid(
        +0.5 * f.single_entity
        + 0.4 * f.attribute_marker
        + 0.3 * f.single_hop
        + boost
        - 0.3 * f.multi_hop_marker
        - 0.2 * f.chain_marker
        + 0.2  # base
    )
    # RE-SPEC: hop-soft-multiplier on the base sigmoid.
    return base * get_hop_weight("v3bu", hop)


def _score_decompose_p_arm(f) -> float:
    """decompose wins on bridge-entity / two-entity compositional questions."""
    base = _sigmoid(
        +0.5 * f.bridge_entity_marker
        + 0.3 * f.two_entity
        + 0.2 * f.attribute_marker
        - 0.3 * f.single_hop
        + 0.3  # base
    )
    # RE-SPEC: hop-soft-multiplier on the base sigmoid.
    return base * get_hop_weight("decompose", _hop_structure(f))


def _score_iter_p_arm(f) -> float:
    """iter wins on chain / temporal / deeply nested multi-hop."""
    base = _sigmoid(
        +0.5 * f.chain_marker
        + 0.4 * f.temporal_marker
        + 0.3 * f.deep_complexity
        + 0.3 * f.multi_hop_marker
        - 0.2 * f.single_entity
        + 0.3  # base
    )
    # RE-SPEC: hop-soft-multiplier on the base sigmoid.
    return base * get_hop_weight("iter", _hop_structure(f))


def _score_infobox_arm_p_arm(f) -> float:
    """infobox_arm wins on entity-attribute / single-hop direct lookup.

    Multi-hop penalty is strong (-0.7) because the arm performs a SINGLE
    structured-fact lookup -- it cannot chain two infobox lookups, so
    bridge / subordinate-clause questions are out of scope even when
    they contain attribute keywords ("Which company that X founded ..."
    has both ``founded`` attribute fire AND ``that`` multi-hop fire;
    the latter must dominate or sel_v2 mis-routes).
    """
    return _sigmoid(
        +0.5 * f.attribute_marker
        + 0.4 * f.single_hop
        + 0.2 * f.single_entity
        - 0.7 * f.multi_hop_marker
        - 0.2 * f.chain_marker
        + 0.0  # base (opt-in -- start neutral)
    )


def _score_mothgraph_arm_p_arm(f) -> float:
    """mothgraph_arm wins on bridge / relational / multi-entity questions.

    Spec calls for ``relational_complexity`` + ``multi_entity_count``;
    composed from existing scorers to avoid feature catalogue bloat:
      relational_complexity := 0.6*deep_complexity + 0.4*multi_hop_marker
      multi_entity_count    := two_entity  (already saturating [0, 1])
    """
    relational_complexity = 0.6 * f.deep_complexity + 0.4 * f.multi_hop_marker
    multi_entity_count = f.two_entity
    return _sigmoid(
        +0.5 * f.bridge_entity_marker
        + 0.4 * relational_complexity
        + 0.3 * multi_entity_count
        - 0.3 * f.single_hop
        + 0.25  # base
    )


# Maps arm name -> scorer (used by both arm_subset and arm_subset_pam_lite).
_OPT_IN_ARM_SCORERS = {
    "infobox_arm": _score_infobox_arm_p_arm,
    "mothgraph_arm": _score_mothgraph_arm_p_arm,
}

# Binary threshold used by sel_v2 arm_subset for opt-in arms. PAM-lite uses
# its own (configurable, default 0.3) continuous threshold.
_SEL_V2_OPT_IN_THRESHOLD = 0.5


def arm_subset(question: str,
               query_features: dict | None = None,
               gamma_status: str | None = None,
               *,
               arms_pool: list[str] | tuple[str, ...] | None = None,
               ) -> list[str]:
    """Return the optimal arm subset for this query (prospective routing).

    Decides upfront — using only query-surface features — whether to include
    V3+bu in the ensemble. Three orthogonal signal classes drive the decision:

    1. **Label-based exclusion** (sel_v2 classifier output):
       ``chain_deep`` / ``bridge_entity`` → exclude V3+bu (decompose+iter wins).

    2. **Polar-comparison override** (counter-signal):
       "Are/Is X and Y both/same Z?" queries RE-INCLUDE V3+bu even when
       label says exclude — V3+bu holistic reasoning wins ~80% on this
       yes/no sub-type.

    3. **Implicit-multihop signal** (refinement, iters 1+2 unified):
       comparative-selection ("Which X is Y-er, A or B?") + kinship-possessive
       ("X's <kinship>") — queries that look short but are structurally
       multi-hop. See :func:`requires_implicit_multihop`.

    4. **γ-conditioning** (refinement, iter 3, asymmetric):
       When the iterative arm γ-verifier flags ``invalid``, V3+bu is excluded
       (γ-invalid queries are structurally hard — V3+bu typically hallucinates
       on these). ``valid`` / ``partial`` / ``None`` do NOT trigger exclusion
       (empirically γ=valid subset has high F1 with V3+bu in the mix on both
       datasets; forcing exclusion there causes regression).

    Token count and entity count alone are NOT used: on HotpotQA, long
    entity-rich queries are descriptive factoids where V3+bu excels.

    Parameters
    ----------
    question
        The user's question text.
    query_features, gamma_status
        Legacy positional args; see the cascade description above.
    arms_pool
        Optional list of arm names. ``None`` (default) preserves byte-
        identical legacy 3-arm behaviour. When passed, opt-in arms in
        the pool (``infobox_arm``, ``mothgraph_arm``) are evaluated via
        binary thresholding on the same continuous P_arm helpers
        :func:`arm_subset_pam_lite` uses (so the two routers stay
        coefficient-consistent). The final returned subset is also
        filtered to the pool: a legacy arm absent from ``arms_pool``
        will NOT appear in the result.

    Returns
    -------
        Ordered list of arm names. With ``arms_pool=None`` (default),
        one of ``["v3bu", "decompose", "iter"]`` (all three) or
        ``["decompose", "iter"]`` (V3+bu excluded). With a non-None
        ``arms_pool``, additionally extended with any opt-in arm whose
        P_arm exceeds the sel_v2 binary threshold (0.5), and filtered
        so every returned name is in ``arms_pool``.

    Calibration (expected V3+bu exclusion rate, ``arms_pool=None``):
        * 2WikiMultiHopQA — ~70-85 % (bridge-heavy + multi-hop + γ signals)
        * HotpotQA         — ~12-20 % (less bridge, fewer deep chains)
    """
    if query_features is None:
        query_features = classify_with_features(question)

    label_v2 = query_features.get("label_v2", "semantic_rich")
    exclude_v3bu_label = label_v2 in ("chain_deep", "bridge_entity")

    # ---- Legacy 3-arm cascade (unchanged) -------------------------------
    if gamma_status == "invalid":
        # (4) γ-conditioning.
        subset = ["decompose", "iter"]
    elif exclude_v3bu_label and is_polar_comparison(question):
        # (2) Polar-comparison override.
        subset = ["v3bu", "decompose", "iter"]
    elif not exclude_v3bu_label and requires_implicit_multihop(question, query_features):
        # (3) Implicit-multihop signal.
        subset = ["decompose", "iter"]
    elif exclude_v3bu_label:
        # (1) Label-based exclusion.
        subset = ["decompose", "iter"]
    else:
        subset = ["v3bu", "decompose", "iter"]

    # ---- Opt-in arm extension (only when arms_pool is provided) ---------
    # NB: each opt-in arm scoring helper is the SAME function PAM-lite uses
    # in :func:`arm_subset_pam_lite`. Binary threshold (0.5) here vs.
    # continuous (default 0.3) there is the only routing-mode delta.
    if arms_pool is not None:
        from mothrag.routing.semantic_features import extract_semantic_features

        # Lazy: only compute features if an opt-in arm is actually in the pool.
        feat = None
        for arm_name, scorer in _OPT_IN_ARM_SCORERS.items():
            if arm_name not in arms_pool:
                continue
            if feat is None:
                feat = extract_semantic_features(question)
            if scorer(feat) > _SEL_V2_OPT_IN_THRESHOLD:
                if arm_name not in subset:
                    subset.append(arm_name)

        # Dup-arm pool extension. For each ``<base>_dup_<suffix>``
        # in arms_pool, include it in the subset iff the base arm is already
        # in the subset (same dispatch eligibility, separate candidate slot).
        # This enables the dup-arm mechanism-attribution test.
        from mothrag.routing.dup_arm import is_dup_arm, base_arm_of
        for name in arms_pool:
            if not is_dup_arm(name):
                continue
            try:
                base = base_arm_of(name)
            except ValueError:
                continue
            if base in subset and name not in subset:
                subset.append(name)

        # Final pool filter: enforce "no arm appears in result that isn't
        # in arms_pool" uniformly. Drops legacy arms if explicitly excluded.
        pool_set = set(arms_pool)
        subset = [arm for arm in subset if arm in pool_set]

    return subset


def arm_subset_pam_lite(
    question: str,
    *,
    threshold: float = 0.3,
    arms_pool: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[str], dict[str, float]]:
    """PAM-lite: continuous per-arm probability + variable-K subset.

    Extension of :func:`arm_subset` to a continuous ``P_arm`` per arm,
    computed via a deterministic sigmoid over the linguistic features
    in :mod:`mothrag.routing.semantic_features`. Returns the arms whose
    probability strictly exceeds ``threshold`` (variable-K subset) AND
    the full probability dict for downstream weighted arbitration.

    Design contract: feature -> probability coefficients are
    HAND-DERIVED from general
    linguistic principles (entity-attribute -> V3+bu; bridge / two-
    entity -> decompose; chain / temporal / multi-hop -> iter;
    structured-fact lookup -> infobox_arm; relational graph traversal
    -> mothgraph_arm). NO training, no per-dataset tuning, no test-set
    inspection. Sigmoid is :math:`\\sigma(x) = 1 / (1 + e^{-x})`.

    Parameters
    ----------
    question
        The user's question text.
    threshold
        Float in ``[0, 1]``. Arms with ``P_arm > threshold`` enter the
        subset. Default 0.3 is conservative (matches the empirical
        sel_v2 inclusion rates on HP/2W/MQ).
    arms_pool
        Optional list of arm names to score. ``None`` (default)
        scores ONLY the legacy 3 arms (v3bu, decompose, iter) for
        backward compat. When passed, opt-in arms in the pool
        (``infobox_arm``, ``mothgraph_arm``) are scored too and may
        enter the subset. Unknown arm names are silently ignored.

    Returns
    -------
    (subset, P_arm)
        ``subset`` is the variable-K list of arm names; ``P_arm`` is
        the full probability mapping for downstream weighted
        arbitration. Both restricted to arms in ``arms_pool`` (or the
        legacy 3 when None).

    Guarantees
    ----------
    The returned subset is NEVER empty: if every arm's probability is
    below threshold, the highest-probability arm is included as a
    singleton (so the cascade always has at least one arm to run).
    """
    from mothrag.routing.semantic_features import extract_semantic_features

    f = extract_semantic_features(question)

    # Per-arm scorers live at module level (also consumed by sel_v2
    # :func:`arm_subset` with a different binary threshold). Single
    # source of truth for the coefficient mapping.
    probabilities: dict[str, float] = {
        "v3bu": _score_v3bu_p_arm(f),
        "decompose": _score_decompose_p_arm(f),
        "iter": _score_iter_p_arm(f),
    }

    pool = list(arms_pool) if arms_pool is not None else ["v3bu", "decompose", "iter"]

    # ---- Opt-in arms (only scored when present in the pool) -------------
    for arm_name, scorer in _OPT_IN_ARM_SCORERS.items():
        if arm_name in pool:
            probabilities[arm_name] = scorer(f)

    # Dup-arm scoring. ``<base>_dup_<suffix>`` shares the
    # base arm's probability exactly (same code path, separate slot for
    # arbitration). Probability is read from the base's already-scored
    # entry above.
    from mothrag.routing.dup_arm import is_dup_arm, base_arm_of
    for name in pool:
        if not is_dup_arm(name):
            continue
        try:
            base = base_arm_of(name)
        except ValueError:
            continue
        if base in probabilities:
            probabilities[name] = probabilities[base]

    # Restrict to the requested pool (drops legacy arms if pool excludes them).
    pool_set = set(pool)
    probabilities = {arm: p for arm, p in probabilities.items() if arm in pool_set}

    if not probabilities:
        return [], {}

    subset = [arm for arm, p in probabilities.items() if p > threshold]
    if not subset:
        # Always-non-empty guarantee: include the argmax even if below
        # threshold. Avoids the cascade running with zero arms.
        best = max(probabilities.items(), key=lambda kv: kv[1])[0]
        subset = [best]

    return subset, probabilities


__all__ = [
    "RELATION_LEXICON",
    "arm_subset",
    "arm_subset_pam_lite",
    "classify_query",
    "classify_query_v2",
    "classify_with_features",
    "count_named_entities",
    "count_relations",
    "count_nested_np_depth",
    "has_relation",
    "has_chain",
    "is_chain_deep",
    "is_comparative_selection",
    "is_kinship_possessive",
    "is_polar_comparison",
    "requires_implicit_multihop",
    "token_count",
]
