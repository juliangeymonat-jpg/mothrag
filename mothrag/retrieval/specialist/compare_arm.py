# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""M8 CompareArm — boolean-intersection retrieval specialist.

Target cluster ``comparison_yes_no`` — "Are the directors of both X and Y from the same
country?" — 233 2W queries (44% of the 2W gap) + 48 MQ. Generic dense retrieval
tends to over-cover ONE of the two compared entities; the boolean answer needs
BOTH entities' attribute docs.

CompareArm:
  1. detect the cluster (comparison marker + ≥2 entities + boolean structure),
  2. split out the two (or more) compared entities,
  3. retrieve each entity SEPARATELY (focused on the compared attribute),
  4. fold the per-entity results into a BALANCED union (round-robin) so both
     entities are guaranteed representation in the final top-K.

$0 LLM: rule-based detection/splitting + the injected dense ``ann_retrieve``
(the same seam as BridgeArm — backend-agnostic, fully offline-testable). The
reader composes the boolean answer downstream from the balanced doc set.

Anti-leak: question text + retrieved passages only; general-purpose, no
per-dataset branching.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

# ann_retrieve(query_text, top_k) -> sequence[Candidate]
AnnRetrieve = Callable[[str, int], Sequence["Candidate"]]


@dataclass(frozen=True)
class Candidate:
    passage_id: str
    text: str = ""
    score: float = 0.0


@dataclass
class CompareConfig:
    per_entity_top_k: int = 15      # dense top-K per compared entity
    final_top_k: int = 10           # balanced-union cap
    max_entities: int = 3           # comparisons are usually 2-way, allow 3


@dataclass
class CompareResult:
    question: str
    entities: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    ranked_passage_ids: list[str] = field(default_factory=list)
    per_entity_passage_ids: dict[str, list[str]] = field(default_factory=dict)
    fired: bool = False             # True iff the two-pass split actually ran


# ---- detection -------------------------------------------------------------

COMPARISON_MARKERS: tuple[str, ...] = (
    r"\bboth\b",
    r"\bsame\b",
    r"\bcompared?\s+(?:to|with)\b",
    r"\bdiffer(?:ent|ence)?\b",
    r"\beither\b",
    r"\bneither\b",
    r"\bmore\s+\w+\s+than\b",
    r"\bless\s+\w+\s+than\b",
    r"\b(?:older|younger|taller|shorter|larger|smaller|longer)\b",
)

_BOOLEAN_Q = re.compile(
    r"\b(are|is|was|were|do|does|did|has|have|had|can|could|will|would)\b.*\?",
    re.IGNORECASE,
)

_QUESTION_WORDS = {
    "are", "is", "was", "were", "do", "does", "did", "has", "have", "had",
    "can", "could", "will", "would", "who", "what", "which", "when", "where",
    "why", "how", "whose", "whom",
}


def has_comparison_marker(question: str) -> bool:
    return any(re.search(p, question or "", re.IGNORECASE) for p in COMPARISON_MARKERS)


def is_comparison_query(question: str, *,
                        ner: Optional[Callable[[str], Sequence[str]]] = None) -> bool:
    """True for the boolean-comparison cluster: marker + ≥2 entities + boolean."""
    if not question:
        return False
    if not has_comparison_marker(question):
        return False
    if not _BOOLEAN_Q.search(question):
        return False
    return len(extract_compared_entities(question, ner=ner)) >= 2


# ---- entity + attribute extraction (deterministic; injectable NER) ---------

_TITLECASE_RUN = re.compile(r"\b([A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*)*)")
_QUOTED = re.compile(r"[\"“]([^\"”]{2,})[\"”]")
_BOTH_AND = re.compile(r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[?,.]|$)", re.IGNORECASE)


def _dedupe_keep_order(items) -> list[str]:
    out, seen = [], set()
    for it in items:
        k = it.strip().lower()
        if it.strip() and k not in seen:
            seen.add(k)
            out.append(it.strip())
    return out


def _strip_leading_question_word(span: str) -> str:
    toks = span.split()
    while toks and toks[0].lower() in _QUESTION_WORDS:
        toks = toks[1:]
    return " ".join(toks)


def extract_compared_entities(
    question: str, *,
    ner: Optional[Callable[[str], Sequence[str]]] = None,
) -> list[str]:
    """Extract the compared entities. Injected NER wins; else deterministic
    title-case / quoted / ``between X and Y`` heuristics."""
    if not question:
        return []
    if ner is not None:
        try:
            ents = _dedupe_keep_order([str(e) for e in ner(question) if str(e).strip()])
            if len(ents) >= 2:
                return ents
        except Exception:  # noqa: BLE001 — fall back to the heuristic
            pass

    spans: list[str] = []
    spans.extend(m.group(1) for m in _QUOTED.finditer(question))
    m = _BOTH_AND.search(question)
    if m:
        spans.extend([m.group(1), m.group(2)])
    for run in _TITLECASE_RUN.findall(question):
        cleaned = _strip_leading_question_word(run)
        if cleaned:
            spans.append(cleaned)
    # Drop spans that are ONLY stop/question words and 1-char noise.
    cleaned = [s for s in spans if len(s) >= 2 and s.lower() not in _QUESTION_WORDS]
    return _dedupe_keep_order(cleaned)


_ATTR_OF = re.compile(r"\bthe\s+([a-z]+?)s?\s+of\b", re.IGNORECASE)
_ATTR_SAME = re.compile(r"\bsame\s+([a-z]+)\b", re.IGNORECASE)
_ATTR_COMPARATIVE = re.compile(
    r"\b(older|younger|taller|shorter|larger|smaller|longer|nationality|"
    r"country|director|author|birthplace|profession)\b", re.IGNORECASE)


def extract_comparison_attributes(question: str) -> list[str]:
    """The attribute(s) being compared (e.g. 'director', 'country') — used to
    focus each entity's retrieval. Best-effort; [] when none found."""
    if not question:
        return []
    attrs: list[str] = []
    attrs += [m.group(1).lower() for m in _ATTR_OF.finditer(question)]
    attrs += [m.group(1).lower() for m in _ATTR_SAME.finditer(question)]
    attrs += [m.group(1).lower() for m in _ATTR_COMPARATIVE.finditer(question)]
    return _dedupe_keep_order(attrs)


# ---- the arm ---------------------------------------------------------------

class CompareArm:
    """Boolean-intersection retrieval specialist. ``ann_retrieve`` is required."""

    name = "compare_arm"

    def __init__(
        self,
        ann_retrieve: AnnRetrieve,
        *,
        config: Optional[CompareConfig] = None,
        ner: Optional[Callable[[str], Sequence[str]]] = None,
    ) -> None:
        if ann_retrieve is None:
            raise ValueError("CompareArm requires an ann_retrieve callable")
        self.ann_retrieve = ann_retrieve
        self.cfg = config or CompareConfig()
        self.ner = ner

    def applicable(self, question: str) -> bool:
        return is_comparison_query(question, ner=self.ner)

    @staticmethod
    def _balanced_union(per_entity: dict[str, list[str]], cap: int) -> list[str]:
        """Round-robin merge so EVERY entity is represented before any entity's
        deeper docs — the boolean-intersection coverage guarantee."""
        ranked: list[str] = []
        seen: set[str] = set()
        cols = [list(v) for v in per_entity.values()]
        depth = max((len(c) for c in cols), default=0)
        for d in range(depth):
            for col in cols:
                if d < len(col):
                    pid = col[d]
                    if pid not in seen:
                        seen.add(pid)
                        ranked.append(pid)
                        if len(ranked) >= cap:
                            return ranked
        return ranked

    def retrieve(self, question: str) -> CompareResult:
        entities = extract_compared_entities(question, ner=self.ner)[: self.cfg.max_entities]
        attributes = extract_comparison_attributes(question)
        result = CompareResult(question=question, entities=entities,
                               attributes=attributes)
        if len(entities) < 2:
            return result   # not splittable → caller falls back to generic

        per_entity: dict[str, list[str]] = {}
        attr_suffix = (" " + " ".join(attributes)) if attributes else ""
        for ent in entities:
            try:
                cands = list(self.ann_retrieve(f"{ent}{attr_suffix}",
                                               self.cfg.per_entity_top_k))
            except Exception:  # noqa: BLE001 — one entity's failure ≠ whole arm
                cands = []
            per_entity[ent] = [str(c.passage_id) for c in cands]

        result.per_entity_passage_ids = per_entity
        result.ranked_passage_ids = self._balanced_union(per_entity, self.cfg.final_top_k)
        result.fired = bool(result.ranked_passage_ids)
        return result


__all__ = [
    "CompareArm",
    "CompareConfig",
    "CompareResult",
    "Candidate",
    "AnnRetrieve",
    "is_comparison_query",
    "has_comparison_marker",
    "extract_compared_entities",
    "extract_comparison_attributes",
]
