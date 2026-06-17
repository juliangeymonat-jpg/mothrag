# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
r"""Deterministic OpenIE-style triple extraction for MothGraphArm.

Two extraction tiers, both pluggable:

1. **Stdlib-only (default)** -- :func:`_regex_extract_triples` harvests
   ``(subject, predicate, object)`` from prose via a conservative set
   of high-precision sentence patterns. No external dependency; runs
   in CI on any machine. Patterns are deliberately narrow (e.g.
   ``"X founded Y"``, ``"X is the Y of Z"``) so precision stays high
   at the cost of recall.

2. **Spacy-accelerated (optional)** -- when ``spacy`` plus an English
   model are importable, :func:`_spacy_extract_triples` walks the
   dependency parse to harvest ``(nsubj, ROOT/aux+verb, dobj/pobj)``
   triples with richer coverage. Falls back to the stdlib tier on any
   import failure so a missing dependency is non-fatal.

Both tiers feed into :func:`extract_triples_from_text` -- the single
public extraction entry point. Callers may inject their own extractor
via :func:`extract_triples` (sequence of chunks -> list of triples)
without touching the per-text tier dispatch.

Per :data:`feedback_no_dataset_specific_training_general_purpose_only_2026_05_20`:
patterns target GENERIC English clause structures (subject + verb +
object). NO per-dataset patterns, NO gold-derived tuning, NO test-set
inspection.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)


# ---- Core data model -------------------------------------------------------

@dataclass(frozen=True)
class RawTriple:
    """Pre-index triple emitted by the extractor.

    Distinct from :class:`mothrag.graph.index.GraphEdge`: the index-
    level edge stores normalised entity keys; ``RawTriple`` carries the
    original surface forms so downstream consumers (e.g. answer
    formatting) see human-readable text.
    """

    subject: str
    predicate: str
    object: str
    source_chunk_id: str = ""
    confidence: float = 0.7


# ---- Normalisation helpers (shared with InfoboxIndex contract) -------------

_WS_RE = re.compile(r"\s+")
_NON_ALPHANUM = re.compile(r"[^\w\s]+", re.UNICODE)


def normalize_entity(s: str) -> str:
    """Surface-form normalisation: ASCII fold, lower, squash WS.

    Aligned with :func:`mothrag.core.retrieval.infobox.normalize_surface`
    so a graph-side entity key compares equal to an infobox-side
    subject key. Lets a downstream consumer cross-reference the two
    indexes without a separate alias table.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = _NON_ALPHANUM.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip().lower()
    return s


# ---- Stdlib extraction tier ------------------------------------------------

# Conservative SVO patterns. Each entry is (compiled_regex, predicate_canonical
# OR group-index). The regex captures (subject, [predicate?], object).
#
# Precision policy: each pattern must have an UNAMBIGUOUS (subject, predicate,
# object) mapping. Multi-clause sentences are NOT decomposed -- the extractor
# refuses ambiguous spans rather than guess.

_CAP_ENTITY = r"(?:[A-Z][\w\.\-]*(?:\s+[A-Z][\w\.\-]*){0,4})"
_LOWER_NP = r"(?:(?:the\s+|a\s+|an\s+)?[a-z][\w\-]+(?:\s+[a-z][\w\-]+){0,4})"
_OBJECT_NP = rf"(?:{_CAP_ENTITY}|{_LOWER_NP})"

# (X) (founded|created|...) (Y)  -> (X, predicate, Y)
_SVO_VERB_PATTERN = re.compile(
    rf"\b({_CAP_ENTITY})\s+"
    r"(founded|created|directed|wrote|composed|invented|discovered|"
    r"acquired|started|published|released|established|signed|elected|"
    r"played|hosted|formed|developed|owned|owns|produced|designed|"
    r"adapted|succeeded|preceded|married|joined|led|leads|managed|"
    r"won|received|earned|achieved|attended|studied)\s+"
    rf"({_OBJECT_NP})",
    re.IGNORECASE,
)

# (X) is/was (the|a|an) (predicate-noun) of (Y)  -> (X, predicate, Y)
# Captures the relation noun explicitly (e.g. "Y is the founder of Z" ->
# (Y, founder, Z)).
_BE_OF_PATTERN = re.compile(
    rf"\b({_CAP_ENTITY})\s+(?:is|was|are|were)\s+(?:the|a|an)\s+"
    r"(\w+(?:\s+\w+){0,2})\s+of\s+"
    rf"({_OBJECT_NP})",
    re.IGNORECASE,
)

# X's Y is Z  -> (X, Y, Z)
_POSSESSIVE_PATTERN = re.compile(
    rf"\b({_CAP_ENTITY})'s\s+(\w+(?:\s+\w+){{0,2}})\s+(?:is|was|are|were)\s+"
    rf"({_OBJECT_NP})",
    re.IGNORECASE,
)

# X was born in/on Y -> (X, born_in / born_on, Y). Kept separate from generic
# "born" verb pattern because the preposition disambiguates the relation.
_BORN_IN_PATTERN = re.compile(
    rf"\b({_CAP_ENTITY})\s+(?:was|is)\s+born\s+in\s+({_OBJECT_NP})",
    re.IGNORECASE,
)
_BORN_ON_PATTERN = re.compile(
    rf"\b({_CAP_ENTITY})\s+(?:was|is)\s+born\s+(?:on\s+)?"
    r"([^.;,(]+?)(?=[\.;,(]|$)",
    re.IGNORECASE,
)

# X died in/on Y -> (X, died_in / died_on, Y).
_DIED_IN_PATTERN = re.compile(
    rf"\b({_CAP_ENTITY})\s+died\s+in\s+({_OBJECT_NP})",
    re.IGNORECASE,
)


_REGEX_PATTERNS: list[tuple[re.Pattern, str | None]] = [
    (_SVO_VERB_PATTERN, None),  # predicate captured in group 2
    (_BE_OF_PATTERN, None),     # predicate captured in group 2
    (_POSSESSIVE_PATTERN, None),  # predicate captured in group 2
    (_BORN_IN_PATTERN, "born_in"),
    (_BORN_ON_PATTERN, "born_on"),
    (_DIED_IN_PATTERN, "died_in"),
]


def _clean_predicate(raw: str) -> str:
    s = (raw or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    return s


def _clean_object(raw: str) -> str:
    s = (raw or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _regex_extract_triples(
    text: str, *, source_chunk_id: str, max_triples: int,
) -> list[RawTriple]:
    """Stdlib regex extractor (always available, no spacy dependency)."""
    if not text:
        return []
    seen: set[tuple[str, str, str]] = set()
    out: list[RawTriple] = []
    for pat, predicate_const in _REGEX_PATTERNS:
        for m in pat.finditer(text):
            if len(out) >= max_triples:
                return out
            groups = m.groups()
            if predicate_const is not None:
                subject = groups[0].strip()
                predicate = predicate_const
                obj = _clean_object(groups[1])
            else:
                subject = groups[0].strip()
                predicate = _clean_predicate(groups[1])
                obj = _clean_object(groups[2])
            if not subject or not predicate or not obj:
                continue
            if len(subject) > 80 or len(obj) > 200:
                continue
            key = (normalize_entity(subject), predicate, normalize_entity(obj))
            if key in seen:
                continue
            seen.add(key)
            out.append(RawTriple(
                subject=subject,
                predicate=predicate,
                object=obj,
                source_chunk_id=source_chunk_id,
                confidence=0.7,
            ))
    return out


# ---- Spacy extraction tier (optional) --------------------------------------

_SPACY_NLP = None
_SPACY_TRIED = False


def _try_load_spacy():
    """Lazy-import spacy with English model. Returns nlp callable or None.

    Cached: a successful load is reused; a failure short-circuits future
    calls so the cost of the import attempt is paid at most once.
    """
    global _SPACY_NLP, _SPACY_TRIED
    if _SPACY_TRIED:
        return _SPACY_NLP
    _SPACY_TRIED = True
    try:
        import spacy  # noqa: F401
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except (OSError, IOError):
            _SPACY_NLP = None
    except ImportError:
        _SPACY_NLP = None
    return _SPACY_NLP


def _spacy_extract_triples(
    text: str, *, source_chunk_id: str, max_triples: int,
) -> list[RawTriple]:
    """Spacy dependency-parse extractor (optional accelerator)."""
    nlp = _try_load_spacy()
    if nlp is None:
        return []
    try:
        doc = nlp(text)
    except Exception:  # noqa: BLE001
        return []
    out: list[RawTriple] = []
    seen: set[tuple[str, str, str]] = set()
    for sent in doc.sents:
        for token in sent:
            if token.dep_ != "ROOT" and token.pos_ != "VERB":
                continue
            subj = None
            obj = None
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subj = " ".join(c.text for c in child.subtree)
                elif child.dep_ in ("dobj", "pobj", "attr", "oprd"):
                    obj = " ".join(c.text for c in child.subtree)
                elif child.dep_ == "prep":
                    for gc in child.children:
                        if gc.dep_ == "pobj":
                            obj = " ".join(c.text for c in gc.subtree)
                            break
            if not subj or not obj:
                continue
            subj = subj.strip()
            obj = obj.strip()
            predicate = _clean_predicate(token.lemma_ or token.text)
            if not subj or not obj or not predicate:
                continue
            if len(subj) > 80 or len(obj) > 200:
                continue
            key = (normalize_entity(subj), predicate, normalize_entity(obj))
            if key in seen:
                continue
            seen.add(key)
            out.append(RawTriple(
                subject=subj,
                predicate=predicate,
                object=obj,
                source_chunk_id=source_chunk_id,
                confidence=0.6,
            ))
            if len(out) >= max_triples:
                return out
    return out


# ---- Public extraction entry points ----------------------------------------

def extract_triples_from_text(
    text: str,
    *,
    source_chunk_id: str = "",
    max_triples: int = 64,
    use_spacy: bool = False,
) -> list[RawTriple]:
    """Extract :class:`RawTriple` objects from one block of text.

    Parameters
    ----------
    text
        The prose / chunk text to harvest.
    source_chunk_id
        Threaded to every emitted triple for provenance.
    max_triples
        Per-text upper bound; protects the graph from blowing up on a
        single dense paragraph.
    use_spacy
        When ``True`` and spacy + an English model are importable, the
        spacy tier runs FIRST and its output is unioned with the stdlib
        tier. When ``False`` (default), only the stdlib tier runs --
        zero-dependency baseline.

    Output triples carry the ORIGINAL surface form for subject /
    object; only the predicate is canonicalised (lowercase
    snake_case). Downstream consumers should normalise via
    :func:`normalize_entity` for index keying.
    """
    if not text:
        return []
    out = _regex_extract_triples(
        text, source_chunk_id=source_chunk_id, max_triples=max_triples,
    )
    if use_spacy:
        remaining = max_triples - len(out)
        if remaining > 0:
            spacy_triples = _spacy_extract_triples(
                text, source_chunk_id=source_chunk_id, max_triples=remaining,
            )
            seen = {(normalize_entity(t.subject), t.predicate,
                     normalize_entity(t.object)) for t in out}
            for t in spacy_triples:
                key = (normalize_entity(t.subject), t.predicate,
                       normalize_entity(t.object))
                if key in seen:
                    continue
                seen.add(key)
                out.append(t)
    return out


def extract_triples(
    chunks: Iterable,
    *,
    chunk_id_attr: str = "chunk_id",
    chunk_text_attr: str = "text",
    max_triples_per_chunk: int = 64,
    use_spacy: bool = False,
) -> list[RawTriple]:
    """Extract triples from a sequence of chunk-like objects.

    Each chunk-like must expose ``text``; ``chunk_id`` is read if
    present, else a positional index is synthesised.
    """
    out: list[RawTriple] = []
    if not chunks:
        return out
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, Mapping):
            text = chunk.get(chunk_text_attr, "")
            cid = chunk.get(chunk_id_attr, None)
        else:
            text = getattr(chunk, chunk_text_attr, "")
            cid = getattr(chunk, chunk_id_attr, None)
        if not text:
            continue
        cid = str(cid) if cid is not None else f"chunk_{i}"
        out.extend(extract_triples_from_text(
            text,
            source_chunk_id=cid,
            max_triples=max_triples_per_chunk,
            use_spacy=use_spacy,
        ))
    return out


__all__ = [
    "RawTriple",
    "extract_triples",
    "extract_triples_from_text",
    "normalize_entity",
]
