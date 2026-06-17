# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""InfoboxIndex -- structured entity-attribute-value retrieval modality.

Wikipedia-style infoboxes (and similar structured-fact tables on every
entity page in many knowledge corpora) are a *higher-precision retrieval
target* than free-text prose for entity-attribute questions:

    "When was X born?"           -> (X, born, 1955-10-28)
    "Who is Y's spouse?"         -> (Y, spouse, Z)
    "What is the capital of Z?"  -> (Z, capital, W)

Free-text dense retrievers see this signal only indirectly: infoboxes
are usually serialised as one large unstructured chunk that has poor
cosine similarity to natural-language questions. The :class:`InfoboxIndex`
exposes the underlying triples as a first-class lookup modality.

Two extraction modes are provided, both optional and pluggable:

1. **Wikitext markup** -- ``extract_wikitext_infobox(text)`` parses the
   canonical ``| key = value`` rows from a ``{{Infobox ...}}`` template.
   Robust to template-name variation; ignores nested templates / wiki
   links inside values (kept raw for the caller to clean).

2. **Natural-language facts** -- ``extract_natural_facts(text, subject)``
   uses simple deterministic patterns to harvest common biographical /
   geographical claims from prose ("X was born in Y on Z", "X is the
   capital of Y", ...). Conservative by design; only the top-N highest-
   confidence patterns are emitted to keep precision over recall.

Both functions return ``list[InfoboxTriple]``; the index is corpus-
agnostic and consumes any iterable of triples. Callers may also feed
pre-extracted triples from external KG sources (Wikidata dumps, DBpedia,
custom enterprise KGs) -- the retrieval surface is the same.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)


# ---- Core data model -------------------------------------------------------

@dataclass(frozen=True)
class InfoboxTriple:
    """An ``(entity, attribute, value)`` fact harvested from a chunk.

    Parameters
    ----------
    subject
        Surface form of the entity the fact is about (e.g. ``"Albert
        Einstein"``). Compared against question-side surface forms via
        :func:`normalize_surface` -- callers should not pre-normalise.
    attribute
        Attribute / relation key (e.g. ``"born"``, ``"spouse"``,
        ``"occupation"``). Normalised to lower snake-case on insertion
        into :class:`InfoboxIndex`.
    value
        Free-text value (e.g. ``"14 March 1879"``, ``"Mileva Maric"``).
        Kept as-is; the caller decides whether to date-normalise or
        entity-link.
    source_chunk_id
        ID of the chunk this triple was harvested from. Enables the
        retriever to materialise the underlying passage when the triple
        is selected (so the reader sees grounded context, not a naked
        fact).
    confidence
        Float in ``[0, 1]``. Wikitext-template triples default to 1.0;
        natural-language pattern matches default to 0.7. Callers may
        pass external KG confidence (e.g. Wikidata rank: preferred=1.0,
        normal=0.6, deprecated=0.0).
    """

    subject: str
    attribute: str
    value: str
    source_chunk_id: str = ""
    confidence: float = 1.0


# ---- Normalisation helpers -------------------------------------------------

_WS_RE = re.compile(r"\s+")
_NON_ALPHANUM = re.compile(r"[^\w\s]+", re.UNICODE)


def normalize_surface(s: str) -> str:
    """Surface-form normalisation: strip diacritics, lower, squash WS.

    Used for the InfoboxIndex subject key + the question-side subject
    extracted before lookup. Matches against Wikipedia's
    case-insensitive title convention while preserving multi-word
    entities (so ``"Albert Einstein"`` matches but ``"einstein"`` does
    not without a separate alias step).
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = _NON_ALPHANUM.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip().lower()
    return s


def normalize_attribute(s: str) -> str:
    """Attribute-key normalisation: snake_case lower, alias collapse.

    ``"Date of birth"`` and ``"Born"`` both collapse to ``"born"``;
    ``"place_of_birth"`` and ``"birthplace"`` collapse to
    ``"birthplace"``; see :data:`_ATTRIBUTE_ALIASES` for the full map.
    """
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^\w_]+", "", s)
    return _ATTRIBUTE_ALIASES.get(s, s)


# Alias collapse for the most common Wikipedia infobox parameter names.
# Conservative: only well-known synonyms; extensible by the caller via
# :func:`InfoboxIndex.register_attribute_alias`.
_ATTRIBUTE_ALIASES: dict[str, str] = {
    "date_of_birth": "born",
    "birth_date": "born",
    "born_date": "born",
    "date_of_death": "died",
    "death_date": "died",
    "place_of_birth": "birthplace",
    "birth_place": "birthplace",
    "place_of_death": "deathplace",
    "death_place": "deathplace",
    "alma_mater": "alma_mater",
    "alma_matter": "alma_mater",
    "occupation": "occupation",
    "profession": "occupation",
    "nationality": "nationality",
    "citizenship": "nationality",
    "country": "country",
    "spouse_s": "spouse",
    "spouses": "spouse",
    "partner": "spouse",
    "children": "children",
    "parents": "parents",
    "parent": "parents",
    "fathers": "father",
    "mothers": "mother",
    "headquartered_in": "headquarters",
    "headquartered": "headquarters",
    "hq": "headquarters",
    "capital_city": "capital",
}


# ---- The index -------------------------------------------------------------

class InfoboxIndex:
    """Subject-attribute keyed lookup over a triple corpus.

    Underlying storage is a dict-of-dict-of-list:
    ``_table[normalized_subject][normalized_attribute] = list[Triple]``.

    Lookup paths:

    1. ``lookup(subject, attribute)`` -- exact subject + exact attribute.
       Returns ``list[InfoboxTriple]`` (multiple values per attribute
       are common: "spouse" with two consecutive marriages, multiple
       "children" in one infobox, ...).
    2. ``entity_attributes(subject)`` -- all attributes recorded for
       one entity. Returns ``dict[attribute_key, list[Triple]]``.
    3. ``find_by_value(value_substring)`` -- reverse lookup; used by
       the natural-language fact extractor to dedupe against existing
       wikitext-template triples.

    The index is **build-once / read-many**. Insertion order is preserved
    so deterministic iteration is guaranteed even for unstable Python
    dict semantics across versions.
    """

    def __init__(self) -> None:
        self._table: dict[str, dict[str, list[InfoboxTriple]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._value_to_subjects: dict[str, set[str]] = defaultdict(set)
        self._chunk_id_to_triples: dict[str, list[InfoboxTriple]] = defaultdict(list)
        self._n_triples = 0
        self._custom_aliases: dict[str, str] = {}

    # ---- Mutation ----------------------------------------------------

    def add(self, triple: InfoboxTriple) -> None:
        """Insert one triple. Idempotent on (subject, attribute, value)."""
        s = normalize_surface(triple.subject)
        a = self._apply_aliases(normalize_attribute(triple.attribute))
        if not s or not a or not triple.value:
            return
        existing = self._table[s][a]
        for t in existing:
            if normalize_surface(t.value) == normalize_surface(triple.value):
                return  # Already present
        canonical = InfoboxTriple(
            subject=triple.subject,
            attribute=a,
            value=triple.value,
            source_chunk_id=triple.source_chunk_id,
            confidence=triple.confidence,
        )
        existing.append(canonical)
        self._value_to_subjects[normalize_surface(triple.value)].add(s)
        if triple.source_chunk_id:
            self._chunk_id_to_triples[triple.source_chunk_id].append(canonical)
        self._n_triples += 1

    def add_many(self, triples: Iterable[InfoboxTriple]) -> None:
        for t in triples:
            self.add(t)

    def register_attribute_alias(self, alias: str, canonical: str) -> None:
        """Map ``alias`` (post-normalisation) to ``canonical`` attribute key.

        Used by callers with domain-specific attribute names (e.g. an
        enterprise KG may name a "headcount" attribute ``"FTEs"``).
        Aliases are evaluated AFTER the built-in normalisation.
        """
        self._custom_aliases[normalize_attribute(alias)] = normalize_attribute(canonical)

    def _apply_aliases(self, attribute_key: str) -> str:
        return self._custom_aliases.get(attribute_key, attribute_key)

    # ---- Read --------------------------------------------------------

    def lookup(self, subject: str, attribute: str) -> list[InfoboxTriple]:
        """Return all triples matching exact subject + exact attribute."""
        s = normalize_surface(subject)
        a = self._apply_aliases(normalize_attribute(attribute))
        return list(self._table.get(s, {}).get(a, ()))

    def entity_attributes(self, subject: str) -> dict[str, list[InfoboxTriple]]:
        """Return all attributes recorded for ``subject``."""
        s = normalize_surface(subject)
        return {a: list(v) for a, v in self._table.get(s, {}).items()}

    def find_by_value(self, value_substring: str) -> list[tuple[str, str]]:
        """Reverse lookup: find ``(subject, attribute)`` pairs whose value
        contains ``value_substring``. Used for dedupe + cross-validation.
        """
        target = normalize_surface(value_substring)
        if not target:
            return []
        hits: list[tuple[str, str]] = []
        for subj, attrs in self._table.items():
            for attr, triples in attrs.items():
                for t in triples:
                    if target in normalize_surface(t.value):
                        hits.append((subj, attr))
                        break
        return hits

    def triples_from_chunk(self, chunk_id: str) -> list[InfoboxTriple]:
        """Return all triples harvested from one source chunk."""
        return list(self._chunk_id_to_triples.get(chunk_id, ()))

    def __len__(self) -> int:
        return self._n_triples

    def __contains__(self, subject: str) -> bool:
        return normalize_surface(subject) in self._table


# ---- Extraction: wikitext infobox templates --------------------------------

# Matches the opening of an Infobox template. Wikipedia variants:
#   {{Infobox person | ...
#   {{Infobox scientist
#   {{infobox university | ...
_INFOBOX_OPEN_RE = re.compile(
    r"\{\{\s*[Ii]nfobox\b[^\|}]*", re.MULTILINE
)
# Matches a "| key = value" row inside the infobox; non-greedy value
# capture stops at the next "\n|" or "}}" or "<!--".
_INFOBOX_ROW_RE = re.compile(
    r"\|\s*(?P<key>[\w\s\-]+?)\s*=\s*(?P<value>.*?)"
    r"(?=(?:\n\s*\|)|(?:\n\s*\}\})|<!--)",
    re.DOTALL,
)
# Wiki-link cleanup: [[Foo|bar]] -> bar; [[Foo]] -> Foo
_WIKILINK_RE = re.compile(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]")
# Curly-brace template cleanup: {{...|val}} -> val; {{date|1879|3|14}} -> "1879 3 14"
_TEMPLATE_RE = re.compile(r"\{\{([^{}]+)\}\}")
# Pipe-template arg split (after _TEMPLATE_RE strips the braces).
_PIPE_SPLIT_RE = re.compile(r"\s*\|\s*")
# Strip HTML comments.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def _clean_wikitext_value(raw: str) -> str:
    """Strip wiki-links + template wrappers; preserve plain text content."""
    if not raw:
        return ""
    s = _HTML_COMMENT_RE.sub("", raw)
    # Iterate template stripping until fixed-point (handles nested).
    prev = None
    while prev != s:
        prev = s
        s = _TEMPLATE_RE.sub(
            lambda m: " ".join(p.strip() for p in _PIPE_SPLIT_RE.split(m.group(1))[1:]
                               if "=" not in p),
            s,
        )
    s = _WIKILINK_RE.sub(r"\1", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def extract_wikitext_infobox(
    text: str,
    *,
    subject: str | None = None,
    source_chunk_id: str = "",
    default_confidence: float = 1.0,
) -> list[InfoboxTriple]:
    """Parse a Wikipedia ``{{Infobox ...}}`` template into triples.

    Parameters
    ----------
    text
        Raw wikitext (may contain prose around the template -- the
        template span is detected by :data:`_INFOBOX_OPEN_RE`).
    subject
        Entity name. If ``None``, attempts to read the first wikilink
        or ``name`` row inside the template; falls back to the first
        line of prose preceding the template.
    source_chunk_id
        Threaded to every emitted triple.
    default_confidence
        Per-triple confidence (default 1.0 for wikitext template rows
        because the structure is explicit and unambiguous).
    """
    if not text or "infobox" not in text.lower():
        return []
    m = _INFOBOX_OPEN_RE.search(text)
    if not m:
        return []
    template_start = m.start()
    depth = 0
    i = template_start
    n = len(text)
    template_end = n
    while i < n:
        if text[i:i+2] == "{{":
            depth += 1
            i += 2
        elif text[i:i+2] == "}}":
            depth -= 1
            i += 2
            if depth == 0:
                template_end = i
                break
        else:
            i += 1
    template_body = text[template_start:template_end]

    if subject is None:
        m_name = re.search(
            r"\|\s*name\s*=\s*(?P<v>[^\n|]+)", template_body, re.MULTILINE
        )
        if m_name:
            subject = _clean_wikitext_value(m_name.group("v")) or None
    if subject is None:
        m_link = _WIKILINK_RE.search(text)
        if m_link:
            subject = m_link.group(1)
    if not subject:
        return []

    triples: list[InfoboxTriple] = []
    for row in _INFOBOX_ROW_RE.finditer(template_body):
        key = row.group("key").strip()
        raw_value = row.group("value")
        value = _clean_wikitext_value(raw_value)
        if not key or not value:
            continue
        if key.lower() in ("name", "image", "caption", "image_size", "alt"):
            continue
        # Apply normalisation + alias collapse here so the returned
        # triples carry the canonical attribute key. InfoboxIndex would
        # re-normalise on insert idempotently, but callers reading
        # triples directly (e.g. for inspection or dedupe) get the
        # canonical form without an extra pass.
        triples.append(InfoboxTriple(
            subject=subject,
            attribute=normalize_attribute(key),
            value=value,
            source_chunk_id=source_chunk_id,
            confidence=default_confidence,
        ))
    return triples


# ---- Extraction: natural-language patterns ---------------------------------

# Conservative biographical / geographical patterns. Each entry is
# (compiled_regex, attribute_name, value_group_index). The regex groups
# capture (subject, value); attribute is fixed per pattern. value_group is
# typically 2 (after the subject capture).
#
# Patterns chosen for precision: each must have an unambiguous mapping to
# (entity, attribute, value). Multi-clause sentences are NOT decomposed -
# the prose-extractor refuses ambiguous sentences rather than guess.

_FACT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(
        r"\b([A-Z][\w\.\- ]+?)\s+(?:was|is)\s+born\s+(?:on\s+)?"
        r"([^.;,(]+?)(?:[\.;,(]|$)",
        re.IGNORECASE,
    ), "born"),
    (re.compile(
        r"\b([A-Z][\w\.\- ]+?)\s+(?:was|is)\s+born\s+in\s+"
        r"([^.;,(]+?)(?:[\.;,(]|$)",
        re.IGNORECASE,
    ), "birthplace"),
    (re.compile(
        r"\b([A-Z][\w\.\- ]+?)\s+died\s+(?:on\s+)?"
        r"([^.;,(]+?)(?:[\.;,(]|$)",
        re.IGNORECASE,
    ), "died"),
    (re.compile(
        r"\b([A-Z][\w\.\- ]+?)\s+died\s+in\s+"
        r"([^.;,(]+?)(?:[\.;,(]|$)",
        re.IGNORECASE,
    ), "deathplace"),
    (re.compile(
        r"\bThe capital of\s+([A-Z][\w\. ]+?)\s+is\s+"
        r"([^.;,(]+?)(?:[\.;,(]|$)",
        re.IGNORECASE,
    ), "capital"),
    (re.compile(
        r"\b([A-Z][\w\.\- ]+?)\s+is\s+the capital of\s+"
        r"([A-Z][\w\. ]+?)(?:[\.;,(]|$)",
        re.IGNORECASE,
    ), "capital_of"),
    (re.compile(
        r"\b([A-Z][\w\.\- ]+?)\s+is\s+(?:an?\s+)?"
        r"(\w+(?:\s+\w+){0,2})\s+(?:actor|actress|director|singer|"
        r"author|writer|scientist|politician|player)\b",
        re.IGNORECASE,
    ), "nationality"),
]


def extract_natural_facts(
    text: str,
    *,
    source_chunk_id: str = "",
    default_confidence: float = 0.7,
    max_facts: int = 32,
) -> list[InfoboxTriple]:
    """Harvest conservative biographical / geographical facts from prose.

    Pattern set is deliberately small and high-precision: a fact only
    fires when one of the unambiguous templates in :data:`_FACT_PATTERNS`
    matches. Subject is captured from the regex; value is the matched
    span; attribute is fixed per pattern.

    Parameters
    ----------
    text
        Raw prose (paragraph or chunk).
    source_chunk_id
        Threaded to every emitted triple.
    default_confidence
        Per-triple confidence (default 0.7 for prose patterns -- lower
        than wikitext template confidence to reflect ambiguity).
    max_facts
        Cap on emitted triples per chunk to bound the index size.
    """
    if not text:
        return []
    triples: list[InfoboxTriple] = []
    for pat, attribute in _FACT_PATTERNS:
        for m in pat.finditer(text):
            if len(triples) >= max_facts:
                break
            subject = m.group(1).strip()
            value = m.group(2).strip()
            if not subject or not value:
                continue
            if len(subject) > 80 or len(value) > 200:
                continue
            triples.append(InfoboxTriple(
                subject=subject,
                attribute=attribute,
                value=value,
                source_chunk_id=source_chunk_id,
                confidence=default_confidence,
            ))
        if len(triples) >= max_facts:
            break
    return triples


# ---- Corpus-level builder --------------------------------------------------

def build_infobox_index_from_chunks(
    chunks: Sequence,
    *,
    wikitext: bool = True,
    natural_language: bool = True,
    chunk_id_attr: str = "chunk_id",
    chunk_text_attr: str = "text",
    subject_attr: str | None = None,
) -> InfoboxIndex:
    """Build an :class:`InfoboxIndex` from a sequence of chunk-like objects.

    Each chunk-like must expose at least ``text``; if ``chunk_id_attr`` is
    set on the object it is used as the source identifier; otherwise the
    chunk's position in the sequence is used.

    Parameters
    ----------
    wikitext
        Run :func:`extract_wikitext_infobox` on every chunk.
    natural_language
        Run :func:`extract_natural_facts` on every chunk.
    subject_attr
        If set, read the chunk attribute of that name to pass as the
        ``subject`` hint to :func:`extract_wikitext_infobox` (improves
        recall on chunks where the template ``name`` row is absent).
    """
    index = InfoboxIndex()
    if not chunks:
        return index
    for i, chunk in enumerate(chunks):
        text = getattr(chunk, chunk_text_attr, "") if not isinstance(chunk, Mapping) \
            else chunk.get(chunk_text_attr, "")
        if not text:
            continue
        cid = getattr(chunk, chunk_id_attr, None) if not isinstance(chunk, Mapping) \
            else chunk.get(chunk_id_attr, None)
        cid = str(cid) if cid is not None else f"chunk_{i}"
        subj_hint = None
        if subject_attr is not None:
            subj_hint = (getattr(chunk, subject_attr, None)
                         if not isinstance(chunk, Mapping)
                         else chunk.get(subject_attr, None))
        if wikitext:
            index.add_many(extract_wikitext_infobox(
                text, subject=subj_hint, source_chunk_id=cid,
            ))
        if natural_language:
            index.add_many(extract_natural_facts(
                text, source_chunk_id=cid,
            ))
    logger.info("build_infobox_index_from_chunks: %d triples from %d chunks",
                len(index), len(chunks))
    return index


__all__ = [
    "InfoboxTriple",
    "InfoboxIndex",
    "normalize_surface",
    "normalize_attribute",
    "extract_wikitext_infobox",
    "extract_natural_facts",
    "build_infobox_index_from_chunks",
]
