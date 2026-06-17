# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""MultiModalRetriever -- dense + structured-infobox blend.

Blends the v0.5.0 alpha :class:`DenseRetriever` with the
:class:`InfoboxIndex` structured lookup over the same chunk corpus.
The two modalities are complementary:

- Dense retrieval recovers free-text passages discussing the question
  semantically. High recall on open-ended Wh-questions; relatively low
  precision on entity-attribute lookups where the dense embedding has
  to "compose" subject + attribute from one chunk.

- Infobox retrieval looks up structured ``(subject, attribute, value)``
  triples by exact subject + attribute key. Near-perfect precision on
  entity-attribute questions; zero recall on open-ended or
  multi-fact-synthesis questions.

The :class:`MultiModalRetriever` extracts ``(subject, attribute)`` hints
from the question (via configurable extractors, with a deterministic
default), looks them up in the :class:`InfoboxIndex`, materialises the
underlying source chunks for the hits, prepends them to the dense top-K
result list, and de-duplicates by chunk-id. The blend is *additive*,
not replacement -- dense recall is preserved; infobox triples enrich
the top-K with high-precision structured matches that dense embeddings
miss.

The hint extractor is pluggable:

- ``mode='deterministic'`` (default) -- regex patterns for
  ``"when was X born"`` / ``"who is the X of Y"`` / ``"what is the
  capital of X"`` style entity-attribute questions. No LLM call.
- ``mode='llm'`` -- callable that takes the question and returns
  ``list[(subject, attribute)]`` candidates. Used when the deployment
  already wires an LLM-NER cache (e.g. the L1C cache in MothRag's eval
  scripts). Failures fall back to the deterministic path silently.

The blend weight ``infobox_top_n_boost`` controls how many infobox
chunks are prepended (default 3); the resulting list always has length
``<= top_k + infobox_top_n_boost`` even when both modalities saturate.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Sequence

from mothrag.core.retrieval.infobox import (
    InfoboxIndex,
    InfoboxTriple,
    normalize_attribute,
    normalize_surface,
)

logger = logging.getLogger(__name__)


# ---- Deterministic question-side hint extractor ----------------------------

# Patterns capture (attribute, subject) for the canonical entity-attribute
# question shapes. Order matters: more specific patterns first so the
# greedy `searcher` doesn't shadow them.
_QUESTION_HINT_PATTERNS: list[tuple[re.Pattern, str, int, int]] = [
    # "When was X born?"  -> (X, born)
    (re.compile(
        r"\b(?:when|on\s+what\s+date)\s+(?:was|were)\s+(?P<subject>[A-Z][\w\.\- ]+?)\s+born",
        re.IGNORECASE,
    ), "born", 0, 1),
    # "Where was X born?" -> (X, birthplace)
    (re.compile(
        r"\bwhere\s+(?:was|were)\s+(?P<subject>[A-Z][\w\.\- ]+?)\s+born",
        re.IGNORECASE,
    ), "birthplace", 0, 1),
    # "Where did X die?" -> (X, deathplace)
    (re.compile(
        r"\bwhere\s+did\s+(?P<subject>[A-Z][\w\.\- ]+?)\s+die",
        re.IGNORECASE,
    ), "deathplace", 0, 1),
    # "When did X die?" -> (X, died)
    (re.compile(
        r"\bwhen\s+did\s+(?P<subject>[A-Z][\w\.\- ]+?)\s+die",
        re.IGNORECASE,
    ), "died", 0, 1),
    # "What is the capital of X?" -> (X, capital)
    (re.compile(
        r"\bwhat\s+is\s+the\s+capital\s+of\s+(?P<subject>[A-Z][\w\.\- ]+)",
        re.IGNORECASE,
    ), "capital", 0, 1),
    # "Who is X's spouse?" -> (X, spouse)  // also handles "wife" / "husband"
    (re.compile(
        r"\bwho\s+is\s+(?P<subject>[A-Z][\w\.\- ]+?)['’]s\s+"
        r"(?P<attribute>spouse|wife|husband|partner)",
        re.IGNORECASE,
    ), None, 0, 1),
    # "Who is the spouse of X?" -> (X, spouse)
    (re.compile(
        r"\bwho\s+is\s+the\s+(?P<attribute>spouse|wife|husband|partner|"
        r"father|mother|son|daughter|director|author|composer|founder|"
        r"president|prime\s+minister|ceo|chairman)\s+of\s+"
        r"(?P<subject>[A-Z][\w\.\- ]+)",
        re.IGNORECASE,
    ), None, 0, 1),
    # "What is X's nationality?" -> (X, nationality) // also occupation
    (re.compile(
        r"\bwhat\s+is\s+(?P<subject>[A-Z][\w\.\- ]+?)['’]s\s+"
        r"(?P<attribute>nationality|occupation|profession|religion)",
        re.IGNORECASE,
    ), None, 0, 1),
]


def extract_question_hints(question: str) -> list[tuple[str, str]]:
    """Deterministic entity-attribute extraction from question text.

    Returns ``list[(subject, attribute)]`` candidates. The list may be
    empty (when the question is open-ended or doesn't match a hint
    template). Each candidate is normalised via the same surface +
    attribute normalisers used by :class:`InfoboxIndex` so the caller
    can issue ``index.lookup(*candidate)`` directly.
    """
    if not question:
        return []
    hints: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pat, fixed_attribute, _, _ in _QUESTION_HINT_PATTERNS:
        for m in pat.finditer(question):
            subject = m.group("subject").strip()
            if fixed_attribute is None:
                attribute = m.group("attribute").strip()
            else:
                attribute = fixed_attribute
            key = (normalize_surface(subject), normalize_attribute(attribute))
            if key in seen or not key[0] or not key[1]:
                continue
            seen.add(key)
            hints.append((subject, attribute))
    return hints


# ---- The retriever ---------------------------------------------------------

class MultiModalRetriever:
    """Dense + infobox blend.

    Parameters
    ----------
    dense
        Any :class:`mothrag.core.retrieval.Retriever` -- typically a
        :class:`DenseRetriever`. The blend is additive: the dense
        retriever's top-K is preserved; the infobox is only enriching.
    infobox_index
        Pre-built :class:`InfoboxIndex`. Build via
        :func:`build_infobox_index_from_chunks` over the same chunk
        corpus the dense retriever was indexed on.
    chunk_provider
        Optional callable ``chunk_id -> Chunk`` used to materialise
        chunks for the infobox hits. If ``None``, the retriever falls
        back to returning whatever chunks the underlying ``dense``
        retriever surfaces -- the infobox triples then act as a
        re-ranking signal only (boost dense scores for chunks whose
        chunk_id matches an infobox hit).
    infobox_top_n_boost
        Number of infobox chunks to prepend to the dense top-K (default
        3). The blended list has length ``<= top_k + infobox_top_n_boost``.
    hint_extractor
        Pluggable replacement for :func:`extract_question_hints`. Must
        return ``list[(subject, attribute)]``. Raised exceptions are
        caught and the deterministic default is used as a fallback.
    """

    name = "dense_plus_infobox"

    def __init__(
        self,
        dense,
        infobox_index: InfoboxIndex,
        *,
        chunk_provider: Callable | None = None,
        infobox_top_n_boost: int = 3,
        hint_extractor: Callable[[str], list[tuple[str, str]]] | None = None,
    ) -> None:
        self.dense = dense
        self.infobox_index = infobox_index
        self.chunk_provider = chunk_provider
        self.infobox_top_n_boost = int(infobox_top_n_boost)
        self._hint_extractor = hint_extractor or extract_question_hints

    # ---- Index passthrough (no infobox-side index step needed; the
    # InfoboxIndex is built separately at corpus-prep time). ----

    def index(self, chunks: Sequence) -> None:
        self.dense.index(chunks)

    # ---- Retrieve ----------------------------------------------------

    def retrieve(self, question: str, *, top_k: int = 10) -> list:
        """Blended retrieval: prepend infobox hits, return dedup'd list."""
        dense_chunks = list(self.dense.retrieve(question, top_k=top_k))
        seen_ids = {self._chunk_id(c) for c in dense_chunks if self._chunk_id(c)}

        infobox_chunks = self._retrieve_infobox(question)
        prepend: list = []
        for c in infobox_chunks:
            cid = self._chunk_id(c)
            if cid and cid not in seen_ids:
                prepend.append(c)
                seen_ids.add(cid)
            if len(prepend) >= self.infobox_top_n_boost:
                break

        return prepend + dense_chunks

    def __len__(self) -> int:
        return len(self.dense)

    # ---- Internals ---------------------------------------------------

    def _retrieve_infobox(self, question: str) -> list:
        """Resolve (subject, attribute) hints into source chunks.

        Returns the *materialised chunks* in priority order:
        higher-confidence triples first. Empty list when no hint
        matches or no chunks resolved.
        """
        try:
            hints = self._hint_extractor(question)
        except Exception as exc:  # noqa: BLE001
            logger.warning("hint_extractor raised %s; falling back to deterministic",
                           exc)
            hints = extract_question_hints(question)
        if not hints:
            return []
        triples: list[InfoboxTriple] = []
        seen_triple: set[tuple[str, str, str]] = set()
        for subject, attribute in hints:
            for t in self.infobox_index.lookup(subject, attribute):
                key = (normalize_surface(t.subject),
                       normalize_attribute(t.attribute),
                       normalize_surface(t.value))
                if key in seen_triple:
                    continue
                seen_triple.add(key)
                triples.append(t)
        triples.sort(key=lambda t: t.confidence, reverse=True)
        if not triples:
            return []
        return self._triples_to_chunks(triples)

    def _triples_to_chunks(self, triples: Sequence[InfoboxTriple]) -> list:
        """Materialise triple -> source chunk (or a synthetic facade).

        When ``chunk_provider`` is set, each triple is hydrated to its
        underlying source chunk so the reader sees grounded passage
        context. When unset, a lightweight synthetic chunk facade is
        emitted containing the triple text as the passage so callers
        without a chunk_provider still get usable surface text in the
        reader prompt.
        """
        out: list = []
        seen_cids: set[str] = set()
        for t in triples:
            if self.chunk_provider is not None and t.source_chunk_id:
                try:
                    chunk = self.chunk_provider(t.source_chunk_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("chunk_provider(%s) raised %s",
                                   t.source_chunk_id, exc)
                    chunk = None
                if chunk is not None:
                    cid = self._chunk_id(chunk) or t.source_chunk_id
                    if cid not in seen_cids:
                        seen_cids.add(cid)
                        out.append(chunk)
                    continue
            # Synthetic chunk facade
            synthetic = _SyntheticInfoboxChunk.from_triple(t)
            cid = synthetic.chunk_id
            if cid not in seen_cids:
                seen_cids.add(cid)
                out.append(synthetic)
        return out

    @staticmethod
    def _chunk_id(chunk) -> str:
        if chunk is None:
            return ""
        cid = getattr(chunk, "chunk_id", None)
        if cid is None and isinstance(chunk, dict):
            cid = chunk.get("chunk_id", "")
        return str(cid or "")


# ---- Synthetic chunk facade ------------------------------------------------

class _SyntheticInfoboxChunk:
    """Lightweight chunk facade for an infobox triple.

    Mirrors the :class:`mothrag.core.api.Chunk` surface that arms
    consume (``chunk_id``, ``text``, ``embedding``, ``metadata``)
    without importing the heavy Chunk dataclass (which would create a
    circular import). Used when no ``chunk_provider`` is wired -- the
    triple itself becomes the passage.
    """

    __slots__ = ("chunk_id", "text", "embedding", "metadata")

    def __init__(self, chunk_id: str, text: str, metadata: dict) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.embedding = None
        self.metadata = metadata

    @classmethod
    def from_triple(cls, t: InfoboxTriple) -> "_SyntheticInfoboxChunk":
        text = f"{t.subject} -- {t.attribute}: {t.value}"
        cid = (
            f"infobox:{normalize_surface(t.subject)[:32]}:"
            f"{normalize_attribute(t.attribute)}"
        )
        meta = {
            "source": "infobox",
            "subject": t.subject,
            "attribute": t.attribute,
            "value": t.value,
            "confidence": t.confidence,
            "source_chunk_id": t.source_chunk_id,
        }
        return cls(chunk_id=cid, text=text, metadata=meta)


__all__ = [
    "MultiModalRetriever",
    "extract_question_hints",
]
