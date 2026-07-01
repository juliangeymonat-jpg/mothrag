# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""High-level MothRAG public API (v0.5.0).

Designed to be the LlamaIndex-style entry point:

    from mothrag import MothRAG

    rag = MothRAG.from_documents("path/to/docs/")
    print(rag.query("What is X?"))

Auto-defaults mirror the production stack (MOTHRAG 1):
embedder = Gemini Embedding 2 (fallback: hash-bucket baseline),
reader   = Llama-3.3-70B-Instruct-Turbo via Together (fallback: echo stub),
vector   = in-memory numpy cosine-similarity index.

All three components are pluggable via the ``embedder=``, ``reader=``,
``vector_db=`` constructor arguments — see :class:`Embedder`,
:class:`Reader`, :class:`VectorStore` protocols below.

Heavy backends (sentence-transformers, openai, google-genai) are
optional extras installed via ``pip install mothrag[prod]``. Without them
the class still imports and constructs successfully (auto-fallbacks
fire on instantiation), so ``from mothrag import MothRAG`` is safe in
minimal environments.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence, runtime_checkable

logger = logging.getLogger(__name__)


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class Document:
    """A source document — text + optional metadata.

    Created by loaders (see :mod:`mothrag.loaders`) or directly by user code.
    """
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return str(self.metadata.get("source", id(self)))


@dataclass
class Chunk:
    """A retrievable chunk derived from a Document during ingestion."""
    text: str
    doc_id: str = ""
    chunk_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # Embedding populated by Embedder during index().
    embedding: list[float] | None = None


@dataclass
class QueryResult:
    """End-to-end query result with provenance for debugging / citation."""
    answer: str
    retrieved_chunks: list[Chunk] = field(default_factory=list)
    arm_used: str = ""
    arm_subset: list[str] = field(default_factory=list)
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# PROTOCOLS — pluggable backend contracts
# ============================================================

@runtime_checkable
class Embedder(Protocol):
    """Encode a batch of texts into fixed-dim vectors."""
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class Reader(Protocol):
    """Answer a question given a list of retrieved passages."""
    def read(self, question: str, passages: Sequence[str]) -> str: ...


@runtime_checkable
class VectorStore(Protocol):
    """Add chunks (with embeddings), retrieve top-K by query embedding."""
    def add(self, chunks: Sequence[Chunk]) -> None: ...
    def retrieve(self, query_embedding: list[float], top_k: int = 10) -> list[Chunk]: ...
    def __len__(self) -> int: ...


@runtime_checkable
class MutableVectorStore(VectorStore, Protocol):
    """A :class:`VectorStore` that also supports removing and replacing
    chunks by id, enabling incremental *updates* (a fact changed) and
    *deletes* (a fact was retracted) without rebuilding the index.

    Append-only stores need not implement this; :meth:`MothRAG.update` and
    :meth:`MothRAG.delete` raise a clear error when the active store does
    not support mutation.
    """
    def delete(self, chunk_ids: Sequence[str]) -> int: ...
    def delete_by_doc(self, doc_id: str) -> int: ...
    def upsert(self, chunks: Sequence[Chunk]) -> None: ...


# ============================================================
# DEFAULT BACKENDS — minimal, no-network, no-LLM fallbacks
# ============================================================

class _HashEmbedder:
    """Deterministic hash-bucket embedder — zero deps, zero network.

    Used as a fallback when neither Gemini nor sentence-transformers is
    installed. NOT for production use; produces 256-dim sparse vectors
    via word-hash modulo and counts. Useful for smoke tests and offline
    development. Buckets via crc32, not ``hash()``: str hash is
    siphash-salted per process, which would make runs non-reproducible and
    silently break any index persisted across processes.
    """
    DIM = 256

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        import math
        import zlib
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.DIM
            tokens = re.findall(r"\w+", text.lower())
            for tok in tokens:
                vec[zlib.crc32(tok.encode("utf-8")) % self.DIM] += 1.0
            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class _EchoReader:
    """Fallback reader — returns the first sentence of the top passage.

    Used when no LLM API key is configured. Allows the pipeline to run
    end-to-end for smoke tests / offline development.
    """
    def read(self, question: str, passages: Sequence[str]) -> str:
        if not passages:
            return "[no passages retrieved]"
        first = passages[0]
        # Return the first sentence (up to ~200 chars).
        match = re.search(r"^[^.!?\n]+[.!?]?", first.strip())
        return (match.group(0) if match else first[:200]).strip()


def _attach_recall_at_k(meta: dict, retrieved: Sequence[Chunk],
                          gold_doc_ids: Sequence[str] | None,
                          top_k: int) -> None:
    """Compute Recall@K when the caller supplies gold_doc_ids.

    Populates meta with three keys (only if gold_doc_ids is non-empty):
      - r_at_k: float — Recall@{top_k} over the retrieved chunks' doc_ids
      - retrieved_doc_ids: ordered list of retrieved doc_ids
      - gold_doc_ids: echoed input (for downstream consumers)

    R@K = |retrieved_doc_ids[:k] ∩ gold| / |gold|. Doc-level not chunk-level —
    if a doc is split into N chunks, hitting any of them counts once.
    """
    if not gold_doc_ids:
        return
    retrieved_doc_ids: list[str] = []
    seen: set[str] = set()
    for c in retrieved:
        did = c.doc_id or ""
        if did and did not in seen:
            seen.add(did)
            retrieved_doc_ids.append(did)
    gold_set = set(gold_doc_ids)
    if not gold_set:
        return
    hits = sum(1 for d in retrieved_doc_ids[:top_k] if d in gold_set)
    meta["r_at_k"] = hits / len(gold_set)
    meta["retrieved_doc_ids"] = retrieved_doc_ids
    meta["gold_doc_ids"] = list(gold_doc_ids)


def _is_uncertain_answer(pred: str) -> bool:
    """Cheap uncertainty heuristic for the iter arm refinement loop.

    Uses the canonical ``ABSTAIN_MARKERS`` set shared with the eval
    pipeline (P24 unification). External callers
    can extend the set via
    :func:`mothrag.core.abstain_markers.is_abstain_marker`'s
    ``extra_markers`` kwarg.
    """
    from mothrag.core.abstain_markers import is_abstain_marker
    return is_abstain_marker(pred)


def _detect_abstention_signal(
    chosen: str, reason: str, c7_info: Any,
) -> str | None:
    """Classify the arbitration outcome into a retry-cascade signal.

    Returns one of the strings in
    :data:`mothrag.core.retry.protocol.ABSTENTION_SIGNALS`, or ``None`` when
    the arbitration produced a confident answer that does not need
    escalation. The function is heuristic and intentionally permissive at
    the v0.5.0-alpha layer; the full γ / H4 / H12 / L4b signals will be
    surfaced cleanly when the production iter pipeline lands in v0.5.1.
    """
    r = (reason or "").lower()
    if "γ_refuse" in r or "gamma" in r or "γ" in r:
        return "gamma_refuse"
    if "h4" in r or "refuse" in r:
        return "h4_refuse"
    if "h12" in r or "how_many" in r:
        return "h12_refuse"
    if isinstance(c7_info, dict):
        if c7_info.get("l4b", {}).get("cancelled"):
            return "iter_abstain"
        if c7_info.get("disagreement") is True:
            return "cross_arm_disagree"
    if _is_uncertain_answer(chosen):
        return "empty_answer"
    return None


class _MemoryVectorStore:
    """In-memory numpy-backed cosine-similarity index. No external deps."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []

    def add(self, chunks: Sequence[Chunk]) -> None:
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk {c.chunk_id} has no embedding")
            self._chunks.append(c)
            self._embeddings.append(c.embedding)

    def retrieve(self, query_embedding: list[float], top_k: int = 10) -> list[Chunk]:
        if not self._chunks:
            return []
        import numpy as np
        q = np.asarray(query_embedding, dtype=np.float32)
        e = np.asarray(self._embeddings, dtype=np.float32)
        # Both already normalized → cosine = dot.
        scores = e @ q
        top = np.argsort(scores)[::-1][:top_k]
        return [self._chunks[int(i)] for i in top]

    def __len__(self) -> int:
        return len(self._chunks)

    def delete(self, chunk_ids: Sequence[str]) -> int:
        """Remove chunks (and their embeddings) by ``chunk_id``.

        Returns the number of chunks actually removed; unknown ids are
        ignored. O(n) over the current index, no rebuild. Keeps the
        ``_chunks`` / ``_embeddings`` parallel lists aligned.
        """
        targets = set(chunk_ids)
        if not targets:
            return 0
        kept_chunks: list[Chunk] = []
        kept_embeddings: list[list[float]] = []
        removed = 0
        for chunk, emb in zip(self._chunks, self._embeddings):
            if chunk.chunk_id in targets:
                removed += 1
            else:
                kept_chunks.append(chunk)
                kept_embeddings.append(emb)
        self._chunks = kept_chunks
        self._embeddings = kept_embeddings
        return removed

    def delete_by_doc(self, doc_id: str) -> int:
        """Remove every chunk belonging to ``doc_id``. Returns the count."""
        return self.delete([c.chunk_id for c in self._chunks if c.doc_id == doc_id])

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        """Replace any existing chunks sharing a ``chunk_id`` with ``chunks``,
        then append. Equivalent to delete-then-add in a single call."""
        self.delete([c.chunk_id for c in chunks if c.chunk_id])
        self.add(chunks)


# ============================================================
# AUTO-DEFAULTS — production-tested MOTHRAG 1 settings
# ============================================================

_DEFAULTS = {
    "top_k_chunks": 10,
    "chunk_size_tokens": 400,
    "chunk_overlap_tokens": 50,
    "use_router": True,
    "embedder_preference": ("gemini-embedding-2", "st-mini", "hash"),
    "reader_preference": ("llama-3.3-70b-together", "echo"),
    # retry-on-abstain cascade. None / [] / "off" disables escalation entirely
    # (terminal-skip behavior). "all" enables the full 7-strategy stack;
    # "sweet_spot" enables #1 + #2 + #4 + #7; a list[str] selects named
    # strategies (soft_fallback is auto-appended as the terminus in loop mode).
    "retry_strategies": "all",
    "retry_budget_limit": 8,
    # Arm pool + execution. ``arms_pool`` = 3 (default, legacy
    # byte-stable v3bu/decompose/iter) or 4 (adds the iter_dup_a PDD arm, a COPY
    # of iter, in the ensemble path). Default stays 3 until the 4-arm pip lift
    # is verified (the pip iter arm is γ-less, so 4-arm changes arbitration
    # without a measured win — opt-in until verified).
    # ``arms_parallel`` (default True) runs the base arms concurrently via a
    # within-query ThreadPoolExecutor (same predictions, faster wall-clock);
    # ``arms_max_workers`` is hard-clamped to the N=4 pool-safety ceiling.
    "arms_pool": 3,
    "arms_parallel": True,
    "arms_max_workers": 4,
    # Deployment mode for the abstention cascade.
    #   "loop" (default): SoftFallback terminal, answer guaranteed non-empty.
    #   "abstention":     terminal abstain allowed (KB-audit / gap-discovery).
    "mode": "loop",
    # Bridge-conditioned arm. Opt-in:
    # default OFF (Haiku judge/SVO/entity calls cost money; needs
    # ANTHROPIC_API_KEY). When ON, a 4th arm runs the tripartite-judge
    # pipeline and competes as an ensemble candidate (primary on bridge_entity
    # / chain_deep cohorts); its top-K bridge-ranked passages are surfaced in
    # QueryResult.metadata for R@5 measurement.
    "use_bridge_arm": False,
    "bridge_judge_model": "claude-haiku-4-5",
    "bridge_max_cost_usd": 10.0,
    # Per-arm gate — accepted for API completeness / forward-compat, but a
    # NO-OP in the pip path: the per-arm bridge gate operates on the bridge
    # retrieval SUBSTRATE, which is EVAL-PATH only (route_prospective.py); pip has
    # no substrate to gate (the pip `use_bridge_arm` is a separate candidate arm).
    # Setting this does not raise and has no effect here. See
    # scripts/route_prospective.py --arm-bridge-qtype-gate for the live feature.
    "arm_bridge_qtype_gate": None,
}


def _resolve_default_embedder() -> Embedder:
    """Try Vertex AI → Gemini Studio → sentence-transformers → hash fallback.

    Resolution order is structured so enterprise GCP customers (Vertex AI,
    GDPR-region projects) take precedence over Studio API keys when both are
    configured. Each step is optional and the chain degrades gracefully.
    """
    if os.environ.get("VERTEX_AI_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT"):
        try:
            from mothrag.embedders import VertexEmbedder
            logger.info("MothRag auto-default embedder: Vertex AI text-embedding-005 "
                        "(Vertex does not yet expose gemini-embedding-2; use Studio for -2)")
            return VertexEmbedder()
        except ImportError:
            logger.debug("VertexEmbedder unavailable (install mothrag[vertex])")
        except Exception as exc:  # noqa: BLE001
            logger.warning("VertexEmbedder init failed (%s); falling through.", exc)
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        try:
            from mothrag.embedders import GeminiEmbedder
            logger.info("MothRag auto-default embedder: Gemini Studio gemini-embedding-2")
            return GeminiEmbedder()
        except ImportError:
            logger.debug("GeminiEmbedder unavailable (install mothrag[gemini])")
    try:
        import sentence_transformers  # noqa: F401
        from mothrag.core._api_adapters import SentenceTransformersEmbedder
        logger.info("MothRag auto-default embedder: sentence-transformers MiniLM-L6")
        return SentenceTransformersEmbedder()
    except ImportError:
        pass
    logger.warning("MothRag auto-default embedder: hash fallback (install mothrag[prod] for production)")
    return _HashEmbedder()


def _resolve_embedder_spec(spec: str) -> Embedder:
    """Resolve a string spec ``"backend"`` or ``"backend:model"`` to an Embedder.

    Supported backends:

    - ``"vertex"``, ``"vertex:text-embedding-005"`` (Vertex does not expose
      ``gemini-embedding-2`` yet; use ``"gemini"`` backend for the -2 model)
    - ``"gemini"``, ``"gemini:gemini-embedding-2"``, ``"gemini:text-embedding-005"``
    - ``"gemini-embedding-2"`` / ``"gemini-2"`` — bare canonical PROD model id
      (shorthand for ``"gemini:gemini-embedding-2"``)
    - ``"openai"``, ``"openai:text-embedding-3-small"``
    - ``"cohere"``, ``"cohere:embed-english-v3.0"``
    - ``"st"``, ``"sentence-transformers"``, ``"st:all-MiniLM-L6-v2"``
    - ``"hash"`` — built-in offline fallback

    All instances are constructed with their adapter defaults; pass an
    instance directly to :class:`MothRAG` for fine-grained config.
    """
    backend, _, model = spec.partition(":")
    backend = backend.strip().lower()
    model = model.strip() or None

    # Accept the bare canonical PROD model id
    # ``gemini-embedding-2`` (alias ``gemini-2``) as a full spec, routing to
    # the Gemini backend. Without this, ``MothRAG(embedder="gemini-embedding-2")``
    # raised "Unknown embedder spec" (partition on ':' made the whole string the
    # backend). Mirrors the corpus-side acceptance in mothrag/eval/pipeline.py
    # and the canonical PROD name. Callers no longer need
    # the ``--embedding gemini`` workaround.
    if model is None and backend in ("gemini-embedding-2", "gemini-2"):
        from mothrag.embedders import GeminiEmbedder
        return GeminiEmbedder(model="gemini-embedding-2")

    if backend == "vertex":
        from mothrag.embedders import VertexEmbedder
        return VertexEmbedder(model=model) if model else VertexEmbedder()
    if backend == "gemini":
        from mothrag.embedders import GeminiEmbedder
        return GeminiEmbedder(model=model) if model else GeminiEmbedder()
    if backend == "openai":
        from mothrag.embedders import OpenAIEmbedder
        return OpenAIEmbedder(model=model) if model else OpenAIEmbedder()
    if backend == "cohere":
        from mothrag.embedders import CohereEmbedder
        return CohereEmbedder(model=model) if model else CohereEmbedder()
    if backend in ("st", "sentence-transformers", "sentencetransformers"):
        from mothrag.embedders import SentenceTransformersEmbedder
        return SentenceTransformersEmbedder(model=model) if model else SentenceTransformersEmbedder()
    if backend == "hash":
        return _HashEmbedder()
    raise ValueError(
        f"Unknown embedder spec: {spec!r}. "
        f"Use 'vertex'|'gemini'|'openai'|'cohere'|'st'|'hash', optionally suffixed ':model'."
    )


def _resolve_default_reader() -> Reader:
    """Try Llama Together / Groq → echo fallback.

    Both echo fallbacks are LOUD and name their own fix: a silent fallback
    here means a user reads a chunk-echo as an LLM answer (the 0.6.0 trap).
    """
    api_key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("GROQ_API_KEY")
    if api_key:
        try:
            from mothrag.core._api_adapters import OpenAICompatibleReader
            base_url = "https://api.together.xyz/v1" if os.environ.get("TOGETHER_API_KEY") else "https://api.groq.com/openai/v1"
            model = "meta-llama/Llama-3.3-70B-Instruct-Turbo" if os.environ.get("TOGETHER_API_KEY") else "llama-3.3-70b-versatile"
            logger.info("MothRag auto-default reader: %s @ %s", model, base_url)
            return OpenAICompatibleReader(model=model, base_url=base_url, api_key=api_key)
        except ImportError:
            key_name = "TOGETHER_API_KEY" if os.environ.get("TOGETHER_API_KEY") else "GROQ_API_KEY"
            logger.warning(
                "MothRag: %s is set but the reader SDK is not installed, so answers "
                "will be the echoed top chunk (NO LLM call). Fix: pip install 'mothrag[openai]'",
                key_name,
            )
            return _EchoReader()
    logger.warning(
        "MothRag auto-default reader: echo fallback -- answers are the echoed top "
        "chunk, NOT LLM-generated (set TOGETHER_API_KEY or GROQ_API_KEY and "
        "pip install 'mothrag[openai]' for production)")
    return _EchoReader()


# ============================================================
# CHUNKING — sentence + size based
# ============================================================

def _simple_chunk(text: str, *, chunk_size_tokens: int = 400,
                  chunk_overlap_tokens: int = 50) -> list[str]:
    """Sentence-aware fixed-size chunker.

    Splits on sentence boundaries (.!?). Greedily packs sentences into
    chunks of ~chunk_size_tokens whitespace tokens with chunk_overlap_tokens
    overlap between adjacent chunks. Tokens = whitespace-separated runs
    (approximate, not GPT-tokenizer; good enough for chunking heuristics).
    """
    if not text.strip():
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for sent in sentences:
        s_tokens = len(sent.split())
        if cur_tokens + s_tokens > chunk_size_tokens and cur:
            chunks.append(" ".join(cur))
            # Overlap: keep the last N tokens worth of sentences.
            overlap: list[str] = []
            o_tokens = 0
            for back in reversed(cur):
                b_tokens = len(back.split())
                if o_tokens + b_tokens > chunk_overlap_tokens:
                    break
                overlap.insert(0, back)
                o_tokens += b_tokens
            cur = overlap
            cur_tokens = o_tokens
        cur.append(sent)
        cur_tokens += s_tokens
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ============================================================
# PUBLIC API — the MothRAG class
# ============================================================

class MothRAG:
    """High-level MothRAG entry point — adaptive multi-arm subset RAG.

    Auto-defaults reproduce the production stack (MOTHRAG 1)
    when API keys / extras are available; degrade gracefully to offline
    fallbacks when not. All three backends (embedder, reader, vector_db)
    are pluggable via constructor args.

    Examples
    --------
    Minimal — auto everything::

        from mothrag import MothRAG
        rag = MothRAG.from_documents(["First doc text.", "Second doc text."])
        print(rag.query("What is mentioned?"))

    Custom reader::

        from mothrag import MothRAG
        from mothrag.core.api import _EchoReader  # or your own
        rag = MothRAG.from_documents("docs/", reader=_EchoReader())

    Hybrid graph retrieval (HippoRAG2 backend)::

        rag = MothRAG.from_documents(
            "docs/", production=True,
            retrieval="hybrid_graph",
            retrieval_config={"save_dir": ".hipporag-cache"},
        )

    """

    # Recognised values for the ``mode`` parameter.
    SUPPORTED_MODES = ("adaptive", "ensemble_arbitrate")
    # Recognised values for the ``retrieval`` keyword.
    SUPPORTED_RETRIEVAL = ("dense", "hybrid_graph", "dense_plus_infobox")

    def __init__(
        self,
        embedder: Embedder | str | None = None,
        reader: Reader | None = None,
        vector_db: VectorStore | None = None,
        *,
        production: bool = False,
        mode: str = "adaptive",
        retry_mode: str = "loop",
        retrieval: str = "dense",
        retriever: Any = None,
        retrieval_config: dict[str, Any] | None = None,
        **config: Any,
    ) -> None:
        """Construct a MothRAG instance.

        Parameters
        ----------
        embedder, reader, vector_db
            Pluggable backends. ``None`` → auto-default chain (Vertex AI →
            Gemini Studio → sentence-transformers → hash for embedder; Llama
            via Together / Groq → echo for reader; in-memory cosine vector
            store).
            ``embedder`` also accepts a string spec like ``"vertex"``,
            ``"vertex:text-embedding-005"``, ``"gemini"``, ``"openai"``,
            ``"st:all-MiniLM-L6-v2"`` — dispatched through
            :func:`_resolve_embedder_spec`. Instances bypass the dispatcher.
        production
            If False (default), ``query()`` runs a single-arm read on the
            top-K retrieved passages (LlamaIndex-style minimal path).
            If True, ``query()`` runs the full MOTHRAG 1 multi-arm
            orchestration whose exact form is selected by the ``mode``
            argument below. Production mode requires a real LLM reader
            (echo fallback works for smoke but produces low-quality
            decompose/iter outputs).
        mode
            Production-mode arm-selection strategy. Only consulted when
            ``production=True``.

            - ``"adaptive"`` (default): MOTHRAG 1 adaptive routing.
              ``arm_subset(question)`` picks the subset of arms to run;
              each arm runs on the shared passages; ``arbitrate_with_c7``
              (or ``arbitrate_excl_v3bu`` when V3+bu is excluded) selects
              the final answer. Same routing decision as
              ``scripts/route_prospective.py`` and
              ``scripts/arbitrate_post.py --auto-arm-subset --use-router-v2``.
            - ``"ensemble_arbitrate"``: always run all three arms
              (V3+bu + decompose + iter), then select via the
              :class:`mothrag.core.arbitrate.DeterministicArbitrator`
              with default weights (gamma=1.0, agree=0.5, faith=0.3).
              Higher per-query compute than adaptive (always-three-arms),
              but skips the routing classifier; recovery is folded into
              the deterministic post-hoc score.
        retry_mode
            Retry-cascade terminal behaviour. Only consulted when the
            retry-on-abstain orchestrator fires.

            - ``"loop"`` (default): SoftFallback terminal, ``qr.answer``
              guaranteed non-empty (customer-facing production).
            - ``"abstention"``: terminal abstain allowed
              (``qr.answer == ""`` + ``terminal_abstain=True``) for
              KB-audit / gap-discovery deployments.
        retrieval
            String dispatcher for the retrieval base. ``"dense"`` (default)
            preserves v0.5.0 alpha behaviour: the configured ``embedder`` +
            ``vector_db`` pair, wrapped by
            :class:`mothrag.core.retrieval.DenseRetriever`. ``"hybrid_graph"``
            constructs a :class:`mothrag.core.retrieval.HybridGraphRetriever`
            that wraps the OSU-NLP-Group HippoRAG2 SDK (requires
            ``pip install mothrag[hybrid-graph]``).
            ``"dense_plus_infobox"`` blends the dense retriever with an
            :class:`mothrag.core.retrieval.InfoboxIndex` of
            ``(subject, attribute, value)`` triples harvested from the
            corpus at ingest time, surfacing high-precision structured
            matches for entity-attribute questions (e.g. "When was X
            born?", "Who is Y's spouse?"). The arms (V3+bu / decompose
            / iter) and arbitration logic are unchanged; only the
            retrieval base differs.
        retriever
            Optional explicit :class:`mothrag.core.retrieval.Retriever`
            instance. When supplied, ``retrieval`` is ignored and the
            instance is used verbatim (useful for tests + custom backends).
        retrieval_config
            Free-form kwargs forwarded to the retrieval-string dispatcher.
            E.g. ``{"save_dir": "/path/to/hipporag-cache",
            "embedding_model_name": "gemini-embedding-2"}`` for
            ``retrieval="hybrid_graph"``.
        config
            Free-form overrides: ``top_k_chunks``, ``chunk_size_tokens``,
            ``chunk_overlap_tokens``, ``use_router``, ``max_iter_steps``,
            ``decompose_max_subq``,
            ``arbitrate_weights={"gamma": ..., "agree": ..., "faith": ...}``,
            ``arbitrate_agreement_threshold`` (default 0.70),
            ``retry_strategies``, ``retry_budget_limit``.
        """
        if isinstance(embedder, str):
            embedder = _resolve_embedder_spec(embedder)
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"MothRAG mode must be one of {self.SUPPORTED_MODES}, got {mode!r}."
            )
        if retry_mode not in ("loop", "abstention"):
            raise ValueError(
                f"MothRAG retry_mode must be 'loop' or 'abstention', got {retry_mode!r}."
            )
        if retrieval not in self.SUPPORTED_RETRIEVAL:
            raise ValueError(
                f"MothRAG retrieval must be one of {self.SUPPORTED_RETRIEVAL}, "
                f"got {retrieval!r}."
            )
        self.embedder: Embedder = embedder or _resolve_default_embedder()
        self.reader: Reader = reader or _resolve_default_reader()
        # Identity check, not ``or``: an empty custom store has len 0 and is
        # falsy, so ``vector_db or _MemoryVectorStore()`` would silently
        # discard a freshly-constructed (still empty) injected store.
        self.vector_db: VectorStore = (
            vector_db if vector_db is not None else _MemoryVectorStore())
        self.production: bool = production
        self.mode: str = mode
        self.retry_mode: str = retry_mode
        self.retrieval: str = retrieval
        self.config: dict[str, Any] = {**_DEFAULTS, **config}
        # Constructor kwargs win over the same keys passed via **config.
        self.config["mode"] = mode
        self.config["retry_mode"] = retry_mode
        self.config["retrieval"] = retrieval
        self.config.setdefault("max_iter_steps", 3)
        self.config.setdefault("decompose_max_subq", 4)
        self.retriever = self._build_retriever(
            retrieval=retrieval,
            retriever=retriever,
            retrieval_config=retrieval_config or {},
        )

    def _build_retriever(
        self, *, retrieval: str, retriever: Any, retrieval_config: dict[str, Any],
    ):
        """Resolve the active :class:`mothrag.core.retrieval.Retriever`.

        Order of precedence:
          1. Explicit ``retriever`` instance wins.
          2. ``retrieval="hybrid_graph"`` → build a
             :class:`HybridGraphRetriever` with ``retrieval_config``
             forwarded as kwargs.
          3. ``retrieval="dense_plus_infobox"`` → wrap a
             :class:`DenseRetriever` in a :class:`MultiModalRetriever`
             with an :class:`InfoboxIndex` that's populated incrementally
             at ingest time. ``retrieval_config`` may set
             ``infobox_top_n_boost`` and ``hint_extractor``.
          4. ``retrieval="dense"`` (default) → wrap the configured
             ``embedder`` + ``vector_db`` in a :class:`DenseRetriever`.
        """
        if retriever is not None:
            return retriever
        if retrieval == "hybrid_graph":
            from mothrag.core.retrieval import HybridGraphRetriever
            kwargs = {
                "embedder": self.embedder,
                "reader_llm": self.reader,
            }
            kwargs.update(retrieval_config)
            return HybridGraphRetriever(**kwargs)
        if retrieval == "dense_plus_infobox":
            from mothrag.core.retrieval import (
                DenseRetriever, InfoboxIndex, MultiModalRetriever,
            )
            dense = DenseRetriever(embedder=self.embedder, vector_db=self.vector_db)
            self._infobox_index = InfoboxIndex()
            # Optional pre-fed triples (external KG sources).
            for t in retrieval_config.get("seed_triples", ()):
                self._infobox_index.add(t)
            # Chunk provider: hydrate triples to their source chunks via
            # the in-memory vector_db keyed lookup.
            def _chunk_provider(chunk_id: str):
                for c in getattr(self.vector_db, "_chunks", []):
                    if getattr(c, "chunk_id", "") == chunk_id:
                        return c
                return None
            return MultiModalRetriever(
                dense=dense,
                infobox_index=self._infobox_index,
                chunk_provider=_chunk_provider,
                infobox_top_n_boost=int(
                    retrieval_config.get("infobox_top_n_boost", 3)
                ),
                hint_extractor=retrieval_config.get("hint_extractor"),
            )
        # default: dense
        from mothrag.core.retrieval import DenseRetriever
        return DenseRetriever(embedder=self.embedder, vector_db=self.vector_db)

    # ------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------

    @classmethod
    def from_documents(
        cls,
        source: str | Path | Sequence[str | Document],
        *,
        embedder: Embedder | str | None = None,
        reader: Reader | None = None,
        vector_db: VectorStore | None = None,
        production: bool = False,
        mode: str = "adaptive",
        retry_mode: str = "loop",
        retrieval: str = "dense",
        retriever: Any = None,
        retrieval_config: dict[str, Any] | None = None,
        **config: Any,
    ) -> "MothRAG":
        """Construct a MothRAG instance and ingest documents in one call.

        Parameters
        ----------
        source : str | Path | list[str] | list[Document]
            - ``str | Path``: filesystem path (file or directory). Uses
              :func:`mothrag.loaders.auto_load` to dispatch loaders.
            - ``list[str]``: raw text snippets — each becomes a Document.
            - ``list[Document]``: already-prepared Documents.
        embedder, reader, vector_db : optional pluggable backends.
        production : route through the full MOTHRAG 1 adaptive multi-arm
            subset orchestration (see :class:`MothRAG` for details).
        **config : extra config (top_k_chunks, chunk_size_tokens, ...).
        """
        rag = cls(
            embedder=embedder, reader=reader, vector_db=vector_db,
            production=production,
            mode=mode, retry_mode=retry_mode,
            retrieval=retrieval, retriever=retriever,
            retrieval_config=retrieval_config,
            **config,
        )
        rag.ingest(source)
        return rag

    def ingest(self, source: str | Path | Sequence[str | Document]) -> None:
        """Add documents to the existing index (incremental ingestion).

        Behaviour depends on ``self.retrieval``:

        - ``"dense"`` (default): chunks are embedded via ``self.embedder``
          and added to ``self.vector_db`` via the
          :class:`DenseRetriever` adapter (identical to the v0.5.0 alpha
          path).
        - ``"hybrid_graph"``: chunks are passed to
          :meth:`HybridGraphRetriever.index`, which delegates to the
          HippoRAG2 SDK. Chunk embeddings are still pre-computed for
          potential reuse by retrieval-side fusion logic but are not
          required by HippoRAG2 (it manages its own embeddings).
        - ``"dense_plus_infobox"``: chunks flow through the dense path
          AND are scanned by :func:`extract_wikitext_infobox` +
          :func:`extract_natural_facts` to populate the
          :class:`InfoboxIndex` attached to the
          :class:`MultiModalRetriever`. Both extractors are best-effort;
          they emit zero triples on prose chunks without recognisable
          structured patterns.
        """
        docs = self._normalize_source(source)
        chunks: list[Chunk] = []
        for doc in docs:
            doc_chunks = _simple_chunk(
                doc.text,
                chunk_size_tokens=self.config["chunk_size_tokens"],
                chunk_overlap_tokens=self.config["chunk_overlap_tokens"],
            )
            for i, ctext in enumerate(doc_chunks):
                chunks.append(Chunk(
                    text=ctext,
                    doc_id=doc.doc_id,
                    chunk_id=f"{doc.doc_id}#chunk{i}",
                    metadata=dict(doc.metadata),
                ))
        if not chunks:
            logger.warning("MothRag.ingest received 0 chunks (empty source).")
            return

        # Pre-compute embeddings for downstream reuse (dense retrieval +
        # cross-arm agreement signals etc.). For hybrid_graph the embeddings
        # are recomputed inside HippoRAG2; the duplication is intentional
        # to keep the chunk objects self-contained.
        embs = self.embedder.embed_batch([c.text for c in chunks])
        for c, e in zip(chunks, embs):
            c.embedding = list(e)

        self.retriever.index(chunks)

        # Incremental InfoboxIndex build for the multi-modal retriever.
        # Same extractors that build_infobox_index_from_chunks calls, but
        # threaded inline so the index updates incrementally across
        # multiple ingest() calls without needing a corpus rebuild.
        if self.retrieval == "dense_plus_infobox":
            from mothrag.core.retrieval.infobox import (
                extract_natural_facts, extract_wikitext_infobox,
            )
            for c in chunks:
                self._infobox_index.add_many(
                    extract_wikitext_infobox(
                        c.text, source_chunk_id=c.chunk_id,
                    )
                )
                self._infobox_index.add_many(
                    extract_natural_facts(
                        c.text, source_chunk_id=c.chunk_id,
                    )
                )

        logger.info(
            "MothRag ingested %d chunks from %d docs via %s retriever (index size now %d).",
            len(chunks), len(docs), self.retrieval, len(self.retriever),
        )

    def delete(self, doc_id: str) -> int:
        """Remove every chunk belonging to ``doc_id`` from the index.

        Incremental retraction of a source document: no index rebuild, no
        graph reconstruction, no retraining. Returns the number of chunks
        removed. Requires ``retrieval='dense'`` and a mutable vector store
        (the default in-memory store qualifies).
        """
        self._require_mutable("delete")
        removed = self.vector_db.delete_by_doc(doc_id)
        logger.info(
            "MothRag deleted %d chunks for doc_id=%r (index size now %d).",
            removed, doc_id, len(self.vector_db),
        )
        return removed

    def update(self, doc_id: str, text: str,
               metadata: dict[str, Any] | None = None) -> int:
        """Replace the content stored under ``doc_id`` with ``text``.

        The superseding-fact operation: the old chunks for ``doc_id`` are
        removed and ``text`` is re-chunked, embedded and appended, so the
        next query reasons over the new content. Incremental: one embedding
        pass over the changed document, no index rebuild, no retraining.

        Returns the number of stale chunks removed (0 if ``doc_id`` was not
        present, in which case this acts as an insert).
        """
        self._require_mutable("update")
        removed = self.delete(doc_id)
        meta = dict(metadata or {})
        meta["source"] = doc_id
        self.ingest([Document(text=text, metadata=meta)])
        return removed

    def _require_mutable(self, op: str) -> None:
        """Guard for :meth:`delete` / :meth:`update`: dense path plus a store
        that supports mutation. Raises a clear error otherwise."""
        if self.retrieval != "dense":
            raise NotImplementedError(
                f"MothRag.{op}() currently supports retrieval='dense'; "
                f"retrieval={self.retrieval!r} maintains extra index state "
                f"(graph / infobox) that incremental {op} does not yet reconcile."
            )
        if not isinstance(self.vector_db, MutableVectorStore):
            raise NotImplementedError(
                f"MothRag.{op}() needs a mutable vector store exposing "
                f"delete()/upsert(); the active store "
                f"{type(self.vector_db).__name__} is append-only."
            )

    @staticmethod
    def _normalize_source(source: str | Path | Sequence[str | Document]) -> list[Document]:
        if isinstance(source, (str, Path)) and (isinstance(source, Path) or "/" in str(source) or "\\" in str(source) or Path(source).exists()):
            from mothrag.loaders import auto_load
            return auto_load(source)
        if isinstance(source, str):
            return [Document(text=source)]
        out: list[Document] = []
        for item in source:
            if isinstance(item, Document):
                out.append(item)
            elif isinstance(item, str):
                out.append(Document(text=item))
            else:
                raise TypeError(f"Unsupported source item: {type(item).__name__}")
        return out

    # ------------------------------------------------------------
    # Query (sync + async + batch)
    # ------------------------------------------------------------

    def query(self, question: str, **kwargs: Any) -> QueryResult:
        """End-to-end query.

        Dispatches to ``_query_production`` when the instance was constructed
        with ``production=True`` (full MOTHRAG 1 adaptive multi-arm subset
        orchestration), else to ``_query_single`` (single-arm read on the
        top-K retrieved passages — the v0.5.0 alpha minimal path).
        """
        if self.production:
            return self._query_production(question, **kwargs)
        return self._query_single(question, **kwargs)

    # ------------------------------------------------------------
    # Single-arm path (production=False)
    # ------------------------------------------------------------

    def _query_single(self, question: str, **kwargs: Any) -> QueryResult:
        """Single-arm read on top-K retrieved passages (v0.5.0 minimal path)."""
        top_k = int(kwargs.get("top_k", self.config["top_k_chunks"]))
        gold_doc_ids = kwargs.get("gold_doc_ids")
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        passages = [c.text for c in retrieved]
        arm_subset_list: list[str] = []
        arm_used = "default"
        if self.config.get("use_router", True):
            try:
                from mothrag.core.query_type_classifier import arm_subset
                arm_subset_list = arm_subset(question)
                arm_used = arm_subset_list[0] if arm_subset_list else "default"
            except Exception as exc:  # noqa: BLE001
                logger.debug("arm_subset routing skipped: %s", exc)
                arm_subset_list = []
        try:
            answer = self.reader.read(question, passages)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Reader failed: %s", exc)
            answer = ""
        meta = {
            "top_k": top_k, "n_passages": len(passages),
            "mode": "single", "retrieval": self.retrieval,
        }
        _attach_recall_at_k(meta, retrieved, gold_doc_ids, top_k)
        return QueryResult(
            answer=answer,
            retrieved_chunks=list(retrieved),
            arm_used=arm_used,
            arm_subset=arm_subset_list,
            metadata=meta,
        )

    # ------------------------------------------------------------
    # Production path (production=True) — adaptive multi-arm subset
    # ------------------------------------------------------------

    def _query_production(self, question: str, **kwargs: Any) -> QueryResult:
        """Full MOTHRAG 1 multi-arm orchestration; dispatches on ``self.mode``.

        ``mode="adaptive"`` (default) -- :meth:`_query_production_adaptive`:
          1. arm_subset(question) decides which subset of arms to run.
          2. Run each selected arm on the corpus (V3+bu / decompose / iter).
          3. arbitrate_with_c7 (or arbitrate_excl_v3bu when V3+bu excluded)
             selects the final answer from the running-arm predictions.

        ``mode="ensemble_arbitrate"`` -- :meth:`_query_production_ensemble`:
          1. Always run all three arms (V3+bu + decompose + iter) on the
             shared retrieved passages -- no arm_subset gating.
          2. Compute per-arm signals (gamma_valid, cross_arm_agreement,
             faith) where available; defaults fill in for missing signals.
          3. :class:`DeterministicArbitrator` picks the highest-scoring
             arm under the configured weights.
        """
        if self.mode == "ensemble_arbitrate":
            return self._query_production_ensemble(question, **kwargs)
        return self._query_production_adaptive(question, **kwargs)

    def _query_production_adaptive(self, question: str, **kwargs: Any) -> QueryResult:
        """Adaptive multi-arm subset orchestration (MOTHRAG 1 default)."""
        from mothrag.core.query_type_classifier import arm_subset as _arm_subset
        from mothrag.core.selective_ensemble import (
            arbitrate_with_c7, arbitrate_excl_v3bu,
        )

        top_k = int(kwargs.get("top_k", self.config["top_k_chunks"]))
        # Shared retrieval via the configured retriever (dense or hybrid_graph).
        # Each arm consumes these passages identically; the iter arm may
        # re-retrieve internally via self.retriever for augmented contexts.
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        passages = [c.text for c in retrieved]
        # Pre-compute q_emb for arms that need it directly (iter accumulator).
        q_emb = self.embedder.embed_batch([question])[0]

        subset = _arm_subset(question)
        v3bu_in = "v3bu" in subset

        v3bu_pred: str | None = None
        dec_pred: str | None = None
        iter_pred: str | None = None
        if v3bu_in:
            v3bu_pred = self._arm_v3bu(question, passages)
        dec_pred = self._arm_decompose(question, passages)
        iter_pred = self._arm_iter(question, passages, q_emb=q_emb, top_k=top_k)

        # Opt-in bridge arm (default OFF). Produces a 4th candidate +
        # a bridge-ranked passage list (surfaced in meta for R@5 measurement).
        bridge_pred: str | None = None
        bridge_pids: list[str] | None = None
        if self.config.get("use_bridge_arm"):
            bridge_pred, bridge_pids = self._arm_bridge(
                question, passages, q_emb=q_emb, top_k=top_k)

        # Arbitrate using existing logic.
        # Aurora L6 C7 phase-cancellation is promoted to PROD. It was once
        # deferred under a "use_c7=False" comment; the reference baselines
        # (HP 0.7770 / 2W 0.7379 / MQ 0.4298) use C7 ON, so a pip-install with
        # it OFF was measuring a different system. Now default ON with
        # c7_trigger="gated". c7_embedder defaults to the constructor-resolved
        # embedder (gemini-embedding-2 via the Studio API).
        if v3bu_in:
            chosen, reason, c7_info = arbitrate_with_c7(
                v3bu_pred=v3bu_pred or "",
                dec_pred=dec_pred or "",
                question=question,
                iter_pred=iter_pred,
                use_c7=True,                    # promoted to PROD
                c7_trigger="gated",
                embedder=self.embedder,
                query_embed=q_emb,
                use_router_v2=True,
                bridge_pred=bridge_pred,        # opt-in (None when OFF)
            )
        else:
            chosen, reason = arbitrate_excl_v3bu(
                dec_pred=dec_pred or "",
                iter_pred=iter_pred,
                question=question,
                v3bu_fallback=None,
            )
            c7_info = None

        # Populate alpha-pipeline L4b state so the
        # ``l4b_anchor_retry`` strategy can fire instead of silently no-opping.
        # Anchors = positional indices into the shared ``passages`` list (top-k
        # by cosine). ``cancelled`` is True when the iter arm returned empty —
        # the substitute for the full L4b temporal-cancellation signal.
        c7_info = c7_info if isinstance(c7_info, dict) else {}
        c7_info["l4b"] = {
            "cancelled": not bool((iter_pred or "").strip()),
            "anchors": list(range(len(passages))),
            "alpha_substitute": True,
        }

        meta: dict[str, Any] = {
            "top_k": top_k,
            "n_passages": len(passages),
            "mode": self.mode,
            "production_strategy": "adaptive",
            "retrieval": self.retrieval,
            "v3bu_pred": v3bu_pred,
            "dec_pred": dec_pred,
            "iter_pred": iter_pred,
            "arbitrate_reason": reason,
            "c7_info": c7_info,
            "bridge_pred": bridge_pred,
            "bridge_ranked_passage_ids": bridge_pids,
        }

        # retry-on-abstain escalation cascade. Only fires when arbitration
        # produced an abstention signal AND the user has not opted out via
        # retry_strategies=None / [] / "off".
        signal = _detect_abstention_signal(chosen, reason, c7_info)
        chosen_final, escalation_meta = self._maybe_escalate(
            question=question, passages=passages, q_emb=q_emb, top_k=top_k,
            subset=subset, v3bu_pred=v3bu_pred, dec_pred=dec_pred, iter_pred=iter_pred,
            chosen=chosen, arbitrate_reason=reason, c7_info=c7_info, signal=signal,
        )
        meta.update(escalation_meta)

        _attach_recall_at_k(meta, retrieved, kwargs.get("gold_doc_ids"), top_k)
        return QueryResult(
            answer=chosen_final,
            retrieved_chunks=list(retrieved),
            arm_used=reason,
            arm_subset=subset,
            metadata=meta,
        )

    def _query_production_ensemble(self, question: str, **kwargs: Any) -> QueryResult:
        """Ensemble-arbitrate path: adaptive subset + arbitrate-when-multi.

        Preserves the Pareto-dominant adaptive routing:
        :func:`arm_subset` picks which arms to run (1, 2, or 3 of
        v3bu / decompose / iter); only the selected arms execute.
        DeterministicArbitrator is then applied *only when the subset
        has size >= 2* -- a single-arm subset returns its arm's answer
        directly with no arbitrate overhead, exactly matching the
        adaptive path's behaviour in that regime.

        The arbitrator's value-add is therefore strictly when the routing
        classifier elects multiple arms; ensemble_arbitrate does NOT
        force always-three-arms and does NOT override the routing
        decision.
        """
        from mothrag.core.arbitrate import DeterministicArbitrator, pairwise_agreement
        from mothrag.core.query_type_classifier import arm_subset as _arm_subset

        top_k = int(kwargs.get("top_k", self.config["top_k_chunks"]))
        q_emb = self.embedder.embed_batch([question])[0]
        retrieved = self.vector_db.retrieve(q_emb, top_k=top_k)
        passages = [c.text for c in retrieved]

        # Adaptive subset routing -- IDENTICAL to the adaptive path.
        subset = _arm_subset(question)
        v3bu_in = "v3bu" in subset

        # Dispatch the base arms through the unified arms_runner:
        # parallel by default (same predictions, faster wall-clock; safe per
        # workflow w9u5xu1oo) with an --arms-serial / arms_parallel=False escape.
        # The decompose + iter arms run unconditionally (mirroring the adaptive
        # path); v3bu only when in the subset. iter_dup_a (PDD 4th arm) is a COPY
        # of iter, added ONLY when arms_pool == 4 (opt-in; default 3 = legacy
        # byte-stable). The arbitrator below is pool-size-agnostic.
        from mothrag.core.arms_runner import ArmSpec, run_arms

        specs: list[ArmSpec[str]] = []
        if v3bu_in:
            specs.append(ArmSpec("v3bu",
                                 fn=lambda: self._arm_v3bu(question, passages)))
        specs.append(ArmSpec("decompose",
                             fn=lambda: self._arm_decompose(question, passages)))
        specs.append(ArmSpec("iter",
                             fn=lambda: self._arm_iter(question, passages,
                                                       q_emb=q_emb, top_k=top_k)))
        if int(self.config.get("arms_pool", 3)) >= 4:
            specs.append(ArmSpec("iter_dup_a", fn=None, is_dup=True, dup_of="iter"))

        ran = run_arms(
            specs,
            parallel=bool(self.config.get("arms_parallel", True)),
            max_workers=int(self.config.get("arms_max_workers", 4)),
        )  # copy_fn=identity: pip arm results are immutable str.

        v3bu_pred: str | None = ran.get("v3bu")
        dec_pred: str | None = ran.get("decompose")
        iter_pred: str | None = ran.get("iter")

        # Collect only the arms that produced a candidate answer (skip None,
        # preserving the prior ``is not None`` semantics).
        candidates: dict[str, str] = {
            name: (pred or "") for name, pred in ran.items() if pred is not None
        }

        # ---- Single-arm subset: direct pass-through (no arbitrate). ----
        if len(candidates) <= 1:
            sole_name, sole_pred = (
                next(iter(candidates.items())) if candidates else ("", "")
            )
            meta: dict[str, Any] = {
                "top_k": top_k,
                "n_passages": len(passages),
                "mode": self.mode,
                "production_strategy": "ensemble_arbitrate",
                "v3bu_pred": v3bu_pred,
                "dec_pred": dec_pred,
                "iter_pred": iter_pred,
                "arm_scores": {sole_name: 0.0} if sole_name else {},
                "selected_arm": sole_name,
                "arbitrate_signal": "single_arm_passthrough",
                "arbitrate_breakdown": {},
                "arbitrate_weights": DeterministicArbitrator().weights,
                "arbitrate_agreement": {sole_name: 0.0} if sole_name else {},
                "subset_size": len(candidates),
            }
            # Compose Path B (single-arm pass-through) with Path C (retry).
            signal = _detect_abstention_signal(
                sole_pred, "single_arm_passthrough", None,
            )
            chosen_final, escalation_meta = self._maybe_escalate(
                question=question, passages=passages, q_emb=q_emb, top_k=top_k,
                subset=list(subset),
                v3bu_pred=v3bu_pred, dec_pred=dec_pred, iter_pred=iter_pred,
                chosen=sole_pred,
                arbitrate_reason="ensemble_arbitrate:single_arm_passthrough",
                c7_info={"arbitrate_signal": "single_arm_passthrough"},
                signal=signal,
            )
            meta.update(escalation_meta)
            _attach_recall_at_k(meta, retrieved, kwargs.get("gold_doc_ids"), top_k)
            return QueryResult(
                answer=chosen_final,
                retrieved_chunks=list(retrieved),
                arm_used=sole_name,
                arm_subset=list(subset),
                metadata=meta,
            )

        # ---- Multi-arm subset: arbitrate over running arms only. ----
        # Cross-arm agreement is computed within the subset (not over the
        # global v3bu/decompose/iter triple) so spectral semantics line up
        # with the actually-executed arms.
        threshold = float(self.config.get("arbitrate_agreement_threshold", 0.70))
        try:
            agreement = pairwise_agreement(
                candidates, embedder=self.embedder, threshold=threshold,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("pairwise_agreement failed (%s); using zeros.", exc)
            agreement = {k: 0.0 for k in candidates}

        gamma_signals = kwargs.get("gamma_signals") or {}
        faith_signals = kwargs.get("faith_signals") or {}

        weight_overrides = self.config.get("arbitrate_weights") or {}
        arbitrator = DeterministicArbitrator(
            w_gamma=float(weight_overrides.get("gamma", 1.0)),
            w_agree=float(weight_overrides.get("agree", 0.5)),
            w_faith=float(weight_overrides.get("faith", 0.3)),
        )
        result = arbitrator.arbitrate(
            answers=candidates,
            gamma_signals=gamma_signals,
            agreement_signals=agreement,
            faith_signals=faith_signals,
        )

        meta: dict[str, Any] = {
            "top_k": top_k,
            "n_passages": len(passages),
            "mode": self.mode,
            "production_strategy": "ensemble_arbitrate",
            "v3bu_pred": v3bu_pred,
            "dec_pred": dec_pred,
            "iter_pred": iter_pred,
            "arm_scores": result.arm_scores,
            "selected_arm": result.selected_arm,
            "arbitrate_signal": result.arbitrate_signal,
            "arbitrate_breakdown": result.component_breakdown,
            "arbitrate_weights": result.weights_used,
            "arbitrate_agreement": agreement,
            "subset_size": len(candidates),
        }

        # Compose Path B (ensemble arbitrate) with Path C (retry-on-abstain
        # orchestrator). The arbitrate_signal "fallback" -- and any
        # uncertainty-template chosen answer -- triggers the retry cascade
        # via the same _maybe_escalate path used by the adaptive route.
        signal = _detect_abstention_signal(
            result.answer, result.arbitrate_signal, None,
        )
        chosen_final, escalation_meta = self._maybe_escalate(
            question=question, passages=passages, q_emb=q_emb, top_k=top_k,
            subset=list(subset),
            v3bu_pred=v3bu_pred, dec_pred=dec_pred, iter_pred=iter_pred,
            chosen=result.answer,
            arbitrate_reason=f"ensemble_arbitrate:{result.arbitrate_signal}",
            c7_info={"arbitrate_signal": result.arbitrate_signal},
            signal=signal,
        )
        meta.update(escalation_meta)

        _attach_recall_at_k(meta, retrieved, kwargs.get("gold_doc_ids"), top_k)
        return QueryResult(
            answer=chosen_final,
            retrieved_chunks=list(retrieved),
            arm_used=result.selected_arm,
            arm_subset=list(subset),
            metadata=meta,
        )

    def _maybe_escalate(
        self,
        *,
        question: str, passages: list[str], q_emb: list[float], top_k: int,
        subset: list[str], v3bu_pred: str | None, dec_pred: str | None,
        iter_pred: str | None,
        chosen: str, arbitrate_reason: str, c7_info: Any, signal: str | None,
    ) -> tuple[str, dict[str, Any]]:
        """Build a RetryContext and run the EscalationOrchestrator.

        Returns ``(final_answer, escalation_metadata)``. Falls through to
        ``(chosen, {})`` (no escalation) when the cascade is disabled or no
        abstention signal was raised.
        """
        preset = self.config.get("retry_strategies")
        if not preset or preset in ("off", "disabled", "none"):
            # retry_mode="abstention" with retry disabled means the original
            # chosen is preserved (no recovery attempted, no terminal abstain) --
            # the user explicitly opted out of escalation.
            return chosen, {
                "retry_mode": self.retry_mode,
                "escalation_applied": [],
                "escalation_recovered_by": None,
                "original_abstention_signal": signal,
                "final_answer_confidence": "high" if signal is None else "low_no_retry",
                "terminal_abstain": False,
            }
        if signal is None:
            return chosen, {
                "retry_mode": self.retry_mode,
                "escalation_applied": [],
                "escalation_recovered_by": None,
                "original_abstention_signal": None,
                "final_answer_confidence": "high",
                "terminal_abstain": False,
            }

        from mothrag.core.retry import (
            RetryContext, build_default_orchestrator,
        )

        ctx = RetryContext(
            question=question,
            passages=list(passages),
            q_emb=list(q_emb),
            top_k=top_k,
            arm_subset=list(subset),
            v3bu_pred=v3bu_pred,
            dec_pred=dec_pred,
            iter_pred=iter_pred,
            chosen=chosen,
            arbitrate_reason=arbitrate_reason,
            c7_info=c7_info,
            abstention_signal=signal,
            budget_limit=int(self.config.get("retry_budget_limit", 8)),
            embedder=self.embedder,
            reader=self.reader,
            vector_db=self.vector_db,
            config=dict(self.config),
            run_arm_iter=self._run_arm_iter_for_retry,
            run_arm_v3bu=self._run_arm_v3bu_for_retry,
            run_arm_decompose=self._run_arm_decompose_for_retry,
            # Surface the iter-cap sidecar set by the last _arm_iter call
            # (which ran earlier in this same query) so the escalation
            # strategy can gate on the literal "iter==cap AND gamma_refuse".
            iter_hit_cap=bool(
                getattr(self, "_last_iter_meta", {}).get("hit_cap", False)
            ),
        )

        try:
            orchestrator = build_default_orchestrator(preset, mode=self.retry_mode)
        except (ValueError, ImportError) as exc:
            logger.warning("retry orchestrator build failed (%s); skipping escalation.", exc)
            return chosen, {
                "retry_mode": self.retry_mode,
                "escalation_applied": [],
                "escalation_recovered_by": None,
                "original_abstention_signal": signal,
                "final_answer_confidence": "low_orchestrator_failed",
                "escalation_error": str(exc),
                "terminal_abstain": False,
            }

        outcome = orchestrator.try_escalate(ctx)
        return outcome.answer, {
            "retry_mode": outcome.mode,
            "escalation_applied": outcome.strategies_tried,
            "escalation_recovered_by": outcome.recovered_by,
            "original_abstention_signal": outcome.original_signal,
            "final_answer_confidence": outcome.final_confidence,
            "escalation_budget_used": outcome.budget_used,
            "terminal_abstain": outcome.terminal_abstain,
        }

    # --- Retry-context arm runner shims (decouple the orchestrator from
    # MothRAG.__init__-time state by exposing config-overridable runners) ---

    def _run_arm_iter_for_retry(
        self, *, question: str, passages: Sequence[str], q_emb: list[float],
        top_k: int, max_steps: int | None = None, l4b_anchor: Any = None,
    ) -> str:
        """Run the iter arm with optional budget overrides for retry strategies.

        When ``l4b_anchor`` is provided (alpha-pipeline L4b anchor swap, the
        fix for a dead-strategy bug), reorder ``passages`` so the
        anchor passage becomes index 0 before delegating to :meth:`_arm_iter`.
        The anchor identifier is interpreted as either a chunk ID present in
        ``self._last_retrieved_chunk_ids`` or a positional index into
        ``passages``; unknown anchors leave ordering untouched.
        """
        passages_list = list(passages)
        passages_list = self._reorder_passages_by_anchor(passages_list, l4b_anchor)
        if max_steps is not None:
            prev = self.config.get("max_iter_steps", 3)
            self.config["max_iter_steps"] = max_steps
            try:
                return self._arm_iter(question, passages_list, q_emb=q_emb, top_k=top_k)
            finally:
                self.config["max_iter_steps"] = prev
        return self._arm_iter(question, passages_list, q_emb=q_emb, top_k=top_k)

    @staticmethod
    def _reorder_passages_by_anchor(
        passages: list[str], anchor: Any,
    ) -> list[str]:
        """Move the passage matching ``anchor`` to index 0 if found.

        ``anchor`` accepted forms:
            * positive integer ``i`` → passage at index ``i`` boosted to 0
            * any other type / out-of-range index → passages returned unchanged

        Helper isolated so the retry-pipeline test can exercise the reorder
        without standing up the whole MothRAG.
        """
        if not passages or anchor is None:
            return passages
        try:
            idx = int(anchor)
        except (TypeError, ValueError):
            return passages
        if idx <= 0 or idx >= len(passages):
            return passages
        return [passages[idx]] + passages[:idx] + passages[idx + 1:]

    def _run_arm_v3bu_for_retry(self, *, question: str, passages: Sequence[str]) -> str:
        return self._arm_v3bu(question, passages)

    def _run_arm_decompose_for_retry(self, *, question: str, passages: Sequence[str]) -> str:
        return self._arm_decompose(question, passages)

    # ------------------------------------------------------------
    # Arm implementations (production mode helpers)
    # ------------------------------------------------------------

    def _arm_v3bu(self, question: str, passages: Sequence[str]) -> str:
        """V3+bu arm — single-shot retrieve + read (top-down anchor + bottom-up boost
        in the full MOTHRAG 1 stack; v0.5.0 alpha uses simple single-pass read).
        """
        try:
            return self.reader.read(question, list(passages))
        except Exception as exc:  # noqa: BLE001
            logger.exception("V3+bu arm reader failed: %s", exc)
            return ""

    def _arm_decompose(self, question: str, passages: Sequence[str]) -> str:
        """Decompose arm — split question into sub-questions, read each, synthesize.

        v0.5.0 alpha: single reader call with a chain-of-thought prompt that
        asks the LLM to break down → answer sub-parts → synthesize. The full
        MOTHRAG 1 decompose arm uses a 2-call pattern (decompose then read);
        the alpha collapses this into a single reasoning prompt to keep
        backend-agnosticism.
        """
        n = self.config.get("decompose_max_subq", 4)
        meta_question = (
            f"Answer the following question by first breaking it into at most "
            f"{n} sub-questions, briefly answering each from the passages, "
            f"and then synthesizing a final 1-2 sentence answer.\n\n"
            f"QUESTION: {question}\n\n"
            f"Reply with ONLY the final synthesized answer."
        )
        try:
            return self.reader.read(meta_question, list(passages))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Decompose arm reader failed: %s", exc)
            return ""

    def _arm_iter(self, question: str, passages: Sequence[str],
                  *, q_emb: list[float], top_k: int) -> str:
        """Iterative arm — refine retrieval over multiple steps using
        accumulated facts.

        v0.5.0 alpha: 2-step loop. Step 1: read on initial top-K → extract
        candidate facts. Step 2: re-retrieve with augmented query
        (question + candidate facts), then re-read.

        The full MOTHRAG 1 iter arm in scripts/route_prospective.py uses up to
        4 iterations + γ verifier; alpha skips γ (deferred to v0.5.1).

        Two patches are wired here: P4 (abstain filter — abstain
        markers are NOT propagated into the accumulator, preventing
        semantically degenerate "Context from prior steps: I don't know"
        pollution) and P8 (few-shot prompt enrichment —
        leading few-shot worked example in the augmented_q template
        encourages multi-hop synthesis on subsequent iterations).
        """
        from mothrag.core.abstain_markers import is_abstain_marker
        max_steps = self.config.get("max_iter_steps", 3)
        # P8 few-shot toggle — default ON. Users wanting the alpha
        # 2-step behaviour without the prompt enrichment can set
        # ``MothRAG(..., config={"iter_use_few_shot": False})``.
        use_few_shot = bool(self.config.get("iter_use_few_shot", True))
        accumulated: list[str] = []
        cur_passages = list(passages)
        last_answer = ""
        # Iter-cap sidecar. The arm returns a bare str so
        # downstream callers cannot tell a confident early return from a
        # cap-exhausted fall-through. Stash that here (overwritten at each
        # return) so _maybe_escalate can populate RetryContext.iter_hit_cap for
        # the cohort gate ("iter==cap AND gamma_refuse").
        self._last_iter_meta: dict[str, Any] = {
            "hit_cap": False, "steps_used": 0, "max_iter_steps": max_steps,
        }
        for step in range(max_steps):
            try:
                if accumulated:
                    # P8 few-shot — frame the accumulator as worked-context
                    # synthesis exemplar so the reader treats prior steps as
                    # composable facts rather than verbatim context.
                    if use_few_shot:
                        augmented_q = (
                            f"{question}\n\n"
                            "Synthesise an answer that composes the prior "
                            "facts when they entail the question; otherwise "
                            "rely on the supplied passages only.\n\n"
                            f"Facts from prior steps: {'; '.join(accumulated)}"
                        )
                    else:
                        augmented_q = (f"{question}\n\nContext from prior "
                                        f"steps: {'; '.join(accumulated)}")
                else:
                    augmented_q = question
                last_answer = self.reader.read(augmented_q, cur_passages)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Iter arm step %d reader failed: %s", step, exc)
                break
            if not last_answer or _is_uncertain_answer(last_answer):
                # P4 abstain filter: only append to the
                # accumulator if ``last_answer`` is NOT an abstain marker.
                # Previously abstain markers leaked into ``accumulated`` and
                # polluted the augmented_q on subsequent iterations.
                if last_answer and not is_abstain_marker(last_answer):
                    accumulated.append(last_answer)
                augmented_q = f"{question} {' '.join(accumulated)}"
                cur_chunks = self.retriever.retrieve(augmented_q, top_k=top_k)
                cur_passages = [c.text for c in cur_chunks]
                continue
            # Confident early return — cap NOT hit.
            self._last_iter_meta = {
                "hit_cap": False, "steps_used": step + 1,
                "max_iter_steps": max_steps,
            }
            return last_answer
        # Loop exhausted all iterations (or broke on a reader error) without a
        # confident early return: the iter arm hit its cap (cohort signal).
        self._last_iter_meta = {
            "hit_cap": True, "steps_used": max_steps, "max_iter_steps": max_steps,
        }
        return last_answer

    def _arm_bridge(
        self, question: str, passages: Sequence[str], *,
        q_emb: list[float], top_k: int,
    ) -> tuple[str, list[str]]:
        """Bridge-conditioned arm (opt-in). Returns (answer, ranked_pids).

        Builds a BridgeArm whose dense ANN reuses ``self.retriever`` (the PROD
        gemini-embedding-2 retriever) for all four retrieval stages, runs the
        Bacellar tripartite-judge pipeline, then reads the top-K bridge-ranked
        passages into an answer via ``self.reader``. The Haiku LLM stages
        degrade gracefully (dense order + passthrough) when ANTHROPIC_API_KEY
        is absent, so this is safe in offline / no-key deployments.
        """
        from mothrag.retrieval.bridge_haiku import BridgeArm, BridgeConfig
        from mothrag.retrieval.bridge_haiku.types import Candidate

        def ann_retrieve(query: str, k: int) -> list:
            chunks = self.retriever.retrieve(query, top_k=k)
            out: list[Candidate] = []
            for i, c in enumerate(chunks):
                pid = (getattr(c, "doc_id", None) or getattr(c, "chunk_id", None)
                       or str(i))
                out.append(Candidate(
                    str(pid), getattr(c, "text", "") or "",
                    float(getattr(c, "score", 0.0) or 0.0)))
            return out

        cfg = BridgeConfig(
            judge_model=self.config.get("bridge_judge_model", "claude-haiku-4-5"),
            max_cost_usd=float(self.config.get("bridge_max_cost_usd", 10.0)),
        )
        try:
            arm = BridgeArm(ann_retrieve, config=cfg, require_backend=False)
            result = arm.retrieve(question)
        except Exception as exc:  # noqa: BLE001 — never break the main pipeline
            logger.warning("bridge arm failed: %s", exc)
            return "", []
        # Materialise an answer from the top bridge-ranked passages.
        bridge_passages = result.ranked_texts or list(passages)
        try:
            answer = self.reader.read(question, bridge_passages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("bridge arm reader failed: %s", exc)
            answer = ""
        return answer, list(result.ranked_passage_ids)

    async def aquery(self, question: str, **kwargs: Any) -> QueryResult:
        """Async wrapper around :meth:`query`. Uses thread pool for I/O."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.query(question, **kwargs))

    def batch_query(
        self,
        questions: Sequence[str],
        *,
        max_workers: int = 4,
        **kwargs: Any,
    ) -> list[QueryResult]:
        """Parallel batched query using a thread pool."""
        if not questions:
            return []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(lambda q: self.query(q, **kwargs), questions))

    # ------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"MothRAG(embedder={type(self.embedder).__name__}, "
                f"reader={type(self.reader).__name__}, "
                f"retrieval={self.retrieval!r}, "
                f"retriever={type(self.retriever).__name__}, "
                f"indexed={len(self.retriever)})")


__all__ = [
    "Chunk",
    "Document",
    "Embedder",
    "MothRAG",
    "QueryResult",
    "Reader",
    "VectorStore",
]
