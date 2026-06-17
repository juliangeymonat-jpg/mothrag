# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Dense ANN backend for BridgeRAG-Haiku.

Provides the ``ann_retrieve(query_text, top_k) -> list[Candidate]`` callable
that :class:`mothrag.retrieval.bridge_haiku.BridgeArm` injects for all four
retrieval stages (hop-1, SVO expansion, dual-entity expansion). The PROD
embedder is ``gemini-embedding-2`` — the same space
the corpus doc-vectors are built in, so cosine == dot product on the
L2-normalised vectors GeminiEmbedder returns.

Corpus doc-vectors can be supplied pre-computed (the
``chunk_vecs_gemini_doc.npy`` cache the eval pipeline already builds) to
avoid re-embedding the corpus; otherwise they are embedded once at
construction. Query vectors are embedded live (RETRIEVAL_QUERY) and cached
per the embedder's own ``MOTHRAG_GEMINI_CACHE_DIR``.

Backend-agnostic: any object exposing ``embed(list[str]) -> list[vector]``
works (GeminiEmbedder in PROD; a deterministic fake in tests), so the whole
bridge pipeline is offline-testable. Anti-leak: corpus text + query text
only; no gold/F1.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np

from mothrag.retrieval.bridge_haiku.types import Candidate


def _coerce_chunk(c: Any) -> tuple[str, str]:
    """Normalise a chunk to ``(passage_id, text)``."""
    if isinstance(c, (tuple, list)) and len(c) >= 2:
        return str(c[0]), str(c[1])
    if isinstance(c, dict):
        # entity_id FIRST: the MOTHRAG corpus keys chunks
        # by a doc-level ``entity_id`` and the gold_doc_ids match on it; without
        # this fallback such chunks coerced to pid=None → R@5 collapsed to 0.
        # Corpora that use passage_id/doc_id/id are unaffected (entity_id
        # absent → fall through unchanged).
        pid = (c.get("entity_id") or c.get("passage_id")
               or c.get("doc_id") or c.get("id"))
        text = c.get("text") or c.get("passage_text") or ""
        return str(pid), str(text)
    raise TypeError(f"unsupported chunk type: {type(c)!r}")


def _l2norm(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class GeminiANNRetriever:
    """Dense cosine ANN over a fixed corpus, embedded with gemini-embedding-2.

    Parameters
    ----------
    chunks
        Corpus as ``[{passage_id, text}, ...]`` or ``[(pid, text), ...]``.
    embedder
        Any object with ``embed(list[str]) -> list[vector]``. Defaults to a
        :class:`mothrag.embedders.GeminiEmbedder` (``gemini-embedding-2``).
    doc_vectors
        Optional pre-computed ``(N, D)`` corpus matrix aligned with ``chunks``
        (e.g. ``chunk_vecs_gemini_doc.npy``). If absent, embedded at init.
    """

    def __init__(
        self,
        chunks: Sequence[Any],
        *,
        embedder: Any = None,
        doc_vectors: Optional[np.ndarray] = None,
    ) -> None:
        pairs = [_coerce_chunk(c) for c in chunks]
        self.passage_ids: list[str] = [p for p, _ in pairs]
        self.texts: list[str] = [t for _, t in pairs]
        if embedder is None:
            from mothrag.embedders import GeminiEmbedder
            embedder = GeminiEmbedder(model="gemini-embedding-2")
        self.embedder = embedder

        if doc_vectors is not None:
            dv = np.asarray(doc_vectors, dtype=np.float32)
            if dv.shape[0] != len(self.texts):
                raise ValueError(
                    f"doc_vectors rows {dv.shape[0]} != corpus size "
                    f"{len(self.texts)}")
        elif self.texts:
            dv = np.asarray(
                [np.asarray(v, dtype=np.float32)
                 for v in self.embedder.embed(self.texts)],
                dtype=np.float32,
            )
        else:
            dv = np.zeros((0, 1), dtype=np.float32)
        # Defensive re-normalise (GeminiEmbedder already L2-normalises; a fake
        # embedder or raw cache may not).
        self.doc_vectors = _l2norm(dv) if dv.size else dv

    def retrieve(self, query: str, top_k: int) -> list[Candidate]:
        if not query or self.doc_vectors.size == 0 or top_k <= 0:
            return []
        qv = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        n = np.linalg.norm(qv) or 1.0
        qv = qv / n
        sims = self.doc_vectors @ qv  # cosine (both L2-normalised)
        k = min(top_k, sims.shape[0])
        # argpartition for top-k then sort those k by score desc (stable).
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx], kind="stable")]
        return [
            Candidate(self.passage_ids[i], self.texts[i], float(sims[i]))
            for i in top_idx
        ]

    # Allow passing the instance directly as BridgeArm's ann_retrieve callable.
    def __call__(self, query: str, top_k: int) -> list[Candidate]:
        return self.retrieve(query, top_k)

    def __len__(self) -> int:
        return len(self.passage_ids)


def build_gemini_ann(
    chunks: Sequence[Any],
    *,
    embedder: Any = None,
    doc_vectors: Optional[np.ndarray] = None,
) -> GeminiANNRetriever:
    """Convenience factory mirroring the other mothrag retrieval builders."""
    return GeminiANNRetriever(chunks, embedder=embedder, doc_vectors=doc_vectors)


__all__ = ["GeminiANNRetriever", "build_gemini_ann"]
