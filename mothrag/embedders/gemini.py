# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Gemini embedder — gemini-embedding-2 (3072-d, current production model).

MothRAG production default for embeddings.

Note: gemini-embedding-001 is deprecated and removed from defaults.
The two models occupy completely different embedding spaces
(cosine similarity ~0.02 on identical text); pre-built corpus indices on
-001 must be re-embedded for -2 queries.
"""

from __future__ import annotations

import numpy as np

from mothrag.embedders.base import EmbedderAdapter, _resolve_api_key

_DIMS = {
    "gemini-embedding-2":   3072,
    "text-embedding-005":   768,
    "text-embedding-004":   768,
}


class GeminiEmbedder(EmbedderAdapter):
    """Google Gemini Embedding API (gemini-embedding-2 / text-embedding-005).

    PROD embedder is ``gemini-embedding-2`` (``gemini-embedding-001``
    references in older docstrings are stale).

    Optional SHA256(model::text) file cache
    eliminates redundant API calls across runs (the MQ query set is
    fixed across all evals → cache hit 100% after pre-warm). Backward
    compatible: pass ``cache_dir=None`` (default) for legacy live-only.
    Anti-leak: cache key namespaces by model; no DS / gold / answer fields.
    """

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-embedding-2",
        *,
        api_key: str | None = None,
        task_type: str = "SEMANTIC_SIMILARITY",
        cache_dir: str | None = None,
    ) -> None:
        try:
            from google import genai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "GeminiEmbedder requires `google-genai`. "
                "Install via `pip install mothrag[gemini]`."
            ) from e
        from google import genai
        self.model = model
        self.task_type = task_type
        self._client = genai.Client(
            api_key=api_key or _resolve_api_key(("GEMINI_API_KEY", "GOOGLE_API_KEY"))
        )
        # Optional file cache
        import os as _os
        # Env var fallback: respect $MOTHRAG_GEMINI_CACHE_DIR if cache_dir None
        if cache_dir is None:
            cache_dir = _os.environ.get("MOTHRAG_GEMINI_CACHE_DIR")
        self._cache_dir = None
        if cache_dir:
            import pathlib
            self._cache_dir = pathlib.Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._n_cache_hits = 0
        self._n_cache_misses = 0

    @property
    def dim(self) -> int:
        return _DIMS.get(self.model, 3072)

    def _cache_path(self, text: str):
        if self._cache_dir is None:
            return None
        import hashlib
        key = hashlib.sha256(f"{self.model}::{text}".encode("utf-8")).hexdigest()
        return self._cache_dir / f"{key}.npy"

    def embed(self, texts, *, batch_size: int = 1):
        from google.genai import types as gtypes
        out = []
        for t in texts:
            # cache lookup
            cp = self._cache_path(t)
            if cp is not None and cp.exists():
                try:
                    v = np.load(cp).astype(np.float32)
                    self._n_cache_hits += 1
                    out.append(v)
                    continue
                except Exception:  # noqa: BLE001 — corrupt cache → live
                    pass
            r = self._client.models.embed_content(
                model=self.model,
                contents=t,
                config=gtypes.EmbedContentConfig(task_type=self.task_type),
            )
            v = np.asarray(r.embeddings[0].values, dtype=np.float32)
            norm = np.linalg.norm(v) or 1.0
            v = v / norm
            self._n_cache_misses += 1
            # Persist
            if cp is not None:
                try:
                    np.save(cp, v)
                except Exception:  # noqa: BLE001 — cache write failure non-fatal
                    pass
            out.append(v)
        return np.stack(out, axis=0) if out else np.zeros((0, self.dim), dtype=np.float32)


__all__ = ["GeminiEmbedder"]
