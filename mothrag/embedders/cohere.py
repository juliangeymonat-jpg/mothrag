# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Cohere embedder — embed-english-v3.0 (1024-d) / multilingual-v3.0."""

from __future__ import annotations

import numpy as np

from mothrag.embedders.base import EmbedderAdapter, _resolve_api_key

_DIMS = {
    "embed-english-v3.0":      1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
}


class CohereEmbedder(EmbedderAdapter):
    """Cohere Embed API v2."""

    name = "cohere"

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        *,
        api_key: str | None = None,
        input_type: str = "search_document",
    ) -> None:
        try:
            import cohere  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "CohereEmbedder requires `cohere`. "
                "Install via `pip install mothrag[cohere]`."
            ) from e
        import cohere
        self.model = model
        self.input_type = input_type
        self._client = cohere.ClientV2(
            api_key=api_key or _resolve_api_key(("COHERE_API_KEY",))
        )

    @property
    def dim(self) -> int:
        return _DIMS.get(self.model, 1024)

    def embed(self, texts, *, batch_size: int = 96):
        out = []
        texts = list(texts)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self._client.embed(
                model=self.model,
                texts=batch,
                input_type=self.input_type,
                embedding_types=["float"],
            )
            for v_list in resp.embeddings.float:
                v = np.asarray(v_list, dtype=np.float32)
                norm = np.linalg.norm(v) or 1.0
                out.append(v / norm)
        return np.stack(out, axis=0) if out else np.zeros((0, self.dim), dtype=np.float32)


__all__ = ["CohereEmbedder"]
