# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""OpenAI embedder — text-embedding-3-small (1536-d) / -large (3072-d)."""

from __future__ import annotations

import numpy as np

from mothrag.embedders.base import EmbedderAdapter, _resolve_api_key

_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(EmbedderAdapter):
    """OpenAI Embeddings API."""

    name = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAIEmbedder requires `openai`. "
                "Install via `pip install mothrag[openai]`."
            ) from e
        self.model = model
        self._client = OpenAI(
            api_key=api_key or _resolve_api_key(("OPENAI_API_KEY",)),
            base_url=base_url,
        )

    @property
    def dim(self) -> int:
        return _DIMS.get(self.model, 1536)

    def embed(self, texts, *, batch_size: int = 100):
        out = []
        texts = list(texts)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            for d in resp.data:
                v = np.asarray(d.embedding, dtype=np.float32)
                norm = np.linalg.norm(v) or 1.0
                out.append(v / norm)
        return np.stack(out, axis=0) if out else np.zeros((0, self.dim), dtype=np.float32)


__all__ = ["OpenAIEmbedder"]
