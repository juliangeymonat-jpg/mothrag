# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""EmbedderAdapter ABC for MothRAG v0.5.0 embedder plugins.

All concrete embedders subclass :class:`EmbedderAdapter` and implement
:meth:`embed`. The legacy :class:`mothrag.core.api.Embedder` Protocol is
satisfied via :meth:`embed_batch` (default impl wraps embed → list[list[float]]).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Sequence


class EmbedderAdapter(ABC):
    """Pluggable text embedder. Subclasses implement :meth:`embed`."""

    name: str = ""
    model: str = ""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Output embedding dimensionality."""

    @abstractmethod
    def embed(self, texts: Sequence[str], *, batch_size: int = 32):
        """Encode texts. Returns array-like, shape (N, dim), L2-normalised float32."""

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Protocol-compatible (mothrag.core.api.Embedder) shim."""
        arr = self.embed(texts)
        return [list(map(float, v)) for v in arr]

    def __call__(self, text: str):
        """Convenience for single text."""
        return self.embed([text])[0]


def _resolve_api_key(env_keys: tuple[str, ...]) -> str:
    for k in env_keys:
        v = os.environ.get(k)
        if v:
            return v
    raise RuntimeError(f"No API key found in env. Tried: {env_keys}.")


__all__ = ["EmbedderAdapter", "_resolve_api_key"]
