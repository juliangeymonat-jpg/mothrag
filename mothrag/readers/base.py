# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""ReaderAdapter ABC + ReaderResponse for MothRAG v0.5.0 reader plugins.

All concrete readers in :mod:`mothrag.readers.*` subclass :class:`ReaderAdapter`
and implement :meth:`complete`. The legacy :class:`mothrag.core.api.Reader`
Protocol is satisfied by exposing :meth:`read` (default impl wraps complete).

Lazy imports: provider SDKs (openai, anthropic, cohere, google-genai) are
imported inside __init__ of each concrete subclass — `from mothrag.readers
import OpenAIReader` does NOT pull SDKs into memory until instantiated.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class ReaderResponse:
    """Provider-agnostic completion result."""
    text: str
    finish_reason: str = "stop"
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    raw: dict = field(default_factory=dict)


class ReaderAdapter(ABC):
    """Pluggable LLM reader. Subclasses implement :meth:`complete`."""

    name: str = ""
    model: str = ""
    supports_thinking: bool = False
    default_max_tokens: int = 1024
    default_temperature: float = 0.0

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        **provider_kwargs,
    ) -> ReaderResponse:
        """Run a chat completion. ``messages`` is OpenAI-format list of dicts."""

    def read(self, question: str, passages: Sequence[str]) -> str:
        """Protocol-compatible single-shot read (Reader Protocol from core.api)."""
        joined = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
        msgs = [
            {"role": "system",
             "content": ("You answer questions using ONLY the provided passages. "
                         "Reply in 1-2 short sentences with the precise answer. "
                         "If the passages do not contain the answer, reply 'Not in passages.'")},
            {"role": "user",
             "content": f"PASSAGES:\n{joined}\n\nQUESTION: {question}\n\nANSWER:"},
        ]
        return self.complete(msgs).text.strip()

    def estimate_cost(self, n_input_tokens: int, n_output_tokens: int) -> float:
        """Override per-provider for $/query estimate. Default returns 0.0."""
        return 0.0


def _resolve_api_key(env_keys: tuple[str, ...]) -> str:
    """Return the first non-empty env var from env_keys, else raise."""
    for k in env_keys:
        v = os.environ.get(k)
        if v:
            return v
    raise RuntimeError(
        f"No API key found in env. Tried: {env_keys}. "
        f"Set one and retry."
    )


__all__ = ["ReaderAdapter", "ReaderResponse", "_resolve_api_key"]
