# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Adapter implementations of Embedder / Reader protocols.

Each adapter lazy-imports its heavy dependency. Users who only need the
public MothRAG class (with offline fallbacks) never trigger these imports.

Adapters provided:
- SentenceTransformersEmbedder  (requires sentence-transformers extra)
- OpenAICompatibleReader        (requires openai extra; works with Together,
                                  Groq, OpenAI, Anthropic-via-proxy endpoints)
"""

from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)


class SentenceTransformersEmbedder:
    """Sentence-Transformers MiniLM-L6 embedder (384-d).

    Default match for MothRAG 1 production "st-mini" baseline. Loads
    weights from HuggingFace on first use.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self._model.encode(list(texts), show_progress_bar=False,
                                         normalize_embeddings=True)
        return [list(map(float, e)) for e in embeddings]


class OpenAICompatibleReader:
    """Reader that calls an OpenAI-format chat completion endpoint.

    Works for OpenAI, Together AI, Groq, OpenRouter, and any other
    OpenAI-compatible API. The default prompt is a single-pass extractive
    style matching MothRAG production V3+bu arm reading step.
    """

    SYSTEM_PROMPT = (
        "You answer questions using ONLY the provided passages. "
        "Reply in 1-2 short sentences with the precise answer extracted "
        "from the passages. If the passages do not contain the answer, "
        "reply 'Not in passages.'"
    )

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAICompatibleReader requires `openai` — install via "
                "`pip install mothrag[openai]` or `pip install openai`"
            ) from exc
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def read(self, question: str, passages: Sequence[str]) -> str:
        joined = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
        user_msg = f"PASSAGES:\n{joined}\n\nQUESTION: {question}\n\nANSWER:"
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.exception("OpenAICompatibleReader call failed: %s", exc)
            return ""


__all__ = [
    "OpenAICompatibleReader",
    "SentenceTransformersEmbedder",
]
