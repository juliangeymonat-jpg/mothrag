# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Groq reader adapter — Llama-3.3-70B-versatile (default MOTHRAG 1 reader).

Uses the openai SDK with Groq's OpenAI-compatible base_url.
"""

from __future__ import annotations

from mothrag.readers.base import ReaderAdapter, ReaderResponse, _resolve_api_key

_PRICING = {
    "llama-3.3-70b-versatile": {"in": 0.59, "out": 0.79},
}


class GroqReader(ReaderAdapter):
    """Groq-hosted reader (OpenAI-compatible API)."""

    name = "groq"

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "GroqReader requires `openai`. Install via "
                "`pip install mothrag[openai]` or `pip install openai`."
            ) from e
        self.model = model
        self._client = OpenAI(
            api_key=api_key or _resolve_api_key(("GROQ_API_KEY",)),
            base_url="https://api.groq.com/openai/v1",
            timeout=timeout,
        )

    def complete(self, messages, *, max_tokens=None, temperature=None,
                 stop=None, **provider_kwargs) -> ReaderResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens or self.default_max_tokens,
            temperature=temperature if temperature is not None else self.default_temperature,
            stop=stop,
            **provider_kwargs,
        )
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return ReaderResponse(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason or "stop",
            n_input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            n_output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            raw={"model": resp.model, "id": resp.id},
        )

    def estimate_cost(self, n_input_tokens, n_output_tokens) -> float:
        p = _PRICING.get(self.model, {"in": 0.0, "out": 0.0})
        return (n_input_tokens / 1e6) * p["in"] + (n_output_tokens / 1e6) * p["out"]


__all__ = ["GroqReader"]
