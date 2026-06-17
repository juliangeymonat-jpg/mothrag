# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""OpenAI reader adapter — gpt-4o, gpt-4o-mini, gpt-5.x.

Lazy imports openai SDK. Raises ImportError with install hint at instantiation.
"""

from __future__ import annotations

from mothrag.readers.base import ReaderAdapter, ReaderResponse, _resolve_api_key

# OpenAI list pricing as of 2026-05 (USD per 1M tokens). Override via subclass.
_PRICING = {
    "gpt-4o":      {"in": 2.50, "out": 10.00},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-5":       {"in": 5.00, "out": 20.00},
}


class OpenAIReader(ReaderAdapter):
    """OpenAI-hosted reader (gpt-4o family)."""

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAIReader requires `openai`. Install via "
                "`pip install mothrag[openai]` or `pip install openai`."
            ) from e
        self.model = model
        self._client = OpenAI(
            api_key=api_key or _resolve_api_key(("OPENAI_API_KEY",)),
            base_url=base_url,
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


__all__ = ["OpenAIReader"]
