# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Cohere reader adapter — command-r-plus, command-r."""

from __future__ import annotations

from mothrag.readers.base import ReaderAdapter, ReaderResponse, _resolve_api_key

_PRICING = {
    "command-r-plus": {"in": 2.50, "out": 10.00},
    "command-r":      {"in": 0.15, "out": 0.60},
}


class CohereReader(ReaderAdapter):
    """Cohere reader (Chat API v2)."""

    name = "cohere"

    def __init__(
        self,
        model: str = "command-r-plus",
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        try:
            import cohere  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "CohereReader requires `cohere`. Install via "
                "`pip install mothrag[cohere]` or `pip install cohere`."
            ) from e
        import cohere
        self.model = model
        self._client = cohere.ClientV2(
            api_key=api_key or _resolve_api_key(("COHERE_API_KEY",)),
            timeout=timeout,
        )

    def complete(self, messages, *, max_tokens=None, temperature=None,
                 stop=None, **provider_kwargs) -> ReaderResponse:
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens or self.default_max_tokens,
            temperature=temperature if temperature is not None else self.default_temperature,
            stop_sequences=stop,
            **provider_kwargs,
        )
        text_parts = []
        for c in resp.message.content or []:
            if getattr(c, "type", "") == "text":
                text_parts.append(c.text)
        usage = getattr(resp, "usage", None)
        billed = getattr(usage, "billed_units", None) if usage else None
        return ReaderResponse(
            text="".join(text_parts),
            finish_reason=resp.finish_reason or "stop",
            n_input_tokens=int(getattr(billed, "input_tokens", 0) or 0) if billed else 0,
            n_output_tokens=int(getattr(billed, "output_tokens", 0) or 0) if billed else 0,
            raw={"id": resp.id, "model": self.model},
        )

    def estimate_cost(self, n_input_tokens, n_output_tokens) -> float:
        p = _PRICING.get(self.model, {"in": 0.0, "out": 0.0})
        return (n_input_tokens / 1e6) * p["in"] + (n_output_tokens / 1e6) * p["out"]


__all__ = ["CohereReader"]
