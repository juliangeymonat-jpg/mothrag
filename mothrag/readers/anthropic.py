# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Anthropic reader adapter — Claude Sonnet 4.6, Haiku 4.5, Opus 4.7."""

from __future__ import annotations

from mothrag.readers.base import ReaderAdapter, ReaderResponse, _resolve_api_key

# Anthropic list pricing 2026-05 (USD per 1M tokens).
_PRICING = {
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "claude-haiku-4-5":  {"in": 0.80, "out":  4.00},
    "claude-opus-4-7":   {"in": 15.00, "out": 75.00},
}


class AnthropicReader(ReaderAdapter):
    """Anthropic Claude reader (Messages API)."""

    name = "anthropic"
    supports_thinking = True

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "AnthropicReader requires `anthropic`. Install via "
                "`pip install mothrag[anthropic]` or `pip install anthropic`."
            ) from e
        self.model = model
        self._client = Anthropic(
            api_key=api_key or _resolve_api_key(("ANTHROPIC_API_KEY",)),
            timeout=timeout,
        )

    def complete(self, messages, *, max_tokens=None, temperature=None,
                 stop=None, **provider_kwargs) -> ReaderResponse:
        # Anthropic separates system from user/assistant messages.
        system = ""
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                system += m["content"] + "\n"
            else:
                user_msgs.append({"role": m["role"], "content": m["content"]})
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens or self.default_max_tokens,
            "messages": user_msgs,
            "temperature": temperature if temperature is not None else self.default_temperature,
        }
        if system:
            kwargs["system"] = system.strip()
        if stop:
            kwargs["stop_sequences"] = stop
        kwargs.update(provider_kwargs)
        resp = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        usage = getattr(resp, "usage", None)
        return ReaderResponse(
            text=text,
            finish_reason=resp.stop_reason or "stop",
            n_input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            n_output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            raw={"model": resp.model, "id": resp.id},
        )

    def estimate_cost(self, n_input_tokens, n_output_tokens) -> float:
        p = _PRICING.get(self.model, {"in": 0.0, "out": 0.0})
        return (n_input_tokens / 1e6) * p["in"] + (n_output_tokens / 1e6) * p["out"]


__all__ = ["AnthropicReader"]
