# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Gemini reader adapter — gemini-2.5-flash, gemini-2.5-pro."""

from __future__ import annotations

from mothrag.readers.base import ReaderAdapter, ReaderResponse, _resolve_api_key

_PRICING = {
    "gemini-2.5-flash": {"in": 0.075, "out": 0.30},
    "gemini-2.5-pro":   {"in": 1.25,  "out": 5.00},
}


class GeminiReader(ReaderAdapter):
    """Google Gemini reader via google-genai SDK."""

    name = "gemini"
    supports_thinking = True

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        api_key: str | None = None,
        thinking_budget: int = 0,
    ) -> None:
        try:
            from google import genai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "GeminiReader requires `google-genai`. Install via "
                "`pip install mothrag[gemini]` or `pip install google-genai`."
            ) from e
        self.model = model
        self.thinking_budget = thinking_budget
        from google import genai
        self._genai = genai
        self._client = genai.Client(
            api_key=api_key or _resolve_api_key(("GEMINI_API_KEY", "GOOGLE_API_KEY"))
        )

    def complete(self, messages, *, max_tokens=None, temperature=None,
                 stop=None, **provider_kwargs) -> ReaderResponse:
        from google.genai import types as gtypes
        # Flatten OpenAI-style messages into a single contents string;
        # Gemini supports system_instruction separately.
        system = ""
        user_text_parts = []
        for m in messages:
            if m["role"] == "system":
                system += m["content"] + "\n"
            elif m["role"] == "user":
                user_text_parts.append(m["content"])
            elif m["role"] == "assistant":
                user_text_parts.append(f"(prior assistant: {m['content']})")
        contents = "\n\n".join(user_text_parts)
        config_kwargs = {
            "max_output_tokens": max_tokens or self.default_max_tokens,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "thinking_config": gtypes.ThinkingConfig(thinking_budget=self.thinking_budget),
        }
        if system:
            config_kwargs["system_instruction"] = system.strip()
        if stop:
            config_kwargs["stop_sequences"] = stop
        resp = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=gtypes.GenerateContentConfig(**config_kwargs),
        )
        usage = getattr(resp, "usage_metadata", None)
        return ReaderResponse(
            text=(resp.text or "").strip(),
            finish_reason="stop",
            n_input_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            n_output_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            raw={"model": self.model},
        )

    def estimate_cost(self, n_input_tokens, n_output_tokens) -> float:
        p = _PRICING.get(self.model, {"in": 0.0, "out": 0.0})
        return (n_input_tokens / 1e6) * p["in"] + (n_output_tokens / 1e6) * p["out"]


__all__ = ["GeminiReader"]
