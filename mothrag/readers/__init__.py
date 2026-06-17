# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""MothRAG reader adapter package.

Each reader subclasses :class:`mothrag.readers.base.ReaderAdapter` and
implements :meth:`complete` against a specific provider's API. All readers
also satisfy the :class:`mothrag.core.api.Reader` Protocol via the inherited
:meth:`read` method.

Lazy imports: provider SDKs (openai, anthropic, cohere, google-genai) are
imported only when the corresponding adapter is *instantiated*. Importing
this package is cheap and never pulls SDKs into memory.

Available adapters:

- :class:`OpenAIReader` — gpt-4o, gpt-4o-mini
- :class:`AnthropicReader` — claude-sonnet-4-6, claude-haiku-4-5, claude-opus-4-7
- :class:`GroqReader` — llama-3.3-70b-versatile (MOTHRAG 1 default)
- :class:`GeminiReader` — gemini-2.5-flash / pro
- :class:`CohereReader` — command-r-plus, command-r

String alias registry — `mothrag.config.registry` will resolve "openai" →
OpenAIReader (planned for Phase 2.5).
"""

from __future__ import annotations

from mothrag.readers.base import ReaderAdapter, ReaderResponse


def _lazy_import(name: str):
    """Defer SDK-heavy imports until the user references a specific class."""
    if name == "OpenAIReader":
        from mothrag.readers.openai import OpenAIReader
        return OpenAIReader
    if name == "AnthropicReader":
        from mothrag.readers.anthropic import AnthropicReader
        return AnthropicReader
    if name == "GroqReader":
        from mothrag.readers.groq import GroqReader
        return GroqReader
    if name == "GeminiReader":
        from mothrag.readers.gemini import GeminiReader
        return GeminiReader
    if name == "CohereReader":
        from mothrag.readers.cohere import CohereReader
        return CohereReader
    raise AttributeError(f"module 'mothrag.readers' has no attribute {name!r}")


def __getattr__(name: str):
    return _lazy_import(name)


__all__ = [
    "ReaderAdapter",
    "ReaderResponse",
    "OpenAIReader",
    "AnthropicReader",
    "GroqReader",
    "GeminiReader",
    "CohereReader",
]
