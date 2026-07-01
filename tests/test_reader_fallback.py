# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Loud-fallback guard for the default reader resolution.

The 0.6.0 trap: a user sets an API key but not the ``[openai]`` extra, the
reader import fails INSIDE the adapter constructor, and the library silently
falls back to the echo reader -- so chunk-echo output masquerades as an LLM
answer. These tests pin the fix: the fallback must WARN and name the exact
install command, in both echo cases.
"""

from __future__ import annotations

import logging

import pytest

from mothrag.core.api import _EchoReader, _resolve_default_reader


def _openai_installed() -> bool:
    try:
        import openai  # noqa: F401

        return True
    except ImportError:
        return False


def test_no_key_fallback_is_loud(monkeypatch, caplog):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING, logger="mothrag.core.api"):
        reader = _resolve_default_reader()
    assert isinstance(reader, _EchoReader)
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "NOT LLM-generated" in messages
    assert "mothrag[openai]" in messages


def test_key_but_no_sdk_fallback_is_loud(monkeypatch, caplog):
    if _openai_installed():
        pytest.skip("openai SDK installed; the missing-SDK branch cannot fire")
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "dummy-key-for-test")
    with caplog.at_level(logging.WARNING, logger="mothrag.core.api"):
        reader = _resolve_default_reader()
    assert isinstance(reader, _EchoReader)
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "GROQ_API_KEY is set but the reader SDK is not installed" in messages
    assert "mothrag[openai]" in messages
