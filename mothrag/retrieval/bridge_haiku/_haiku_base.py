# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Shared Haiku backend for the BridgeRAG-Haiku LLM stages.

Each LLM stage (SVO generation, entity extraction, tripartite judge) is a
thin subclass of :class:`HaikuBackend`. The single network seam is
:meth:`_call` — override it in tests to return canned responses without
hitting the Anthropic API (mirrors the LLMRelationExtractor ``_call_llm``
test pattern).

Anti-leak: no gold/F1/dataset args. Cost is tracked per call.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

# Retry-with-backoff on transient Anthropic errors.
# 3 retries after the initial attempt, exponential sleeps 1s / 4s / 16s.
RETRY_BACKOFF_S: tuple[float, ...] = (1.0, 4.0, 16.0)
_TRANSIENT_CODES = frozenset({429, 500, 502, 503, 504})
_TRANSIENT_MARKERS = (
    "429", "500", "502", "503", "504", "rate limit", "ratelimit",
    "overloaded", "timeout", "timed out", "temporarily", "try again",
    "service unavailable",
)


def is_transient_api_error(exc: BaseException) -> bool:
    """True for retryable Anthropic errors (429 / 5xx / overloaded / timeout)."""
    for attr in ("status_code", "status", "code"):
        code = getattr(exc, attr, None)
        try:
            if int(code) in _TRANSIENT_CODES:
                return True
        except (TypeError, ValueError):
            pass
    s = str(exc).lower()
    return any(m in s for m in _TRANSIENT_MARKERS)


class HaikuBackend:
    """Lazily-constructed Claude Haiku backend with a mockable call seam."""

    def __init__(
        self,
        *,
        reader=None,
        model: str = "claude-haiku-4-5",
        api_key: str | None = None,
        provider: str = "anthropic",
        require_backend: bool = True,
    ) -> None:
        self.model = model
        self._api_key = api_key
        # Provider switch so the bridge LLM stages (SVO / entity / judge) can
        # run on Gemini (commodity Flash) instead of Anthropic Haiku.
        # Default "anthropic" → byte-identical to the Haiku config.
        self._provider = (provider or "anthropic").lower()
        self._reader = reader
        self._require_backend = require_backend
        if reader is None and require_backend:
            # Build eagerly so misconfiguration fails fast (parity with the
            # LLMRelationExtractor require_backend contract).
            self._reader = self._build_reader()

    # ---- backend construction (overridable) -------------------------
    def _build_reader(self):
        if self._provider == "gemini":
            from mothrag.readers.gemini import GeminiReader

            return GeminiReader(model=self.model, api_key=self._api_key)
        from mothrag.readers.anthropic import AnthropicReader

        return AnthropicReader(model=self.model, api_key=self._api_key)

    @property
    def reader(self):
        if self._reader is None:
            self._reader = self._build_reader()
        return self._reader

    # ---- the single network seam (override in tests) ----------------
    def _call(
        self, system: str, user: str, *, max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> tuple[str, int, int]:
        """Send a (system, user) prompt to Haiku; return (text, n_in, n_out).

        Tests override this to inject canned responses. The default routes
        through the Anthropic reader's ``complete()`` (via :meth:`_call_once`),
        wrapped in retry-with-backoff: a transient 429 /
        5xx / overloaded / timeout is retried up to ``len(RETRY_BACKOFF_S)``
        times with exponential sleeps; a non-transient error raises
        immediately; an exhausted transient error re-raises the last exception
        (the calling stage then records the failure + degrades gracefully).
        """
        last_exc: BaseException | None = None
        for attempt in range(len(RETRY_BACKOFF_S) + 1):
            try:
                return self._call_once(
                    system, user, max_tokens=max_tokens, temperature=temperature)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if (attempt >= len(RETRY_BACKOFF_S)
                        or not is_transient_api_error(exc)):
                    raise
                wait = RETRY_BACKOFF_S[attempt]
                logger.warning("Haiku transient error (attempt %d/%d): %s — "
                               "retrying in %.0fs", attempt + 1,
                               len(RETRY_BACKOFF_S) + 1, exc, wait)
                time.sleep(wait)
        raise last_exc  # pragma: no cover — loop always returns or raises

    def _call_once(
        self, system: str, user: str, *, max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> tuple[str, int, int]:
        """One network call to the Anthropic reader (no retry). Mockable seam."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        resp = self.reader.complete(
            messages, max_tokens=max_tokens, temperature=temperature,
        )
        return (
            (resp.text or "").strip(),
            int(getattr(resp, "n_input_tokens", 0) or 0),
            int(getattr(resp, "n_output_tokens", 0) or 0),
        )

    def _cost(self, n_in: int, n_out: int) -> float:
        try:
            return float(self.reader.estimate_cost(n_in, n_out))
        except Exception:  # noqa: BLE001
            return 0.0


__all__ = ["HaikuBackend", "is_transient_api_error", "RETRY_BACKOFF_S"]
