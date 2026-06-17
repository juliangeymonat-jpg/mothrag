# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Allowlist-based validation for LLM API ``base_url`` arguments.

Background
----------

Throughout MothRAG, several entry points accept a ``base_url`` argument that
is forwarded to ``OpenAI(api_key=..., base_url=...)`` (or the equivalent
client). If the URL is taken from untrusted input (a CLI flag, a config file,
an env-driven default that a user can flip), an attacker can set it to a host
they control and the next API call will leak the user's API key in the
``Authorization`` header.

Mitigation
----------

This module exposes :func:`validate_base_url`, which checks the host against
:data:`ALLOWED_HOSTS` (a small set of well-known LLM API hosts). Callers must
explicitly opt in via ``allow_custom_endpoint=True`` to use any other host
(e.g. self-hosted vLLM, internal proxy). Even with opt-in, a warning is
logged so the choice is auditable.

Local development endpoints (``localhost`` / ``127.0.0.1``) are exempt from
the HTTPS requirement so users can test against a local server without TLS.

Allowed hosts
-------------

.. data:: ALLOWED_HOSTS

   - ``api.together.xyz``
   - ``api.openai.com``
   - ``api.anthropic.com``
   - ``generativelanguage.googleapis.com``
   - ``api.groq.com``
   - ``api.mistral.ai``
   - ``openrouter.ai``
"""

from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import urlparse


logger = logging.getLogger("mothrag.security")


ALLOWED_HOSTS = frozenset({
    "api.together.xyz",
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.groq.com",
    "api.mistral.ai",
    "openrouter.ai",
})


_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def validate_base_url(
    base_url: Optional[str],
    *,
    allow_custom_endpoint: bool = False,
) -> Optional[str]:
    """Validate a ``base_url`` against the allowlist.

    Args:
        base_url: The URL to validate. ``None`` or empty string passes through
            unchanged (caller will fall back to the SDK's default endpoint).
        allow_custom_endpoint: When ``True``, any non-allowlisted host is
            permitted but a warning is logged on
            ``logging.getLogger("mothrag.security")``. When ``False`` (the
            default), non-allowlisted hosts raise :class:`ValueError`.

    Returns:
        The URL unchanged if valid (or ``None``/empty if nothing was passed).

    Raises:
        ValueError: If ``base_url`` is set to a non-allowlisted host and
            ``allow_custom_endpoint`` is ``False``, OR if the scheme is not
            ``https`` for a non-local host.

    Notes:
        Allowed hosts are listed in :data:`ALLOWED_HOSTS`. ``localhost`` and
        ``127.0.0.1`` are accepted under either ``http`` or ``https`` so
        local development against a self-hosted endpoint works without TLS.
    """
    if base_url is None or base_url == "":
        return base_url

    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    scheme = (parsed.scheme or "").lower()

    if not host:
        raise ValueError(
            f"Invalid base_url {base_url!r}: could not parse a host. "
            "Expected something like 'https://api.openai.com/v1'."
        )

    is_local = host in _LOCAL_HOSTS

    # Scheme check — https required for any remote host. Local dev may use http.
    if not is_local and scheme != "https":
        raise ValueError(
            f"Refusing base_url {base_url!r}: scheme must be 'https' for "
            f"remote hosts (got {scheme!r}). API keys would be sent in the "
            "clear over plain HTTP. Use https:// or run against "
            "localhost/127.0.0.1 for local development."
        )
    if is_local and scheme not in ("http", "https"):
        raise ValueError(
            f"Refusing base_url {base_url!r}: scheme must be 'http' or "
            f"'https' (got {scheme!r})."
        )

    if host in ALLOWED_HOSTS or is_local:
        return base_url

    if allow_custom_endpoint:
        logger.warning(
            "Using non-allowlisted endpoint %s — your API key will be sent "
            "to this host. Verify it is trustworthy.",
            host,
        )
        return base_url

    allowed_sorted = ", ".join(sorted(ALLOWED_HOSTS))
    raise ValueError(
        f"Refusing base_url {base_url!r}: host {host!r} is not in the "
        f"MothRAG allowlist. Allowed hosts: {allowed_sorted}. If you really "
        "need to point at a custom inference endpoint (e.g. self-hosted "
        "vLLM, internal proxy), pass allow_custom_endpoint=True explicitly."
    )


__all__ = ["ALLOWED_HOSTS", "validate_base_url"]
