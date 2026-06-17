# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for the ``base_url`` allowlist guard (mothrag.utils.url_safety).

These tests pin the security contract:

  * Allowlisted hosts are accepted unchanged.
  * Disallowed hosts raise ``ValueError`` by default (fail-closed).
  * The opt-in ``allow_custom_endpoint=True`` flips it to "warn-and-pass".
  * ``None`` / empty pass through (lets the SDK use its own default).
  * ``localhost`` / ``127.0.0.1`` are accepted for dev (HTTP allowed).
  * Plain ``http://`` is rejected for any remote host.
"""

import logging

import pytest

from mothrag.utils.url_safety import ALLOWED_HOSTS, validate_base_url


@pytest.mark.parametrize("host", sorted(ALLOWED_HOSTS))
def test_allowed_host_passes(host):
    url = f"https://{host}/v1"
    assert validate_base_url(url) == url


def test_disallowed_host_raises_by_default():
    with pytest.raises(ValueError) as excinfo:
        validate_base_url("https://evil.example.com/v1")
    msg = str(excinfo.value)
    # Helpful error: names the bad host, lists allowed hosts, mentions opt-in.
    assert "evil.example.com" in msg
    assert "api.openai.com" in msg
    assert "allow_custom_endpoint" in msg


def test_disallowed_host_with_opt_in_passes_with_warning(caplog):
    url = "https://my-internal-proxy.corp.local/v1"
    with caplog.at_level(logging.WARNING, logger="mothrag.security"):
        out = validate_base_url(url, allow_custom_endpoint=True)
    assert out == url
    assert any(
        "my-internal-proxy.corp.local" in r.getMessage()
        and r.levelno == logging.WARNING
        for r in caplog.records
    )


def test_none_returns_none():
    assert validate_base_url(None) is None


def test_empty_string_returns_empty():
    assert validate_base_url("") == ""


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/v1",
        "http://127.0.0.1:8080/v1",
        "https://localhost/v1",
    ],
)
def test_localhost_allowed_for_dev(url):
    assert validate_base_url(url) == url


def test_http_scheme_rejected_for_remote():
    with pytest.raises(ValueError) as excinfo:
        validate_base_url("http://api.openai.com/v1")
    assert "https" in str(excinfo.value).lower()


def test_https_scheme_required_for_remote():
    # Even with opt-in, plain HTTP to a remote host is refused (would leak
    # the key in the clear).
    with pytest.raises(ValueError):
        validate_base_url(
            "http://my-internal-proxy.corp.local/v1",
            allow_custom_endpoint=True,
        )


def test_invalid_url_no_host_raises():
    with pytest.raises(ValueError):
        validate_base_url("not-a-url")
