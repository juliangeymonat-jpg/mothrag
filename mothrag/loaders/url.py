# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""URL loader for MothRAG.

Fetches a remote URL (HTTP/HTTPS), dispatches to the right parser based
on Content-Type:

- text/html  → BeautifulSoup → visible-text Document
- text/plain → Document with raw text
- application/json → :func:`mothrag.loaders.json_loader.load_json` semantics
- application/pdf → :func:`mothrag.loaders.pdf.load_pdf` (via tempfile)

Uses ``requests`` (core dependency). The HTML branch needs
``beautifulsoup4`` (``mothrag[html]``); the PDF branch needs ``pypdf``
(``mothrag[pdf]``).

Usage::

    from mothrag.loaders import load_url
    docs = load_url("https://en.wikipedia.org/wiki/Python_(programming_language)")
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from mothrag.core.api import Document

logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT = 20  # seconds — conservative for large pages / slow servers
DEFAULT_USER_AGENT = (
    "mothrag/0.5.0 (+https://github.com/juliangeymonat-jpg/mothrag) "
    "python-requests"
)


def load_url(url: str, *, timeout: int = DEFAULT_TIMEOUT,
              headers: dict[str, str] | None = None,
              follow_redirects: bool = True) -> list[Document]:
    """Fetch ``url`` and convert to one or more :class:`Document`.

    Parameters
    ----------
    url
        Fully-qualified http(s) URL.
    timeout
        Per-request timeout in seconds.
    headers
        Extra HTTP headers (User-Agent is set automatically if not
        present).
    follow_redirects
        Whether to follow HTTP 3xx (default True).

    Raises
    ------
    ImportError
        If ``requests`` is missing (should never happen — it's a core dep).
    ValueError
        If ``url`` is not a valid http(s) URL.
    """
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "URL loader requires `requests` (should be a core dep)"
        ) from exc

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Only http/https URLs supported; got scheme={parsed.scheme!r}"
        )

    req_headers = dict(headers or {})
    req_headers.setdefault("User-Agent", DEFAULT_USER_AGENT)
    req_headers.setdefault("Accept", "text/html,application/json,application/pdf,text/plain;q=0.9,*/*;q=0.5")

    resp = requests.get(url, headers=req_headers, timeout=timeout,
                         allow_redirects=follow_redirects)
    resp.raise_for_status()

    ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    size_bytes = len(resp.content)

    if ctype.startswith("text/html"):
        from mothrag.loaders.html import _parse_html
        return [_parse_html(resp.text, source=url, size_bytes=size_bytes)]

    if ctype.startswith("application/json"):
        records = resp.json()
        # Re-use json_loader's record→Document logic
        from mothrag.loaders.json_loader import _records_to_documents
        return _records_to_documents(records, source=url)

    if ctype.startswith("application/pdf"):
        # Write bytes to temp file and reuse the PDF loader
        from mothrag.loaders.pdf import load_pdf
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name
        try:
            docs = load_pdf(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        # Patch source metadata to the original URL
        for d in docs:
            d.metadata["source"] = url
            d.metadata["fetched_from"] = url
        return docs

    if ctype.startswith("text/") or ctype == "":
        # Plain text or unknown → treat as text
        return [Document(text=resp.text, metadata={
            "source": url,
            "format": "url-text",
            "content_type": ctype,
            "size_bytes": size_bytes,
        })]

    # Unknown binary content type — refuse rather than guess
    raise ValueError(
        f"Unsupported Content-Type {ctype!r} for URL {url}. "
        "Pre-extract text externally and pass strings."
    )


__all__ = ["load_url"]
