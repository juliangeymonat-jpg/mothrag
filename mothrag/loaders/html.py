# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""HTML loader for MothRAG (uses BeautifulSoup, optional dependency).

Usage::

    from mothrag.loaders import load_html
    docs = load_html("page.html")

If ``beautifulsoup4`` is not installed, raises ``ImportError`` with the
install hint ``pip install mothrag[html]``. The loader strips ``<script>``,
``<style>``, ``<noscript>`` tags and returns the visible-text content
as a single :class:`Document` per file, preserving page-title and
meta-description as metadata when available.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mothrag.core.api import Document

logger = logging.getLogger(__name__)

_STRIP_TAGS = ("script", "style", "noscript", "iframe", "svg", "head", "header", "footer", "nav")


def load_html(path: str | Path, *, encoding: str = "utf-8",
               strip_tags: tuple[str, ...] = _STRIP_TAGS) -> list[Document]:
    """Load an ``.html`` / ``.htm`` file as a Document.

    Parameters
    ----------
    path
        Path to the HTML file.
    encoding
        File encoding (default UTF-8 with replacement on decode errors).
    strip_tags
        Tags to remove BEFORE text extraction. Default strips script/
        style/noscript/iframe/svg/head/header/footer/nav.

    Raises
    ------
    ImportError
        If ``beautifulsoup4`` is not installed.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "HTML loader requires `beautifulsoup4` — install via "
            "`pip install mothrag[html]` or `pip install beautifulsoup4`"
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"HTML not found: {p}")

    raw = p.read_text(encoding=encoding, errors="replace")
    return [_parse_html(raw, source=str(p), filename=p.name,
                        size_bytes=p.stat().st_size, strip_tags=strip_tags)]


def _parse_html(raw: str, *, source: str, filename: str | None = None,
                 size_bytes: int | None = None,
                 strip_tags: tuple[str, ...] = _STRIP_TAGS) -> Document:
    """Internal HTML → Document parser shared with the URL loader."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(raw, "html.parser")
    for tag_name in strip_tags:
        for el in soup.find_all(tag_name):
            el.decompose()
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None
    desc_tag = soup.find("meta", attrs={"name": "description"})
    description = desc_tag.get("content", "") if desc_tag else ""
    text = soup.get_text(separator="\n", strip=True)

    meta: dict[str, object] = {
        "source": source,
        "format": "html",
    }
    if filename is not None:
        meta["filename"] = filename
    if size_bytes is not None:
        meta["size_bytes"] = size_bytes
    if title:
        meta["title"] = title
    if description:
        meta["description"] = description

    return Document(text=text, metadata=meta)


__all__ = ["load_html"]
