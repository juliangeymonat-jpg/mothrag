# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""PDF loader for MothRAG (uses pypdf, an optional dependency).

Usage::

    from mothrag.loaders import load_pdf
    docs = load_pdf("paper.pdf")

If ``pypdf`` is not installed, raises ``ImportError`` with the install
hint ``pip install mothrag[pdf]``. The loader extracts plain text per
page, joins pages with double newlines, and emits ONE :class:`Document`
per file. Per-page splitting can be achieved by passing
``per_page=True``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from mothrag.core.api import Document

logger = logging.getLogger(__name__)


def load_pdf(path: str | Path, *, per_page: bool = False,
              encoding_errors: str = "ignore") -> list[Document]:
    """Load a ``.pdf`` file as a Document (or list of per-page Documents).

    Parameters
    ----------
    path
        Path to the PDF file.
    per_page
        If True, emit one Document per page (metadata.page = 1-indexed page
        number). If False (default), emit a single Document concatenating
        all pages with double-newline separators.
    encoding_errors
        Strategy for non-decodable bytes inside extracted text. Passed
        through to PyPDF's PageObject.extract_text where supported.

    Raises
    ------
    ImportError
        If ``pypdf`` is not installed.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF loader requires `pypdf` — install via "
            "`pip install mothrag[pdf]` or `pip install pypdf`"
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")

    reader = PdfReader(str(p))
    n_pages = len(reader.pages)
    pages_text: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("PDF page %d extraction failed (%s) — skipping", i, exc)
            text = ""
        pages_text.append(text.strip())

    base_meta = {
        "source": str(p),
        "filename": p.name,
        "format": "pdf",
        "size_bytes": p.stat().st_size,
        "n_pages": n_pages,
    }

    if per_page:
        out: list[Document] = []
        for i, text in enumerate(pages_text, start=1):
            if not text:
                continue
            meta = dict(base_meta)
            meta["page"] = i
            out.append(Document(text=text, metadata=meta))
        return out

    joined = "\n\n".join(pt for pt in pages_text if pt)
    if not joined:
        logger.warning("PDF %s extracted 0 chars (likely scanned/image-only)", p)
    return [Document(text=joined, metadata=base_meta)]


__all__ = ["load_pdf"]
