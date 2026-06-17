# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Document loaders for MothRAG.

The :func:`auto_load` dispatcher inspects a path (or list of paths) and
routes to the appropriate per-format loader. PDF / HTML / DOCX loaders
are intentionally deferred to future versions; v0.5.0 ships:

- ``text``: ``.txt``, ``.md``, ``.markdown`` plain text
- ``json``: ``.json`` / ``.jsonl`` structured records

Each loader returns a list of :class:`mothrag.core.api.Document`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from mothrag.core.api import Document
from mothrag.loaders.text import load_text
from mothrag.loaders.json_loader import load_json
from mothrag.loaders.pdf import load_pdf
from mothrag.loaders.html import load_html
from mothrag.loaders.url import load_url


_EXT_DISPATCH = {
    ".txt": load_text,
    ".md": load_text,
    ".markdown": load_text,
    ".json": load_json,
    ".jsonl": load_json,
    ".pdf": load_pdf,
    ".html": load_html,
    ".htm": load_html,
}

# Format-specific deferrals remaining for future versions
_DEFERRED = {
    ".docx": "DOCX loader deferred — pre-extract text externally and pass strings via MothRAG.from_documents",
    ".epub": "EPUB loader deferred — pre-extract text externally and pass strings",
    ".pptx": "PPTX loader deferred — pre-extract text externally and pass strings",
}


def auto_load(source: str | Path | Iterable[str | Path]) -> list[Document]:
    """Auto-dispatch a path (file or directory) to the right loader.

    Behavior:
    - File path: loaded by extension.
    - Directory: recursively walks for supported extensions and concatenates.
    - List of paths: concatenates results.

    Raises ``ValueError`` for unrecognized extensions; ``NotImplementedError``
    with a helpful message for deferred formats (.pdf, .html, .docx).
    """
    if isinstance(source, (str, Path)):
        return _load_one(Path(source))
    out: list[Document] = []
    for item in source:
        out.extend(_load_one(Path(item)))
    return out


def _load_one(path: Path) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if path.is_dir():
        out: list[Document] = []
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in _EXT_DISPATCH:
                out.extend(_load_one(child))
        return out
    suf = path.suffix.lower()
    if suf in _DEFERRED:
        raise NotImplementedError(_DEFERRED[suf])
    if suf in _EXT_DISPATCH:
        return _EXT_DISPATCH[suf](path)
    raise ValueError(f"Unrecognized file extension '{suf}' for {path}. "
                     f"Supported: {sorted(_EXT_DISPATCH)}; deferred: {sorted(_DEFERRED)}")


__all__ = ["auto_load", "load_text", "load_json", "load_pdf", "load_html", "load_url"]
