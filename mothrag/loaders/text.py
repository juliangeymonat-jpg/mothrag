# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Plain text / Markdown loader for MothRAG.

Reads the entire file as a single Document with source-path metadata.
For very large files (>10MB), consider pre-chunking externally — MothRAG
will handle chunking inside ingest() but holds the full text in memory
during ingestion.
"""

from __future__ import annotations

from pathlib import Path

from mothrag.core.api import Document


def load_text(path: str | Path, *, encoding: str = "utf-8") -> list[Document]:
    """Load a single ``.txt`` / ``.md`` file as a Document."""
    p = Path(path)
    text = p.read_text(encoding=encoding, errors="replace")
    return [Document(
        text=text,
        metadata={
            "source": str(p),
            "filename": p.name,
            "format": "markdown" if p.suffix.lower() in (".md", ".markdown") else "text",
            "size_bytes": p.stat().st_size,
        },
    )]


__all__ = ["load_text"]
