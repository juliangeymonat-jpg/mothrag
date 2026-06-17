# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""JSON / JSONL loader for MothRAG.

Supported shapes:
- ``.json`` containing a list of strings → one Document per string
- ``.json`` containing a list of objects → each object becomes a Document;
    looks for a "text" / "content" / "body" field (first match wins); all
    other top-level fields are preserved as metadata
- ``.json`` containing a single object → one Document (same field lookup)
- ``.jsonl``: one JSON record per line, otherwise as above

For arbitrary nested shapes, transform externally and pass List[Document]
directly to MothRAG.from_documents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mothrag.core.api import Document


_TEXT_KEYS = ("text", "content", "body", "passage", "answer")


def load_json(path: str | Path, *, encoding: str = "utf-8") -> list[Document]:
    """Load a ``.json`` or ``.jsonl`` file as a list of Documents."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".jsonl":
        records = []
        with p.open("r", encoding=encoding) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        raw = p.read_text(encoding=encoding)
        records = json.loads(raw)
    return _records_to_documents(records, source=str(p))


def _records_to_documents(records: Any, source: str) -> list[Document]:
    if isinstance(records, list):
        out: list[Document] = []
        for i, rec in enumerate(records):
            doc = _record_to_document(rec, source=source, index=i)
            if doc is not None:
                out.append(doc)
        return out
    if isinstance(records, dict):
        doc = _record_to_document(records, source=source, index=0)
        return [doc] if doc is not None else []
    if isinstance(records, str):
        return [Document(text=records, metadata={"source": source})]
    raise ValueError(f"Unsupported JSON top-level type {type(records).__name__} in {source}")


def _record_to_document(rec: Any, *, source: str, index: int) -> Document | None:
    if isinstance(rec, str):
        return Document(text=rec, metadata={"source": source, "index": index})
    if not isinstance(rec, dict):
        return None
    text: str | None = None
    for key in _TEXT_KEYS:
        v = rec.get(key)
        if isinstance(v, str) and v.strip():
            text = v
            break
    if text is None:
        # No recognized text field — stringify the whole record.
        text = json.dumps(rec, ensure_ascii=False)
    meta = {k: v for k, v in rec.items() if not isinstance(v, str) or k not in _TEXT_KEYS}
    meta.setdefault("source", source)
    meta.setdefault("index", index)
    return Document(text=text, metadata=meta)


__all__ = ["load_json"]
