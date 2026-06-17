# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""ann._coerce_chunk entity_id fallback (R@5 bug fix).

The MOTHRAG corpus keys chunks by a doc-level ``entity_id``; without it in the
passage-id resolution chain such chunks coerced to ``pid="None"`` and R@5
collapsed to 0. Pins the entity_id-FIRST precedence and that other corpora are
unaffected.
"""
from __future__ import annotations

from mothrag.retrieval.bridge_haiku.ann import _coerce_chunk


def test_entity_id_only_chunk_returns_non_none_pid():
    pid, text = _coerce_chunk({"entity_id": "E42", "text": "body"})
    assert pid == "E42"
    assert pid not in (None, "None")
    assert text == "body"


def test_entity_id_takes_precedence():
    # The MOTHRAG gold matches on entity_id even when passage_id co-exists.
    pid, _ = _coerce_chunk({"entity_id": "E1", "passage_id": "p9",
                            "doc_id": "d9", "id": "i9", "text": "t"})
    assert pid == "E1"


def test_passage_id_corpora_unaffected():
    # No entity_id => fall through to the legacy chain (no regression).
    assert _coerce_chunk({"passage_id": "p3", "text": "t"})[0] == "p3"
    assert _coerce_chunk({"doc_id": "d3", "text": "t"})[0] == "d3"
    assert _coerce_chunk({"id": "i3", "text": "t"})[0] == "i3"


def test_tuple_and_list_chunks_still_work():
    assert _coerce_chunk(("p1", "txt")) == ("p1", "txt")
    assert _coerce_chunk(["p2", "txt2"]) == ("p2", "txt2")
