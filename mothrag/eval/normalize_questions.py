# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Normalize question-set formats from MuSiQue / 2WikiMultiHopQA to HotpotQA-style.

The MothRAG pipeline expects each question dict to carry:
  - ``"_id"``: unique identifier
  - ``"question"``: the question text
  - ``"answer"``: gold answer string
  - ``"supporting_facts"``: list of ``[title, sentence_idx]`` tuples

MuSiQue native format uses ``"id"``, ``"paragraphs"`` (with ``is_supporting`` flags).
2WikiMultiHopQA native format is HotpotQA-compatible already.
"""

import json
from pathlib import Path


def normalize_musique(q: dict) -> dict:
    """MuSiQue -> HotpotQA-style.

    MuSiQue ``paragraphs`` field:
      ``[{"idx": int, "title": str, "paragraph_text": str, "is_supporting": bool}, ...]``
    HotpotQA ``supporting_facts``: ``[[title, sentence_idx], ...]``.

    Synthesises supporting_facts as ``[[title, 0], ...]`` for paragraphs marked
    ``is_supporting=True``.
    """
    supporting_facts = []
    for p in q.get("paragraphs", []):
        if p.get("is_supporting"):
            supporting_facts.append([p["title"], 0])

    return {
        "_id": q["id"],
        "question": q["question"],
        "answer": q["answer"],
        "supporting_facts": supporting_facts,
        "answer_aliases": q.get("answer_aliases", []),
        "_native_format": "musique",
        "_question_decomposition": q.get("question_decomposition"),
    }


def normalize_2wiki(q: dict) -> dict:
    """2WikiMultiHopQA -> HotpotQA-style. Mostly a passthrough with metadata preserved."""
    return {
        "_id": q["_id"],
        "question": q["question"],
        "answer": q["answer"],
        "supporting_facts": q.get("supporting_facts", []),
        "_native_format": "2wiki",
        "_type": q.get("type"),
        "_evidences": q.get("evidences"),
    }


def normalize_file(input_path: str | Path, output_path: str | Path,
                   fmt: str) -> int:
    """Normalise a question file in-place and return the question count."""
    questions = json.loads(Path(input_path).read_text(encoding="utf-8"))

    if fmt == "musique":
        normalized = [normalize_musique(q) for q in questions]
    elif fmt == "2wiki":
        normalized = [normalize_2wiki(q) for q in questions]
    elif fmt == "hotpotqa":
        normalized = questions
    else:
        raise ValueError(f"Unknown format: {fmt}")

    Path(output_path).write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return len(normalized)
