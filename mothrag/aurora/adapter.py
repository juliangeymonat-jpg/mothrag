# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Adapter: MothRAG retrieval/reader formats <-> Aurora ProofTree formats.

Bridges between:
- MothRAG ``MothRAGPipeline.retrieve()`` output (chunk indices into
  ``pipeline.chunks_by_id``) and the ``list[{doc_id, text}]`` format expected
  by ``aurora.proof_tree_user_prompt`` and ``aurora.verify_proof_tree``.
- Reader JSON response and the ``ProofTree`` dataclass via
  :func:`prooftree_from_dict`.
"""
from __future__ import annotations

import json
from typing import Optional

from mothrag.aurora.rules import ProofTree, ProofStep, Source


def mothrag_passages_to_aurora(chunk_indices, pipeline) -> list[dict]:
    """Convert MothRAG retrieval indices into Aurora-format passages.

    Each passage dict has ``doc_id`` (from ``chunks_by_id[cid]['entity_id']``
    when present, else the chunk id) and ``text`` (chunk text).
    """
    out = []
    for ci in chunk_indices:
        cid = pipeline.chunk_ids[ci]
        chunk = pipeline.chunks_by_id[cid]
        doc_id = chunk.get("entity_id") or chunk.get("doc_id") or str(cid)
        out.append({"doc_id": doc_id, "text": chunk.get("text", "")})
    return out


def chunks_to_aurora(chunk_dicts: list[dict]) -> list[dict]:
    """Convert pre-materialized chunk dicts (e.g. from a partial-output JSON)
    into Aurora-format passages."""
    out = []
    for c in chunk_dicts:
        doc_id = c.get("entity_id") or c.get("doc_id") or c.get("id") or ""
        out.append({"doc_id": str(doc_id), "text": c.get("text", "")})
    return out


def _source_from_dict(d) -> Source:
    if d is None:
        return Source(doc_id="", span_text="", char_offset=None)
    if isinstance(d, str):
        # R1 follow-up (HP-569 crash): readers sometimes emit sources as bare
        # strings — "sources": ["Cesar Millan"] — instead of objects. Treat the
        # string as the doc_id; the empty span makes the step unverifiable
        # (γ-invalid) downstream instead of AttributeError-killing the query.
        return Source(doc_id=d.strip(), span_text="", char_offset=None)
    if not isinstance(d, dict):
        return Source(doc_id=str(d), span_text="", char_offset=None)
    return Source(
        doc_id=str(d.get("doc_id", "")),
        span_text=str(d.get("span_text", "")),
        char_offset=d.get("char_offset"),
    )


def _step_from_dict(d) -> ProofStep:
    # The `full` proof-tree prompt
    # (PROOF_TREE_SYSTEM_PROMPT) instructs the reader to emit a lookup's
    # provenance under the SINGULAR key "source" (an object), while the dataclass
    # + verifier expect "sources" (a list). That schema mismatch made every
    # lookup parse to sources=[] → verifier "no source for lookup step" → γ=invalid
    # ~100% on every `full`-variant run (V4 finals + smokes). Tolerate BOTH keys:
    # the singular "source" is wrapped into a one-element list. The `llama` prompt
    # already emits plural "sources" → this branch is a no-op there.
    _raw = d.get("sources")
    if not _raw and d.get("source") is not None:
        _raw = [d["source"]]
    # HP-569 crash hardening: "sources" may arrive as a bare string or a single
    # object rather than a list — wrap instead of iterating a str char-by-char.
    if isinstance(_raw, (str, dict)):
        _raw = [_raw]
    sources = [_source_from_dict(s) for s in (_raw or [])]
    return ProofStep(
        step=int(d.get("step", 0)),
        rule=str(d.get("rule", "")),
        predicate=d.get("predicate"),
        subject=d.get("subject"),
        object=d.get("object"),
        claim_text=str(d.get("claim_text", "")),
        sources=sources,
        inputs=list(d.get("inputs") or []),
        type_tag=d.get("type_tag"),
        verifier_status=d.get("verifier_status"),
        verifier_reason=d.get("verifier_reason"),
    )


def prooftree_from_dict(d: dict, qid: str = "") -> ProofTree:
    """Build a ProofTree from a parsed JSON dict.

    Robust to keys like ``naturalised_answer`` (UK), missing optional fields,
    and absence of ``qid`` (falls back to the supplied default).
    """
    # Non-dict steps (e.g. a stray string in "steps") are dropped, not crashed
    # on: a malformed step yields γ-invalid downstream, never a dead query.
    steps = [_step_from_dict(s) for s in (d.get("steps") or [])
             if isinstance(s, dict)]
    nat_ans = (d.get("naturalized_answer")
               or d.get("naturalised_answer")
               or d.get("answer")
               or "")
    return ProofTree(
        qid=str(d.get("qid", qid)),
        steps=steps,
        naturalized_answer=str(nat_ans) if nat_ans is not None else "",
        is_complete=bool(d.get("is_complete", False)),
        refusal_reason=d.get("refusal_reason"),
        overall_status=d.get("overall_status"),
    )


def parse_reader_prooftree_json(raw: str, qid: str = "") -> Optional[ProofTree]:
    """Parse a reader response into a ProofTree.

    Tolerates raw text wrapped in markdown code fences ``` ```json … ``` ``` or
    leading prose before the first ``{`` brace. Returns ``None`` on any parse
    failure (caller decides fallback behavior — typically treat as ``refuse``).
    """
    if not raw:
        return None
    s = raw.strip()
    # Strip markdown fences
    if s.startswith("```"):
        # remove first fence line
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s[: -3]
    # Trim to first {...} balanced object
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    blob = s[start:end]
    try:
        obj = json.loads(blob)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return prooftree_from_dict(obj, qid=qid)
