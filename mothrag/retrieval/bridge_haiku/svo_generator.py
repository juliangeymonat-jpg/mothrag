# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Stage 2 — SVO (subject-verb-object) hop-2 query generation.

Given the question ``q`` and the hop-1 bridge passage ``b``, Haiku
generates ``N`` subject-verb-object queries that name the next-hop
relation to retrieve. Each SVO query is then embedded + ANN-expanded
(Stage 2 retrieval, wired in ``bridge_arm``).

Bacellar §3.2. Cost ~$0.0005/query at Haiku rates.
"""
from __future__ import annotations

import json
import logging
import re

from mothrag.retrieval.bridge_haiku._haiku_base import (
    HaikuBackend,
    is_transient_api_error,
)

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a multi-hop retrieval query planner. Given a QUESTION and a "
    "BRIDGE passage that establishes an intermediate fact, produce short "
    "subject-verb-object (SVO) search queries that name the NEXT hop needed "
    "to answer the question. Each query must be a concise factual phrase "
    "(not a question), grounded in entities from the bridge. "
    "Return ONLY a JSON array of strings, no prose."
)

_USER_TEMPLATE = (
    "QUESTION: {question}\n\n"
    "BRIDGE PASSAGE: {bridge}\n\n"
    "Produce exactly {n} SVO queries as a JSON array of strings."
)

# Opt-in v2 prompt (default stays v1, the validated config). Tighter
# constraints + a 1-shot exemplar to improve hop-targeting + parse
# determinism, while keeping the SAME JSON-array output contract (so
# parse_svo_response is unchanged).
_SYSTEM_V2 = (
    "You plan the NEXT retrieval hop for multi-hop QA. Input: a QUESTION and a "
    "BRIDGE passage stating an intermediate fact. Output: terse subject-verb-"
    "object search phrases (NOT questions) that, if retrieved, would supply the "
    "fact still missing to answer the QUESTION. Rules: (1) reuse the concrete "
    "entities named in the BRIDGE; (2) each phrase ≤ 8 words; (3) no "
    "duplicates; (4) output ONLY a JSON array of strings — no prose, no code "
    "fences, no keys.\n"
    'Example — QUESTION: "What country is the director of Pulgasari from?" '
    'BRIDGE: "Pulgasari was directed by Shin Sang-ok." '
    '→ ["Shin Sang-ok nationality", "Shin Sang-ok country of origin"]'
)

_USER_TEMPLATE_V2 = (
    "QUESTION: {question}\n"
    "BRIDGE PASSAGE: {bridge}\n"
    "Return EXACTLY {n} SVO phrases as a JSON array of strings."
)

_PROMPTS = {
    "v1": (_SYSTEM, _USER_TEMPLATE),
    "v2": (_SYSTEM_V2, _USER_TEMPLATE_V2),
}


def _select_prompt(variant: str):
    return _PROMPTS.get(variant, _PROMPTS["v1"])

# Tolerant JSON-array extraction (LLMs sometimes wrap in ```json fences).
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def parse_svo_response(text: str, *, max_n: int) -> list[str]:
    """Parse a JSON array of SVO strings from a (possibly fenced) response."""
    if not text:
        return []
    m = _ARRAY_RE.search(text)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except (ValueError, TypeError):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in arr:
        if not isinstance(item, str):
            continue
        s = item.strip()
        key = s.lower()
        if s and key not in seen:
            seen.add(key)
            out.append(s)
        if len(out) >= max_n:
            break
    return out


class SVOQueryGenerator(HaikuBackend):
    """Haiku SVO hop-2 query generator."""

    def __init__(self, *, prompt_variant: str = "v1", **kwargs) -> None:
        super().__init__(**kwargs)
        self.prompt_variant = prompt_variant

    def generate(self, question: str, bridge: str, *, n: int = 3,
                 stats=None) -> tuple[list[str], int, int, float]:
        """Return ``(svo_queries, n_in, n_out, cost)``.

        Falls back to ``[question]`` (single passthrough query) on an empty
        / unparseable response so the pipeline always has at least one
        hop-2 query.
        """
        if not question:
            return [], 0, 0, 0.0
        system, user_tmpl = _select_prompt(self.prompt_variant)
        user = user_tmpl.format(question=question, bridge=bridge or "", n=n)
        try:
            text, n_in, n_out = self._call(system, user, max_tokens=256)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SVO generation failed: %s", exc)
            if stats is not None:
                stats.add_failure("svo", is_5xx=is_transient_api_error(exc))
            return [question], 0, 0, 0.0
        queries = parse_svo_response(text, max_n=n) or [question]
        cost = self._cost(n_in, n_out)
        if stats is not None:
            stats.add_call("svo", n_in, n_out, cost)
        return queries, n_in, n_out, cost


__all__ = ["SVOQueryGenerator", "parse_svo_response"]
