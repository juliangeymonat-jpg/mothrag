# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Stage 3 — dual-entity extraction.

From the bridge passage ``b`` and question ``q``, Haiku extracts the two
entities that anchor the reasoning chain: ``e1`` (the intermediate entity
the hop-1 bridge establishes) and ``e2`` (the target entity the question
asks about). Each entity is then ANN-expanded independently (Stage 3
retrieval, wired in ``bridge_arm``).

Bacellar §3.3. Cost ~$0.0005/query at Haiku rates.
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
    "You extract the two bridge entities for multi-hop retrieval. Given a "
    "QUESTION and a BRIDGE passage, return e1 (the intermediate entity the "
    "bridge establishes, linking hop-1 to hop-2) and e2 (the target entity "
    "the question ultimately asks about). Return ONLY a JSON object "
    '{"e1": "...", "e2": "..."} with surface forms, no prose. Use an empty '
    "string for an entity you cannot identify."
)

_USER_TEMPLATE = (
    "QUESTION: {question}\n\nBRIDGE PASSAGE: {bridge}\n\n"
    'Return {{"e1": "...", "e2": "..."}}.'
)

# Opt-in v2 (default stays v1, the validated config). Same JSON-object
# contract (parse_entity_response unchanged); 1-shot exemplar + tighter role
# definitions to reduce e1/e2 confusion.
_SYSTEM_V2 = (
    "Extract the two bridge entities for a multi-hop question. e1 = the "
    "INTERMEDIATE entity the BRIDGE passage establishes (it links hop-1 to "
    "hop-2). e2 = the TARGET entity the QUESTION ultimately asks about. Use the "
    "exact surface forms; use an empty string for one you cannot identify. "
    'Output ONLY a JSON object {"e1": "...", "e2": "..."} — no prose, no fences.'
    '\nExample — QUESTION: "What country is the director of Pulgasari from?" '
    'BRIDGE: "Pulgasari was directed by Shin Sang-ok." '
    '→ {"e1": "Shin Sang-ok", "e2": "country"}'
)

_USER_TEMPLATE_V2 = (
    "QUESTION: {question}\nBRIDGE PASSAGE: {bridge}\n"
    'Return {{"e1": "...", "e2": "..."}}.'
)

_PROMPTS = {
    "v1": (_SYSTEM, _USER_TEMPLATE),
    "v2": (_SYSTEM_V2, _USER_TEMPLATE_V2),
}


def _select_prompt(variant: str):
    return _PROMPTS.get(variant, _PROMPTS["v1"])

_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_entity_response(text: str) -> tuple[str, str]:
    """Parse ``{"e1","e2"}`` from a (possibly fenced) response. ('','') on fail."""
    if not text:
        return "", ""
    m = _OBJ_RE.search(text)
    if not m:
        return "", ""
    try:
        obj = json.loads(m.group(0))
    except (ValueError, TypeError):
        return "", ""
    if not isinstance(obj, dict):
        return "", ""
    e1 = obj.get("e1") or ""
    e2 = obj.get("e2") or ""
    return (str(e1).strip(), str(e2).strip())


class DualEntityExtractor(HaikuBackend):
    """Haiku dual-entity (e1, e2) extractor."""

    def __init__(self, *, prompt_variant: str = "v1", **kwargs) -> None:
        super().__init__(**kwargs)
        self.prompt_variant = prompt_variant

    def extract(self, question: str, bridge: str,
                stats=None) -> tuple[tuple[str, str], int, int, float]:
        """Return ``((e1, e2), n_in, n_out, cost)``. ('','') on failure."""
        if not (question and bridge):
            return ("", ""), 0, 0, 0.0
        system, user_tmpl = _select_prompt(self.prompt_variant)
        user = user_tmpl.format(question=question, bridge=bridge)
        try:
            text, n_in, n_out = self._call(system, user, max_tokens=128)
        except Exception as exc:  # noqa: BLE001
            logger.warning("entity extraction failed: %s", exc)
            if stats is not None:
                stats.add_failure("entity", is_5xx=is_transient_api_error(exc))
            return ("", ""), 0, 0, 0.0
        ents = parse_entity_response(text)
        cost = self._cost(n_in, n_out)
        if stats is not None:
            stats.add_call("entity", n_in, n_out, cost)
        return ents, n_in, n_out, cost


__all__ = ["DualEntityExtractor", "parse_entity_response"]
