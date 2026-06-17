# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Stage 4 — bridge-conditioned tripartite judge.

The heart of Bacellar's mechanism: every pooled candidate ``c_i`` is scored
``s(q, b, e1, e2, c_i) in [0, 10]`` for how well it advances the reasoning
chain *through the bridge*, conditioned jointly on the question, the bridge
passage, and the two bridge entities (hence "tripartite": question / bridge /
candidate, anchored by the entity pair).

All candidates are scored in a SINGLE batched call (one prompt enumerates
the whole pool) so the question+bridge+entity prefix is shared — cheap at
Haiku rates (~$0.001-0.003 for a 20-candidate pool). The response is a JSON
array of scores aligned to candidate index.

Bacellar §3.4.
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
    "You are a multi-hop relevance judge. You are given a QUESTION, a BRIDGE "
    "passage that establishes intermediate entities {e1, e2}, and a numbered "
    "list of CANDIDATE passages. Score EACH candidate from 0 to 10 for how "
    "much it helps answer the QUESTION by continuing the reasoning chain "
    "through the bridge (a candidate that names the target fact about e2, "
    "reached via e1, scores high; an off-topic candidate scores low). "
    "Return ONLY a JSON array of numbers, one per candidate, in the same "
    "order. No prose, no keys."
)

_USER_TEMPLATE = (
    "QUESTION: {question}\n\n"
    "BRIDGE PASSAGE: {bridge}\n"
    "BRIDGE ENTITIES: e1={e1!r}, e2={e2!r}\n\n"
    "CANDIDATES:\n{candidates}\n\n"
    "Return a JSON array of {n} numbers in [0, 10], one score per candidate "
    "in order."
)

# Opt-in v2 (default stays v1, the validated config). Same JSON-number-array
# contract (parse_judge_scores unchanged); an explicit scoring rubric + 1-shot
# exemplar to sharpen the high/low separation.
_SYSTEM_V2 = (
    "You score how well each CANDIDATE passage advances a multi-hop reasoning "
    "chain. You get a QUESTION, a BRIDGE passage establishing entities "
    "{e1 (intermediate), e2 (target)}, and numbered CANDIDATES. Score each 0-10:"
    " 9-10 = states the target fact about e2 reached via e1; 5-8 = on-chain "
    "context about e1/e2 but not the final fact; 1-4 = same topic, wrong link; "
    "0 = off-topic. Output ONLY a JSON array of N numbers in candidate order — "
    "no prose, no keys, no fences.\n"
    "Example — 3 candidates, scores → [9, 4, 0]"
)

_USER_TEMPLATE_V2 = (
    "QUESTION: {question}\n"
    "BRIDGE: {bridge}\n"
    "ENTITIES: e1={e1!r} e2={e2!r}\n"
    "CANDIDATES:\n{candidates}\n"
    "Return a JSON array of EXACTLY {n} numbers in [0,10], one per candidate."
)

_PROMPTS = {
    "v1": (_SYSTEM, _USER_TEMPLATE),
    "v2": (_SYSTEM_V2, _USER_TEMPLATE_V2),
}


def _select_prompt(variant: str):
    return _PROMPTS.get(variant, _PROMPTS["v1"])

_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def parse_judge_scores(text: str, *, n: int, lo: float = 0.0,
                       hi: float = 10.0) -> list[float]:
    """Parse ``n`` numeric scores from a (possibly fenced) JSON array.

    Clamps to ``[lo, hi]``. Pads with the neutral midpoint and truncates so
    the result is exactly length ``n`` (robust to the judge returning too
    few / too many). Non-numeric entries become the midpoint.
    """
    mid = (lo + hi) / 2.0
    if n <= 0:
        return []
    if not text:
        return [mid] * n
    m = _ARRAY_RE.search(text)
    if not m:
        return [mid] * n
    try:
        arr = json.loads(m.group(0))
    except (ValueError, TypeError):
        return [mid] * n
    if not isinstance(arr, list):
        return [mid] * n
    out: list[float] = []
    for v in arr[:n]:
        try:
            x = float(v)
        except (ValueError, TypeError):
            x = mid
        out.append(max(lo, min(hi, x)))
    while len(out) < n:
        out.append(mid)
    return out


class TripartiteJudge(HaikuBackend):
    """Haiku bridge-conditioned batched relevance judge."""

    def __init__(self, *, prompt_variant: str = "v1", **kwargs) -> None:
        super().__init__(**kwargs)
        self.prompt_variant = prompt_variant

    def score(
        self, question: str, bridge: str, e1: str, e2: str,
        candidate_texts: list[str], *, lo: float = 0.0, hi: float = 10.0,
        max_tokens: int = 1024, stats=None,
    ) -> tuple[list[float], int, int, float]:
        """Score all candidates in one batched call.

        Returns ``(scores, n_in, n_out, cost)`` where ``scores`` is aligned
        with ``candidate_texts`` and has the same length. On failure every
        candidate gets the neutral midpoint (no candidate unfairly killed).
        """
        n = len(candidate_texts)
        mid = (lo + hi) / 2.0
        if n == 0:
            return [], 0, 0, 0.0
        listing = "\n".join(
            f"[{i + 1}] {t}" for i, t in enumerate(candidate_texts)
        )
        system, user_tmpl = _select_prompt(self.prompt_variant)
        user = user_tmpl.format(
            question=question, bridge=bridge or "", e1=e1, e2=e2,
            candidates=listing, n=n,
        )
        try:
            text, n_in, n_out = self._call(system, user, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001
            logger.warning("tripartite judge failed: %s", exc)
            if stats is not None:
                stats.add_failure("judge", is_5xx=is_transient_api_error(exc))
            return [mid] * n, 0, 0, 0.0
        scores = parse_judge_scores(text, n=n, lo=lo, hi=hi)
        cost = self._cost(n_in, n_out)
        if stats is not None:
            stats.add_call("judge", n_in, n_out, cost)
        return scores, n_in, n_out, cost


__all__ = ["TripartiteJudge", "parse_judge_scores"]
