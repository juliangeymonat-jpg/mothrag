# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Context-aware sub-question generation for the Iterative Ragnatela.

The loop's feedback signal: from the UNCERTAIN (MID) and ANTI-CONTEXT (LOW)
answers of the current pool, generate targeted sub-questions whose re-retrieval
brings the evidence needed to resolve them — raising those answers' γ on the
next iteration. HIGH-band (fact) answers are already resolved and generate no
sub-questions.

An optional ``llm`` callable produces the sub-questions; without it (or on
error) a deterministic, anti-leak template fallback keeps the loop offline-
testable. Anti-leak: question text + arm answers only.
"""
from __future__ import annotations

from typing import Callable, Optional

from mothrag.iterative_ragnatela.gamma_pooling import normalize_answer
from mothrag.iterative_ragnatela.types import PoolOutcome, RagnatelaConfig

# llm(question, outcome, cfg) -> list[str]
SubQLLM = Callable[[str, PoolOutcome, RagnatelaConfig], list]


def generate_sub_questions(
    question: str,
    outcome: PoolOutcome,
    cfg: RagnatelaConfig,
    *,
    llm: Optional[SubQLLM] = None,
) -> list[str]:
    """Return up to ``cfg.n_sub_questions`` context-aware sub-questions."""
    targets = list(outcome.mid) + list(outcome.low)   # uncertain first, then anti
    if not targets:
        return []

    if llm is not None:
        try:
            out = [str(s).strip() for s in llm(question, outcome, cfg) if str(s).strip()]
            if out:
                return out[: cfg.n_sub_questions]
        except Exception:  # noqa: BLE001 — never let the loop die on the LLM
            pass

    # Deterministic fallback: one verification probe per uncertain/anti answer.
    subs: list[str] = []
    seen: set[str] = set()
    for a in targets:
        frag = a.answer.strip() or a.arm
        s = (f"What evidence confirms or refutes that the answer to "
             f"\"{question.strip()}\" is \"{frag}\"?")
        key = normalize_answer(s)
        if key in seen:
            continue
        seen.add(key)
        subs.append(s)
        if len(subs) >= cfg.n_sub_questions:
            break
    return subs


__all__ = ["generate_sub_questions", "SubQLLM"]
