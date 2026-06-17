# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Query decomposition + synthesis for multi-hop questions.

Pattern (StepChain-style):
  Multi-hop Q: "Who was born first, X or Y?"
  -> DECOMPOSE LLM call
  Sub-Qs: ["When was X born?", "When was Y born?"]
  -> per sub-Q: retrieval pipeline + reader (single-hop)
  Sub-As: [("When was X born?", "1950"), ("When was Y born?", "1940")]
  -> SYNTHESIZE LLM call
  Final answer: "Y" (because 1940 < 1950)

Decompose and synthesize are separate few-shot LLM calls.

Usage::

    from mothrag.core.decompose import decompose_question, synthesize_answer
    sub_qs = decompose_question(client, model, q_text)
    sub_qa = [(sq, run_pipeline(sq)) for sq in sub_qs]
    final = synthesize_answer(client, model, q_text, sub_qa)
"""

from __future__ import annotations

import json
import re


DECOMPOSE_SYSTEM = """You decompose multi-hop Wikipedia questions into atomic sub-questions.

A multi-hop question needs 2-4 atomic facts combined. Each sub-question must be answerable by ONE Wikipedia passage about ONE entity.

OUTPUT: STRICT JSON list of strings. NO other text, NO explanation, NO markdown fences.

Examples:

Q: Who was born first, Karen Joy Fowler or Bruce Chatwin?
["When was Karen Joy Fowler born?", "When was Bruce Chatwin born?"]

Q: In what city are both the Nusretiye Clock Tower and the Eski Imaret Mosque located?
["Where is the Nusretiye Clock Tower located?", "Where is the Eski Imaret Mosque located?"]

Q: Are Colocasia and Coronilla both flowering plants?
["Is Colocasia a flowering plant?", "Is Coronilla a flowering plant?"]

Q: Who is a winner of the 2013 6 Hours of Silverstone a co-commentator for?
["Who won the 2013 6 Hours of Silverstone?", "Who is that winner a co-commentator for?"]

Q: The Distribution of Industry act was passed by a man who was prime minister when?
["Who passed the Distribution of Industry act?", "When was that person prime minister?"]

Q: Bytham Castle is a castle in the civil parish of how many houses?
["What civil parish is Bytham Castle in?", "How many houses are in that civil parish?"]

Q: What screenplay was worked on by both Edward Carfagno and Miklos Rozsa?
["What screenplays did Edward Carfagno work on?", "What screenplays did Miklos Rozsa work on?"]"""


SYNTHESIZE_SYSTEM = """You combine sub-question answers into a final HotpotQA-style answer.

ANSWER FORMAT (strict):
- Yes/no question: reply EXACTLY "yes" or "no" (lowercase, no period).
- All other: COPY VERBATIM from sub-answers (do NOT abbreviate). If combining, take the most direct verbatim span.
- No prefix ("The answer is", "Yes,"), no trailing period.
- If sub-answers are insufficient or contradict, reply "Not in passages".

Examples:

Original: Who was born first, X or Y?
Sub-answers:
- When was X born? -> 1950
- When was Y born? -> 1940
Reasoning: 1940 < 1950, so Y born first.
Final answer: Y

Original: Are X and Y both fruits?
Sub-answers:
- Is X a fruit? -> yes
- Is Y a fruit? -> yes
Final answer: yes

Original: In what city are X and Y both located?
Sub-answers:
- Where is X located? -> Istanbul
- Where is Y located? -> Istanbul
Final answer: Istanbul

Original: When was the prime minister X in office?
Sub-answers:
- Who was prime minister X? -> Clement Attlee
- When was Clement Attlee in office? -> 1945 to 1951
Final answer: 1945 to 1951"""


def decompose_question(client, model, question: str, max_tokens: int = 200,
                       temperature: float = 0.0) -> list[str]:
    """Return list of sub-questions. Falls back to ``[question]`` on parse failure."""
    sub_qs, _ = decompose_question_with_usage(client, model, question, max_tokens, temperature)
    return sub_qs


def decompose_question_with_usage(client, model, question: str, max_tokens: int = 200,
                                  temperature: float = 0.0):
    """Return ``(sub_questions, usage)`` where ``usage = {prompt_tokens, completion_tokens, latency_s}``."""
    import time as _t
    msgs = [
        {"role": "system", "content": DECOMPOSE_SYSTEM},
        {"role": "user", "content": f"Q: {question}"},
    ]
    t0 = _t.time()
    resp = client.chat.completions.create(
        model=model, messages=msgs, max_tokens=max_tokens, temperature=temperature
    )
    latency = _t.time() - t0
    raw = resp.choices[0].message.content.strip()
    sub_qs = _parse_sub_qs(raw)
    if not sub_qs:
        sub_qs = [question]
    u = getattr(resp, "usage", None)
    pt = int(getattr(u, "prompt_tokens", 0) or 0) if u else 0
    ct = int(getattr(u, "completion_tokens", 0) or 0) if u else 0
    return sub_qs, {"prompt_tokens": pt, "completion_tokens": ct, "latency_s": float(latency)}


def _parse_sub_qs(raw: str) -> list[str]:
    """Robust JSON list parser; handles markdown fences and extra text."""
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    lb, rb = raw.find("["), raw.rfind("]")
    if lb < 0 or rb <= lb:
        return []
    candidate = raw[lb : rb + 1]
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list) and all(isinstance(x, str) and x.strip() for x in parsed):
            return [x.strip() for x in parsed]
    except json.JSONDecodeError:
        pass
    return []


def synthesize_answer(client, model, original_q: str,
                      sub_qa_pairs: list[tuple[str, str]],
                      max_tokens: int = 96, temperature: float = 0.0) -> str:
    """Combine sub-Q+A pairs into final answer."""
    ans, _ = synthesize_answer_with_usage(client, model, original_q, sub_qa_pairs, max_tokens, temperature)
    return ans


def synthesize_answer_with_usage(client, model, original_q: str,
                                 sub_qa_pairs: list[tuple[str, str]],
                                 max_tokens: int = 96, temperature: float = 0.0):
    """Return ``(answer, usage)`` where ``usage = {prompt_tokens, completion_tokens, latency_s}``."""
    import time as _t
    sub_text = "\n".join(f"- {sq} -> {sa}" for sq, sa in sub_qa_pairs)
    user_msg = f"Original: {original_q}\nSub-answers:\n{sub_text}\n\nFinal answer:"
    msgs = [
        {"role": "system", "content": SYNTHESIZE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    t0 = _t.time()
    resp = client.chat.completions.create(
        model=model, messages=msgs, max_tokens=max_tokens, temperature=temperature
    )
    latency = _t.time() - t0
    raw = resp.choices[0].message.content.strip()
    for marker in ("Final answer:", "Final Answer:", "FINAL ANSWER:"):
        if marker in raw:
            raw = raw.split(marker, 1)[-1].strip()
            break
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    ans = (lines[0] if lines else "").rstrip(".").strip()
    u = getattr(resp, "usage", None)
    pt = int(getattr(u, "prompt_tokens", 0) or 0) if u else 0
    ct = int(getattr(u, "completion_tokens", 0) or 0) if u else 0
    return ans, {"prompt_tokens": pt, "completion_tokens": ct, "latency_s": float(latency)}


REFINE_SYSTEM = """You produce the most concise verbatim span from passages that answers a question.

INPUT: original question, a CANDIDATE answer (from a prior synthesis), and the relevant passages.
TASK: judge whether the candidate is the best answer. If yes, repeat it verbatim. If you can find a more concise verbatim span in the passages that better matches HotpotQA-style gold answers, return that span instead.

STRICT FORMAT:
- Output ONLY the answer text. No prefix, no trailing period, no commentary.
- Yes/no question: reply EXACTLY "yes" or "no".
- Entity / location / date / number: shortest verbatim span from passages that answers the question.
- If candidate already matches the most concise span, return it unchanged.
- If the passages do not contain a better answer, return the candidate.
- If candidate is "Not in passages" and passages do contain an answer, return the new span; otherwise return "Not in passages"."""


def refine_answer_with_usage(client, model, original_q: str, candidate: str,
                              passages: list[str], max_tokens: int = 64,
                              temperature: float = 0.0):
    """CM-2 second pass: given ``(q, candidate, passages)``, produce concise verbatim span.

    Returns ``(refined_answer, usage)``. Safety net handled by caller: empty/identical
    output should fall back to candidate.
    """
    import time as _t
    ctx = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    user_msg = (
        f"Question: {original_q}\n\n"
        f"Candidate answer: {candidate}\n\n"
        f"Passages:\n{ctx}\n\n"
        f"Refined answer:"
    )
    msgs = [
        {"role": "system", "content": REFINE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    t0 = _t.time()
    resp = client.chat.completions.create(
        model=model, messages=msgs, max_tokens=max_tokens, temperature=temperature
    )
    latency = _t.time() - t0
    raw = resp.choices[0].message.content.strip()
    for marker in ("Refined answer:", "Refined Answer:", "Final answer:"):
        if marker in raw:
            raw = raw.split(marker, 1)[-1].strip()
            break
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    ans = (lines[0] if lines else "").rstrip(".").strip().strip('"').strip("'")
    u = getattr(resp, "usage", None)
    pt = int(getattr(u, "prompt_tokens", 0) or 0) if u else 0
    ct = int(getattr(u, "completion_tokens", 0) or 0) if u else 0
    return ans, {"prompt_tokens": pt, "completion_tokens": ct, "latency_s": float(latency)}
