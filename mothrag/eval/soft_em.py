# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Soft EM via cross-family LLM judge.

HotpotQA gold answers contain format inconsistencies that penalise semantically
equivalent paraphrases under hard EM:

  - ``pred="10"`` vs ``gold="ten"``                        -> hard EM = 0
  - ``pred="Boston"`` vs ``gold="Boston, Massachusetts"``  -> hard EM = 0

Soft EM asks an LLM judge whether ``pred`` is semantically equivalent to
``gold``. Reporting both hard and soft EM is best practice for the paper.

Cross-family judges (Gemini, Claude, GPT-4o-mini) reduce intra-family bias
relative to a single in-family judge.
"""

import json
import os
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

from mothrag.eval.metrics import em_score, f1_score, normalize_answer
from mothrag.utils.url_safety import validate_base_url


JUDGE_SYSTEM = """You are a HotpotQA semantic equivalence judge. Given a question, a gold answer, and a predicted answer, decide if the prediction is SEMANTICALLY equivalent to gold.

Rules:
- "10" vs "ten": yes (same number, different format)
- "Boston" vs "Boston, Massachusetts": yes (Boston is part of MA, the city is unambiguous)
- "Pavel Alexandrov" vs "Pavel Sergeyevich Alexandrov": yes (full name vs short, same person)
- "1945" vs "1945 to 1951": NO (a specific year vs a range — different)
- "Captain America" vs "superhero roles as the Marvel Comics": NO (specific character vs general role)
- "Sherlock Holmes" vs "Sir Arthur Conan Doyle": NO (character vs author)
- "yes" vs "Yes, they are": yes (same yes/no with verbose form)
- "Justin Spitzer" vs "Justin Spitzer": yes (identical)
- "" or "Not in passages" vs anything: NO (no answer is not equivalent)

Output ONLY one word: yes or no (lowercase, nothing else)."""


def soft_em_score(client, model: str, question: str, gold: str, pred: str,
                  cache: Optional[dict] = None,
                  provider: str = "openai") -> float:
    """Returns 1.0 (equivalent) or 0.0 (not).

    ``provider``:
      - ``openai``: client is an OpenAI-compatible client (Together AI, OpenAI, ...)
      - ``gemini``: client is ``google.genai.Client``
    """
    if not pred or not pred.strip():
        return 0.0
    if normalize_answer(pred) == normalize_answer(gold):
        return 1.0  # trivially equivalent (hard EM = 1)
    if cache is not None:
        key = (gold.lower().strip(), pred.lower().strip())
        if key in cache:
            return cache[key]

    user_msg = f"Question: {question}\nGold: {gold!r}\nPredicted: {pred!r}\n\nSemantically equivalent?"

    if provider == "gemini":
        full_prompt = f"{JUDGE_SYSTEM}\n\n{user_msg}"
        resp = client.models.generate_content(
            model=model, contents=full_prompt,
            config={"temperature": 0.0},
        )
        text = resp.text
        if text is None and getattr(resp, "candidates", None):
            for cand in resp.candidates:
                if cand.content and cand.content.parts:
                    text = cand.content.parts[0].text
                    break
        raw = (text or "").strip().lower()
    else:
        msgs = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        resp = client.chat.completions.create(model=model, messages=msgs,
                                              max_tokens=8, temperature=0)
        raw = resp.choices[0].message.content.strip().lower()

    result = 1.0 if raw.startswith("yes") else 0.0
    if cache is not None:
        cache[(gold.lower().strip(), pred.lower().strip())] = result
    return result


def score_predictions(predictions: list[dict], judge_client, judge_model: str,
                      cache_path: Optional[str | Path] = None,
                      provider: str = "openai") -> dict:
    """Score a list of ``{"question", "gold", "pred"}`` dicts.

    Returns ``{"hard_em", "soft_em", "soft_em_delta", "n", "per_question"}``.
    Caches judge calls to ``cache_path`` (per-provider) when supplied.
    """
    cache: dict[tuple[str, str], float] = {}
    if cache_path:
        cp = Path(cache_path)
        if cp.exists():
            try:
                raw = json.loads(cp.read_text(encoding="utf-8"))
                cache = {tuple(json.loads(k)): v for k, v in raw.items()}
            except Exception:
                cache = {}

    n = len(predictions)
    hard_total = soft_total = 0.0
    out_per_q = []
    for r in predictions:
        gold, pred = r["gold"], r.get("pred", "")
        hard = em_score(pred, gold)
        if hard >= 1.0:
            soft = 1.0
        else:
            try:
                soft = soft_em_score(judge_client, judge_model, r.get("question", ""),
                                     gold, pred, cache=cache, provider=provider)
            except Exception:  # noqa: BLE001
                soft = hard
        hard_total += hard
        soft_total += soft
        out_per_q.append({**r, "hard_em": hard, "soft_em": soft})

    if cache_path:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(
            {json.dumps(list(k)): v for k, v in cache.items()}, indent=2,
        ))

    hem = hard_total / max(1, n)
    sem = soft_total / max(1, n)
    return {
        "n": n,
        "hard_em": hem,
        "soft_em": sem,
        "soft_em_delta": sem - hem,
        "per_question": out_per_q,
    }


def gemini_client(api_key: Optional[str] = None,
                  *,
                  allow_custom_endpoint: bool = False):
    """Return a configured ``google.genai.Client`` (lazy import).

    Requires ``pip install mothrag[gemini]`` (adds ``google-genai``).

    ``allow_custom_endpoint`` is accepted for API symmetry with
    :func:`together_client`; the Gemini SDK does not expose a user-controlled
    ``base_url`` here, so the parameter is currently a no-op.
    """
    del allow_custom_endpoint  # reserved for future use; no base_url to validate
    from google import genai  # type: ignore

    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY required for Gemini judge")
    return genai.Client(api_key=api_key)


def together_client(api_key: Optional[str] = None,
                    base_url: str = "https://api.together.xyz/v1",
                    *,
                    allow_custom_endpoint: bool = False):
    """Return a configured Together AI / OpenAI-compatible client.

    ``base_url`` is validated against :data:`mothrag.utils.url_safety.ALLOWED_HOSTS`
    to prevent accidental API-key exfiltration. Pass
    ``allow_custom_endpoint=True`` to opt into a self-hosted / proxy endpoint
    (a warning is logged in that case).
    """
    from openai import OpenAI

    api_key = api_key or os.getenv("TOGETHER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY or OPENAI_API_KEY required")
    validated_base_url = validate_base_url(
        base_url, allow_custom_endpoint=allow_custom_endpoint,
    )
    return OpenAI(api_key=api_key, base_url=validated_base_url)
