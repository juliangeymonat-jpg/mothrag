# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Faithfulness LLM judge — RAG groundedness eval (RAGAS-style, multi-judge).

Judges whether each predicted answer is GROUNDED in the retrieved passages
(YES / PARTIAL / NO), then aggregates a faithfulness_score per stack.

Score mapping for aggregation:
  YES     -> 1.0
  PARTIAL -> 0.5
  NO      -> 0.0

``faithfulness_score = mean across queries`` (0.0-1.0, RAGAS-style)
``hallucination_rate = 1.0 - faithfulness_score``

Special handling:
- ``"Not in passages"`` / empty pred -> labelled NO (system correctly refused),
  but treated as ``abstain`` — counted separately and excluded from the
  strict score (also report inclusive variant for transparency).
"""

import json
from pathlib import Path
from typing import Optional


JUDGE_SYSTEM = """You are a faithfulness judge for a RAG question-answering system.
Given a question, the retrieved passages, and a predicted answer, decide if the predicted answer's
claim is GROUNDED in the passages or HALLUCINATED.

Output ONE of three labels:
- YES: predicted answer's claim explicitly appears in or is directly supported by the passages
- PARTIAL: predicted answer is partially supported (some claim in passages, some inferred)
- NO: predicted answer is not supported by passages (hallucination, parametric memory, or refusal)

Special cases:
- "Not in passages" or empty pred: output NO (system correctly refused)
- Prediction matches passage text verbatim or near-verbatim: YES
- Prediction is a reasonable inference from multiple passages (multi-hop): YES
- Prediction is correct factually but not derivable from passages: NO (parametric memory, would be hallucination if passages were the only source)

Output ONLY one word: yes, partial, or no (lowercase, no punctuation)."""


ABSTAIN_PATTERNS = (
    "not in passages",
    "not in the passages",
    "no answer",
    "cannot determine",
    "i don't know",
    "i do not know",
)


def is_abstain(pred: str) -> bool:
    if not pred or not pred.strip():
        return True
    p = pred.strip().lower()
    return any(pat in p for pat in ABSTAIN_PATTERNS)


def _label_to_score(label: str) -> float:
    label = (label or "").strip().lower()
    if label.startswith("yes"):
        return 1.0
    if label.startswith("partial"):
        return 0.5
    return 0.0


def _judge_disk_cache_path(question: str, pred: str, model: str):
    """SHA256-keyed file cache for Gemini/OpenAI judge calls.

    Cache hit when SAME (question, pred, model) triple appears across runs.
    Large-scale re-runs with the same config get massive cache hit; A/B
    runs with config-drift get partial.

    Cache dir from ``$MOTHRAG_JUDGE_CACHE_DIR`` env var; None = no cache.
    Anti-leak: hash key is SHA256(model::question::pred) — no DS / gold fields.
    Stored value is the float score (0.0 / 0.5 / 1.0).
    """
    import hashlib
    import os as _os
    import pathlib
    cache_root = _os.environ.get("MOTHRAG_JUDGE_CACHE_DIR")
    if not cache_root:
        return None
    p = pathlib.Path(cache_root)
    p.mkdir(parents=True, exist_ok=True)
    key_str = f"{model}::{question.lower().strip()}::{pred.lower().strip()}"
    key = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
    return p / f"{key}.json"


def faithfulness_score(client, model: str, question: str, passages: list[str],
                       pred: str, cache: Optional[dict] = None,
                       provider: str = "openai") -> tuple[float, str]:
    """Returns ``(score, label)`` where ``score`` is in {0.0, 0.5, 1.0}."""
    if is_abstain(pred):
        return 0.0, "no"
    # In-memory cache (legacy)
    if cache is not None:
        key = (question.lower().strip(), pred.lower().strip())
        if key in cache:
            score = cache[key]
            label = "yes" if score == 1.0 else ("partial" if score == 0.5 else "no")
            return score, label
    # Disk cache (env-var-gated)
    disk_path = _judge_disk_cache_path(question, pred, model)
    if disk_path is not None and disk_path.exists():
        try:
            import json as _json
            score = float(_json.loads(disk_path.read_text(encoding="utf-8"))["score"])
            label = "yes" if score == 1.0 else ("partial" if score == 0.5 else "no")
            return score, label
        except Exception:  # noqa: BLE001 — corrupt cache → live
            pass

    ctx = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    user_msg = (
        f"Passages:\n{ctx}\n\n"
        f"Question: {question}\n\n"
        f"Predicted answer: {pred!r}\n\n"
        f"Faithfulness label:"
    )

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

    score = _label_to_score(raw)
    if cache is not None:
        cache[(question.lower().strip(), pred.lower().strip())] = score
    # Write to disk cache too (env-var-gated)
    if disk_path is not None:
        try:
            import json as _json
            disk_path.write_text(_json.dumps({"score": score, "label": raw}),
                                  encoding="utf-8")
        except Exception:  # noqa: BLE001 — cache write failure non-fatal
            pass
    label = "yes" if score == 1.0 else ("partial" if score == 0.5 else "no")
    return score, label


def score_faithfulness(predictions: list[dict], judge_client, judge_model: str,
                       cache_path: Optional[str | Path] = None,
                       provider: str = "openai") -> dict:
    """Score a list of ``{"question", "passages", "pred"}`` dicts.

    Returns aggregated faithfulness + hallucination + abstain stats.
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
    n_abstain = 0
    score_total_inclusive = 0.0
    score_total_strict = 0.0
    out_per_q = []
    for r in predictions:
        pred = r.get("pred", "")
        passages = r.get("passages", [])
        question = r.get("question", "")
        if is_abstain(pred):
            n_abstain += 1
            score, label = 0.0, "no"
        else:
            try:
                score, label = faithfulness_score(
                    judge_client, judge_model, question, passages, pred,
                    cache=cache, provider=provider,
                )
            except Exception:  # noqa: BLE001
                score, label = 0.0, "no"
            score_total_strict += score
        score_total_inclusive += score
        out_per_q.append({**r, "faithfulness": score, "faithfulness_label": label})

    if cache_path:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(
            {json.dumps(list(k)): v for k, v in cache.items()}, indent=2,
        ))

    n_strict = max(1, n - n_abstain)
    return {
        "n": n,
        "n_abstain": n_abstain,
        "faithfulness_inclusive": score_total_inclusive / max(1, n),
        "faithfulness_strict": score_total_strict / n_strict,
        "hallucination_rate_inclusive": 1.0 - (score_total_inclusive / max(1, n)),
        "hallucination_rate_strict": 1.0 - (score_total_strict / n_strict),
        "per_question": out_per_q,
    }
