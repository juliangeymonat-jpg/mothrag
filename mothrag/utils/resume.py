# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Resume-from-partial helpers for long-running eval jobs.

Eval JSONs follow this shape::

    {
      "summary": {...},
      "per_question": [{"qid": "...", "em": ..., "f1": ..., ...}, ...]
    }

``resume_partial_eval`` returns the set of already-processed ``qid`` plus the
running aggregate counters, allowing the caller to skip completed questions
and accumulate further results on top of the existing JSON.
"""

import json
from pathlib import Path


def resume_partial_eval(out_path: str | Path, *, fields: tuple[str, ...] = ("em", "f1")
                         ) -> tuple[list[dict], set[str], dict[str, float]]:
    """Load a partial eval JSON.

    Returns ``(per_question, done_qids, totals)`` where ``totals`` sums each
    field in ``fields`` across already-processed questions. If the file does
    not exist or fails to parse, returns empty containers.
    """
    p = Path(out_path)
    if not p.exists():
        return [], set(), {f: 0.0 for f in fields}
    try:
        existing = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return [], set(), {f: 0.0 for f in fields}

    per_q = existing.get("per_question", [])
    done = {r["qid"] for r in per_q if "qid" in r}
    totals = {f: float(sum(r.get(f, 0.0) for r in per_q)) for f in fields}
    return per_q, done, totals


def save_partial_eval(out_path: str | Path,
                       summary: dict,
                       per_question: list[dict]) -> None:
    """Atomically persist a partial eval JSON."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"summary": summary, "per_question": per_question}, indent=2))
