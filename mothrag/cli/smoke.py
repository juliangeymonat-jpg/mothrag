# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Smoke test CLI: run MothRAG on a tiny HotpotQA-style subset.

Usage::

    mothrag-smoke --corpus path/to/data_wiki_hotpotqa --questions path/to/hotpotqa.json --max 5

Prints per-question EM/F1, retrieval recall, latency, and an aggregate summary.
Requires a configured reader API key (``TOGETHER_API_KEY`` by default).
"""

import argparse
import json
import sys
from pathlib import Path

from mothrag.api.simple import run as simple_run
from mothrag.eval.metrics import em_score, f1_score


def main():
    ap = argparse.ArgumentParser(description="MothRAG smoke test on a HotpotQA-style subset.")
    ap.add_argument("--corpus", required=True,
                    help="Preprocessed corpus dir (entities.json, edges.json, chunks.jsonl)")
    ap.add_argument("--questions", required=True,
                    help="HotpotQA-style questions JSON (list of {_id, question, answer, ...})")
    ap.add_argument("--max", type=int, default=5,
                    help="Number of questions to evaluate (default 5).")
    ap.add_argument("--reader", default="llama-3.3-70b",
                    help="Reader alias (see mothrag.api.simple.READER_ALIASES) or raw model id.")
    ap.add_argument("--config", default="default",
                    help="Pipeline config preset (default | fast | high-quality).")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    selected = questions[: args.max]
    if not selected:
        print("No questions to evaluate.", file=sys.stderr)
        return 1

    qtexts = [q["question"] for q in selected]
    print(f"Running MothRAG ({args.config}) on {len(qtexts)} questions with reader={args.reader} ...",
          file=sys.stderr)
    results = simple_run(qtexts, corpus_path=args.corpus, reader=args.reader, config=args.config)

    em_total = f1_total = 0.0
    for q, r in zip(selected, results):
        gold = q.get("answer", "")
        em = em_score(r.answer, gold)
        f1 = f1_score(r.answer, gold)
        em_total += em
        f1_total += f1
        print(f"  [{q.get('_id', '?')}] EM={em:.0f} F1={f1:.2f}  pred={r.answer!r}  gold={gold!r}  ({r.latency_s:.2f}s)")

    n = len(selected)
    print(f"\nSummary: n={n} EM={em_total/n:.3f} F1={f1_total/n:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
