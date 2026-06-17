# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Run MothRAG on a HotpotQA-style question file and print EM/F1.

Same workflow as ``mothrag-smoke`` but as a script — useful as a starting
point for paper-style evaluations or custom metrics.
"""

import argparse
import json
import time
from pathlib import Path

from mothrag.api.simple import run as simple_run
from mothrag.eval.metrics import em_score, f1_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="./data_hotpot_mini")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--max", type=int, default=20)
    ap.add_argument("--config", default="default")
    ap.add_argument("--reader", default="llama-3.3-70b")
    ap.add_argument("--out", default=None,
                    help="Optional: write per-question results JSON to this path.")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))[: args.max]
    qtexts = [q["question"] for q in questions]

    t0 = time.time()
    results = simple_run(qtexts, corpus_path=args.corpus,
                          reader=args.reader, config=args.config)
    elapsed = time.time() - t0

    em_total = f1_total = 0.0
    per_q = []
    for q, r in zip(questions, results):
        gold = q.get("answer", "")
        em = em_score(r.answer, gold)
        f1 = f1_score(r.answer, gold)
        em_total += em
        f1_total += f1
        per_q.append({
            "qid": q.get("_id"), "question": q["question"],
            "gold": gold, "pred": r.answer, "em": em, "f1": f1,
            "latency_s": r.latency_s, "route": r.route,
        })

    n = len(per_q)
    summary = {"n": n, "em": em_total / n, "f1": f1_total / n,
               "elapsed_s": elapsed,
               "config": args.config, "reader": args.reader}
    print(json.dumps(summary, indent=2))

    if args.out:
        Path(args.out).write_text(
            json.dumps({"summary": summary, "per_question": per_q}, indent=2)
        )
        print(f"Saved per-question results to {args.out}")


if __name__ == "__main__":
    main()
