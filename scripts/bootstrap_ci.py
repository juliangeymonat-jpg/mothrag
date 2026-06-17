"""Bootstrap 95% confidence intervals for MothRAG headline numbers.

Resamples per-question results with replacement N times (default 1000),
computes percentile CIs for EM, F1.

Usage:
  python scripts/bootstrap_ci.py --input <prediction_file.json> [--metric em|f1|both] [--n 1000]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
from pathlib import Path


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def em(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = set(p) & set(g)
    matches = sum(min(p.count(t), g.count(t)) for t in common)
    if matches == 0:
        return 0.0
    pr = matches / len(p)
    rc = matches / len(g)
    return 2 * pr * rc / (pr + rc)


def bootstrap_ci(per_question_metrics: list[float],
                 n_resamples: int = 1000,
                 ci_pct: float = 0.95,
                 seed: int = 42) -> tuple[float, float, float]:
    """Returns (mean, ci_lo, ci_hi)."""
    random.seed(seed)
    n = len(per_question_metrics)
    means = []
    for _ in range(n_resamples):
        sample = [per_question_metrics[random.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - ci_pct) / 2
    lo = means[int(alpha * n_resamples)]
    hi = means[int((1 - alpha) * n_resamples)]
    return sum(per_question_metrics) / n, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Prediction JSON file")
    ap.add_argument("--metric", default="both", choices=["em", "f1", "both"])
    ap.add_argument("--n", type=int, default=1000, help="Number of bootstrap resamples")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    per_q = data.get("per_question", [])

    em_vals, f1_vals = [], []
    for r in per_q:
        gold = r.get("gold", "") or r.get("answer", "")
        pred = r.get("pred", "") or r.get("prediction", "")
        # Use precomputed if available, else compute
        em_v = r.get("em")
        if em_v is None:
            em_v = em(pred, gold)
        em_vals.append(em_v)
        f1_v = r.get("f1")
        if f1_v is None:
            f1_v = f1(pred, gold)
        f1_vals.append(f1_v)

    n = len(per_q)
    print(f"\n=== Bootstrap CI (n={n}, n_resamples={args.n}, 95% CI) ===")
    print(f"File: {args.input}\n")

    if args.metric in ("em", "both"):
        mean, lo, hi = bootstrap_ci(em_vals, args.n)
        print(f"  EM  = {mean:.4f}  [{lo:.4f}, {hi:.4f}]  (CI width = {(hi-lo)*100:.2f}pp)")

    if args.metric in ("f1", "both"):
        mean, lo, hi = bootstrap_ci(f1_vals, args.n)
        print(f"  F1  = {mean:.4f}  [{lo:.4f}, {hi:.4f}]  (CI width = {(hi-lo)*100:.2f}pp)")

    print()


if __name__ == "__main__":
    main()
