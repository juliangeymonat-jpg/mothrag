# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Canonical HotpotQA / SQuAD evaluation metrics.

Yang et al. 2018 normalisation: lowercase + punctuation removal + article
removal (``a/an/the``) + whitespace collapse.

All papers comparing on HotpotQA should use this exact normalisation; deviating
gives non-comparable numbers.
"""

import math
import re
import string
from collections import Counter

import numpy as np


def normalize_answer(s: str) -> str:
    """Lower, strip punctuation, drop articles, collapse whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def em_score(prediction: str, ground_truth: str) -> float:
    """Exact-match (after normalisation)."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 (after normalisation)."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_tokens)
    r = num_same / len(gt_tokens)
    return 2 * p * r / (p + r)


def recall_at_k(predicted: list[str], ground_truth: list[str], k: int) -> float:
    if not ground_truth:
        return 0.0
    pred_set = set(predicted[:k])
    gt_set = set(ground_truth)
    return len(pred_set & gt_set) / len(gt_set)


def ndcg_at_k(predicted: list[str], ground_truth: list[str], k: int) -> float:
    """Binary-relevance NDCG@k."""
    gt_set = set(ground_truth)
    dcg = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in gt_set:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def aggregate_metrics(per_query_results: list[dict]) -> dict:
    """Aggregate per-query result dicts into per-category summary stats."""
    if not per_query_results:
        return {}
    out = {}
    by_cat: dict[str, list[dict]] = {}
    for r in per_query_results:
        by_cat.setdefault(r.get("category", "all"), []).append(r)
        by_cat.setdefault("all", []).append(r)
    for cat, items in by_cat.items():
        out[cat] = {
            "n": len(items),
            "recall@5": float(np.mean([r.get("recall@5", 0.0) for r in items])),
            "recall@10": float(np.mean([r.get("recall@10", 0.0) for r in items])),
            "ndcg@5": float(np.mean([r.get("ndcg@5", 0.0) for r in items])),
            "latency_ms_p50": percentile([r.get("latency_ms", 0.0) for r in items], 50),
            "latency_ms_p95": percentile([r.get("latency_ms", 0.0) for r in items], 95),
            "latency_ms_p99": percentile([r.get("latency_ms", 0.0) for r in items], 99),
            "tokens_in_avg": float(np.mean([r.get("tokens_in", 0) for r in items])),
            "tokens_out_avg": float(np.mean([r.get("tokens_out", 0) for r in items])),
            "cost_cents_avg": float(np.mean([r.get("cost_cents", 0) for r in items])),
        }
    return out
