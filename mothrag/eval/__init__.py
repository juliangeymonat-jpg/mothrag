# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Evaluation utilities: HotpotQA-style metrics, soft EM, faithfulness, latency."""

from mothrag.eval.metrics import (
    normalize_answer,
    em_score,
    f1_score,
    recall_at_k,
    ndcg_at_k,
    percentile,
    aggregate_metrics,
)
from mothrag.eval.soft_em import (
    soft_em_score,
    score_predictions,
    JUDGE_SYSTEM as SOFT_EM_JUDGE_SYSTEM,
)
from mothrag.eval.faithfulness import (
    faithfulness_score,
    score_faithfulness,
    JUDGE_SYSTEM as FAITHFULNESS_JUDGE_SYSTEM,
)
from mothrag.eval.normalize_questions import (
    normalize_musique,
    normalize_2wiki,
)
from mothrag.eval.latency import analyze as analyze_latency

__all__ = [
    "normalize_answer",
    "em_score",
    "f1_score",
    "recall_at_k",
    "ndcg_at_k",
    "percentile",
    "aggregate_metrics",
    "soft_em_score",
    "score_predictions",
    "SOFT_EM_JUDGE_SYSTEM",
    "faithfulness_score",
    "score_faithfulness",
    "FAITHFULNESS_JUDGE_SYSTEM",
    "normalize_musique",
    "normalize_2wiki",
    "analyze_latency",
]
