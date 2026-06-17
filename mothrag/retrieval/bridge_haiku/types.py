# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Shared types for the BridgeRAG-Haiku single-arm POC.

Implements the data contracts for the tripartite-judge bridge-conditioned
retrieval mechanism of Bacellar (arXiv 2604.03384v2, "BRIDGERAG:
Training-Free Bridge-Conditioned Retrieval for Multi-Hop QA"), with Claude
Haiku as the cost-grade judge in place of the paper's self-hosted Llama-3.3
70B AWQ.

Anti-leak: no gold / F1 / dataset fields anywhere. General-purpose — no
per-dataset branching.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Candidate:
    """A retrieval candidate passage.

    ``passage_id`` is the corpus-stable id used for the R@K metric;
    ``text`` is the passage body fed to the judge; ``ann_score`` is the
    raw cosine/IP similarity from the dense ANN stage (used for the SVO
    similarity PIT in fusion).
    """

    passage_id: str
    text: str
    ann_score: float = 0.0


@dataclass
class BridgeStats:
    """Per-run cost + call telemetry (mirrors LLMExtractorStats convention)."""

    n_svo_calls: int = 0
    n_entity_calls: int = 0
    n_judge_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    # Per-stage failure counters (a silenced Exception +
    # neutral fallback is no longer invisible). ``haiku_5xx_count`` counts
    # transient Anthropic 429/503 that exhausted the retry-with-backoff.
    svo_failures: int = 0
    entity_failures: int = 0
    judge_failures: int = 0
    haiku_5xx_count: int = 0

    def add_call(self, kind: str, n_in: int, n_out: int, cost: float) -> None:
        if kind == "svo":
            self.n_svo_calls += 1
        elif kind == "entity":
            self.n_entity_calls += 1
        elif kind == "judge":
            self.n_judge_calls += 1
        self.input_tokens += int(n_in)
        self.output_tokens += int(n_out)
        self.estimated_cost_usd += float(cost)

    def add_failure(self, kind: str, *, is_5xx: bool = False) -> None:
        if kind == "svo":
            self.svo_failures += 1
        elif kind == "entity":
            self.entity_failures += 1
        elif kind == "judge":
            self.judge_failures += 1
        if is_5xx:
            self.haiku_5xx_count += 1


@dataclass
class BridgeConfig:
    """Knobs for the 5-stage bridge pipeline (paper defaults).

    All keyword-defaulted; general-purpose. The Haiku judge model is
    pinned but overridable for the Sonnet-upgrade ablation in the
    decision gate.
    """

    # Stage 1 — hop-1 ANN
    hop1_top_k: int = 5              # K1: top-K1 for hop-1; top-1 = bridge `b`
    # Stage 2 — SVO expansion
    n_svo_queries: int = 3          # N SVO queries generated from (q, b)
    svo_top_k: int = 10             # ANN top-K per SVO query
    svo_pool_cap: int = 15          # SVO-15 union cap
    # Stage 3 — dual-entity expansion
    entity_top_k: int = 5           # ANN top-K per extracted entity (e1, e2)
    pool_cap: int = 20             # final pool P cap (SVO-15 ∪ e1-5 ∪ e2-5)
    # Stage 4 — tripartite judge
    judge_model: str = "claude-haiku-4-5"
    # Provider for the bridge LLM stages (SVO / entity / judge).
    # "anthropic" (default) or "gemini" (Flash).
    judge_provider: str = "anthropic"
    judge_max_tokens: int = 1024
    # Haiku prompt variant for the SVO / entity / judge stages.
    # "v1" = the validated prompts (default, no regression); "v2" =
    # tightened + few-shot variants.
    prompt_variant: str = "v1"
    judge_score_min: float = 0.0
    judge_score_max: float = 10.0
    # Stage 5 — PIT fusion
    alpha: float = 0.1             # f = (1-alpha)*PIT_judge + alpha*PIT_svo
    final_top_k: int = 5           # return top-5 by fused score
    # cost guard (single-arm; runner enforces the cross-query total)
    max_cost_usd: float = 10.0


@dataclass
class BridgeResult:
    """Output of one bridge-arm retrieval."""

    question: str
    bridge_passage_id: Optional[str]
    entities: tuple[str, ...] = ()
    svo_queries: tuple[str, ...] = ()
    ranked_passage_ids: list[str] = field(default_factory=list)
    ranked_texts: list[str] = field(default_factory=list)
    pool_size: int = 0
    stats: BridgeStats = field(default_factory=BridgeStats)


__all__ = ["Candidate", "BridgeStats", "BridgeConfig", "BridgeResult"]
