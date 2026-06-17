# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""BridgeRAG-Haiku single-arm orchestrator.

Ties the five stages of Bacellar's bridge-conditioned retrieval into one
``bridge_arm.retrieve(question) -> BridgeResult``:

    1. hop-1 ANN            -> bridge passage ``b`` (top-1)
    2. SVO hop-2 expansion  -> N SVO queries -> ANN union (SVO-pool)
    3. dual-entity expansion-> (e1, e2) -> per-entity ANN -> pool P
    4. tripartite judge     -> s(q, b, e1, e2, c_i) for each c_i in P
    5. PIT fusion           -> rank by (1-alpha)*PIT_judge + alpha*PIT_svo

The dense ANN is injected as a callable ``ann_retrieve(query_text, top_k)
-> list[Candidate]`` so this orchestrator is backend-agnostic and fully
unit-testable without a live index. The PROD gemini-embedding-2 ANN is wired
in; the three LLM stages default to Claude Haiku but accept injected
(mock) instances.

Cost guard: the per-query running cost is accumulated in ``BridgeStats``;
once it would exceed ``config.max_cost_usd`` the remaining LLM stages are
skipped and the arm degrades to the dense pool (never silently overspends).

Anti-leak: operates on question text + retrieved passages only; no gold/F1.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Callable, Optional, Sequence

from mothrag.retrieval.bridge_haiku.entity_extractor import DualEntityExtractor
from mothrag.retrieval.bridge_haiku.pit_fusion import rank_candidates
from mothrag.retrieval.bridge_haiku.svo_generator import SVOQueryGenerator
from mothrag.retrieval.bridge_haiku.tripartite_judge import TripartiteJudge
from mothrag.retrieval.bridge_haiku.types import (
    BridgeConfig,
    BridgeResult,
    BridgeStats,
    Candidate,
)

logger = logging.getLogger(__name__)

AnnRetrieve = Callable[[str, int], Sequence[Candidate]]


class BridgeArmDegraded(RuntimeError):
    """Raised when a bridge LLM stage's POST-RETRY failure rate exceeds the
    fail-fast threshold over the sliding window — the fire is producing
    garbage, so stop loudly instead of silently degrading for 19 min.
    Callers must let this propagate (NOT swallow it with the per-query
    graceful-degrade except)."""


class BridgeArm:
    """Bridge-conditioned single arm. ``ann_retrieve`` is required."""

    name = "bridge_arm"

    # Cross-query fail-fast: if a stage's POST-RETRY
    # failure rate exceeds ``_DEGRADE_RATE`` over the last ``_DEGRADE_WINDOW``
    # calls (once at least ``_DEGRADE_MIN_SAMPLES`` are observed), raise
    # BridgeArmDegraded. The BridgeArm instance persists across queries (it is
    # built once per runner), so these windows span the whole fire.
    _DEGRADE_WINDOW = 50
    _DEGRADE_RATE = 0.10
    _DEGRADE_MIN_SAMPLES = 20

    def __init__(
        self,
        ann_retrieve: AnnRetrieve,
        *,
        config: Optional[BridgeConfig] = None,
        svo_generator: Optional[SVOQueryGenerator] = None,
        entity_extractor: Optional[DualEntityExtractor] = None,
        judge: Optional[TripartiteJudge] = None,
        require_backend: bool = True,
    ) -> None:
        if ann_retrieve is None:
            raise ValueError("BridgeArm requires an ann_retrieve callable")
        self.ann_retrieve = ann_retrieve
        self.cfg = config or BridgeConfig()
        _variant = getattr(self.cfg, "prompt_variant", "v1")
        _provider = getattr(self.cfg, "judge_provider", "anthropic")
        self.svo = svo_generator or SVOQueryGenerator(
            model=self.cfg.judge_model, provider=_provider,
            require_backend=require_backend, prompt_variant=_variant)
        self.entities = entity_extractor or DualEntityExtractor(
            model=self.cfg.judge_model, provider=_provider,
            require_backend=require_backend, prompt_variant=_variant)
        self.judge = judge or TripartiteJudge(
            model=self.cfg.judge_model, provider=_provider,
            require_backend=require_backend, prompt_variant=_variant)
        self._fail_windows: dict[str, deque] = {
            stage: deque(maxlen=self._DEGRADE_WINDOW)
            for stage in ("svo", "entity", "judge")
        }

    def _record(self, stage: str, failed: bool) -> None:
        """Track a stage call outcome; fail fast on systemic degradation."""
        dq = self._fail_windows[stage]
        dq.append(1 if failed else 0)
        if len(dq) >= self._DEGRADE_MIN_SAMPLES:
            n_fail = sum(dq)
            rate = n_fail / len(dq)
            if rate > self._DEGRADE_RATE:
                raise BridgeArmDegraded(
                    f"bridge '{stage}' stage post-retry failure rate "
                    f"{n_fail}/{len(dq)} = {rate:.0%} > {self._DEGRADE_RATE:.0%} "
                    f"over the last {len(dq)} calls — failing fast (anti-waste)")

    # ---- internal: dedup-cap union of candidate lists ---------------
    @staticmethod
    def _union(pools: Sequence[Sequence[Candidate]], cap: int) -> list[Candidate]:
        out: list[Candidate] = []
        seen: set[str] = set()
        for pool in pools:
            for c in pool:
                if c.passage_id in seen:
                    continue
                seen.add(c.passage_id)
                out.append(c)
                if len(out) >= cap:
                    return out
        return out

    def _over_budget(self, stats: BridgeStats) -> bool:
        return stats.estimated_cost_usd >= self.cfg.max_cost_usd

    # ---- public entry point -----------------------------------------
    def retrieve(self, question: str) -> BridgeResult:
        cfg = self.cfg
        stats = BridgeStats()
        result = BridgeResult(question=question, bridge_passage_id=None, stats=stats)
        if not question:
            return result

        # Stage 1 — hop-1 ANN -> bridge passage (top-1)
        hop1 = list(self.ann_retrieve(question, cfg.hop1_top_k))
        if not hop1:
            return result
        bridge = hop1[0]
        result.bridge_passage_id = bridge.passage_id

        # Stage 2 — SVO hop-2 expansion
        svo_pools: list[Sequence[Candidate]] = []
        if not self._over_budget(stats):
            _f0 = stats.svo_failures
            svo_queries, *_ = self.svo.generate(
                question, bridge.text, n=cfg.n_svo_queries, stats=stats)
            self._record("svo", stats.svo_failures > _f0)
            result.svo_queries = tuple(svo_queries)
            for sq in svo_queries:
                svo_pools.append(list(self.ann_retrieve(sq, cfg.svo_top_k)))
        svo_pool = self._union(svo_pools, cfg.svo_pool_cap)

        # Stage 3 — dual-entity expansion
        entity_pools: list[Sequence[Candidate]] = []
        if not self._over_budget(stats):
            _f0 = stats.entity_failures
            (e1, e2), *_ = self.entities.extract(question, bridge.text, stats=stats)
            self._record("entity", stats.entity_failures > _f0)
            result.entities = (e1, e2)
            for ent in (e1, e2):
                if ent:
                    entity_pools.append(list(self.ann_retrieve(ent, cfg.entity_top_k)))

        # Pool P = hop1 ∪ SVO-pool ∪ entity-pools  (bridge stays in the pool)
        pool = self._union([hop1, svo_pool, *entity_pools], cfg.pool_cap)
        result.pool_size = len(pool)
        if not pool:
            return result

        pool_ids = [c.passage_id for c in pool]
        svo_scores = [c.ann_score for c in pool]   # dense similarity channel

        # Stage 4 — tripartite judge (batched). Budget-guarded: if over, fall
        # back to the dense ANN order (judge_scores = ann_score so PIT_judge
        # mirrors PIT_svo and the fused rank == dense rank).
        if self._over_budget(stats):
            judge_scores = list(svo_scores)
            logger.info("bridge_arm: cost cap reached, skipping judge "
                        "(est $%.4f >= $%.2f)", stats.estimated_cost_usd,
                        cfg.max_cost_usd)
        else:
            (e1, e2) = result.entities if result.entities else ("", "")
            _f0 = stats.judge_failures
            judge_scores, *_ = self.judge.score(
                question, bridge.text, e1, e2, [c.text for c in pool],
                lo=cfg.judge_score_min, hi=cfg.judge_score_max,
                max_tokens=cfg.judge_max_tokens, stats=stats)
            self._record("judge", stats.judge_failures > _f0)

        # Stage 5 — PIT fusion + final ranking
        result.ranked_passage_ids = rank_candidates(
            pool_ids, judge_scores, svo_scores,
            alpha=cfg.alpha, top_k=cfg.final_top_k)
        pid_to_text = {c.passage_id: c.text for c in pool}
        result.ranked_texts = [pid_to_text.get(pid, "")
                               for pid in result.ranked_passage_ids]
        return result


__all__ = ["BridgeArm", "AnnRetrieve"]
