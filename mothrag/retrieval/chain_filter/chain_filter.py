# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""ChainFilter v0.1 — γ-weighted, hop-gated post-retrieval fact-coverage filter.

NeocorRAG (WWW 2026, arXiv 2604.27852v1) wins F1 on multi-hop QA with an
LLM fact-filter that keeps only the query-relevant facts (triples) before the
reader. We REINVENT this component with our own design twists — NOT a copy of
their DSPy prompt — as an opt-in POST-retrieval reranker over the bridge_top100:

    bridge_top100 (list[Candidate])  ->  ChainFilter.filter(question, cands)  ->  top-5

It is a FILTER/reranker (reshapes the candidate set feeding the reader/arms),
**NOT a 5th arm** — pool-safety preserved by construction.

Our two IP twists vs NeocorRAG's flat "keep ≤4 facts":
  (i)  γ-feedback weighted fact validity — each kept fact's contribution is
       weighted by its γ band (HIGH=fact / MID=uncertain / LOW=anti-context,
       excluded). A passage supported by a CONFIDENT, well-grounded fact ranks
       above one that merely lexically overlaps a low-γ (likely contradicted)
       fact. Reuses ``iterative_ragnatela.gamma_pooling.classify_band`` verbatim.
  (iii) hop-count conditional gating on INPUT FEATURES only — the filter fires
        only on structurally multi-hop questions (``hop_count >= hop_gate_min``
        OR ``is_chain_deep``), derived from question text by
        ``query_type_classifier``. Single-hop questions BYPASS untouched, so
        single-hop accuracy cannot regress and nothing fires needlessly.
        Anti-leak: never a dataset / corpus signal — dataset asymmetry emerges
        organically from the input features.

Everything that touches an LLM (OpenIE triple extraction, the ≤K fact filter,
the γ scorer) is an INJECTED callable with a deterministic, LLM-free default, so
the whole filter is offline-testable and BUILDs with no live fire. Production
swaps the real OpenIE client / γ verifier / LLM fact-selector into the seams.

Graceful degrade: ANY failure (or the gate not firing) returns the input
candidates' top-``top_k_out`` unchanged — the filter never drops the candidate
set, so a ChainFilter fault can only ever no-op, never break retrieval.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from mothrag.iterative_ragnatela.gamma_pooling import classify_band
from mothrag.iterative_ragnatela.types import GammaBand, RagnatelaConfig
from mothrag.retrieval.bridge_haiku.pit_fusion import pit_fuse

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")

# Fact = [subject, predicate, object]
Fact = Sequence[str]
# gamma_scorer(question, fact, candidate_text) -> float in [0, 1]
GammaScorer = Callable[[str, Fact, str], float]
# fact_filter(question, facts) -> kept facts (<= max)
FactFilter = Callable[[str, Sequence[Fact]], Sequence[Fact]]
# triple_extractor(text) -> list[Fact]
TripleExtractor = Callable[[str], Sequence[Fact]]
# classify_fn(question) -> features dict (query_type_classifier.classify_with_features)
ClassifyFn = Callable[[str], dict]


@dataclass
class ChainFilterConfig:
    """Knobs for ChainFilter. All keyword-defaulted; general-purpose."""

    top_k_in: int = 100          # consider this many bridge candidates
    top_k_out: int = 5           # emit this many (R@5 metric)
    max_facts_kept: int = 4      # ≤K query-relevant facts (mirrors NeocorRAG ≤4)
    hop_gate_min: int = 2        # fire only when hop_count >= this (twist iii)
    alpha_ann: float = 0.1       # PIT fuse weight on the dense ann channel
    # γ band weights (twist i): facts carry forward, uncertain = partial credit,
    # anti-context = excluded (a contradicted fact must NOT boost a passage).
    w_high: float = 1.0
    w_mid: float = 0.5
    w_low: float = 0.0
    enabled: bool = False        # opt-in; default OFF == byte-identical passthrough
    # Cohort opt-out. MQ chain_deep is extraction-bound; the OpenIE
    # triple-extract can discard answer-bearing chunks there, so this skips the
    # ChainFilter rerank (pure passthrough) on the chain_deep cohort while
    # leaving every other cohort on the legacy path. Default OFF =
    # byte-identical. Gated by --use-chainfilter-cohort-gate.
    cohort_gate_skip_chain_deep: bool = False
    gamma_cfg: RagnatelaConfig = field(default_factory=RagnatelaConfig)


# --------------------------------------------------------------------------
# deterministic, LLM-free defaults (so BUILD + tests need no live fire)
# --------------------------------------------------------------------------

def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _fact_tokens(fact: Fact) -> set[str]:
    """Tokens of a fact's subject + object (the entities that should ground it)."""
    parts = list(fact)
    ends = (parts[0] if parts else "", parts[-1] if len(parts) > 1 else "")
    return _tokens(" ".join(ends))


def default_gamma_scorer(question: str, fact: Fact, candidate_text: str) -> float:
    """Deterministic γ proxy ∈ [0, 1]: how grounded the fact is in the passage.

    Jaccard of the fact's entity tokens against the candidate text — a confident
    (well-supported) fact has its subject/object present in the passage. Used for
    offline BUILD/tests; production injects the real γ verifier
    (gamma_l4b_andgate valid≈0.9 / partial≈0.5 / invalid≈0.1)."""
    ft = _fact_tokens(fact)
    if not ft:
        return 0.0
    ct = _tokens(candidate_text)
    if not ct:
        return 0.0
    inter = len(ft & ct)
    union = len(ft | ct)
    return inter / union if union else 0.0


def default_fact_filter(max_facts: int) -> FactFilter:
    """LLM-free ≤K fact selector: keep the facts whose entity tokens overlap the
    question most (a relevance proxy for the DSPy fact-filter). Deterministic."""

    def _filter(question: str, facts: Sequence[Fact]) -> list[Fact]:
        q = _tokens(question)
        scored = []
        seen: set[tuple] = set()
        for f in facts:
            key = tuple(str(x).lower() for x in f)
            if key in seen:
                continue
            seen.add(key)
            overlap = len(_fact_tokens(f) & q)
            scored.append((overlap, f))
        # keep only facts with SOME question relevance; strongest first, stable.
        scored = [sf for sf in scored if sf[0] > 0]
        scored.sort(key=lambda sf: sf[0], reverse=True)
        return [f for _, f in scored[:max_facts]]

    return _filter


def _candidate_supports(fact: Fact, candidate_text: str) -> bool:
    """A candidate supports a fact iff BOTH the subject and object entities
    appear in its text (a minimal chain-membership test)."""
    parts = list(fact)
    if not parts:
        return False
    subj = _tokens(parts[0])
    obj = _tokens(parts[-1]) if len(parts) > 1 else set()
    ct = _tokens(candidate_text)
    subj_ok = bool(subj) and subj <= ct
    obj_ok = (not obj) or (obj <= ct)
    return subj_ok and obj_ok


# --------------------------------------------------------------------------
# the filter
# --------------------------------------------------------------------------

class ChainFilter:
    """γ-weighted, hop-gated post-retrieval fact-coverage reranker (NOT an arm)."""

    def __init__(
        self,
        *,
        config: Optional[ChainFilterConfig] = None,
        triple_extractor: Optional[TripleExtractor] = None,
        fact_filter: Optional[FactFilter] = None,
        gamma_scorer: Optional[GammaScorer] = None,
        classify_fn: Optional[ClassifyFn] = None,
    ) -> None:
        self.cfg = config or ChainFilterConfig()
        self.triple_extractor = triple_extractor      # text -> [Fact]; None ⇒ no facts
        self.fact_filter = fact_filter or default_fact_filter(self.cfg.max_facts_kept)
        self.gamma_scorer = gamma_scorer or default_gamma_scorer
        if classify_fn is not None:
            self._classify = classify_fn
        else:
            from mothrag.core.query_type_classifier import classify_with_features
            self._classify = classify_with_features
        # Behavioural telemetry (so an evaluation can prove ON != OFF).
        self.counters: dict[str, int] = {
            "gate_fired": 0,
            "gate_skipped_single_hop": 0,
            "triples_extracted": 0,
            "kept_facts": 0,
            "zero_chain_support_passthrough": 0,
            "exception_passthrough": 0,
            "reranked": 0,            # gate fired AND the order actually changed
            # Chunks whose bridge_score prior (>1.0) actually scaled the
            # chain-density (the pipelined hand-off).
            "bridge_prior_applied": 0,
            # Queries where the chain_deep cohort opt-out skipped the
            # ChainFilter rerank (pure passthrough).
            "cohort_skipped_chain_deep": 0,
        }

    def reset_counters(self) -> None:
        for k in self.counters:
            self.counters[k] = 0

    # ---- twist (iii): input-feature hop gate ----------------------------
    def hop_count(self, features: dict) -> int:
        return max(int(features.get("np_depth", 0) or 0),
                   int(features.get("n_relations", 0) or 0))

    def gate_fires(self, question: str) -> bool:
        """True iff the question is structurally multi-hop (input features only)."""
        if not self.cfg.enabled:
            return False
        try:
            feats = self._classify(question)
        except Exception:  # noqa: BLE001 — a flaky classifier must not break retrieval
            return False
        return (self.hop_count(feats) >= self.cfg.hop_gate_min
                or bool(feats.get("has_chain")))

    def _qtype(self, question: str):
        """Input-feature cohort label (``label_v2``) for the cohort gate;
        None on classifier failure (fail open
        to the legacy ChainFilter path, never suppress on error)."""
        try:
            return self._classify(question).get("label_v2")
        except Exception:  # noqa: BLE001
            return None

    # ---- twist (i): γ-band-weighted fact-coverage score -----------------
    def _band_weight(self, gamma: float) -> float:
        band = classify_band(gamma, self.cfg.gamma_cfg)
        if band is GammaBand.HIGH:
            return self.cfg.w_high
        if band is GammaBand.MID:
            return self.cfg.w_mid
        return self.cfg.w_low

    def _chain_score(self, question: str, candidate: Any, kept: Sequence[Fact]) -> float:
        text = getattr(candidate, "text", "") or ""
        total = 0.0
        for fact in kept:
            if not _candidate_supports(fact, text):
                continue
            g = self.gamma_scorer(question, fact, text)
            g = 0.0 if g < 0.0 else 1.0 if g > 1.0 else g
            total += self._band_weight(g) * g
        # Bridge → ChainFilter pipelined hand-off.
        # The bridge substrate's per-chunk score rides in as a MULTIPLICATIVE
        # PRIOR on the chain-density (early fusion, complementary to the late
        # ann/PIT fusion this same chunk already feeds). NOT flag-gated: it is
        # active iff the producer attached a ``bridge_score``; absent (None) or
        # exactly 1.0 ⇒ byte-identical legacy density. Defensive float-coerce so
        # a non-numeric prior degrades to legacy rather than raising.
        prior = getattr(candidate, "bridge_score", None)
        if prior is not None:
            try:
                p = float(prior)
            except (TypeError, ValueError):
                p = 1.0
            if p != 1.0:
                total *= p
                if p > 1.0:
                    self.counters["bridge_prior_applied"] += 1
        return total

    # ---- public entry point ---------------------------------------------
    def filter(self, question: str, candidates: Sequence[Any]) -> list[Any]:
        """Reshape ``candidates`` (bridge_top100) into the filtered top-K.

        Pass-through (top_k_out, unchanged order) when the gate does not fire or
        anything goes wrong — never drops the candidate set.
        """
        cands = list(candidates)
        out_k = self.cfg.top_k_out
        # Chain_deep cohort opt-out: skip the
        # rerank entirely (pure passthrough) before any classification/extraction
        # work, so the extraction-bound chain_deep cohort keeps its bridge order.
        if (self.cfg.enabled and cands and self.cfg.cohort_gate_skip_chain_deep
                and self._qtype(question) == "chain_deep"):
            self.counters["cohort_skipped_chain_deep"] += 1
            return cands[:out_k]
        if not cands or not self.gate_fires(question):
            # gate did not fire — single-hop bypass (only counted when ENABLED,
            # so an all-OFF run shows 0 activity = ON-vs-OFF is observable).
            if self.cfg.enabled and cands:
                self.counters["gate_skipped_single_hop"] += 1
            return cands[:out_k]
        self.counters["gate_fired"] += 1
        try:
            pool = cands[: self.cfg.top_k_in]
            # 1. triples over the pool (injected extractor; None ⇒ degrade).
            if self.triple_extractor is None:
                self.counters["exception_passthrough"] += 1
                return cands[:out_k]
            all_facts: list[Fact] = []
            for c in pool:
                try:
                    facts = [list(f) for f in self.triple_extractor(getattr(c, "text", "") or "")]
                except Exception:  # noqa: BLE001 — one passage's extraction failing ≠ abort
                    facts = []
                all_facts.extend(facts)
            self.counters["triples_extracted"] += len(all_facts)
            if not all_facts:
                self.counters["zero_chain_support_passthrough"] += 1
                return cands[:out_k]
            # 2. ≤K query-relevant facts (injected filter; deterministic default).
            kept = list(self.fact_filter(question, all_facts))
            self.counters["kept_facts"] += len(kept)
            if not kept:
                self.counters["zero_chain_support_passthrough"] += 1
                return cands[:out_k]
            # 3. γ-band-weighted fact-coverage per candidate (twist i).
            chain_scores = [self._chain_score(question, c, kept) for c in pool]
            if not any(s > 0 for s in chain_scores):
                # nothing in the pool supports a kept fact → don't reshuffle blindly.
                self.counters["zero_chain_support_passthrough"] += 1
                return cands[:out_k]
            # 4. PIT-fuse chain validity with the dense ann channel, rank, top-K.
            ann_scores = [float(getattr(c, "ann_score", 0.0) or 0.0) for c in pool]
            fused = pit_fuse(chain_scores, ann_scores, alpha=self.cfg.alpha_ann)
            order = sorted(range(len(pool)), key=lambda i: fused[i], reverse=True)
            out = [pool[i] for i in order[:out_k]]
            if [c.passage_id for c in out] != [
                    c.passage_id for c in pool[:out_k]]:
                self.counters["reranked"] += 1     # order genuinely changed
            return out
        except Exception:  # noqa: BLE001 — any fault ⇒ safe passthrough
            logger.warning("ChainFilter.filter failed; passthrough top-%d", out_k,
                           exc_info=True)
            self.counters["exception_passthrough"] += 1
            return cands[:out_k]


__all__ = ["ChainFilter", "ChainFilterConfig", "default_gamma_scorer",
           "default_fact_filter"]
