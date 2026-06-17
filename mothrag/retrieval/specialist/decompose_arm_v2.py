# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""M8 DecomposeArm 2.0 — chain-coherent compositional decomposition.

Target cluster ``compositional_attribute`` — "What year did a director of a North Korean
cinema film get kidnapped?" — 157 MQ queries (21.5% of the MQ gap) + the
3hop/4+hop residual after M5-M7.

The "2.0" over the production decompose arm is the **chain-coherence
validator**: after retrieving per sub-question, it checks that each hop's answer
actually THREADS into the next hop's retrieved context (shared entity). A
decomposition that drifts (sub-Q2's context never mentions sub-Q1's answer)
yields an incoherent chain → the arm signals a fallback to the generic stack
instead of recomposing a broken chain into a confidently-wrong answer.

  decompose (LLM, chained placeholders) → per-sub-Q retrieve → chain-coherence
  validate → recompose (balanced union) | fallback-if-broken

Backend-agnostic: the decomposer (LLM), per-sub-Q retriever, the per-sub-Q
answerer (reader) and the NER are all INJECTED callables, so the arm — and
especially the coherence validator — is fully offline-testable. Anti-leak:
question text + answers + retrieved context only; no gold, no per-dataset
branching.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from mothrag.retrieval.specialist.compare_arm import Candidate

# decomposer(question) -> list[sub_question]        (sub-Qs may carry {prev}/{1}.. placeholders)
Decomposer = Callable[[str], Sequence[str]]
# retriever(sub_question, top_k) -> sequence[Candidate]
SubQRetriever = Callable[[str, int], Sequence[Candidate]]
# answerer(sub_question, context_texts) -> str
SubQAnswerer = Callable[[str, Sequence[str]], str]
# ner(text) -> sequence[str]
NER = Callable[[str], Sequence[str]]


@dataclass
class DecomposeConfig:
    per_subq_top_k: int = 5
    final_top_k: int = 10
    max_sub_questions: int = 4
    min_chain_overlap: int = 1     # shared entities required for a coherent link


@dataclass
class HopResult:
    sub_question: str
    answer: str = ""
    passage_ids: list[str] = field(default_factory=list)
    context_texts: list[str] = field(default_factory=list)


@dataclass
class ChainCoherence:
    links: list[bool] = field(default_factory=list)        # per consecutive hop-pair
    shared: list[list[str]] = field(default_factory=list)  # shared entities per link
    coherent: bool = True
    broken_at: Optional[int] = None                        # index of first broken link


@dataclass
class DecomposeResult:
    question: str
    sub_questions: list[str] = field(default_factory=list)
    hops: list[HopResult] = field(default_factory=list)
    coherence: ChainCoherence = field(default_factory=ChainCoherence)
    ranked_passage_ids: list[str] = field(default_factory=list)
    fallback: bool = False     # True → chain broke (or undecomposable): use M7 alone


# ---- detection -------------------------------------------------------------

_WH = re.compile(r"\b(what|who|which|when|where|how\s+many|whose|whom)\b", re.IGNORECASE)
_COMPOSITIONAL_PATTERNS: tuple[str, ...] = (
    r"\bof\s+(?:a|an)\b",                       # "a director of a film" (indefinite chain)
    r"(?:\bof\s+the\b[^?]*?){2,}",              # 2+ nested "of the"
    r"\b(?:who|which|that)\b[^?]*\bof\b",       # relative clause threading into "of"
    r"\bin\s+the\s+\w+\s+(?:where|in\s+which)\b",
)


def contains_compositional_markers(question: str) -> bool:
    if not question:
        return False
    if not _WH.search(question):
        return False
    return any(re.search(p, question, re.IGNORECASE) for p in _COMPOSITIONAL_PATTERNS)


def needs_decomposition(question: str, gold_n_estimate: int = 3) -> bool:
    """Fire on estimated 3+ supporting docs OR compositional structure."""
    return gold_n_estimate >= 3 or contains_compositional_markers(question)


# ---- entity extraction for chain coherence ---------------------------------

_TITLECASE = re.compile(r"\b([A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*)*)")
_NUMBER = re.compile(r"\b(\d{2,4})\b")
_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "is", "was",
    "what", "who", "which", "when", "where", "did", "do", "does", "year",
}


def chain_key_terms(text: str, *, ner: Optional[NER] = None) -> set[str]:
    """Salient terms used to test whether an answer threads into a context:
    injected NER entities, else title-case spans + multi-digit numbers + the
    answer's significant content tokens (lower-cased)."""
    if not text:
        return set()
    terms: set[str] = set()
    if ner is not None:
        try:
            terms |= {str(e).strip().lower() for e in ner(text) if str(e).strip()}
        except Exception:  # noqa: BLE001 — fall through to the heuristic
            pass
    terms |= {m.group(1).strip().lower() for m in _TITLECASE.finditer(text)}
    terms |= {m.group(1) for m in _NUMBER.finditer(text)}
    # significant content tokens (short answers are often a single token).
    for tok in re.findall(r"[A-Za-z][A-Za-z'-]+", text):
        low = tok.lower()
        if len(low) >= 4 and low not in _STOP:
            terms.add(low)
    return {t for t in terms if t}


def _context_blob(texts: Sequence[str]) -> str:
    return " \n ".join(t for t in texts if t).lower()


def validate_chain_coherence(
    hops: Sequence[HopResult], *,
    ner: Optional[NER] = None,
    min_overlap: int = 1,
) -> ChainCoherence:
    """A chain is coherent iff every hop's answer reappears (≥ ``min_overlap``
    shared terms) in the NEXT hop's retrieved context — i.e. the decomposition
    actually links hop to hop. Fewer than 2 hops is trivially coherent."""
    coh = ChainCoherence()
    if len(hops) < 2:
        return coh
    for i in range(len(hops) - 1):
        ans_terms = chain_key_terms(hops[i].answer, ner=ner)
        nxt_blob = _context_blob(hops[i + 1].context_texts)
        shared = sorted(t for t in ans_terms if t and t in nxt_blob)
        link_ok = len(shared) >= min_overlap
        coh.links.append(link_ok)
        coh.shared.append(shared)
        if not link_ok and coh.broken_at is None:
            coh.broken_at = i
    coh.coherent = all(coh.links) if coh.links else True
    return coh


# ---- placeholder chaining --------------------------------------------------

_PLACEHOLDER = re.compile(r"\{(?:prev|answer|\d+)\}", re.IGNORECASE)


def _resolve_placeholders(sub_q: str, prior_answers: list[str]) -> str:
    """Replace {prev}/{answer}/{N} with prior hop answers (1-indexed for {N})."""
    if not _PLACEHOLDER.search(sub_q):
        return sub_q

    def repl(m: re.Match) -> str:
        tok = m.group(0)[1:-1].lower()
        if tok in ("prev", "answer"):
            return prior_answers[-1] if prior_answers else ""
        try:
            idx = int(tok) - 1
            return prior_answers[idx] if 0 <= idx < len(prior_answers) else ""
        except ValueError:
            return ""

    return _PLACEHOLDER.sub(repl, sub_q).strip()


# ---- the arm ---------------------------------------------------------------

class DecomposeArmV2:
    """Chain-coherent compositional decomposition arm. ``retriever`` required."""

    name = "decompose_arm_v2"

    def __init__(
        self,
        retriever: SubQRetriever,
        *,
        decomposer: Optional[Decomposer] = None,
        answerer: Optional[SubQAnswerer] = None,
        ner: Optional[NER] = None,
        config: Optional[DecomposeConfig] = None,
    ) -> None:
        if retriever is None:
            raise ValueError("DecomposeArmV2 requires a retriever callable")
        self.retriever = retriever
        self.decomposer = decomposer
        self.answerer = answerer
        self.ner = ner
        self.cfg = config or DecomposeConfig()

    def applicable(self, question: str, gold_n_estimate: int = 3) -> bool:
        return needs_decomposition(question, gold_n_estimate)

    def _decompose(self, question: str) -> list[str]:
        if self.decomposer is not None:
            try:
                subs = [str(s).strip() for s in self.decomposer(question) if str(s).strip()]
                if subs:
                    return subs[: self.cfg.max_sub_questions]
            except Exception:  # noqa: BLE001 — fall back to single-hop
                pass
        return [question]   # undecomposable → single hop (validator → trivially coherent)

    @staticmethod
    def _balanced_union(hops: Sequence[HopResult], cap: int) -> list[str]:
        ranked: list[str] = []
        seen: set[str] = set()
        cols = [list(h.passage_ids) for h in hops]
        depth = max((len(c) for c in cols), default=0)
        for d in range(depth):
            for col in cols:
                if d < len(col) and col[d] not in seen:
                    seen.add(col[d])
                    ranked.append(col[d])
                    if len(ranked) >= cap:
                        return ranked
        return ranked

    def retrieve(self, question: str) -> DecomposeResult:
        sub_qs = self._decompose(question)
        result = DecomposeResult(question=question, sub_questions=list(sub_qs))
        prior_answers: list[str] = []
        for sq in sub_qs:
            resolved = _resolve_placeholders(sq, prior_answers)
            try:
                cands = list(self.retriever(resolved, self.cfg.per_subq_top_k))
            except Exception:  # noqa: BLE001 — one hop's retrieval failure ≠ whole arm
                cands = []
            ctx = [getattr(c, "text", "") or "" for c in cands]
            ans = ""
            if self.answerer is not None:
                try:
                    ans = str(self.answerer(resolved, ctx) or "")
                except Exception:  # noqa: BLE001
                    ans = ""
            prior_answers.append(ans)
            result.hops.append(HopResult(
                sub_question=resolved, answer=ans,
                passage_ids=[str(c.passage_id) for c in cands], context_texts=ctx))

        result.coherence = validate_chain_coherence(
            result.hops, ner=self.ner, min_overlap=self.cfg.min_chain_overlap)
        # A broken chain (or a single undecomposable hop) → fall back to M7.
        result.fallback = (not result.coherence.coherent) or (len(sub_qs) < 2)
        result.ranked_passage_ids = self._balanced_union(result.hops, self.cfg.final_top_k)
        return result


__all__ = [
    "DecomposeArmV2",
    "DecomposeConfig",
    "DecomposeResult",
    "HopResult",
    "ChainCoherence",
    "needs_decomposition",
    "contains_compositional_markers",
    "validate_chain_coherence",
    "chain_key_terms",
]
