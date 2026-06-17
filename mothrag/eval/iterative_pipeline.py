# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Iterative retrieval loop for multi-hop chaining.

Wraps a one-shot :class:`mothrag.eval.pipeline.MothRAGPipeline` in a state-aware
loop that lets the reader request additional retrieval passes when it cannot
yet answer the question. Designed to lift recall on 4-hop benchmarks
(MuSiQue) where the one-shot retriever caps at R@10 ~0.60.

Loop structure::

    Q  --(retrieve)-->  passages_1
    Q  --(intermediate reader, accumulated_facts=[])-->  ANSWER | MISSING+NEXT_QUERY
    NEXT_QUERY  --(retrieve)-->  passages_2  (union with passages_1, cap=top_k_total)
    Q  --(intermediate reader, accumulated_facts=[fact_1, ...])-->  ANSWER | MISSING+...
    ...
    On final iteration (or stop_early ANSWER), call :meth:`MothRAGPipeline.read`
    with the union of retrieved passages to produce the final answer.

The intermediate-reader prompt is symmetric ANSWER/MISSING (no anchoring bias)
and explicit chain-of-thought (EXTRACT -> INTEGRATE -> ASSESS -> OUTPUT). See
:data:`INTERMEDIATE_SYSTEM`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from mothrag.eval.pipeline import (
    MothRAGPipeline,
    AnswerInfo,
    parse_reader_output,
    _reader_kwargs,
    call_reader_with_usage,
    make_reader_messages,
)


# ---- Intermediate reader prompt (symmetric, with explicit scaffolding) ----

INTERMEDIATE_SYSTEM = """You are solving a multi-hop question iteratively. You have access to passages retrieved across one or more retrieval passes.

Work through this carefully and produce structured output.

1. EXTRACT — list the facts in the current passages that are directly relevant to the question. Use bullet points; quote spans verbatim where possible.

2. INTEGRATE — combine these facts with the facts you already extracted in previous iterations. State, in one sentence, what you now know toward answering the question.

3. ASSESS — can you give a confident, complete answer right now? Consider: do you have all the intermediate entities the question chains through?

4. OUTPUT — choose exactly one of the two formats below, on its own line(s):

   If you can answer:
   ANSWER: <your answer — verbatim span or yes/no>

   If you cannot answer yet:
   MISSING: <one specific entity or fact, e.g. "the nationality of the director of Parasite", NOT "more information about the film">
   NEXT_QUERY: <reformulated search query that will find that fact>

Important: the two output formats are equally acceptable. Do not bias toward MISSING when you have enough information, and do not bias toward ANSWER when you do not. Be honest about what the passages support.
"""


# P8 — few-shot variant of INTERMEDIATE_SYSTEM (StepChain parity composite).
# Adds 2 synthetic, non-MQ exemplars before the scaffold instructions. Anti-leak:
# exemplars are HAND-WRITTEN from public knowledge, NEVER from gold MQ/HP/2W; no
# StepChain text copied. Active only when `use_stepchain_parity_composite=True`.
INTERMEDIATE_SYSTEM_FEW_SHOT = """You are solving a multi-hop question iteratively. You have access to passages retrieved across one or more retrieval passes.

Work through this carefully and produce structured output.

1. EXTRACT — list the facts in the current passages that are directly relevant to the question. Use bullet points; quote spans verbatim where possible.

2. INTEGRATE — combine these facts with the facts you already extracted in previous iterations. State, in one sentence, what you now know toward answering the question.

3. ASSESS — can you give a confident, complete answer right now? Consider: do you have all the intermediate entities the question chains through?

4. OUTPUT — choose exactly one of the two formats below, on its own line(s):

   If you can answer:
   ANSWER: <your answer — verbatim span or yes/no>

   If you cannot answer yet:
   MISSING: <one specific entity or fact>
   NEXT_QUERY: <reformulated search query that will find that fact>
   NEXT_ENTITY: <a single named entity to search for, if known; one token or short proper noun>

Important: the two output formats are equally acceptable. Be honest about what the passages support.

---

EXAMPLE 1 (bridge entity, 2 hops):

Original question: What is the capital of the country where the inventor of the telephone was born?

Iteration: 1/5
Current retrieved passages:
[1] Alexander Graham Bell, born in Edinburgh in 1847, is credited with inventing the telephone.

EXTRACT:
- Bell was born in Edinburgh (passage 1).

INTEGRATE: Edinburgh is in Scotland, but I do not yet have the capital fact.

ASSESS: Not yet — I have the bridge (birthplace = Edinburgh → Scotland → UK) but no capital passage.

OUTPUT:
MISSING: the capital of the United Kingdom (or Scotland, if the question is about the country of birth)
NEXT_QUERY: capital of United Kingdom
NEXT_ENTITY: United Kingdom

---

EXAMPLE 2 (comparison, numeric attribute):

Original question: Which is older: the Eiffel Tower or the Brooklyn Bridge?

Iteration: 1/5
Current retrieved passages:
[1] The Brooklyn Bridge opened on May 24, 1883.
[2] The Eiffel Tower was completed on March 31, 1889.

EXTRACT:
- Brooklyn Bridge opened 1883 (passage 1).
- Eiffel Tower completed 1889 (passage 2).

INTEGRATE: 1883 < 1889 → Brooklyn Bridge is older.

ASSESS: Yes — both dates are present and comparable.

OUTPUT:
ANSWER: the Brooklyn Bridge

---

Now solve the actual question below using the same scaffold.
"""


# P4 / P24 unification — abstain-marker set used to filter
# toxic faith-loop reformulation. Re-exported from the canonical shared module
# `mothrag.core.abstain_markers` so eval-pipeline + pip-install api.py
# honour the same source of truth.
from mothrag.core.abstain_markers import (  # noqa: E402
    ABSTAIN_MARKERS,
    is_abstain_marker as _is_abstain_marker,
)


# P7 — entity-seeded re-retrieval query helper (StepChain parity).
# Replaces claim-shaped reformulation (`"Find passages that ground the claim: <claim>"`)
# with entity-shaped query (`"<question> <entity1> <entity2> ..."`). Matches the
# "stepped chain" paradigm where iter N+1 seeds on extracted bridge entity, not
# on the verbose claim sentence. Anti-leak: extraction via pipe._link_entities,
# never reads gold passages or answer.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")


def _extract_entities_from_text(text: str, max_entities: int = 3) -> list[str]:
    """Cheap proper-noun extractor — fallback when pipe._link_entities unavailable.

    Returns up to ``max_entities`` capitalized spans, dedup'd, in order of
    appearance. Sentence-initial wh-words filtered.
    """
    if not text:
        return []
    spans = _PROPER_NOUN_RE.findall(text)
    wh = {"Who", "What", "Where", "When", "Why", "How", "Which", "Find", "The", "A", "An"}
    out: list[str] = []
    seen: set[str] = set()
    for s in spans:
        if s in wh or s in seen:
            continue
        out.append(s)
        seen.add(s)
        if len(out) >= max_entities:
            break
    return out


def _entity_seeded_next_query(question: str, cue_text: str,
                              pipe_link_entities=None,
                              max_entities: int = 3) -> str:
    """Build entity-seeded next query: original question + extracted entities.

    ``pipe_link_entities`` is an optional callable (typically
    ``pipe._link_entities``) that returns a list of entity strings. When None
    or returns empty, falls back to the cheap proper-noun extractor.
    """
    entities: list[str] = []
    if pipe_link_entities is not None:
        try:
            entities = list(pipe_link_entities(cue_text) or [])[:max_entities]
        except Exception:  # noqa: BLE001
            entities = []
    if not entities:
        entities = _extract_entities_from_text(cue_text, max_entities=max_entities)
    if not entities:
        return question
    return f"{question} {' '.join(entities)}"


# Graph-aware-iter → reformulation entity propagation. The cross-iter
# accumulated entities already seed the next retrieval; this also weaves them
# into the γ-loop reformulation TEXT, so a
# refuse / partial re-query carries the broader entity context instead of one
# ungrounded claim cue at a time.
_L2_REFORM_ENTITY_CAP = 8  # entities prepended to a reformulation (prompt budget)


def _accumulated_entity_next_query(question: str, accumulated_entities: list,
                                   *, cue: str = "",
                                   cap: int = _L2_REFORM_ENTITY_CAP) -> str:
    """Weave accumulated graph-aware entities into a reformulation query.

    ``cue`` present (partial / invalid) → focus the broadened retrieval on the
    ungrounded step; absent (refuse — whole tree rejected, no cue) → pure
    broadening. Caller MUST invoke this ONLY when ``accumulated_entities`` is
    non-empty; empty ⇒ caller keeps its legacy string (byte-identical legacy
    behaviour, so graph-aware-OFF runs are unaffected).
    """
    ents = ", ".join(str(e) for e in accumulated_entities[:cap])
    if cue:
        return f"{question} (entities seen: {ents}, focus: {cue})"
    return (f"{question} (entities seen: {ents}; alternative phrasings, "
            f"different keywords)")


# Per-cohort γ-loop retrigger cap. CONSERVATIVE revision (v1 cut semantic_rich
# to 1 → MQ -3.6pp / 2W -3.1pp on the bulk cohort). v2: ONLY chain_deep gets the
# EXTRA retry (3); semantic_rich / bridge_entity / other all stay at the
# baseline 2 so the bulk cohort is never starved of retries.
_ADAPTIVE_GAMMA_RETRIGGER_CAP = {"chain_deep": 3, "bridge_entity": 2,
                                 "semantic_rich": 2}


def _effective_gamma_retrigger_cap(cfg, qtype, composite: bool) -> int:
    """Resolve the γ-loop retrigger cap for this run.

    ``use_adaptive_gamma_retrigger`` ON ⇒ per-cohort cap. v2-CONSERVATIVE: only
    chain_deep gets the EXTRA retry (3, so deep chains are not abandoned at the
    fixed 2); semantic_rich=2 / bridge_entity=2 / other=2 stay at the baseline
    (v1's semantic_rich=1 starved the bulk cohort — MQ/2W regression). OFF ⇒ the
    legacy composite-aware fixed cap (byte-identical)."""
    if getattr(cfg, "use_adaptive_gamma_retrigger", False):
        return _ADAPTIVE_GAMMA_RETRIGGER_CAP.get(qtype, 2)
    return (cfg.composite_gamma_max_retrigger if composite
            else cfg.gamma_max_retrigger)


def _faithfulness_gamma_coord_skip(cfg, gamma_final_status, *,
                                   gamma_refuse_loop_exhausted: bool = False,
                                   qtype=None, iter_count: int = 0):
    """Decide whether to skip the faithfulness LLM re-check, returning WHICH
    branch fired:

      * ``"clean_valid"`` — the current proof tree is γ=``"valid"``: already
        γ-verified, faithfulness is redundant.
      * ``"exhausted_safe"`` — the γ-refuse-loop is exhausted AND the cohort is
        NOT ``chain_deep`` AND we are past the first iteration (``iter_count >= 2``):
        a non-chain_deep answer that survived ≥2 iterations to cap is safe to
        accept without the extra check. chain_deep is PRESERVED here because its
        cap-hit answers are exactly the recovery scenarios that still need the
        faithfulness gate (v1 skipped 60.7%% on MQ — over-aggressive on
        the recovery cohort).
      * ``None`` — run the real faithfulness check (legacy path).

    Default OFF (flag) ⇒ always ``None`` ⇒ byte-identical legacy."""
    if not getattr(cfg, "use_faithfulness_gamma_coord", False):
        return None
    if gamma_final_status == "valid":
        return "clean_valid"
    if (gamma_refuse_loop_exhausted and qtype != "chain_deep"
            and iter_count >= 2):
        return "exhausted_safe"
    return None


# P9 — relevance-aware accumulation: rerank accumulated passages by
# cosine similarity to original question (not FIFO insertion order) on cap-hit.
# Falls back to no-op if embedder unavailable.
#
# Passage embedding cache. Without cache, n iter × m accum_texts → n·m embed
# calls on identical texts.
# Cache keyed by SHA256(text); only NEW passages embedded each iter.
# LRU-bounded to MAX entries to prevent memory blow-up across queries.
# Anti-leak: pure memoization (text → vec); no gold / DS / answer info touched.
from collections import OrderedDict as _OrderedDict
_PASSAGE_EMB_CACHE: "_OrderedDict[str, object]" = _OrderedDict()
_PASSAGE_EMB_CACHE_MAX = 20000  # ~240MB at 3072-d float32


def _passage_cache_key(text: str, model: str = "") -> str:
    """Model-namespaced cache key.

    Mirrors the gemini.py pattern: ``SHA256(model::text)``. Empty
    ``model`` keeps backward compat with callers that didn't namespace
    (legacy behavior — single-embedder runs unaffected). Pass
    the embedder's model id when calling from multi-embedder contexts
    to avoid cross-model cache collisions.
    """
    import hashlib
    key_str = f"{model}::{text}" if model else text
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


def _passage_cache_get(key: str):
    v = _PASSAGE_EMB_CACHE.get(key)
    if v is not None:
        _PASSAGE_EMB_CACHE.move_to_end(key)  # LRU touch
    return v


def _passage_cache_put(key: str, vec) -> None:
    _PASSAGE_EMB_CACHE[key] = vec
    _PASSAGE_EMB_CACHE.move_to_end(key)
    while len(_PASSAGE_EMB_CACHE) > _PASSAGE_EMB_CACHE_MAX:
        _PASSAGE_EMB_CACHE.popitem(last=False)


def _rerank_accum_passages(
    accum_ids: list[str], accum_texts: list[str],
    question: str, *, embed_fn=None, top_k: int = 20,
    embed_model: str = "",
) -> tuple[list[str], list[str]]:
    """Reorder (accum_ids, accum_texts) by cosine sim to question; truncate to top_k.

    ``embed_fn`` signature: ``texts: list[str] -> np.ndarray (N, D)``.
    If None or accumulator empty, returns inputs truncated to top_k (legacy
    FIFO behavior).

    Passage embeddings cached via SHA256(``embed_model``::text) — only NEW
    texts embedded each iter (5-10× speedup). Query always fresh.
    ``embed_model`` namespaces the cache to avoid collisions if
    callers ever swap embedder mid-run; defaults to empty for backward
    compat (single-embedder runs see no change).
    """
    n = len(accum_ids)
    if n == 0 or embed_fn is None:
        return accum_ids[:top_k], accum_texts[:top_k]
    try:
        import numpy as np
        # Partition accum_texts: cached vs new
        keys = [_passage_cache_key(t, embed_model) for t in accum_texts]
        new_idx = [i for i, k in enumerate(keys) if _passage_cache_get(k) is None]
        # Embed query + only NEW passages in a single batch (one API roundtrip)
        new_texts = [accum_texts[i] for i in new_idx]
        batch = [question] + new_texts
        emb = embed_fn(batch)
        q_vec = emb[0]
        # Persist new passage vectors
        for j, i in enumerate(new_idx):
            _passage_cache_put(keys[i], emb[1 + j])
        # Assemble full passage matrix from cache
        p_vecs = np.stack([_passage_cache_get(k) for k in keys], axis=0)
        # Normalize for cosine
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        p_norms = p_vecs / (np.linalg.norm(p_vecs, axis=1, keepdims=True) + 1e-9)
        scores = p_norms @ q_norm
        order = np.argsort(-scores)[:top_k]
        ids = [accum_ids[i] for i in order]
        texts = [accum_texts[i] for i in order]
        return ids, texts
    except Exception:  # noqa: BLE001 — fall through to FIFO truncation
        return accum_ids[:top_k], accum_texts[:top_k]


def _passage_text(p) -> str:
    """Text body of a passage that may be a plain string or an Aurora-shape
    dict (proof-tree path). Used by path-serialize ordering."""
    if isinstance(p, dict):
        for k in ("text", "passage", "content", "chunk"):
            v = p.get(k)
            if v:
                return str(v)
        return str(p)
    return str(p)


def _path_serialize_order(passages: list, spine_entities: list) -> list:
    """Reorder passages along a multi-hop spine.

    ``spine_entities`` is the cross-iteration accumulated-entity list (discovery
    order ~ hop order: hop-1 / bridge entity first). Each passage is ranked by
    the EARLIEST spine entity it mentions, so hop-1 evidence precedes hop-2;
    ties keep retrieval order (stable sort on original index). Passages
    mentioning no spine entity sink to the end in their original order. Pure
    (no LLM); graceful no-op when there is no spine or <2 passages. Returns the
    SAME passage objects in a new order (never drops/duplicates)."""
    if not spine_entities or len(passages) < 2:
        return list(passages)
    spine = [str(e).lower() for e in spine_entities if e]
    if not spine:
        return list(passages)

    def _key(ip):
        i, p = ip
        tl = _passage_text(p).lower()
        for r, e in enumerate(spine):
            if e in tl:
                return (r, i)
        return (len(spine), i)

    return [p for _, p in sorted(enumerate(passages), key=_key)]


def _intermediate_user_msg(question: str, passages: list[str], iteration: int,
                           max_iterations: int, accumulated_facts: list[str],
                           *, path_serialize: bool = False,
                           spine_entities: list | None = None) -> str:
    # Present evidence as a hop-ordered path (spine = accumulated_entities)
    # instead of raw retrieval order. Default OFF → byte-identical legacy
    # bag-of-passages.
    path_on = bool(path_serialize and spine_entities)
    ordered = _path_serialize_order(passages, spine_entities) if path_on else passages
    facts_block = ("\n".join(f"- {f}" for f in accumulated_facts)
                   if accumulated_facts else "(none yet)")
    ctx = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(ordered))
    ctx_label = ("Retrieved passages, ordered along the reasoning path "
                 "(hop-1 / bridge evidence first):"
                 if path_on else "Current retrieved passages:")
    return (
        f"Original question: {question}\n"
        f"Iteration: {iteration}/{max_iterations}\n"
        f"Facts extracted in previous iterations:\n{facts_block}\n\n"
        f"{ctx_label}\n{ctx}\n\n"
        f"Now perform EXTRACT, INTEGRATE, ASSESS, OUTPUT."
    )


# ---- Output parsing ----

_ANSWER_RE = re.compile(r"^\s*ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_MISSING_RE = re.compile(r"^\s*MISSING:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NEXT_QUERY_RE = re.compile(r"^\s*NEXT_QUERY:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_EXTRACT_BULLET_RE = re.compile(r"^\s*[-*]\s+(.+?)\s*$", re.MULTILINE)
_EXTRACT_HEADER_RE = re.compile(
    r"(?:^|\n)\s*1[\.\)]\s*EXTRACT[^\n]*\n(.+?)(?=\n\s*2[\.\)]\s*INTEGRATE|\Z)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class IntermediateOutput:
    raw: str
    answer: Optional[str] = None
    missing: Optional[str] = None
    next_query: Optional[str] = None
    extracted_facts: list[str] = field(default_factory=list)
    is_terminal: bool = False  # True iff ANSWER was produced

    @property
    def has_answer(self) -> bool:
        return self.answer is not None and self.answer.strip() != ""


def parse_intermediate(raw: str) -> IntermediateOutput:
    """Parse the structured EXTRACT/INTEGRATE/ASSESS/OUTPUT response.

    Robust to ordering variations and minor formatting drift. If both ANSWER
    and MISSING are present, ANSWER wins (the model self-resolved late).
    """
    out = IntermediateOutput(raw=raw)
    m_ans = _ANSWER_RE.search(raw)
    if m_ans:
        out.answer = m_ans.group(1).strip().rstrip(".").strip()
        out.is_terminal = True
    m_miss = _MISSING_RE.search(raw)
    if m_miss:
        out.missing = m_miss.group(1).strip()
    m_nq = _NEXT_QUERY_RE.search(raw)
    if m_nq:
        out.next_query = m_nq.group(1).strip()

    # Fact extraction: prefer the EXTRACT block, fall back to all bullet lines.
    m_block = _EXTRACT_HEADER_RE.search(raw)
    block = m_block.group(1) if m_block else raw
    bullets = [b.strip() for b in _EXTRACT_BULLET_RE.findall(block) if b.strip()]
    out.extracted_facts = bullets[:6]  # cap to keep accumulator small
    return out


# ---- Iterative pipeline ----

@dataclass
class IterativeConfig:
    max_iterations: int = 4
    top_k_total: int = 15
    stop_early: bool = True
    intermediate_max_tokens: int = 600
    intermediate_temperature: float = 0.0
    # Tier-conditional reader for final synthesis only
    final_reader_client: Optional[object] = None  # OpenAI client override
    final_reader_model: Optional[str] = None  # model name override
    final_reader_max_tokens: int = 384  # standard final answer budget
    # Graph-aware iter retrieval: propagate accumulated entities
    use_graph_aware_iter: bool = False  # default off for ablation parity
    # Order reader evidence along the multi-hop spine
    # (accumulated_entities) instead of raw retrieval order, at all reader
    # sites (intermediate / proof-tree / final). Default OFF = byte-identical
    # bag-of-passages, no new LLM calls.
    path_serialize: bool = False
    max_accumulated_entities: int = 32  # cap to keep retrieval focused
    # Faithfulness loop: gate ANSWER acceptance on grounded check
    use_faithfulness_loop: bool = False  # default off for ablation parity
    judge_client: Optional[object] = None  # OpenAI/Gemini client for judge
    judge_model: Optional[str] = None  # judge model name (e.g. llama-3.3-70b)
    judge_provider: str = "openai"  # "openai" or "gemini"
    judge_min_score: float = 1.0  # require YES (1.0) to accept; 0.5 to accept PARTIAL
    max_judge_iterations: Optional[int] = None  # cap loop extension; None = max_iterations+1
    # Aurora γ verifier mode (alternative to single-judge gate above)
    use_gamma_verifier: bool = False  # ProofTree reader + deterministic γ verifier
    # When ON, the final-iteration verified proof tree
    # (per-step rule / verifier_status / verifier_reason + is_complete + raw) is
    # captured into the per-q output, closing the telemetry gap (per_question
    # otherwise saves only gamma_final_status). Diagnostic only; default OFF so
    # production JSONs are not bloated.
    gamma_diagnostic_dump: bool = False
    # Relax object grounding from "token in cited span" to
    # "tokens covered by the passage" (fixes the dominant secondary γ-invalid
    # mode: descriptive/boolean objects). Default OFF = byte-identical.
    use_relaxed_object_span: bool = False
    use_gamma_liberal: bool = False  # liberal variant when γ-strict invalid
    gamma_max_retrigger: int = 2  # cap for L4 loop retrigger on partial/invalid
    # Adaptive per-cohort retrigger cap. When ON, the γ-loop retrigger cap is
    # set by the input-feature cohort: chain_deep=3 (deep chains abandoned too
    # early at the fixed 2), and bridge_entity=2 / semantic_rich=2 / other=2 all
    # at baseline. CONSERVATIVE: ONLY chain_deep gets the extra retry; v1's
    # semantic_rich=1 starved the bulk cohort (MQ -3.6pp / 2W -3.1pp).
    # Default OFF ⇒ legacy fixed cap, byte-identical.
    use_adaptive_gamma_retrigger: bool = False
    # Faithfulness ↔ γ-refuse-loop coordination.
    # When ON, skip the faithfulness check once the γ-refuse-loop is exhausted
    # (cap hit with an answer accepted via the P11 free-text fallback): the chains
    # were already γ-verified, so faithfulness is redundant. Default OFF.
    use_faithfulness_gamma_coord: bool = False
    gamma_prompt_variant: str = "full"  # "full" (5-rule, Anthropic-tuned) | "llama" (2-rule + example)
    # γ-loop-trigger architecture fix.
    # When True, γ="refuse" triggers an additional loop iteration (reformulate
    # with broader phrasing) instead of immediately emitting "Not in passages".
    # Default flipped True after validation showed +6.54pp F1 paired MQ vs
    # baseline 0.4254. New canonical MQ iter Llama Bmode ≈ 0.4884 (baseline +
    # paired delta).
    use_gamma_refuse_loop: bool = True
    # P5 — cap-branch SoftFallback. Targets the 14 stuck-refuse
    # cohort identified in the audit. When True AND cap fires AND
    # naturalized_answer is empty (γ refuse with no tree answer), fall through
    # to free-text reader on accumulated passages instead of emitting
    # "Not in passages" (which scores F1≈0). Strictly more conservative than
    # use_gamma_router=True (which fires free-text reader BEFORE checking
    # naturalized_answer — empirically harmful, +0.93pp only).
    # Evaluation verdict: ΔF1 -2.20pp paired → flag stays default False.
    cap_branch_softfallback: bool = False
    # Bug-pattern hunt composite (P11+P14+P15 here +
    # P12+P13 in route_prospective.py + P24 in selective_ensemble.py).
    # 6 patches gated under a single composite flag for
    # composite-then-bisect evaluation. Default False until verdict.
    use_bug_pattern_wave_a: bool = False
    # P11 individual toggle. The composite gated 6 patches
    # behind a single flag, blocking single-patch ablation. This flag activates
    # P11 (γ-cap free-text-reader fallback) STANDALONE without P12/P13/P14/
    # P15/P24 — enables tests to isolate the P11 contribution. When
    # use_bug_pattern_wave_a=True, P11 fires regardless of this flag
    # (composite supersedes). Default False.
    use_p11_gamma_cap_fallback: bool = False
    # Disable-P11 ablation override. Forces P11 OFF even when
    # use_bug_pattern_wave_a=True (composite-supersedes-individual-toggle
    # was a one-way gate; this re-introduces the OFF direction for the
    # "FULL extras MINUS P11" ablation). Default False (no override).
    disable_p11_gamma_cap_fallback: bool = False
    # Individual toggles for P12/P13/P15/P24.
    # Same composite-OR-individual pattern as use_p11_gamma_cap_fallback,
    # enabling per-patch ablation cells (p12_only / p13_only /
    # p15_only / p24_only) without requiring use_bug_pattern_wave_a. When
    # use_bug_pattern_wave_a=True the patches fire regardless of these
    # flags (composite supersedes). Default False.
    use_p12_decompose_collapse_cap: bool = False
    use_p13_sub_q_abstain_filter:   bool = False
    use_p15_gamma_gated_naturalize: bool = False
    use_p24_unified_abstain_markers: bool = False
    # StepChain parity composite (P6 + P7 + P8 + P9 under one flag).
    # Composite-then-bisect strategy: paired MQ evaluation to validate
    # aggregate lift before single-patch ship.
    # When True: bumps iter cap (P6), entity-seeds re-retrieval (P7), uses
    # few-shot INTERMEDIATE_SYSTEM (P8), raises top_k_total + reranks on cap
    # hit (P9). Default False until verdict.
    use_stepchain_parity_composite: bool = False
    # P6 — iter cap raise (only used when composite flag True)
    composite_max_iterations: int = 5
    composite_gamma_max_retrigger: int = 3
    # P9 — context accumulation raise + relevance rerank (composite-gated)
    composite_top_k_total: int = 20
    # P9 — optional embedder for relevance-rerank on cap-hit. If None, falls
    # back to FIFO truncation (no rerank). Injected at construction time;
    # signature: list[str] -> np.ndarray (N, D).
    composite_rerank_embed_fn: Optional[Callable[[list[str]], object]] = None
    # γ-as-router: when γ verifier exhausts retries and stays invalid,
    # instead of emitting "Not in passages" or partial naturalized_answer, fall
    # back to free-text V3+bu reader over accumulated passages. Recovers cases
    # where γ-strict rejects but free-text reader has the correct answer.
    use_gamma_router: bool = False
    # C7 phase-cancellation is NOT wired here — it lives at the ensemble
    # arbitrate layer (mothrag.core.selective_ensemble.arbitrate_with_c7) where
    # multiple chain candidates exist (V3+bu / decompose / iterative). The
    # iterative loop produces only one chain and has no rejected_chains to feed.
    # Within-iterative-loop C7 (temporal stability axis, complementary
    # to the ensemble axis). Chosen = final iter answer; rejected = prior iter
    # answers (deduplicated). Centroid bisection (query_embed=None) since this
    # is a temporal-stability signal, not query-relevance.
    use_c7_iter: bool = False
    c7_iter_trigger: str = "gated"  # "gated" (γ partial/invalid only) | "blanket"
    c7_iter_embedder: Optional[Callable[[list[str]], object]] = None  # (list[str]) -> ndarray (K, D)


@dataclass
class IterativeAnswerInfo(AnswerInfo):
    iterations_used: int = 0
    per_iteration_route: list[str] = field(default_factory=list)
    per_iteration_query: list[str] = field(default_factory=list)
    per_iteration_top1_conf: list[float] = field(default_factory=list)
    per_iteration_terminal: list[bool] = field(default_factory=list)
    per_iteration_n_new_chunks: list[int] = field(default_factory=list)
    accumulated_facts: list[str] = field(default_factory=list)
    accumulated_entities: list[str] = field(default_factory=list)
    per_iteration_n_seed_entities: list[int] = field(default_factory=list)
    # Total accumulated entities woven into the
    # γ-loop reformulations across the run (0 when graph-aware-iter is off / no
    # entities yet → legacy reformulation).
    l2_reformulation_entities_total: int = 0
    # The retrigger cap actually used this run
    # (adaptive per-cohort value when --use-adaptive-gamma-retrigger, else the
    # legacy fixed/composite cap).
    gamma_retrigger_cap_used: int = 0
    # Faithfulness ↔ γ-refuse coordination.
    faithfulness_active: bool = False
    faithfulness_skipped_clean_valid: bool = False
    faithfulness_skipped_exhausted_safe: bool = False
    # Final-iter verified proof tree (per-step reason +
    # is_complete + raw). None unless --gamma-diagnostic-dump.
    gamma_diagnostic: Optional[dict] = None
    # Object grounding match-mode counts on the final tree.
    object_relaxed_match_count: int = 0
    object_exact_match_count: int = 0
    # Entries may be None when the faithfulness gate is skipped
    # (no answer this iter, or use_faithfulness_loop=False). Padded so
    # len(faith_*) == len(per_iteration_terminal) per iter for downstream
    # index-based zip / telemetry consumers (per-patch attribution).
    per_iteration_faithfulness_score: list[Optional[float]] = field(default_factory=list)
    per_iteration_faithfulness_label: list[Optional[str]] = field(default_factory=list)
    # Aurora γ verifier audit (γ mode)
    per_iteration_gamma_status: list[str] = field(default_factory=list)
    gamma_final_status: Optional[str] = None
    # Aurora within-iterative-loop C7 audit (temporal stability axis)
    per_iteration_answer: list[Optional[str]] = field(default_factory=list)
    c7_iter_kept: Optional[bool] = None
    c7_iter_info: Optional[dict] = None
    # Per-patch activation telemetry + iter trace.
    # patch_activations: dict[str, bool] — did each patch fire at any iter?
    # iter_trace: list[dict] per iter — {iter, gamma_status, query, passages_n,
    #                                    patches_active_this_iter: list[str]}.
    # Anti-leak: only internal-state fields (no gold/F1/dataset).
    patch_activations: dict = field(default_factory=dict)
    iter_trace: list = field(default_factory=list)


class IterativeMothRAG:
    """State-aware iterative wrapper around :class:`MothRAGPipeline`.

    Reuses the underlying pipeline's retriever, reader client, reader model
    and final-answer prompt. Only the per-pass intermediate prompt is new.
    """

    def __init__(self, pipeline: MothRAGPipeline,
                 config: Optional[IterativeConfig] = None):
        self.pipeline = pipeline
        self.config = config or IterativeConfig()

    # ---- Internal: call the intermediate reader ----

    def _call_intermediate(self, question: str, passages: list[str],
                           iteration: int, accumulated_facts: list[str],
                           *, spine_entities: list | None = None
                           ) -> tuple[IntermediateOutput, dict]:
        cfg = self.config
        # P8 — composite flag toggles few-shot system prompt.
        sys_prompt = (INTERMEDIATE_SYSTEM_FEW_SHOT
                      if cfg.use_stepchain_parity_composite
                      else INTERMEDIATE_SYSTEM)
        # P6 — display effective max in user message so model sees composite cap
        effective_max = (cfg.composite_max_iterations
                         if cfg.use_stepchain_parity_composite
                         else cfg.max_iterations)
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": _intermediate_user_msg(
                question, passages, iteration, effective_max, accumulated_facts,
                path_serialize=cfg.path_serialize, spine_entities=spine_entities,
            )},
        ]
        kwargs = _reader_kwargs(self.pipeline.reader_model, msgs,
                                cfg.intermediate_max_tokens,
                                cfg.intermediate_temperature)
        t0 = time.time()
        resp = self.pipeline.reader_client.chat.completions.create(**kwargs)
        latency = time.time() - t0
        text = resp.choices[0].message.content.strip()
        u = getattr(resp, "usage", None)
        usage = {
            "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0) if u else 0,
            "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0) if u else 0,
            "latency_s": float(latency),
        }
        return parse_intermediate(text), usage

    # ---- Internal: union retrieval state with cap ----

    @staticmethod
    def _merge_chunks(accum_ids: list[str], accum_texts: list[str],
                      seen: set[str], new_ids: list[str], new_texts: list[str],
                      cap: int) -> tuple[list[str], list[str]]:
        for cid, txt in zip(new_ids, new_texts):
            if len(accum_ids) >= cap:
                break
            if cid in seen:
                continue
            seen.add(cid)
            accum_ids.append(cid)
            accum_texts.append(txt)
        return accum_ids, accum_texts

    # ---- Internal: Aurora γ verifier mode (faithfulness-gate alternative) ----

    def _call_gamma_intermediate(self, question: str, passages: list[dict],
                                  iteration: int, *,
                                  spine_entities: list | None = None
                                  ) -> tuple["object", dict]:
        """Run reader with PROOF_TREE prompt; parse JSON; verify deterministically.

        Returns ``(verified_tree_or_None, usage_dict)``. Caller inspects
        ``verified_tree.overall_status`` for routing (valid/partial/invalid/refuse).
        """
        from mothrag.aurora import (PROOF_TREE_SYSTEM_PROMPT,
                                     PROOF_TREE_SYSTEM_PROMPT_LLAMA,
                                     proof_tree_user_prompt, verify_proof_tree)
        from mothrag.aurora.adapter import parse_reader_prooftree_json
        from mothrag.aurora.verifier_liberal import liberal_overall_status
        cfg = self.config
        sys_prompt = (PROOF_TREE_SYSTEM_PROMPT_LLAMA
                      if cfg.gamma_prompt_variant == "llama"
                      else PROOF_TREE_SYSTEM_PROMPT)
        # Order proof-tree evidence along the spine too
        # (default OFF → passages unchanged). Verification below stays on the
        # original `passages` (span lookup is order-independent).
        pt_passages = (_path_serialize_order(passages, spine_entities)
                       if cfg.path_serialize and spine_entities else passages)
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": proof_tree_user_prompt(question, pt_passages)},
        ]
        kwargs = _reader_kwargs(self.pipeline.reader_model, msgs,
                                cfg.intermediate_max_tokens,
                                cfg.intermediate_temperature)
        t0 = time.time()
        try:
            resp = self.pipeline.reader_client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            return None, {"prompt_tokens": 0, "completion_tokens": 0,
                          "latency_s": time.time() - t0, "error": str(exc)[:120]}
        latency = time.time() - t0
        text = resp.choices[0].message.content.strip()
        u = getattr(resp, "usage", None)
        usage = {
            "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0) if u else 0,
            "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0) if u else 0,
            "latency_s": float(latency),
            "raw": text,
        }

        tree = parse_reader_prooftree_json(text, qid=str(iteration))
        if tree is None:
            return None, usage
        verified = verify_proof_tree(tree, passages,
                                     use_relaxed_object_span=cfg.use_relaxed_object_span)
        # Optional liberal pass when strict is invalid + naturalized_answer present
        if (cfg.use_gamma_liberal and verified.overall_status == "invalid"
                and verified.naturalized_answer):
            from dataclasses import asdict
            try:
                lib_status = liberal_overall_status(asdict(verified))
                if lib_status in ("valid", "partial"):
                    verified.overall_status = lib_status
                    usage["liberal_promoted_from"] = "invalid"
            except Exception:  # noqa: BLE001
                pass
        return verified, usage

    # ---- Internal: Aurora L4b within-iterative-loop C7 (temporal stability) ----

    def _apply_c7_iter(self, iteration_candidates: list[Optional[str]],
                       final_answer: str,
                       gamma_status: Optional[str] = None
                       ) -> tuple[Optional[bool], Optional[dict]]:
        """Apply C7 phase-cancellation to candidate answers across L1 iterations.

        L4b applies C7 phase-cancellation to candidate answers from successive
        iterations of the L1 iterative chain extension. ``chosen`` = final
        iteration answer; ``rejected`` = answers from prior iterations
        (deduplicated by ``normalize_answer``). C7 measures temporal stability:
        when chosen is kept, the answer has stabilized across iterations
        (high confidence); when cancelled, the answer is still oscillating
        (low confidence). This is structurally orthogonal to L6 cross-pipeline
        ensemble C7 — same primitive, different axis (temporal vs ensemble).

        Centroid bisection (``query_embed=None``) is used here intentionally:
        temporal stability is a property of the answer trajectory, not of
        query-relevance projection.

        Returns ``(chosen_kept, c7_info)`` or ``(None, None)`` when skipped
        (master switch off / no embedder / gated+γ-valid / max_iter<2 /
        no distinct rejected candidates).
        """
        from mothrag.core.selective_ensemble import normalize_answer
        cfg = self.config
        if not cfg.use_c7_iter or cfg.c7_iter_embedder is None:
            return None, None
        if cfg.max_iterations < 2:
            return None, None  # degenerate — no temporal axis
        if cfg.c7_iter_trigger == "gated" and gamma_status not in ("partial", "invalid"):
            return None, None
        if not final_answer:
            return None, None

        final_norm = normalize_answer(final_answer)
        seen: set[str] = set()
        rejected: list[str] = []
        for cand in iteration_candidates:
            if not cand:
                continue
            cand_norm = normalize_answer(cand)
            if not cand_norm or cand_norm == final_norm or cand_norm in seen:
                continue
            seen.add(cand_norm)
            rejected.append(cand)

        if not rejected:
            return None, None  # only 1 unique answer across iterations

        from mothrag.aurora import c7_aurora_rejected_chains
        try:
            c7_info = c7_aurora_rejected_chains(final_answer, rejected,
                                                  cfg.c7_iter_embedder,
                                                  query_embed=None)
        except Exception as exc:  # noqa: BLE001
            c7_info = {"error": str(exc)[:160], "chosen_kept": True}
        return c7_info.get("chosen_kept"), c7_info

    # ---- Internal: faithfulness gate ----

    def _check_faithfulness(self, question: str, passages: list[str],
                            pred: str) -> tuple[float, str]:
        """Returns ``(score, label)`` from a single judge call. Score in {0, 0.5, 1}."""
        from mothrag.eval.faithfulness import faithfulness_score
        cfg = self.config
        try:
            return faithfulness_score(
                cfg.judge_client, cfg.judge_model,
                question, passages, pred,
                cache=None, provider=cfg.judge_provider,
            )
        except Exception:  # noqa: BLE001
            return 0.0, "no"

    # ---- Internal: final synthesis via tier-conditional override ----

    def _call_final_with_override(self, question: str, passages: list[str],
                                   client, model: str, max_tokens: int
                                   ) -> tuple[str, str, dict]:
        """Replicate :meth:`MothRAGPipeline.read` using a stronger override reader.

        Used for final synthesis only (cheap reader handles intermediate hops).
        Single-shot (no n-sample majority) — strong reader doesn't need it.
        """
        prompt_version = self.pipeline.config.reader_prompt
        msgs = make_reader_messages(question, passages, prompt_version)
        raw, usage = call_reader_with_usage(client, model, msgs,
                                             max_tokens=max_tokens,
                                             temperature=0.0)
        return parse_reader_output(raw, prompt_version), raw, usage

    # ---- Public: answer one question with iterative retrieval ----

    def answer(self, question: str) -> IterativeAnswerInfo:
        cfg = self.config
        pipe = self.pipeline

        # StepChain parity composite flag. Computes effective
        # caps when active; otherwise uses legacy IterativeConfig fields.
        composite = cfg.use_stepchain_parity_composite
        effective_max_iter = (cfg.composite_max_iterations if composite
                              else cfg.max_iterations)
        # Shared cohort label (classify_query_v2, input-feature only, anti-leak).
        # Used by the per-cohort retrigger cap AND the chain_deep
        # faithfulness-preservation guard. Computed ONCE when EITHER flag is on;
        # OFF ⇒ None ⇒ no classification (byte-id legacy — the cap helper
        # ignores qtype when adaptive is off).
        _cohort_qtype = None
        if cfg.use_adaptive_gamma_retrigger or cfg.use_faithfulness_gamma_coord:
            try:
                from mothrag.core.query_type_classifier import classify_query_v2
                _cohort_qtype = classify_query_v2(question)
            except Exception:  # noqa: BLE001 — never let classification break the loop
                _cohort_qtype = None
        effective_gamma_cap = _effective_gamma_retrigger_cap(
            cfg, _cohort_qtype, composite)
        effective_top_k = (cfg.composite_top_k_total if composite
                           else cfg.top_k_total)

        accum_ids: list[str] = []
        accum_texts: list[str] = []
        seen_ids: set[str] = set()
        accumulated_facts: list[str] = []
        accumulated_entities: list[str] = []  # cross-iter entity tracking
        l2_reform_ent_total = 0  # telemetry: entities woven into reformulations
        # γ-refuse-loop exhaustion tracking + faithfulness-coordination telemetry.
        gamma_refuse_loop_exhausted = False
        _gamma_diag = None  # final-iter tree capture
        _obj_relaxed_n = 0   # object grounding match-mode counts
        _obj_exact_n = 0
        faithfulness_active = False
        faithfulness_skipped_clean_valid = False
        faithfulness_skipped_exhausted_safe = False
        per_iter_route: list[str] = []
        per_iter_query: list[str] = []
        per_iter_conf: list[float] = []
        per_iter_terminal: list[bool] = []
        per_iter_new: list[int] = []
        per_iter_n_seed: list[int] = []
        per_iter_faith_score: list[float] = []
        per_iter_faith_label: list[str] = []
        per_iter_gamma_status: list[str] = []
        gamma_final_status: Optional[str] = None
        per_iter_answer: list[Optional[str]] = []  # answer candidates

        total_pt = total_ct = 0
        total_lat = 0.0
        current_q = question

        early_answer: Optional[str] = None
        early_raw: str = ""

        # Pad faith lists so they're always == len(terminal).
        # Called at every iter_trace append site (end-of-iter + 3 break sites).
        def _pad_faith_lists():
            while len(per_iter_faith_score) < len(per_iter_terminal):
                per_iter_faith_score.append(None)  # type: ignore[arg-type]
                per_iter_faith_label.append(None)  # type: ignore[arg-type]

        # Per-patch telemetry. Track which patches fired this query.
        patch_fired: dict[str, bool] = {
            "P6_cap_raise_triggered": False,
            "P7_entity_seeded_used": False,
            "P8_few_shot_applied": False,
            "P9_rerank_triggered": False,
            "P11_gamma_cap_fallback_fired": False,
            "P14_faith_loop_second_chance": False,
            "wave_a_active": bool(cfg.use_bug_pattern_wave_a),
            "stepchain_composite_active": bool(composite),
        }
        iter_trace: list[dict] = []
        # P8 (few-shot) is a config-time activation when composite enabled +
        # the non-γ intermediate path is taken. We mark it eagerly here since
        # the sys_prompt swap happens unconditionally inside _call_intermediate.
        if composite:
            patch_fired["P8_few_shot_applied"] = True

        for it in range(1, effective_max_iter + 1):
            patches_this_iter: list[str] = []
            # P6 — cap raise: any iter beyond legacy max_iterations counts as raised.
            if composite and it > cfg.max_iterations:
                patch_fired["P6_cap_raise_triggered"] = True
                patches_this_iter.append("P6")
            # 1. Retrieve with current_q. Graph-aware iter: pass accumulated entity seeds.
            seeds = (accumulated_entities[: cfg.max_accumulated_entities]
                     if cfg.use_graph_aware_iter and accumulated_entities else None)
            per_iter_n_seed.append(len(seeds) if seeds else 0)
            if seeds:
                top_idx, route, conf = pipe.retrieve(current_q, entity_seeds=seeds)
            else:
                top_idx, route, conf = pipe.retrieve(current_q)
            new_ids = [pipe.chunk_ids[ci] for ci in top_idx]
            new_texts = [pipe.chunks_by_id[cid]["text"] for cid in new_ids]
            before = len(accum_ids)
            self._merge_chunks(accum_ids, accum_texts, seen_ids, new_ids, new_texts,
                               effective_top_k)
            # P9 — when composite + cap hit, rerank accumulator by cosine to question
            if composite and len(accum_ids) >= effective_top_k and cfg.composite_rerank_embed_fn:
                accum_ids, accum_texts = _rerank_accum_passages(
                    accum_ids, accum_texts, question,
                    embed_fn=cfg.composite_rerank_embed_fn,
                    top_k=effective_top_k,
                )
                # Refresh seen set after rerank truncation
                seen_ids = set(accum_ids)
                patch_fired["P9_rerank_triggered"] = True
                patches_this_iter.append("P9")
            n_new = len(accum_ids) - before
            per_iter_route.append(route)
            per_iter_query.append(current_q)
            per_iter_conf.append(conf)
            per_iter_new.append(n_new)

            # 2. Intermediate reader. The γ-mode replaces free-text intermediate
            # with PROOF_TREE-structured reader + deterministic verifier.
            verified_tree = None  # L4b: capture γ tree candidate across branches
            if cfg.use_gamma_verifier:
                # γ-mode requires Aurora-shape passages (list[dict])
                from mothrag.aurora.adapter import mothrag_passages_to_aurora
                aurora_passages = mothrag_passages_to_aurora(
                    [pipe.chunk_ids.index(cid) for cid in accum_ids], pipe)
                verified_tree, usage = self._call_gamma_intermediate(
                    question, aurora_passages, it,
                    spine_entities=accumulated_entities,
                )
                total_pt += usage.get("prompt_tokens", 0)
                total_ct += usage.get("completion_tokens", 0)
                total_lat += usage.get("latency_s", 0.0)
                # Capture the (final-iteration) tree so
                # the per-q output carries per-step verifier_reason + is_complete.
                # Overwritten each iter → the last iteration's tree persists.
                if cfg.gamma_diagnostic_dump:
                    if verified_tree is None:
                        _gamma_diag = {"overall_status": "parse_fail",
                                       "is_complete": None, "steps": [],
                                       "raw": (usage.get("raw", "") or "")[:1500]}
                    else:
                        _gamma_diag = {
                            "overall_status": verified_tree.overall_status,
                            "is_complete": bool(verified_tree.is_complete),
                            "naturalized_answer": verified_tree.naturalized_answer,
                            "steps": [{"rule": s.rule,
                                       "verifier_status": s.verifier_status,
                                       "verifier_reason": s.verifier_reason,
                                       "object_match_mode": s.object_match_mode}
                                      for s in verified_tree.steps],
                            "raw": (usage.get("raw", "") or "")[:1500]}
                if verified_tree is None:
                    # Reader output unparseable → treat as refuse, continue loop
                    per_iter_gamma_status.append("parse_fail")
                    per_iter_terminal.append(False)
                    inter = IntermediateOutput(raw=usage.get("raw", ""))
                else:
                    status = verified_tree.overall_status or "refuse"
                    per_iter_gamma_status.append(status)
                    gamma_final_status = status
                    # Per-q object grounding match-mode counts
                    # (final iteration's tree persists, matching gamma_final_status).
                    _obj_relaxed_n = sum(1 for s in verified_tree.steps
                                         if s.object_match_mode == "relaxed")
                    _obj_exact_n = sum(1 for s in verified_tree.steps
                                       if s.object_match_mode == "exact")
                    # Construct an IntermediateOutput-shape proxy for downstream code
                    inter = IntermediateOutput(raw=usage.get("raw", ""))
                    if status == "valid":
                        inter.answer = verified_tree.naturalized_answer or ""
                        inter.is_terminal = bool(inter.answer)
                    elif status == "refuse" and not cfg.use_gamma_refuse_loop:
                        # Legacy branch (default): refuse → terminal abstain
                        inter.answer = "Not in passages"
                        inter.is_terminal = True  # clean abstention
                    elif status == "refuse" and cfg.use_gamma_refuse_loop:
                        # P1: refuse triggers loop instead of terminating.
                        # γ rejected entire tree → no ungrounded "cue" available;
                        # broaden the query to expand retrieval coverage.
                        inter.is_terminal = False
                        # Weave accumulated graph-aware entities into the
                        # broadening (cross-iter context) when present; else legacy.
                        if accumulated_entities:
                            inter.next_query = _accumulated_entity_next_query(
                                question, accumulated_entities)
                            l2_reform_ent_total += min(len(accumulated_entities),
                                                       _L2_REFORM_ENTITY_CAP)
                        else:
                            inter.next_query = (
                                f"{question} (alternative phrasings, related entities, "
                                f"different keywords)"
                            )
                    else:
                        # partial / invalid → continue loop
                        inter.is_terminal = False
                        # Reformulate query to focus on missing/ungrounded steps
                        ungrounded = [s for s in verified_tree.steps
                                      if (s.verifier_status or "valid") != "valid"]
                        if ungrounded:
                            cue = ungrounded[0].claim_text or ungrounded[0].rule
                            # P7 — composite uses entity-seeded next-query (StepChain
                            # parity). Falls back to claim-shaped reformulation when
                            # flag is False or no entities extractable.
                            if composite:
                                inter.next_query = _entity_seeded_next_query(
                                    question, cue or "",
                                    pipe_link_entities=getattr(pipe, "_link_entities", None),
                                )
                                patch_fired["P7_entity_seeded_used"] = True
                                if "P7" not in patches_this_iter:
                                    patches_this_iter.append("P7")
                            else:
                                # FIX #2: prepend accumulated graph-aware entities
                                # to the claim cue (cross-iter context) when present;
                                # else the legacy claim-only reformulation.
                                if accumulated_entities:
                                    inter.next_query = _accumulated_entity_next_query(
                                        question, accumulated_entities, cue=cue or "")
                                    l2_reform_ent_total += min(
                                        len(accumulated_entities), _L2_REFORM_ENTITY_CAP)
                                else:
                                    inter.next_query = (f"{question} Find passages that "
                                                        f"ground: {cue}")
                    # extract facts from verified steps for L2 entity propagation
                    for s in verified_tree.steps:
                        if (s.verifier_status or "valid") == "valid" and s.claim_text:
                            inter.extracted_facts.append(s.claim_text)
                    # Cap retrigger (composite-aware via effective_gamma_cap)
                    if it >= effective_gamma_cap and not inter.is_terminal:
                        # FIX D — the γ-refuse-loop is now exhausted (cap hit with
                        # no terminal answer); used to skip the redundant
                        # faithfulness re-check at the loop-exit gate below.
                        gamma_refuse_loop_exhausted = True
                        # P11 — bug-pattern: prefer free-text
                        # reader on accumulated context UNCONDITIONALLY when
                        # accum_texts present. Wraps in try/except → naturalized_answer
                        # fallback → "Not in passages" as absolute last resort.
                        # P15 — also gate the legacy naturalized-only branch on
                        # gamma_status=="valid" to avoid drift (only emit nat answer
                        # when γ approved it).
                        # The disable override beats both wave_a
                        # and the individual P11 toggle. Required for the
                        # FULL-extras-MINUS-P11 ablation cells.
                        _p11_active = (
                            (cfg.use_bug_pattern_wave_a
                             or cfg.use_p11_gamma_cap_fallback)
                            and not cfg.disable_p11_gamma_cap_fallback
                        )
                        if _p11_active and accum_texts:
                            patch_fired["P11_gamma_cap_fallback_fired"] = True
                            if "P11" not in patches_this_iter:
                                patches_this_iter.append("P11")
                            try:
                                fb_ans, _fb_raw, fb_usage = pipe.read(
                                    question, accum_texts)
                                inter.answer = fb_ans
                                total_pt += fb_usage.get("prompt_tokens", 0)
                                total_ct += fb_usage.get("completion_tokens", 0)
                                total_lat += fb_usage.get("latency_s", 0.0)
                                inter.is_terminal = True
                            except Exception:  # noqa: BLE001
                                if verified_tree.naturalized_answer:
                                    inter.answer = verified_tree.naturalized_answer
                                    inter.is_terminal = True
                                else:
                                    inter.answer = "Not in passages"
                                    inter.is_terminal = True
                        elif cfg.use_gamma_router and accum_texts:
                            try:
                                fb_ans, _fb_raw, fb_usage = pipe.read(
                                    question, accum_texts)
                                inter.answer = fb_ans
                                total_pt += fb_usage.get("prompt_tokens", 0)
                                total_ct += fb_usage.get("completion_tokens", 0)
                                total_lat += fb_usage.get("latency_s", 0.0)
                                inter.is_terminal = True
                            except Exception:  # noqa: BLE001
                                if verified_tree.naturalized_answer:
                                    inter.answer = verified_tree.naturalized_answer
                                    inter.is_terminal = True
                                else:
                                    inter.answer = "Not in passages"
                                    inter.is_terminal = True
                        elif verified_tree.naturalized_answer:
                            inter.answer = verified_tree.naturalized_answer
                            inter.is_terminal = True
                        elif cfg.cap_branch_softfallback and accum_texts:
                            # P5: refuse-aware free-text reader fallback.
                            # Only kicks in when use_gamma_router is OFF AND γ tree
                            # has no naturalized_answer (i.e. pure refuse). Targets
                            # the 14 stuck-refuse cohort identified in the audit
                            # without P3's regression pattern (which fires free-text
                            # reader on every cap, including ones where nat is good).
                            try:
                                fb_ans, _fb_raw, fb_usage = pipe.read(
                                    question, accum_texts)
                                inter.answer = fb_ans
                                total_pt += fb_usage.get("prompt_tokens", 0)
                                total_ct += fb_usage.get("completion_tokens", 0)
                                total_lat += fb_usage.get("latency_s", 0.0)
                                inter.is_terminal = True
                            except Exception:  # noqa: BLE001
                                inter.answer = "Not in passages"
                                inter.is_terminal = True
                        else:
                            inter.answer = "Not in passages"
                            inter.is_terminal = True
                per_iter_terminal.append(inter.is_terminal)
            else:
                inter, usage = self._call_intermediate(
                    question, accum_texts, it, accumulated_facts,
                    spine_entities=accumulated_entities,
                )
                total_pt += usage["prompt_tokens"]
                total_ct += usage["completion_tokens"]
                total_lat += usage["latency_s"]
                per_iter_terminal.append(inter.is_terminal)

            # L4b: capture this iteration's candidate answer (for temporal C7).
            # In γ-mode use naturalized_answer when present (covers partial/invalid
            # trees too); otherwise the free-text ANSWER block.
            iter_candidate: Optional[str] = None
            if verified_tree is not None and verified_tree.naturalized_answer:
                iter_candidate = verified_tree.naturalized_answer
            elif inter.answer:
                iter_candidate = inter.answer
            per_iter_answer.append(iter_candidate)

            # 3. Accumulate facts (deduped, capped).
            for f in inter.extracted_facts:
                if f and f not in accumulated_facts:
                    accumulated_facts.append(f)
            if len(accumulated_facts) > 24:
                accumulated_facts = accumulated_facts[-24:]

            # Graph-aware iter: extract entities from facts + new query for next-iter seeding
            if cfg.use_graph_aware_iter:
                plugin = getattr(pipe, "plugin", None)
                ent_dict = getattr(pipe, "entities_by_id", None)
                if plugin is not None and ent_dict is not None:
                    seen_ent: set[str] = set(accumulated_entities)
                    for fact in inter.extracted_facts:
                        try:
                            for e in pipe._link_entities(fact):
                                if e not in seen_ent:
                                    seen_ent.add(e)
                                    accumulated_entities.append(e)
                        except Exception:  # noqa: BLE001
                            pass
                    if inter.next_query:
                        try:
                            for e in pipe._link_entities(inter.next_query):
                                if e not in seen_ent:
                                    seen_ent.add(e)
                                    accumulated_entities.append(e)
                        except Exception:  # noqa: BLE001
                            pass

            # 4. Decide loop exit.
            if inter.has_answer and cfg.stop_early:
                # Faithfulness gate before accepting ANSWER as terminal
                if (cfg.use_faithfulness_loop
                        and cfg.judge_client is not None
                        and cfg.judge_model is not None):
                    # Faithfulness ↔ γ coordination, two safe-skip branches:
                    #  clean_valid    → γ=valid (already γ-verified, redundant)
                    #  exhausted_safe → γ-loop exhausted AND qtype != chain_deep
                    #                   AND iter>=2 (non-chain_deep cap-hit is safe)
                    # chain_deep cap-hit answers KEEP the faithfulness gate (the
                    # recovery cohort). Either skip treats the answer as grounded
                    # (fscore = judge_min_score) ⇒ loop-exit byte-identical to a
                    # faithfulness PASS. Default OFF ⇒ legacy.
                    _faith_skip = _faithfulness_gamma_coord_skip(
                        cfg, gamma_final_status,
                        gamma_refuse_loop_exhausted=gamma_refuse_loop_exhausted,
                        qtype=_cohort_qtype, iter_count=it)
                    if _faith_skip is not None:
                        fscore, flabel = cfg.judge_min_score, "skipped_" + _faith_skip
                        if _faith_skip == "clean_valid":
                            faithfulness_skipped_clean_valid = True
                        else:
                            faithfulness_skipped_exhausted_safe = True
                    else:
                        fscore, flabel = self._check_faithfulness(
                            question, accum_texts, inter.answer)
                        faithfulness_active = True
                    per_iter_faith_score.append(fscore)
                    per_iter_faith_label.append(flabel)
                    cap = (cfg.max_judge_iterations
                           if cfg.max_judge_iterations is not None
                           else effective_max_iter + 1)
                    # Trace early-break paths so final iter is always recorded
                    def _record_trace_then_break():
                        iter_trace.append({
                            "iter": it,
                            "gamma_status": (per_iter_gamma_status[-1]
                                             if per_iter_gamma_status else None),
                            "query": current_q,
                            "passages_n": len(accum_texts),
                            "patches_active_this_iter": list(patches_this_iter),
                        })
                    if fscore >= cfg.judge_min_score or it >= cap:
                        # P14 — bug-pattern: on cap-exhausted
                        # AND ungrounded (fscore<judge_min_score), drop the answer
                        # and break to let final pipe.read run on accum_texts
                        # (majority sampling = different decoding, second chance).
                        # Grounded path (fscore meets threshold) accepts normally.
                        if (cfg.use_bug_pattern_wave_a
                                and it >= cap
                                and fscore < cfg.judge_min_score):
                            patch_fired["P14_faith_loop_second_chance"] = True
                            if "P14" not in patches_this_iter:
                                patches_this_iter.append("P14")
                            early_answer = None
                            early_raw = ""
                            # Record trace before break so final iter is included
                            iter_trace.append({
                                "iter": it,
                                "gamma_status": (per_iter_gamma_status[-1]
                                                 if per_iter_gamma_status else None),
                                "query": current_q,
                                "passages_n": len(accum_texts),
                                "patches_active_this_iter": list(patches_this_iter),
                            })
                            _pad_faith_lists()
                            break
                        # Grounded enough OR ran out of patience: accept
                        early_answer = inter.answer
                        early_raw = inter.raw
                        _record_trace_then_break()
                        _pad_faith_lists()
                        break
                    # Not grounded: drop ANSWER, continue with reformulated query
                    # Use missing/next_query if model gave them, else focus on grounding
                    if not inter.next_query and not inter.missing:
                        # P4: filter toxic "Not in passages" reformulation.
                        # Emitting `next_query = "Find passages
                        # that ground the claim: Not in passages"` pollutes iter=2+
                        # retrieval with semantically degenerate context. Skip the
                        # reformulation when answer is a known-abstain marker so
                        # the outer loop falls back to the original question (L631-633).
                        if not _is_abstain_marker(inter.answer):
                            # P7 — composite uses entity-seeded reformulation
                            if composite:
                                inter.next_query = _entity_seeded_next_query(
                                    question, inter.answer or "",
                                    pipe_link_entities=getattr(pipe, "_link_entities", None),
                                )
                                patch_fired["P7_entity_seeded_used"] = True
                                if "P7" not in patches_this_iter:
                                    patches_this_iter.append("P7")
                            else:
                                inter.next_query = (f"Find passages that ground the claim: "
                                                    f"{inter.answer}")
                else:
                    early_answer = inter.answer
                    early_raw = inter.raw
                    iter_trace.append({
                        "iter": it,
                        "gamma_status": (per_iter_gamma_status[-1]
                                         if per_iter_gamma_status else None),
                        "query": current_q,
                        "passages_n": len(accum_texts),
                        "patches_active_this_iter": list(patches_this_iter),
                    })
                    _pad_faith_lists()
                    break

            # Append per-iter trace BEFORE reformulation so
            # `query` reflects the query used to retrieve this iter's passages.
            iter_trace.append({
                "iter": it,
                "gamma_status": (per_iter_gamma_status[-1]
                                 if per_iter_gamma_status else None),
                "query": current_q,
                "passages_n": len(accum_texts),
                "patches_active_this_iter": list(patches_this_iter),
            })
            _pad_faith_lists()

            # 5. Reformulate query for next pass.
            if inter.next_query:
                current_q = inter.next_query
            elif inter.missing:
                current_q = f"{question} Focus on: {inter.missing}"
            else:
                # No useful guidance — keep original question (rare, drift fallback).
                current_q = question

        # ---- Final answer ----
        if early_answer is not None:
            ans = early_answer
            raw = early_raw
            final_usage = {"prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0}
        else:
            # Final canonical reader pass over the accumulated passage set.
            # If a stronger reader override is configured, route final
            # synthesis to it (cheap reader still handled intermediate hops).
            # Order final-synthesis evidence along the
            # multi-hop spine too (default OFF → accum_texts unchanged).
            final_passages = (
                _path_serialize_order(accum_texts, accumulated_entities)
                if cfg.path_serialize else accum_texts)
            if (cfg.final_reader_client is not None
                    and cfg.final_reader_model is not None):
                ans, raw, final_usage = self._call_final_with_override(
                    question, final_passages,
                    cfg.final_reader_client, cfg.final_reader_model,
                    cfg.final_reader_max_tokens,
                )
            else:
                ans, raw, final_usage = pipe.read(question, final_passages)
            total_pt += final_usage["prompt_tokens"]
            total_ct += final_usage["completion_tokens"]
            total_lat += final_usage["latency_s"]

        # ---- Aurora L4b: within-iterative-loop C7 (temporal stability) ----
        c7_iter_kept, c7_iter_info = self._apply_c7_iter(
            per_iter_answer, ans, gamma_final_status,
        )

        return IterativeAnswerInfo(
            answer=ans,
            raw=raw,
            passages=list(accum_texts),
            retrieved_chunk_ids=list(accum_ids),
            route=per_iter_route[0] if per_iter_route else "",
            top1_conf=per_iter_conf[0] if per_iter_conf else 0.0,
            latency_s=float(total_lat),
            prompt_tokens=int(total_pt),
            completion_tokens=int(total_ct),
            iterations_used=len(per_iter_route),
            per_iteration_route=per_iter_route,
            per_iteration_query=per_iter_query,
            per_iteration_top1_conf=per_iter_conf,
            per_iteration_terminal=per_iter_terminal,
            per_iteration_n_new_chunks=per_iter_new,
            accumulated_facts=list(accumulated_facts),
            accumulated_entities=list(accumulated_entities),
            per_iteration_n_seed_entities=per_iter_n_seed,
            l2_reformulation_entities_total=l2_reform_ent_total,
            gamma_retrigger_cap_used=effective_gamma_cap,
            faithfulness_active=faithfulness_active,
            faithfulness_skipped_clean_valid=faithfulness_skipped_clean_valid,
            faithfulness_skipped_exhausted_safe=faithfulness_skipped_exhausted_safe,
            gamma_diagnostic=_gamma_diag,
            object_relaxed_match_count=_obj_relaxed_n,
            object_exact_match_count=_obj_exact_n,
            per_iteration_faithfulness_score=per_iter_faith_score,
            per_iteration_faithfulness_label=per_iter_faith_label,
            per_iteration_gamma_status=per_iter_gamma_status,
            gamma_final_status=gamma_final_status,
            per_iteration_answer=list(per_iter_answer),
            c7_iter_kept=c7_iter_kept,
            c7_iter_info=c7_iter_info,
            patch_activations=dict(patch_fired),
            iter_trace=list(iter_trace),
        )


__all__ = [
    "INTERMEDIATE_SYSTEM",
    "IntermediateOutput",
    "IterativeConfig",
    "IterativeAnswerInfo",
    "IterativeMothRAG",
    "parse_intermediate",
]
