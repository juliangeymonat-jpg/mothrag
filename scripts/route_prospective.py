# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Paper B prospective routing driver.

Per-query, the sel_v2 query-type classifier picks UPFRONT which single arm
(or sel_v1 ensemble) to run, instead of running all three and arbitrating
post-hoc. Cheaper deployment than ``scripts/eval_wiki_iterative.py`` +
``scripts/arbitrate_post.py`` chained.

Routing (mode v2, default)::

    chain_deep      -> ITER full stack (IterativeMothRAG, all γ + L4b knobs)
    bridge_entity   -> DECOMPOSE only
    semantic_rich   -> sel_v1 (V3+bu + DECOMPOSE -> selective_arbitrate)

Routing (mode strict)::

    chain_deep      -> ITER full stack
    bridge_entity   -> DECOMPOSE only
    semantic_rich   -> V3+bu only (no arbitration)

NO L6 C7 cancellation (L6 is a cross-3-arm ensemble primitive — incompatible
with prospective single-arm dispatch).

CLI is a strict superset of ``scripts/eval_wiki_iterative.py``: only ``--mode``
is new. All iter-stack flags (γ verifier, γ router, faithfulness loop, L4b
C7-iter, L1C NER cache, etc.) carry through to the iter arm when it fires.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mothrag.core.decompose import (
    decompose_question_with_usage,
    synthesize_answer_with_usage,
)
from mothrag.core.query_type_classifier import (
    arm_subset,
    classify_query_v2,
    classify_with_features,
    is_polar_comparison,
    requires_implicit_multihop,
)
from mothrag.core.selective_ensemble import (
    arbitrate_excl_v3bu,
    is_uncertain,
    selective_arbitrate,
)
from mothrag.eval.iterative_pipeline import IterativeConfig, IterativeMothRAG
from mothrag.eval.metrics import em_score, f1_score
from mothrag.eval.pipeline import (
    MothRAGPipeline,
    PipelineConfig,
    call_reader_with_usage,
    make_reader_messages,
    parse_reader_output,
)


PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
}

PROVIDER_DEFAULT_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
}


def build_final_reader_client(provider: str | None, base_url: str | None,
                               api_key_env: str | None, model_name: str):
    """Mirror of eval_wiki_iterative.build_final_reader_client (keep parity)."""
    if base_url is None:
        if provider is None:
            raise SystemExit("--final-reader-model set but neither "
                             "--final-reader-provider nor --final-reader-base-url provided")
        if provider not in PROVIDER_BASE_URLS:
            raise SystemExit(f"unknown --final-reader-provider {provider!r}; "
                             f"choices: {sorted(PROVIDER_BASE_URLS)}")
        base_url = PROVIDER_BASE_URLS[provider]

    key_env = api_key_env or (PROVIDER_DEFAULT_KEY_ENV.get(provider) if provider else None)
    if not key_env:
        raise SystemExit("--final-reader-api-key-env not set and could not be inferred from provider")
    api_key = os.environ.get(key_env)
    if not api_key:
        raise SystemExit(f"--final-reader env var {key_env} is empty")

    from openai import OpenAI
    print(f"[setup] final reader: provider={provider} model={model_name} "
          f"base_url={base_url} key_env={key_env}")
    return OpenAI(api_key=api_key, base_url=base_url)


def build_c7_iter_embedder(name: str):
    """Mirror of eval_wiki_iterative.build_c7_iter_embedder (keep parity)."""
    import numpy as np
    if name in ("gemini-embedding-2", "gemini"):
        from google import genai
        from google.genai import types as gtypes
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("--c7-iter-embedder gemini-embedding-2 requires GEMINI_API_KEY env")
        client = genai.Client(api_key=api_key)

        def embed_batch(texts: list[str]) -> "np.ndarray":
            embs = []
            for t in texts:
                try:
                    r = client.models.embed_content(
                        model="gemini-embedding-2",
                        contents=t,
                        config=gtypes.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
                    )
                    embs.append(np.asarray(r.embeddings[0].values, dtype=np.float32))
                except Exception:  # noqa: BLE001
                    embs.append(np.zeros(3072, dtype=np.float32))
            return np.stack(embs, axis=0)

        print(f"[setup] L4b C7-iter embedder: gemini-embedding-2 (3072-d)")
        return embed_batch

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(name)

    def embed_batch_st(texts: list[str]) -> "np.ndarray":
        return np.asarray(model.encode(texts, show_progress_bar=False), dtype=np.float32)

    print(f"[setup] L4b C7-iter embedder: SentenceTransformer({name})")
    return embed_batch_st


def load_queryset(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        if path.suffix.lower() == ".jsonl":
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        else:
            obj = json.load(f)
            rows = obj if isinstance(obj, list) else obj.get("queries", [])
    return rows


def best_em_f1(pred: str, golds: list[str]) -> tuple[float, float]:
    if not golds:
        return 0.0, 0.0
    em = max(em_score(pred, g) for g in golds)
    f1 = max(f1_score(pred, g) for g in golds)
    return em, f1


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    if not gold_ids:
        return float("nan")
    top = set(retrieved_ids[:k])
    hit = len([g for g in gold_ids if g in top])
    return hit / len(gold_ids)


# ---- Per-arm runners ------------------------------------------------------

# The installed bridge substrate when a PER-ARM gate is
# active (else None), so the arm runners can mark which arm's retrievals follow.
# Set once in main() before the query loop. None in uniform / no-substrate mode →
# _mark_arm is a no-op → zero behaviour change.
_ACTIVE_BRIDGE_SUBSTRATE: "_BridgeSubstrate | None" = None


def _set_active_substrate(sub) -> None:
    global _ACTIVE_BRIDGE_SUBSTRATE
    _ACTIVE_BRIDGE_SUBSTRATE = sub


def _mark_arm(arm: str) -> None:
    """Tell the active per-arm bridge substrate which arm is about to retrieve.
    No-op unless a per-arm gate is installed. THREAD-LOCAL on the substrate, so it
    is race-free under the within-query arm-parallel pool (each arm thread marks
    itself)."""
    sub = _ACTIVE_BRIDGE_SUBSTRATE
    if sub is not None:
        sub.set_current_arm(arm)


def _run_v3bu(pipeline: MothRAGPipeline, question: str) -> dict:
    """V3+bu single-shot. Returns dict with pred + cost telemetry."""
    _mark_arm("v3bu")
    t0 = time.time()
    info = pipeline.answer(question)
    return {
        "pred": info.answer,
        "retrieved_chunk_ids": list(info.retrieved_chunk_ids),
        "n_llm_calls": int(pipeline.config.reader_n_samples
                           if isinstance(pipeline.config.reader_temperature, (int, float))
                           else len(pipeline.config.reader_temperature)),
        "prompt_tokens": int(info.prompt_tokens),
        "completion_tokens": int(info.completion_tokens),
        "latency_s": float(info.latency_s if info.latency_s else (time.time() - t0)),
    }


def _run_decompose(pipeline: MothRAGPipeline, question: str,
                   reader_model: str, top_k_subq: int) -> dict:
    """Decompose -> per sub-Q retrieve+read -> synthesize. Cost-tracked."""
    _mark_arm("decompose")
    t0 = time.time()
    n_calls = 0
    pt = ct = 0
    lat = 0.0
    retrieved: list[str] = []

    # 1. Decompose
    try:
        sub_qs, u = decompose_question_with_usage(
            pipeline.reader_client, reader_model, question)
        n_calls += 1
        pt += u["prompt_tokens"]
        ct += u["completion_tokens"]
        lat += u["latency_s"]
    except Exception as exc:  # noqa: BLE001
        return {
            "pred": "",
            "retrieved_chunk_ids": [],
            "n_llm_calls": n_calls,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "latency_s": float(time.time() - t0),
            "error": f"decompose: {type(exc).__name__}: {exc}",
        }
    # P12 bug-pattern: truncate decompositions with >6 sub_qs instead of
    # collapsing to [question]. Was a SILENT V3+bu duplicate on 4-hop MQ
    # chains. Gated under env var to preserve composite-then-bisect verdict.
    _wave_a = os.environ.get("MOTHRAG_BUG_PATTERN_WAVE_A") == "1"
    if not sub_qs:
        sub_qs = [question]
    elif len(sub_qs) > 6:
        if _wave_a:
            sub_qs = sub_qs[:6]  # P12: truncate, preserve first 6
        else:
            sub_qs = [question]  # legacy: collapse to single-shot

    # 2. Per sub-Q: retrieve + read (carry prior context across hops)
    sub_qa: list[tuple[str, str]] = []
    for sq_idx, sq in enumerate(sub_qs):
        if sq_idx > 0 and sub_qa:
            # P13 bug-pattern: filter sub-Q abstain/empty from prior_facts
            # and prior_entities so they don't propagate as legitimate
            # context for the next sub-Q (Pattern B toxic value).
            if _wave_a:
                clean = [(psq, psa) for psq, psa in sub_qa
                         if not is_uncertain(psa)]
                if clean:
                    prior_facts = "; ".join(f"{psq.rstrip('?')}: {psa}"
                                            for psq, psa in clean)
                    prior_entities = " ".join(psa for _, psa in clean
                                              if len(psa) < 60)
                else:
                    prior_facts = ""
                    prior_entities = ""
            else:
                prior_facts = "; ".join(f"{psq.rstrip('?')}: {psa}"
                                        for psq, psa in sub_qa)
                prior_entities = " ".join(psa for _, psa in sub_qa
                                          if len(psa) < 60)
            if prior_entities:
                retrieval_q = f"{sq} ({prior_entities})"
            else:
                retrieval_q = sq
            if prior_facts:
                reader_q = f"{sq}\n(Context from prior sub-answers: {prior_facts})"
            else:
                reader_q = sq
        else:
            retrieval_q = sq
            reader_q = sq

        try:
            top_idx, _route, _conf = pipeline.retrieve(retrieval_q)
        except Exception as exc:  # noqa: BLE001
            top_idx = []
        passages = [pipeline.chunks_by_id[pipeline.chunk_ids[ci]]["text"]
                    for ci in top_idx[:top_k_subq]]
        for ci in top_idx[:top_k_subq]:
            cid = pipeline.chunk_ids[ci]
            if cid not in retrieved:
                retrieved.append(cid)

        # Sub-Q reader: use v2 prompt (single-hop extractive — same as
        # eval_wiki_decompose.py default; cheaper than v3-think for sub-hops)
        try:
            sub_a_raw, u = call_reader_with_usage(
                pipeline.reader_client, reader_model,
                make_reader_messages(reader_q, passages, "v2"),
                max_tokens=64,
            )
            n_calls += 1
            pt += u["prompt_tokens"]
            ct += u["completion_tokens"]
            lat += u["latency_s"]
            sub_a = parse_reader_output(sub_a_raw, "v2")
        except Exception:  # noqa: BLE001
            sub_a = ""
        sub_qa.append((sq, sub_a))

    # 3. Synthesize
    try:
        final_ans, u = synthesize_answer_with_usage(
            pipeline.reader_client, reader_model, question, sub_qa)
        n_calls += 1
        pt += u["prompt_tokens"]
        ct += u["completion_tokens"]
        lat += u["latency_s"]
    except Exception as exc:  # noqa: BLE001
        final_ans = sub_qa[-1][1] if sub_qa else ""

    return {
        "pred": final_ans,
        "sub_qs": sub_qs,
        "sub_qa": sub_qa,
        "retrieved_chunk_ids": retrieved,
        "n_llm_calls": n_calls,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "latency_s": float(lat if lat else (time.time() - t0)),
    }


def _run_iter(runner: IterativeMothRAG, question: str) -> dict:
    """Iterative full stack — γ verifier + L4b C7 + faithfulness loop etc."""
    _mark_arm("iter")
    info = runner.answer(question)
    iters = max(int(info.iterations_used), 1)
    # Iter calls: 1 intermediate per iter + 1 final synthesis (when not stop_early
    # terminal) + faithfulness judge calls. Use iters as conservative lower bound.
    n_calls = iters
    return {
        "pred": info.answer,
        "retrieved_chunk_ids": list(info.retrieved_chunk_ids),
        "n_llm_calls": n_calls,
        "prompt_tokens": int(info.prompt_tokens),
        "completion_tokens": int(info.completion_tokens),
        "latency_s": float(info.latency_s),
        "iterations_used": iters,
        "gamma_final_status": info.gamma_final_status,
        "c7_iter_kept": info.c7_iter_kept,
        # FIX C — adaptive retrigger cap used this query.
        "gamma_retrigger_cap_used": getattr(info, "gamma_retrigger_cap_used", None),
        # FIX D — faithfulness γ-coord telemetry
        # (two safe-skip branches: clean_valid + exhausted_safe).
        "faithfulness_active": getattr(info, "faithfulness_active", None),
        "faithfulness_skipped_clean_valid":
            getattr(info, "faithfulness_skipped_clean_valid", None),
        "faithfulness_skipped_exhausted_safe":
            getattr(info, "faithfulness_skipped_exhausted_safe", None),
        # final-iter proof tree (None unless dump flag).
        "gamma_diagnostic": getattr(info, "gamma_diagnostic", None),
        # object grounding match-mode counts.
        "object_relaxed_match_count": getattr(info, "object_relaxed_match_count", 0),
        "object_exact_match_count": getattr(info, "object_exact_match_count", 0),
    }


# ---- ensemble_arbitrate + retry-on-abstain wiring -------------------------

def _arbitrate_candidates(
    pipeline: "MothRAGPipeline",
    *,
    candidates: dict[str, dict],
    iter_gamma_status: str | None = None,
    arm_probabilities: dict | None = None,
    w_gamma: float = 1.0,
    w_agree: float = 0.5,
    w_faith: float = 0.3,
    simulate_n_cap: int | None = None,
    use_gamma_aware_pdd: bool = False,
    qtype: str | None = None,
) -> dict:
    """Arbitrate among candidate arm outputs and merge their telemetry.

    Shared by :func:`_run_ensemble_arbitrate` (ensemble_arbitrate mode)
    and the v2-mode opt-in arm composition. Single source of truth for
    the gamma + cross-arm agreement + P_arm arbitration logic so the
    two callers cannot drift.

    Parameters
    ----------
    pipeline
        MothRAGPipeline (used to source the embedder for pairwise
        cross-arm agreement).
    candidates
        ``{arm_name: result_dict}`` with the per-arm result shape used
        by the legacy 3-arm runners (pred, retrieved_chunk_ids,
        n_llm_calls, prompt_tokens, completion_tokens, latency_s).
    iter_gamma_status
        Optional γ status from the iter arm (``"valid"`` / ``"partial"``
        / ``"invalid"`` / ``None``). Threaded to the arbitrator as a
        signal on the ``iter`` candidate.
    arm_probabilities
        Optional PAM-lite ``{arm_name: P_arm in [0,1]}`` dict.
        Multiplicatively modulates per-arm signal score.

    Returns
    -------
        Merged result dict matching :func:`_run_ensemble_arbitrate`'s
        return shape's intersection (pred + retrieval ids + cost +
        latency + selected_arm + arbitrate_signal + arm_scores). The
        caller layers on per-mode fields (subset, arm_subset, etc.)
        as needed.
    """
    # The γ + cross-arm-agreement + P_arm scoring core now lives in the unified
    # mothrag.core.arms_runner.arbitrate_pool (single source of truth shared with
    # the pip path). This delegation is byte-identical to the prior inline logic
    # (same answers / iter-γ / pairwise_agreement@0.70 / simulate_n_cap /
    # DeterministicArbitrator(w_gamma,w_agree,w_faith)).
    from mothrag.core.arms_runner import arbitrate_pool, gamma_aware_pdd_should_skip

    if not candidates:
        return {
            "pred": "",
            "retrieved_chunk_ids": [],
            "n_llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_s": 0.0,
            "selected_arm": "",
            "arbitrate_signal": "fallback",
            "arm_scores": {},
            "pdd_active": False,
            "pdd_skipped_chain_deep_valid": False,
            "pdd_preserved_semantic_rich": False,
        }

    # γ-aware gating telemetry: decide via the SAME shared predicate
    # arbitrate_pool uses, so the recorded counters never drift from the actual
    # pool composition. ``pdd_present`` = a dup arm exists this query.
    # PDD cohort gate (reversed variant): ``_pdd_preserved_semantic_rich`` = the
    # dup WOULD have been dropped (flag on + iter γ=valid + present) but the
    # NON-chain_deep cohort gate kept it. This variant reverses the original: the
    # dup is now dropped ONLY on chain_deep (ensemble vote noisy there) and
    # PRESERVED on the semantic_rich bulk (semantic_rich needs PDD). The three
    # counters are mutually exclusive at the "fired" level: skipped XOR preserved
    # XOR (neither → active by default).
    from mothrag.routing.dup_arm import is_dup_arm as _is_dup_arm
    _pdd_present = any(_is_dup_arm(n) for n in candidates)
    _pdd_skipped = gamma_aware_pdd_should_skip(
        candidates, iter_gamma_status, qtype, enabled=use_gamma_aware_pdd)
    _pdd_preserved_semantic_rich = (
        bool(use_gamma_aware_pdd) and iter_gamma_status == "valid"
        and qtype != "chain_deep" and _pdd_present)
    _pdd_active = _pdd_present and not _pdd_skipped

    embedder = (
        getattr(pipeline, "embedder_model", None)
        or _PipelineEmbedderShim(pipeline)
    )
    result = arbitrate_pool(
        candidates,
        pred_of=lambda info: info.get("pred") or "",
        embedder=embedder,
        iter_gamma_status=iter_gamma_status,
        arm_probabilities=arm_probabilities,
        w_gamma=w_gamma, w_agree=w_agree, w_faith=w_faith,
        simulate_n_cap=simulate_n_cap,
        use_gamma_aware_pdd=use_gamma_aware_pdd,
        qtype=qtype,
    )

    # Merge retrieved chunk ids (preserve order; dedupe).
    retrieved: list[str] = []
    seen: set[str] = set()
    for name in candidates:
        for cid in candidates[name].get("retrieved_chunk_ids", []):
            if cid not in seen:
                retrieved.append(cid)
                seen.add(cid)

    return {
        "pred": result.answer,
        "retrieved_chunk_ids": retrieved,
        "n_llm_calls": sum(c.get("n_llm_calls", 0) for c in candidates.values()),
        "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in candidates.values()),
        "completion_tokens": sum(c.get("completion_tokens", 0) for c in candidates.values()),
        "latency_s": sum(c.get("latency_s", 0.0) for c in candidates.values()),
        "selected_arm": result.selected_arm,
        "arbitrate_signal": result.arbitrate_signal,
        "arm_scores": result.arm_scores,
        # γ-aware gating + PDD cohort-gated telemetry.
        "pdd_active": _pdd_active,
        "pdd_skipped_chain_deep_valid": _pdd_skipped,
        "pdd_preserved_semantic_rich": _pdd_preserved_semantic_rich,
    }


def _build_specialist_router(pipeline: "MothRAGPipeline", reader_model: str,
                             top_k_subq: int):
    """Assemble the pool-safe ``SpecialistSlotRouter`` wired
    to the live pipeline. The M8 specialists are RETRIEVAL shapers; the
    polymorphic ``decompose`` slot SUBSTITUTES one of them on its cohort
    (comparison / compositional, input-feature classified) so the arm pool stays
    exactly 4. Built only when ``--use-m8-specialists`` is set (default OFF →
    this is never constructed and the slot is the generic decompose arm)."""
    from types import SimpleNamespace

    from mothrag.routing import SpecialistSlotRouter
    from mothrag.routing.specialist_slot_adapters import (
        make_reader_slot_reader, make_specialist_slot_runner,
    )
    from mothrag.retrieval.specialist.compare_arm import (
        CompareArm, is_comparison_query,
    )
    from mothrag.retrieval.specialist.decompose_arm_v2 import (
        DecomposeArmV2, contains_compositional_markers, needs_decomposition,
    )

    def _ann(query: str, k: int):
        """pipeline.retrieve indices -> [obj(passage_id, text)] for the specialists."""
        try:
            top_idx, *_ = pipeline.retrieve(query)
        except Exception:  # noqa: BLE001
            return []
        out = []
        for ci in list(top_idx)[:k]:
            cid = pipeline.chunk_ids[ci]
            out.append(SimpleNamespace(
                passage_id=cid,
                text=(pipeline.chunks_by_id.get(cid, {}) or {}).get("text", "")))
        return out

    def _decomposer(q: str):
        try:
            subs, _u = decompose_question_with_usage(
                pipeline.reader_client, reader_model, q)
            return list(subs or [])
        except Exception:  # noqa: BLE001
            return []

    shim = _PipelineReaderShim(pipeline, reader_model)
    read_slot = make_reader_slot_reader(
        reader=shim,
        fetch_texts=lambda ids: [
            (pipeline.chunks_by_id.get(i, {}) or {}).get("text", "")
            for i in ids if i in pipeline.chunks_by_id])

    compare_runner = make_specialist_slot_runner(
        specialist=CompareArm(_ann), read_slot=read_slot, name="compare_arm")
    decompose_runner = make_specialist_slot_runner(
        specialist=DecomposeArmV2(_ann, decomposer=_decomposer, answerer=shim.read),
        read_slot=read_slot, name="decompose_arm_v2")

    return SpecialistSlotRouter(
        compare_arm=compare_runner,
        decompose_arm_v2=decompose_runner,
        is_comparison=is_comparison_query,
        is_compositional=lambda q: needs_decomposition(q)
        or contains_compositional_markers(q),
        enabled=True,
    )


def _run_ensemble_arbitrate(
    pipeline: "MothRAGPipeline",
    iter_runner: "IterativeMothRAG",
    question: str,
    reader_model: str,
    top_k_subq: int,
    *,
    arms_pool: list[str] | None = None,
    opt_in_arms: dict | None = None,
    specialist_router=None,
    router: str = "sel_v2",
    pam_lite_threshold: float = 0.3,
    # mechanism ablation flags
    w_gamma: float = 1.0,
    w_agree: float = 0.5,
    w_faith: float = 0.3,
    simulate_n_cap: int | None = None,
    dup_random_answer: bool = False,
    use_gamma_aware_pdd: bool = False,
    qtype: str | None = None,
    precomputed_iter: dict | None = None,
) -> dict:
    """Adaptive subset routing + arbitrate-when-multi.

    PRESERVES the Pareto-dominant adaptive routing
    (+0.15pp F1 / -38-55% compute):

      1. ``arm_subset(question)`` picks 1, 2, or 3 arms to run.
      2. Only the selected arms execute.
      3. If subset size == 1 -> return that arm's answer directly
         (no arbitrate overhead, identical cost+F1 to adaptive path).
      4. If subset size >= 2 -> apply DeterministicArbitrator over the
         running subset only (NOT the full v3bu/decompose/iter triple).

    Arbitrate is therefore a *value-add when the routing classifier
    elects multiple arms*, NOT a replacement for sel_v2 routing.
    Default mode='v2' (adaptive) keeps the production behaviour exactly;
    mode='ensemble_arbitrate' adds arbitrate-when-multi on top.
    """
    from mothrag.core.arbitrate import DeterministicArbitrator
    from mothrag.core.query_type_classifier import arm_subset as _arm_subset

    # PAM-lite extension: when router='pam_lite', use continuous-
    # probability subset + thread P_arm into arbitration. When
    # router='sel_v2' (default), arm_probabilities stays empty and
    # arbitration behaves byte-identical to the prior version. Both
    # paths thread arms_pool so opt-in arms (infobox_arm, mothgraph_arm)
    # can enter the subset under either router.
    if router == "pam_lite":
        try:
            from mothrag.core.query_type_classifier import arm_subset_pam_lite
            subset_list, arm_probabilities = arm_subset_pam_lite(
                question,
                threshold=pam_lite_threshold,
                arms_pool=arms_pool,
            )
            subset = list(subset_list)
        except Exception:  # noqa: BLE001
            subset = _arm_subset(question, arms_pool=arms_pool)
            arm_probabilities = {}
    else:
        subset = _arm_subset(question, arms_pool=arms_pool)
        arm_probabilities = {}

    # Pool-safety axiom: the legacy V3+bu
    # execution gate (v3bu_in) MUST be decided by LEGACY-ONLY routing
    # (3-arm pool semantics), regardless of arms_pool. If we use the
    # FULL subset (which may include opt-in arms above threshold),
    # then in the edge case where PAM-lite's always-non-empty
    # argmax-fallback picks an opt-in arm (e.g. P_infobox > P_v3bu
    # when all legacy P are below threshold), v3bu_in flips False ->
    # V3+bu skipped in 5-arm pool but RUNS in 3-arm pool ->
    # arbitration candidates differ -> F1 differs on the F1=1 cohort.
    #
    # Empirical: 5/50 MQ T1 queries diverge (10%)
    # despite 0% opt-in fire rate. Root cause: argmax-fallback over
    # 5-arm pool picks opt-in when legacy below threshold.
    #
    # Fix: re-derive v3bu_in from a LEGACY-ONLY subset call.
    if router == "pam_lite":
        try:
            legacy_subset, _ = arm_subset_pam_lite(
                question, threshold=pam_lite_threshold,
                # NB no arms_pool -> default legacy 3-arm
            )
            v3bu_in = "v3bu" in legacy_subset
        except Exception:  # noqa: BLE001
            v3bu_in = "v3bu" in _arm_subset(question)
    else:
        # sel_v2 legacy cascade: arms_pool extension only ADDS opt-in
        # arms; never removes legacy arms. v3bu_in is identical
        # regardless of arms_pool. But for consistency with the
        # PAM-lite branch, derive from legacy-only call.
        v3bu_in = "v3bu" in _arm_subset(question)

    # Run the same arms that the adaptive path would run: v3bu only when
    # included in the subset; decompose and iter always run (adaptive
    # rationale -- both are routing-cheap given they share retrieval).
    a_v3 = _run_v3bu(pipeline, question) if v3bu_in else None
    # The decompose SLOT is polymorphic. With the M8
    # specialist router installed, a comparison / compositional question fills
    # the slot from CompareArm / DecomposeArm 2.0 by SUBSTITUTION (pool stays 4);
    # the specialist declining (no fire / fallback) degrades to the generic
    # decompose arm. Default (router None) is byte-identical to the prior line.
    def _generic_decompose(q, **_kw):
        return _run_decompose(pipeline, q, reader_model, top_k_subq)
    if specialist_router is not None and getattr(specialist_router, "enabled", False):
        a_de, _slot_dec = specialist_router.run_decompose_slot(
            question, generic_runner=_generic_decompose)
        if _slot_dec.is_specialist and isinstance(a_de, dict):
            a_de = dict(a_de)
            _m = dict(a_de.get("metadata") or {})
            _m["decompose_slot_arm"] = _slot_dec.arm_name
            _m["decompose_slot_qtype"] = _slot_dec.qtype
            a_de["metadata"] = _m
    else:
        a_de = _generic_decompose(question)
    # FIX B — reuse the dense iter probe when the bridge
    # cohort gate skipped the bridge (the pool would otherwise recompute an
    # identical dense iter). When the bridge fired, precomputed_iter is None and
    # iter re-runs against the bridged retrieval.
    a_it = (precomputed_iter if precomputed_iter is not None
            else _run_iter(iter_runner, question))

    candidates: dict[str, dict] = {}
    if a_v3 is not None:
        candidates["v3bu"] = a_v3
    candidates["decompose"] = a_de
    candidates["iter"] = a_it

    iter_gamma = a_it.get("gamma_final_status")

    # ---- dup-arm registration ---------------------------------------
    # When ``arms_pool`` contains entries of the form
    # ``<base>_dup_<suffix>`` (e.g. ``v3bu_dup_a``), populate a SEPARATE
    # candidate entry that re-uses the base arm's already-computed
    # result. By design these duplicates produce identical predictions
    # to their base -- the mechanism-attribution test
    # measures the dispatch-diversification effect this creates in
    # arbitration's pairwise agreement (NOT a pool-safety violation:
    # the dup IS fired; its prediction IS the base's by construction).
    if arms_pool:
        from mothrag.routing.dup_arm import is_dup_arm, base_arm_of
        _base_to_result = {
            "v3bu":      a_v3 if v3bu_in else None,
            "decompose": a_de,
            "iter":      a_it,
        }
        for name in arms_pool:
            if not is_dup_arm(name):
                continue
            try:
                base = base_arm_of(name)
            except ValueError:
                continue
            base_result = _base_to_result.get(base)
            if base_result is None:
                # Base arm didn't run (e.g., v3bu_excluded for v3bu_dup_*).
                # Skip: pool-safety -- dup cannot fire when base didn't.
                continue
            # Copy the base result under the dup arm_id so arbitration
            # sees a separate slot. Mark metadata so downstream
            # consumers can filter dup slots if needed.
            dup_result = dict(base_result)
            metadata = dict(dup_result.get("metadata") or {})
            metadata["dup_of"] = base
            metadata["dup_arm_id"] = name
            dup_result["metadata"] = metadata
            candidates[name] = dup_result

    # ---- opt-in arm pool composition ------------------
    # For each arm in ``arms_pool`` beyond the legacy three, check if
    # it's wired in ``opt_in_arms`` and applicable to ``question``; run
    # it and include in the candidates dict for arbitration. Backward
    # compat: when arms_pool is None / equals legacy three, this block
    # is a no-op.
    if arms_pool and opt_in_arms:
        for arm_name in arms_pool:
            if arm_name in ("v3bu", "decompose", "iter"):
                continue
            arm = opt_in_arms.get(arm_name)
            if arm is None:
                continue
            try:
                if not arm.applicable(question):
                    continue
            except Exception:  # noqa: BLE001
                continue
            try:
                result = arm.run(question, reader_model=reader_model)
            except Exception:  # noqa: BLE001
                continue
            if not result.pred:
                continue
            # Pool-safety axiom: an opt-in arm's soft
            # fallback result MUST NOT enter the arbitration candidate
            # set. The fallback's pred is typically a DUPLICATE of a
            # legacy arm's answer (e.g. MothGraphArm fallback delegates
            # to V3+bu); including it as a separate candidate inflates
            # pairwise_agreement -> spurious consensus boost ->
            # legitimate disagreeing arms (e.g. iter with the correct
            # answer) lose arbitration. Empirical: MQ
            # F1=1 cohort -23/-26pp regression.
            if result.metadata.get("is_fallback"):
                continue
            candidates[arm_name] = {
                "pred": result.pred,
                "retrieved_chunk_ids": list(result.retrieved_chunk_ids),
                "n_llm_calls": int(result.n_llm_calls),
                "prompt_tokens": int(result.prompt_tokens),
                "completion_tokens": int(result.completion_tokens),
                "latency_s": float(result.latency_s),
                "metadata": dict(result.metadata),
            }

    # ---- Single-arm subset: pass-through (no arbitrate). ------------
    if len(candidates) <= 1:
        sole_name, sole = (
            next(iter(candidates.items())) if candidates else ("", {"pred": ""})
        )
        return {
            "pred": sole.get("pred", "") if sole else "",
            "retrieved_chunk_ids": list(sole.get("retrieved_chunk_ids", [])) if sole else [],
            "n_llm_calls": sole.get("n_llm_calls", 0) if sole else 0,
            "prompt_tokens": sole.get("prompt_tokens", 0) if sole else 0,
            "completion_tokens": sole.get("completion_tokens", 0) if sole else 0,
            "latency_s": sole.get("latency_s", 0.0) if sole else 0.0,
            "iterations_used": (
                sole.get("iterations_used", 0) if sole_name == "iter" else 0
            ),
            "gamma_final_status": (
                sole.get("gamma_final_status") if sole_name == "iter" else None
            ),
            "selected_arm": sole_name,
            "arbitrate_signal": "single_arm_passthrough",
            "arm_scores": {sole_name: 0.0} if sole_name else {},
            "v3bu_pred": (a_v3 or {}).get("pred"),
            "dec_pred": (a_de or {}).get("pred"),
            "iter_pred": (a_it or {}).get("pred"),
            "subset_size": len(candidates),
            "arm_subset": list(subset),
        }

    # ---- Multi-arm subset: arbitrate over the running arms only. ----
    # Ablation B: replace dup-arm answers with a random
    # picks from other arms in the pool so the dup no longer matches
    # its base via cosine. Isolates the agreement-match component of
    # the PDD mechanism (see pam_lite_mechanism.py step 5).
    if dup_random_answer:
        try:
            from mothrag.routing.dup_arm import is_dup_arm
        except ImportError:
            is_dup_arm = lambda n: "_dup_" in n  # noqa: E731 (defensive)
        import random as _r047
        _rng047 = _r047.Random(0)
        other_preds = [
            c.get("pred", "") for n, c in candidates.items()
            if not is_dup_arm(n) and c.get("pred")
        ]
        if other_preds:
            for name in list(candidates.keys()):
                if is_dup_arm(name):
                    candidates[name] = dict(candidates[name])
                    candidates[name]["pred"] = _rng047.choice(other_preds)

    arb = _arbitrate_candidates(
        pipeline,
        candidates=candidates,
        iter_gamma_status=iter_gamma,
        arm_probabilities=arm_probabilities,
        w_gamma=w_gamma,
        w_agree=w_agree,
        w_faith=w_faith,
        simulate_n_cap=simulate_n_cap,
        use_gamma_aware_pdd=use_gamma_aware_pdd,
        qtype=qtype,
    )

    return {
        "pred": arb["pred"],
        "retrieved_chunk_ids": arb["retrieved_chunk_ids"],
        "n_llm_calls": arb["n_llm_calls"],
        "prompt_tokens": arb["prompt_tokens"],
        "completion_tokens": arb["completion_tokens"],
        "latency_s": arb["latency_s"],
        "iterations_used": a_it.get("iterations_used", 0),
        "gamma_final_status": iter_gamma,
        "selected_arm": arb["selected_arm"],
        "arbitrate_signal": arb["arbitrate_signal"],
        "arm_scores": arb["arm_scores"],
        "v3bu_pred": (a_v3 or {}).get("pred"),
        "dec_pred": a_de.get("pred"),
        "iter_pred": a_it.get("pred"),
        "subset_size": len(candidates),
        "arm_subset": list(subset),
        # γ-aware gating + PDD cohort-gated telemetry (pass-through).
        "pdd_active": arb.get("pdd_active", False),
        "pdd_skipped_chain_deep_valid": arb.get("pdd_skipped_chain_deep_valid", False),
        "pdd_preserved_semantic_rich": arb.get("pdd_preserved_semantic_rich", False),
        # FIX C/D — iter-side telemetry pass-through.
        "gamma_retrigger_cap_used": a_it.get("gamma_retrigger_cap_used"),
        "faithfulness_active": a_it.get("faithfulness_active"),
        "faithfulness_skipped_clean_valid":
            a_it.get("faithfulness_skipped_clean_valid"),
        "faithfulness_skipped_exhausted_safe":
            a_it.get("faithfulness_skipped_exhausted_safe"),
        "gamma_diagnostic": a_it.get("gamma_diagnostic"),
        "object_relaxed_match_count": a_it.get("object_relaxed_match_count", 0),
        "object_exact_match_count": a_it.get("object_exact_match_count", 0),
    }


class _PipelineEmbedderShim:
    """Adapter exposing the MothRAGPipeline's embedder via embed_batch."""

    def __init__(self, pipeline: "MothRAGPipeline") -> None:
        self._pipeline = pipeline

    def embed_batch(self, texts):
        # Prefer the query_embedder (single-text path used at retrieve()
        # time); fall back to whatever batch encoder is exposed.
        qe = getattr(self._pipeline, "query_embedder", None)
        if qe is not None:
            import numpy as np
            return np.stack([np.asarray(qe(t), dtype=np.float32)
                             for t in texts], axis=0)
        fn = getattr(self._pipeline, "encode_batch", None) \
            or getattr(self._pipeline, "encode", None) \
            or getattr(self._pipeline, "embed_batch", None)
        if fn is None:
            raise RuntimeError("MothRAGPipeline has no embed_batch / encode method")
        return fn(list(texts))


# ---- opt-in arm pool helpers ----------------------------------------------

def _parse_arms_pool(spec: str) -> list[str]:
    """Parse the ``--arms-pool`` CLI value into an ordered, deduped list.

    Default pool: ``v3bu,decompose,iter``. Opt-in additions: ``infobox_arm``,
    ``bm25_arm``. Unknown names are passed through so future arms compose
    without code changes here.

    Also accepts dup-arm names of the form ``<base>_dup_<suffix>`` for the
    dispatch-diversification test. Dup names are passed through; malformed
    ``*_dup_*`` strings raise.
    """
    if not spec or not spec.strip():
        return ["v3bu", "decompose", "iter"]
    from mothrag.routing.dup_arm import validate_dup_arm_name
    names: list[str] = []
    seen: set[str] = set()
    for raw in spec.split(","):
        n = raw.strip().lower()
        if not n or n in seen:
            continue
        # Surface malformed dup-arm names early at parse time.
        if "_dup_" in n:
            validate_dup_arm_name(n)
        seen.add(n)
        names.append(n)
    return names


def _build_infobox_arm_from_pipeline(pipeline):
    """Construct an :class:`InfoboxArm` over the pipeline's existing chunks.

    Builds a fresh :class:`InfoboxIndex` from the pipeline's prose chunks
    (wikitext + natural-fact extractors) and returns an InfoboxArm wrapper.
    Returns ``None`` when zero triples are harvestable -- the arm would
    always decline, no point including it in the pool.

    NB: this is INDEPENDENT of the ``_InfoboxGate`` mechanism. The gate
    injects synthetic infobox chunks into the dense retrieval index;
    InfoboxArm uses the same harvested triples for DIRECT
    structured-fact lookup with NO LLM call. Both can coexist.
    """
    from mothrag.arms import InfoboxArm
    from mothrag.core.retrieval import (
        InfoboxIndex, extract_natural_facts, extract_wikitext_infobox,
    )

    index = InfoboxIndex()
    for cid in list(pipeline.chunk_ids):
        text = pipeline.chunks_by_id.get(cid, {}).get("text", "")
        if not text:
            continue
        # Skip synthetic infobox chunks (C3 augmentation already
        # injected them; harvesting them again would double-count).
        if cid.startswith("infobox:"):
            continue
        index.add_many(extract_wikitext_infobox(text, source_chunk_id=cid))
        index.add_many(extract_natural_facts(text, source_chunk_id=cid))

    if len(index) == 0:
        return None
    return InfoboxArm(infobox_index=index)


def _build_mothgraph_arm_from_pipeline(
    pipeline,
    reader_model: str,
    *,
    max_iters: int = 3,
    base_depth: int = 2,
    top_k: int = 8,
    use_spacy: bool = False,
):
    """Construct a :class:`MothGraphArm` over the pipeline's prose chunks.

    Builds a fresh :class:`GraphIndex` via :func:`extract_triples` over
    the pipeline's prose, wires a dense-fallback callable that delegates
    to the V3+bu arm so the soft-fallback path returns a real reader
    answer when no anchor / no valid path is available, and returns the
    composed :class:`MothGraphArm`.

    Returns ``None`` when zero triples are harvestable -- the arm would
    always fall back, so no point including it in the pool.
    """
    from mothrag.arms import MothGraphArm
    from mothrag.arms.base import ArmResult
    from mothrag.graph import build_graph_index_from_chunks

    # Wrap pipeline chunks as id+text dicts; build_graph_index_from_chunks
    # accepts mapping-style or attribute-style chunk objects.
    chunks: list[dict] = []
    for cid in list(pipeline.chunk_ids):
        if cid.startswith("infobox:"):
            continue
        text = pipeline.chunks_by_id.get(cid, {}).get("text", "")
        if not text:
            continue
        chunks.append({"chunk_id": cid, "text": text})

    graph_index = build_graph_index_from_chunks(
        chunks, use_spacy=use_spacy,
    )
    if len(graph_index) == 0:
        return None

    def _dense_fallback(question: str) -> ArmResult:
        try:
            v3 = _run_v3bu(pipeline, question)
        except Exception:  # noqa: BLE001
            return ArmResult(pred="")
        return ArmResult(
            pred=v3.get("pred", "") or "",
            retrieved_chunk_ids=list(v3.get("retrieved_chunk_ids", [])),
            n_llm_calls=int(v3.get("n_llm_calls", 0)),
            prompt_tokens=int(v3.get("prompt_tokens", 0)),
            completion_tokens=int(v3.get("completion_tokens", 0)),
            latency_s=float(v3.get("latency_s", 0.0)),
            metadata={"fallback_path": "v3bu"},
        )

    return MothGraphArm(
        graph_index=graph_index,
        dense_fallback=_dense_fallback,
        max_iters=max_iters,
        base_depth=base_depth,
        top_k=top_k,
    )


# ---- full pipeline-ctx adapters for #8 + #9 --------------------------------

class _PipelineReaderShim:
    """Adapter exposing the MothRAGPipeline's reader_client via .read().

    Strategy #8 (ActiveGapQuery) and #9 (SubQuestionRerouteCascade) require
    a ``ctx.reader`` with a ``.read(question, passages) -> str`` surface for
    self-introspection / decomposition / composition LLM calls. The
    production stack exposes only ``pipeline.reader_client`` (an
    OpenAI-compatible client) and the call helpers in
    :mod:`mothrag.eval.pipeline`. This shim wraps those into the abstract
    Reader surface that the strategies consume.

    Uses the ``v2`` reader prompt (single-hop extractive), matching the
    style of the decompose arm's sub-question reader for prompt-shape
    consistency. Max tokens kept terse (256) -- the strategy callers
    parse short answers / decomposition lists; long reasoning is not
    needed.
    """

    def __init__(
        self,
        pipeline: "MothRAGPipeline",
        reader_model: str,
        *,
        max_tokens: int = 256,
    ) -> None:
        self._pipeline = pipeline
        self._reader_model = reader_model
        self._max_tokens = int(max_tokens)

    def read(self, question: str, passages) -> str:
        msgs = make_reader_messages(
            question, list(passages), "v2",
        )
        try:
            raw, _usage = call_reader_with_usage(
                self._pipeline.reader_client, self._reader_model, msgs,
                max_tokens=self._max_tokens,
            )
            return parse_reader_output(raw, "v2") or raw or ""
        except Exception:  # noqa: BLE001
            return ""


class _PipelineVdbShim:
    """Adapter exposing the MothRAGPipeline's chunk index as a VectorStore.

    Strategy #8 (ActiveGapQuery) calls ``ctx.vector_db.retrieve(q_emb,
    top_k)`` for targeted re-retrieval on the self-introspected gap
    query. The production stack does NOT expose an embedding-keyed
    retrieve method on the pipeline directly (it owns the chunk
    embedding matrix internally); this shim re-implements
    "embedding-dot-product top-K" using the pipeline's
    ``chunk_vecs`` + ``chunk_ids`` + ``chunks_by_id`` triple and returns
    a list of chunk-like objects with the ``.text`` attribute the
    strategies consume.
    """

    def __init__(self, pipeline: "MothRAGPipeline") -> None:
        self._pipeline = pipeline

    def retrieve(self, q_emb, top_k: int = 5):
        import numpy as np
        chunk_vecs = getattr(self._pipeline, "chunk_vecs", None)
        if chunk_vecs is None or len(chunk_vecs) == 0:
            return []
        q = np.asarray(q_emb, dtype=np.float32)
        if q.ndim != 1:
            return []
        scores = chunk_vecs @ q
        n = min(int(top_k), len(scores))
        if n <= 0:
            return []
        top_idx = np.argsort(-scores)[:n]
        out = []
        for ci in top_idx:
            cid = self._pipeline.chunk_ids[int(ci)]
            text = self._pipeline.chunks_by_id[cid]["text"]
            out.append(_PipelineChunk(chunk_id=cid, text=text))
        return out

    def __len__(self) -> int:
        return len(getattr(self._pipeline, "chunk_ids", ()))


class _PipelineChunk:
    """Minimal chunk-like surface (.chunk_id + .text) for shim returns."""

    __slots__ = ("chunk_id", "text")

    def __init__(self, chunk_id: str, text: str) -> None:
        self.chunk_id = chunk_id
        self.text = text


def _resolve_arm_subset(
    question: str,
    *,
    arms_pool: list[str] | None = None,
) -> list[str]:
    """Compute the actual sel_v2 a priori arm choice for the question.

    Replaces the previous hard-coded ['v3bu', 'decompose', 'iter']
    placeholder in ``_maybe_run_escalation``. Strategies #8 ActiveGapQuery
    + #9 SubQuestionRerouteCascade consume ``ctx.arm_subset`` to pick the
    runner that respects the production sel_v2 a priori dispatch (see
    feature/active-gap-query-strategy commit 0a673c8 for the architectural
    contract).

    When ``arms_pool`` is provided, sel_v2 also evaluates opt-in arms
    (``infobox_arm``, ``mothgraph_arm``) against their binary-threshold
    triggers and includes them in the returned subset when applicable.
    """
    try:
        result = arm_subset(question, arms_pool=arms_pool)
        return list(result) or ["v3bu"]
    except Exception:  # noqa: BLE001
        return ["v3bu"]


def _resolve_arm_subset_with_router(
    question: str,
    *,
    router: str = "sel_v2",
    pam_lite_threshold: float = 0.3,
    arms_pool: list[str] | None = None,
) -> tuple[list[str], dict[str, float]]:
    """PAM-lite-aware variant of :func:`_resolve_arm_subset`.

    Returns ``(subset, arm_probabilities)``. For ``router='sel_v2'``,
    ``arm_probabilities`` is an empty dict (signal-only arbitration,
    no P_arm modulation; preserves baseline behaviour byte-for-byte).
    For ``router='pam_lite'``, returns the continuous probabilities
    from :func:`arm_subset_pam_lite`.

    When ``arms_pool`` is provided, both routers thread it through so
    opt-in arms can enter the returned subset (subject to each router's
    threshold rule).
    """
    if router == "pam_lite":
        try:
            from mothrag.core.query_type_classifier import arm_subset_pam_lite
            subset, probabilities = arm_subset_pam_lite(
                question, threshold=pam_lite_threshold, arms_pool=arms_pool,
            )
            return list(subset), dict(probabilities)
        except Exception:  # noqa: BLE001
            return _resolve_arm_subset(question, arms_pool=arms_pool), {}
    return _resolve_arm_subset(question, arms_pool=arms_pool), {}


def _augment_pipeline_with_infobox(pipeline, *, top_n_boost: int = 3) -> int:
    """Harvest infobox triples + inject synthetic chunks into pipeline.

    Scans every chunk in ``pipeline.chunks_by_id`` with the wikitext +
    natural-language fact extractors and appends one synthetic chunk per
    extracted triple to the pipeline's dense index. Each synthetic chunk
    has ``chunk_id`` of the form ``"infobox:<subj>:<attr>"`` and text
    ``"<subj> -- <attr>: <value>"`` so the dense retriever ranks it
    naturally by cosine similarity to the question.

    Returns the number of synthetic infobox chunks added. ``top_n_boost``
    is accepted for forward-compat with the high-level
    :class:`MultiModalRetriever` (it controls the prepend count in the
    retriever-level blend) but is not directly consumed here -- the
    pipeline's existing ``top_k_chunks`` already governs how many chunks
    the arms see; infobox chunks compete with prose on equal footing.
    """
    import numpy as np

    from mothrag.core.retrieval import (
        InfoboxIndex,
        extract_natural_facts,
        extract_wikitext_infobox,
    )

    # 1. Build InfoboxIndex from existing chunks.
    index = InfoboxIndex()
    for cid in list(pipeline.chunk_ids):
        text = pipeline.chunks_by_id.get(cid, {}).get("text", "")
        if not text:
            continue
        index.add_many(extract_wikitext_infobox(text, source_chunk_id=cid))
        index.add_many(extract_natural_facts(text, source_chunk_id=cid))

    if len(index) == 0:
        print(
            f"[setup] dense_plus_infobox: 0 triples harvested "
            f"(corpus has no recognisable infobox / fact patterns); "
            f"falling back to dense-only behaviour."
        )
        return 0

    # 2. Synthesise chunk entries and embed in batch.
    synth_texts: list[str] = []
    synth_ids: list[str] = []
    seen_ids: set[str] = set(pipeline.chunk_ids)
    for subj_norm, attrs in index._table.items():  # noqa: SLF001
        for attr, triples in attrs.items():
            for t in triples:
                cid = f"infobox:{subj_norm[:32]}:{attr}"
                if cid in seen_ids:
                    cid = f"{cid}#{len(synth_ids)}"
                seen_ids.add(cid)
                synth_texts.append(f"{t.subject} -- {attr}: {t.value}")
                synth_ids.append(cid)

    if not synth_texts:
        return 0

    # 3. Embed the synthetic texts via the pipeline's query_embedder.
    qe = getattr(pipeline, "query_embedder", None)
    if qe is None:
        print("[setup] dense_plus_infobox: pipeline has no query_embedder; "
              "skipping augmentation.")
        return 0
    synth_vecs = np.stack(
        [np.asarray(qe(t), dtype=np.float32) for t in synth_texts],
        axis=0,
    )

    # 4. Append to the pipeline's dense surface.
    pipeline.chunk_vecs = np.concatenate(
        [pipeline.chunk_vecs, synth_vecs], axis=0,
    )
    pipeline.chunk_ids = list(pipeline.chunk_ids) + synth_ids
    for cid, text in zip(synth_ids, synth_texts):
        pipeline.chunks_by_id[cid] = {
            "text": text, "chunk_id": cid, "metadata": {"source": "infobox"},
        }

    print(
        f"[setup] dense_plus_infobox: harvested {len(index)} triples; "
        f"appended {len(synth_ids)} synthetic infobox chunks "
        f"(total chunks now {len(pipeline.chunk_ids)})."
    )
    return len(synth_ids)


# ---- router-gated infobox state-swap --------------------------------------

def _snapshot_pipeline_state(pipeline) -> dict:
    """Capture the pipeline's current chunk-surface state.

    Used as the "plain" baseline before
    :func:`_augment_pipeline_with_infobox` mutates the pipeline. The
    router-gated dispatch in the main loop swaps back to this snapshot
    when :func:`is_entity_attribute_query` returns False, restoring
    plain-dense retrieval for multi-hop / chain queries.
    """
    return {
        "chunk_vecs": pipeline.chunk_vecs.copy(),
        "chunk_ids": list(pipeline.chunk_ids),
        "chunks_by_id": dict(pipeline.chunks_by_id),
    }


def _apply_pipeline_state(pipeline, state: dict) -> None:
    """Reassign the pipeline's chunk-surface to ``state`` (in-place).

    Cheap: numpy array reassignment + dict / list rebinding. No copy
    on the hot path -- the snapshot dicts and arrays are already
    independent of subsequent mutations because
    :func:`_snapshot_pipeline_state` deep-copied them at setup.
    """
    pipeline.chunk_vecs = state["chunk_vecs"]
    pipeline.chunk_ids = list(state["chunk_ids"])
    pipeline.chunks_by_id = dict(state["chunks_by_id"])


class _InfoboxGate:
    """Per-query state-swap controller for router-gated infobox dispatch.

    At setup time, ``__init__`` snapshots the plain (pre-augmentation)
    pipeline state and triggers the augmentation; ``plain_state`` and
    ``augmented_state`` then hold the two configurations the gate can
    swap between.

    ``decide(question)`` calls
    :func:`mothrag.routing.is_entity_attribute_query` and swaps the
    pipeline's chunk surface to either the augmented or the plain
    state. Returns ``(fired: bool, reason: str)`` for telemetry.
    """

    def __init__(self, pipeline, *, top_n_boost: int) -> None:
        self._pipeline = pipeline
        self.plain_state = _snapshot_pipeline_state(pipeline)
        n_added = _augment_pipeline_with_infobox(
            pipeline, top_n_boost=top_n_boost,
        )
        self.augmented_state = _snapshot_pipeline_state(pipeline)
        self.n_infobox_chunks = n_added
        self._current = "augmented"

    def decide(self, question: str) -> tuple[bool, str]:
        from mothrag.routing import is_entity_attribute_query
        fire = is_entity_attribute_query(question)
        target = "augmented" if fire else "plain"
        if target != self._current:
            state = self.augmented_state if fire else self.plain_state
            _apply_pipeline_state(self._pipeline, state)
            self._current = target
        reason = "entity_attribute" if fire else "multi_hop_or_default"
        return fire, reason


def _maybe_run_escalation(
    *,
    pipeline: "MothRAGPipeline",
    iter_runner: "IterativeMothRAG",
    question: str,
    reader_model: str,
    top_k_subq: int,
    pred: str,
    gamma_status,
    arm_outputs: dict,
    args,
):
    """Wrap a per-query result with the Path C EscalationOrchestrator.

    Fires only when ``args.retry_strategies`` is set AND the chosen
    answer carries an abstention signal (γ_refuse / empty / uncertain).
    Returns ``(final_pred, escalation_meta)``. When the cascade declines
    or escalation is disabled, returns the original ``pred`` unchanged.
    """
    if not args.retry_strategies:
        return pred, {}

    from mothrag.core.api import _detect_abstention_signal
    from mothrag.core.retry import RetryContext, build_default_orchestrator

    signal = _detect_abstention_signal(
        pred, f"gamma_status={gamma_status}", {"gamma_status": gamma_status},
    )
    if signal is None:
        return pred, {"escalation_applied": [], "escalation_recovered_by": None}

    # Build a complete RetryContext: real reader + vector_db
    # adapters wrapping the production pipeline so #8 ActiveGapQuery + #9
    # SubQuestionRerouteCascade can actually fire (the prior placeholder
    # plumbing made these strategies decline at applicable()).
    embedder = getattr(pipeline, "embedder_model", None) \
        or _PipelineEmbedderShim(pipeline)
    reader_shim = _PipelineReaderShim(pipeline, reader_model)
    vector_db_shim = _PipelineVdbShim(pipeline)

    # Re-retrieve fresh top-K passages for the question so strategies that
    # consume ctx.passages (notably #8's gap-augmented re-read) start from
    # a realistic context, not an empty list.
    try:
        top_chunk_idx, _route, _conf = pipeline.retrieve(question)
        ctx_passages = [
            pipeline.chunks_by_id[pipeline.chunk_ids[ci]]["text"]
            for ci in top_chunk_idx[: args.top_k_chunks]
        ]
        q_emb = embedder.embed_batch([question])[0]
        try:
            q_emb_list = list(q_emb)
        except TypeError:
            q_emb_list = list(getattr(q_emb, "tolist", lambda: [])())
    except Exception:  # noqa: BLE001
        ctx_passages = []
        q_emb_list = []

    # sel_v2's a priori arm choice for this question. Strategies #8 + #9
    # respect this to pick the runner (no "different arm" forcing per
    # the architectural contract).
    ctx_arm_subset = _resolve_arm_subset(question)

    layers = tuple(s.strip() for s in args.sub_question_layers.split(",") if s.strip())
    if args.use_spectral and "spectral" not in layers:
        layers = layers + ("spectral",)

    # Per-call arm runner shims wired to the production helpers.
    def _run_arm_v3bu(*, question, passages):  # noqa: ARG001
        info = _run_v3bu(pipeline, question)
        return info["pred"]

    def _run_arm_decompose(*, question, passages):  # noqa: ARG001
        info = _run_decompose(pipeline, question, reader_model, top_k_subq)
        return info["pred"]

    def _run_arm_iter(*, question, passages, q_emb=None, top_k=None,  # noqa: ARG001
                      max_steps=None, l4b_anchor=None,  # noqa: ARG001
                      bottom_up_boost=None):  # noqa: ARG001
        # NB: bottom_up_boost is accepted for forward-compat with
        # RerouteIterWithBoostStrategy but the production iter runner
        # threads the boost via the iter_cfg already wired upstream.
        info = _run_iter(iter_runner, question)
        return info["pred"]

    ctx = RetryContext(
        question=question,
        passages=ctx_passages,
        q_emb=q_emb_list,
        top_k=args.top_k_chunks,
        arm_subset=ctx_arm_subset,
        v3bu_pred=arm_outputs.get("v3bu_pred"),
        dec_pred=arm_outputs.get("dec_pred"),
        iter_pred=arm_outputs.get("iter_pred"),
        chosen=pred,
        arbitrate_reason=f"gamma_status={gamma_status}",
        c7_info={"gamma_status": gamma_status},
        abstention_signal=signal,
        budget_limit=int(args.retry_budget_limit),
        embedder=embedder,
        reader=reader_shim,
        vector_db=vector_db_shim,
        config={
            "sub_question_layers": layers,
            "sub_question_max_depth": args.sub_question_max_depth,
            "sub_question_max_sub_questions": args.sub_question_max_sub_questions,
            "active_gap_max_rounds": args.active_gap_max_rounds,
            "active_gap_max_passages_per_round": args.active_gap_max_passages_per_round,
        },
        run_arm_iter=_run_arm_iter,
        run_arm_v3bu=_run_arm_v3bu,
        run_arm_decompose=_run_arm_decompose,
    )

    # Parse retry_strategies spec: preset alias or comma-separated list.
    preset_arg = args.retry_strategies.strip()
    if "," in preset_arg:
        preset = [s.strip() for s in preset_arg.split(",") if s.strip()]
    else:
        preset = preset_arg
    try:
        orch = build_default_orchestrator(preset, mode=args.retry_mode)
    except (ValueError, ImportError) as exc:
        return pred, {
            "escalation_applied": [],
            "escalation_recovered_by": None,
            "escalation_error": f"{type(exc).__name__}: {exc}",
        }

    outcome = orch.try_escalate(ctx)
    return outcome.answer or pred, {
        "escalation_applied": outcome.strategies_tried,
        "escalation_recovered_by": outcome.recovered_by,
        "original_abstention_signal": outcome.original_signal,
        "final_answer_confidence": outcome.final_confidence,
        "escalation_budget_used": outcome.budget_used,
        "terminal_abstain": outcome.terminal_abstain,
    }


# ---- bridge retrieval SUBSTRATE (upstream of arms) ------------------------

# Per-arm bridge qtype gate.
_ARM_GATE_ARMS = ("v3bu", "decompose", "iter", "iter_dup_a")
_ARM_GATE_VALUES = ("none", "semantic_rich_only", "exclude_bridge_entity")


def _parse_arm_bridge_gates(spec: str) -> dict:
    """Parse ``--arm-bridge-qtype-gate`` into ``{arm: gate}``.

    Form (STRICT, gate values verbatim — NO aliases like 'exclbe')::

        v3bu:none,decompose:none,iter:exclude_bridge_entity,iter_dup_a:exclude_bridge_entity

    Raises ``ValueError`` with a clear message on any malformed input or
    architecturally-impossible coherence violation. The per-arm gate is a
    BUILD-time (CLI) input keyed by ARM NAME + the query's QTYPE (input-feature,
    ``classify_query_v2``); it NEVER sees a dataset / corpus name (anti-leak).

    Coherence (enforced — not optional, these are architectural facts):
      * ``iter_dup_a`` is a DUP that copies ``iter``'s already-computed result
        (``mothrag.core.arms_runner``: a dup is never recomputed) — it has no
        independent retrieval to gate, so its gate MUST equal ``iter``'s.
      * In pool/ensemble mode ``v3bu`` and ``decompose`` are reader-only and share
        ONE pre-pool retrieval (only ``iter`` re-retrieves inside the pool), so
        their gates MUST be equal.
    """
    gates: dict = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if part.count(":") != 1:
            raise ValueError(
                f"--arm-bridge-qtype-gate: entry {part!r} must be 'arm:gate'")
        arm, gate = (x.strip() for x in part.split(":"))
        if arm not in _ARM_GATE_ARMS:
            raise ValueError(
                f"--arm-bridge-qtype-gate: unknown arm {arm!r}; expected one of "
                f"{_ARM_GATE_ARMS}")
        if gate not in _ARM_GATE_VALUES:
            raise ValueError(
                f"--arm-bridge-qtype-gate: arm {arm!r} has unknown gate {gate!r}; "
                f"expected one of {_ARM_GATE_VALUES} (verbatim, no aliases)")
        if arm in gates:
            raise ValueError(f"--arm-bridge-qtype-gate: duplicate arm {arm!r}")
        gates[arm] = gate
    missing = [a for a in _ARM_GATE_ARMS if a not in gates]
    if missing:
        raise ValueError(
            f"--arm-bridge-qtype-gate: missing arm(s) {missing}; all of "
            f"{_ARM_GATE_ARMS} are required")
    if gates["iter_dup_a"] != gates["iter"]:
        raise ValueError(
            "--arm-bridge-qtype-gate: iter_dup_a is a duplicate of iter and must "
            f"share its gate (got iter={gates['iter']!r}, "
            f"iter_dup_a={gates['iter_dup_a']!r})")
    if gates["v3bu"] != gates["decompose"]:
        raise ValueError(
            "--arm-bridge-qtype-gate: in pool mode v3bu and decompose share the "
            f"pre-pool retrieval and must share their gate (got v3bu={gates['v3bu']!r}, "
            f"decompose={gates['decompose']!r})")
    return gates


# Bridge → ChainFilter pipelined hand-off. The
# bridge rank is normalized to a gentle BOOST-ONLY multiplicative prior in
# ``[1.0, 1.0 + _BRIDGE_PRIOR_GAIN]``: the top bridge chunk gets the full boost,
# the last gets 1.0 (neutral, never penalized). ChainFilter._chain_score
# multiplies the chain-density by this prior (early fusion). Conservative gain —
# the same bridge rank already rides the late ann/PIT channel, so this only adds
# a modest early-fusion gradient (not a second full-weight vote).
_BRIDGE_PRIOR_GAIN = 0.5


def _hop_count_proxy(feat: dict) -> int:
    """Input-feature hop-count proxy (anti-leak: question features only, never a
    DS/corpus/gold signal). A query is single-hop (``1``) iff it has at most one
    relation AND no chain cue; anything with a chain cue or ≥2 relations is
    multi-hop (``2``). Used by the FIX B bridge gate to restrict skipping to
    the EASIEST single-hop semantic_rich queries only."""
    if feat.get("has_chain") or int(feat.get("n_relations", 0)) >= 2:
        return 2
    return 1


def _bridge_cohort_should_skip(gamma_proxy, qtype, *, enabled: bool,
                               hop_count=None) -> bool:
    """FIX B — pure predicate for the γ-aware bridge cohort gate. CONSERVATIVE
    revision: skip the bridge expansion iff the flag is on AND the first-pass
    dense probe is γ=``valid`` (retrieval already clean) AND the cohort is
    ``semantic_rich`` AND ``hop_count == 1`` (the EASIEST single-hop queries
    only). An earlier revision skipped on every non-``bridge_entity`` γ=valid
    query (~60% skip rate) → it pulled the bridge from multi-hop
    semantic_rich/chain_deep bulk where it still helps (MQ -3.6pp / 2W -3.1pp).
    Default / non-valid / non-semantic_rich / multi-hop ⇒ ``False`` ⇒ legacy
    bridge.prepare()."""
    return bool(enabled and gamma_proxy == "valid"
                and qtype == "semantic_rich" and hop_count == 1)


class _BridgeSubstrate:
    """Bridge retrieval SUBSTRATE upstream of the 4-arm pool (NOT a 5th arm).

    The bridge is the retrieval *substrate* that the existing pool
    (``v3bu`` / ``decompose`` / ``iter`` / ``iter_dup_a`` PDD) reads from —
    it is NOT an arm/candidate. With the substrate installed, retrievals are
    reshaped by the tripartite-judge bridge pipeline — a multi-query
    ``gemini``-ANN fusion over the SAME corpus vectors the dense path uses,
    plus Claude-Haiku SVO / dual-entity / judge stages. The arm POOL is
    untouched, so there is ZERO pool-safety violation (the destructive-
    interference failure mode of adding a real arm cannot occur). This is the
    clean A/B the experiment needs: 4-arm + bridge substrate vs 4-arm + dense
    substrate (``--use-bridge-substrate`` ON vs OFF, everything else equal).

    Scope (``--bridge-substrate-scope``):
      * ``"primary"`` (default) — only the PRIMARY,
        seed-free retrieval of each top-level question is bridged; sub-question
        (decompose) and iter-refinement (seeded) retrievals fall through to
        plain dense. Cheapest (1 bridge run / query); mirrors api.py's
        "retrieve once, share across arms" model.
      * ``"all"`` — EVERY retrieval (primary + sub-Q + iter-refinement) is
        bridged. The bridge replaces the dense retrieval mechanism wholesale
        (incl. iter's accumulated-entity seed expansion, which it supersedes
        with its own SVO / dual-entity expansion). Richer but costlier
        (~one bridge run per distinct query string; deduped via a per-run
        cache and bounded by the cross-query cost cap).

    Wiring: wraps ``pipeline.retrieve``. :meth:`prepare` runs the bridge for
    the top-level question (per-query telemetry) and warms the cache; the
    wrapped ``retrieve`` serves bridge rankings (chunk indices, truncated to
    the dense ``top_k_chunks``) per the scope rule, else delegates to the
    original dense ``retrieve``. All bridge runs are cached per query string
    (``_ranking_cache``) so identical queries never re-spend. Graceful
    degradation: any bridge failure or a breached cost cap reverts that query
    to dense — never silently overspends, never breaks the eval loop.
    Anti-leak: question text + passages only.
    """

    _MISSING = object()

    _QTYPE_GATES = ("none", "semantic_rich_only", "exclude_bridge_entity")

    def __init__(self, pipeline, *, judge_model: str, max_cost_usd: float,
                 judge_provider: str = "anthropic",
                 require_backend: bool = True, scope: str = "primary",
                 qtype_gate: str = "none", chain_filter=None,
                 per_arm_gates: dict | None = None) -> None:
        from mothrag.retrieval.bridge_haiku import BridgeArm, BridgeConfig
        from mothrag.retrieval.bridge_haiku.types import Candidate

        if scope not in ("primary", "all"):
            raise ValueError(f"bridge substrate scope must be 'primary' or "
                             f"'all', got {scope!r}")
        if qtype_gate not in self._QTYPE_GATES:
            raise ValueError(f"bridge substrate qtype_gate must be one of "
                             f"{self._QTYPE_GATES}, got {qtype_gate!r}")
        self._scope = scope
        self._qtype_gate = qtype_gate
        # Per-arm gate: when set, the gate is decided PER ARM (the active
        # arm sets itself via set_current_arm before its retrievals). current_arm
        # is THREAD-LOCAL so the within-query arm-parallel pool can't race
        # on it (each arm thread reads its own value). None → uniform qtype_gate.
        self._per_arm_gates = dict(per_arm_gates) if per_arm_gates else None
        import threading
        self._current_arm = threading.local()
        # Per top-level question: per-arm bridge fire/skip decisions (telemetry).
        self._bridge_arm_decisions: dict = {}
        # Per top-level question: whether the qtype gate suppressed the bridge.
        self._gate_skip = False
        self._gate_qtype: str | None = None
        self._pipeline = pipeline
        self._Candidate = Candidate
        # Capture the bound dense retrieve BEFORE we shadow the attribute.
        self._orig_retrieve = pipeline.retrieve
        self._pid_to_idx = {pid: i for i, pid in enumerate(pipeline.chunk_ids)}
        self._chain_filter = chain_filter   # ChainFilter reranker (or None)
        self._current_q: str | None = None
        # Per-query-string cache: query -> (idx | None, bridge_passage_id).
        # Shared across the whole run so identical sub-Qs never re-spend.
        self._ranking_cache: dict[str, tuple] = {}

        self.max_cost_usd = float(max_cost_usd)
        self.total_cost_usd = 0.0
        self.n_bridge_runs = 0
        self.n_fallback = 0
        self.n_cost_capped = 0
        # Cross-query bridge stage failure totals.
        self._svo_failures = 0
        self._entity_failures = 0
        self._judge_failures = 0
        self._haiku_5xx = 0

        # Return as many ranked passages as the dense path would, so the A/B is
        # fair on passage COUNT and not just order.
        top_k_chunks = int(getattr(pipeline.config, "top_k_chunks", 10) or 10)
        final_top_k = max(top_k_chunks, 5)
        self._cfg = BridgeConfig(
            judge_model=judge_model,
            judge_provider=judge_provider,
            max_cost_usd=float(max_cost_usd),
            final_top_k=final_top_k,
            pool_cap=max(20, final_top_k),
        )
        self._arm = BridgeArm(self._ann_retrieve, config=self._cfg,
                              require_backend=require_backend)
        # Install the substrate. From here every ``pipeline.retrieve`` call
        # routes through :meth:`_wrapped_retrieve`.
        pipeline.retrieve = self._wrapped_retrieve

    # -- dense ANN over the pipeline's OWN corpus vectors (same embedding
    #    space as the dense substrate; for the live fire that is
    #    gemini-embedding-2 via --embedding). Reused for every bridge sub-query.
    def _ann_retrieve(self, query: str, k: int):
        import numpy as np
        qv = np.asarray(self._pipeline.query_embedder(query), dtype=np.float32)
        scores = self._pipeline.chunk_vecs @ qv
        n = min(int(k), len(scores))
        if n <= 0:
            return []
        top = np.argsort(-scores)[:n]
        out = []
        for ci in top:
            ci = int(ci)
            cid = self._pipeline.chunk_ids[ci]
            text = self._pipeline.chunks_by_id[cid]["text"]
            out.append(self._Candidate(cid, text, float(scores[ci])))
        return out

    def _bridge_ranking(self, query: str):
        """Bridge ranking for ``query`` as ``(idx | None, info)``.

        Cached per query string; cost-capped; degrades to dense (``None``) on
        empty query, breached cap, bridge error, or empty ranking. Shared by
        :meth:`prepare` (primary, telemetry) and :meth:`_wrapped_retrieve`
        (any scoped retrieval).
        """
        cached = self._ranking_cache.get(query, self._MISSING)
        if cached is not self._MISSING:
            idx, pid = cached
            return idx, {"fired": bool(idx),
                         "reason": "bridge_cached" if idx else "dense_cached",
                         "bridge_passage_id": pid, "cost_usd": 0.0,
                         "n_ranked": len(idx) if idx else 0}
        if not query:
            return None, {"fired": False, "reason": "empty_query",
                          "bridge_passage_id": None, "cost_usd": 0.0,
                          "n_ranked": 0}
        if self.total_cost_usd >= self.max_cost_usd:
            self.n_cost_capped += 1
            self._ranking_cache[query] = (None, None)
            return None, {"fired": False, "reason": "cost_capped",
                          "bridge_passage_id": None, "cost_usd": 0.0,
                          "n_ranked": 0}
        from mothrag.retrieval.bridge_haiku import BridgeArmDegraded
        try:
            res = self._arm.retrieve(query)
        except BridgeArmDegraded:
            # Systemic degradation: fail fast + loud, do NOT silently fall
            # through to dense for the rest of the run.
            raise
        except Exception as exc:  # noqa: BLE001 — isolated failure: degrade
            self.n_fallback += 1
            self._ranking_cache[query] = (None, None)
            return None, {"fired": False,
                          "reason": f"bridge_error:{type(exc).__name__}",
                          "bridge_passage_id": None, "cost_usd": 0.0,
                          "n_ranked": 0}
        cost = float(res.stats.estimated_cost_usd)
        self.total_cost_usd += cost
        self.n_bridge_runs += 1
        self._svo_failures += int(res.stats.svo_failures)
        self._entity_failures += int(res.stats.entity_failures)
        self._judge_failures += int(res.stats.judge_failures)
        self._haiku_5xx += int(res.stats.haiku_5xx_count)
        idx = [self._pid_to_idx[p] for p in res.ranked_passage_ids
               if p in self._pid_to_idx] or None
        # ChainFilter reshapes the bridge ranking in-place.
        # The hop-gate runs on the TOP-LEVEL question (``self._current_q``); a
        # single-hop question / no-support pool degrades to the bridge order
        # (passthrough), so ON never regresses single-hop. POST-retrieval
        # reranker, NOT a 5th arm.
        if self._chain_filter is not None and idx:
            idx = self._apply_chain_filter(res.ranked_passage_ids, idx)
        if idx is None:
            self.n_fallback += 1
        pid = res.bridge_passage_id if idx else None
        self._ranking_cache[query] = (idx, pid)
        return idx, {"fired": bool(idx),
                     "reason": "bridge_substrate" if idx else "empty_ranking",
                     "bridge_passage_id": pid, "cost_usd": round(cost, 5),
                     "n_ranked": len(idx) if idx else 0}

    def _apply_chain_filter(self, ranked_pids, idx):
        """Run ChainFilter over the bridge ranking.

        Reconstructs lightweight bridge candidates (``passage_id`` / ``text`` /
        ``ann_score`` = the descending bridge-rank score), calls
        ``chain_filter.filter(top_level_question, candidates)``, and remaps the
        filtered passages back to corpus indices. On an empty / unchanged result
        the original ``idx`` is preserved (passthrough — never drops the set)."""
        from types import SimpleNamespace
        pids = [p for p in ranked_pids if p in self._pid_to_idx]
        if not pids:
            return idx
        n = len(pids)
        cands = [SimpleNamespace(
            passage_id=p,
            text=(self._pipeline.chunks_by_id.get(p, {}) or {}).get("text", ""),
            ann_score=float(n - r),    # preserve bridge order in the ann channel
            # FIX #1: the same bridge rank also rides forward as a normalized
            # boost-only prior on the chain-density (top→full boost, last→1.0).
            bridge_score=(1.0 + _BRIDGE_PRIOR_GAIN * ((n - 1 - r) / (n - 1))
                          if n > 1 else 1.0),
        ) for r, p in enumerate(pids)]
        try:
            filtered = self._chain_filter.filter(self._current_q or "", cands)
        except Exception:  # noqa: BLE001 — a filter fault must not break retrieval
            return idx
        new_idx = [self._pid_to_idx[c.passage_id] for c in filtered
                   if c.passage_id in self._pid_to_idx]
        return new_idx or idx

    def _gate_allows(self, question: str):
        """Qtype gate decision for the TOP-LEVEL ``question``.

        Returns ``(allow: bool, qtype: str | None)``. Input-feature gating only
        (``classify_query_v2`` on the question text — anti-leak safe, already on
        the hot path); NEVER a DS/corpus-name signal. Cross-DS evidence: the
        bridge HELPS ``semantic_rich`` (+2 to +4pp) and HURTS ``bridge_entity``
        (-3.7 to -14.8pp), so the gate confines the
        bridge to the cohorts it helps. Classifier failure never blocks (fail
        open to the bridge).
        """
        if self._qtype_gate == "none":
            return True, None
        try:
            qtype = classify_query_v2(question)
        except Exception:  # noqa: BLE001 — never break on classifier failure
            return True, None
        if self._qtype_gate == "semantic_rich_only":
            return (qtype == "semantic_rich"), qtype
        if self._qtype_gate == "exclude_bridge_entity":
            return (qtype != "bridge_entity"), qtype
        return True, qtype

    @staticmethod
    def _gate_allows_for(gate: str, qtype) -> bool:
        """Does ``gate`` allow the bridge for a query of (precomputed) ``qtype``?

        Same input-feature semantics as :meth:`_gate_allows`, but parameterised by
        the gate (for the per-arm path, where each arm carries its own gate and the
        qtype is classified ONCE per top-level question). Fail-open on a None qtype
        (classifier failure never suppresses the bridge)."""
        if gate == "none":
            return True
        if qtype is None:
            return True
        if gate == "semantic_rich_only":
            return qtype == "semantic_rich"
        if gate == "exclude_bridge_entity":
            return qtype != "bridge_entity"
        return True

    def set_current_arm(self, arm: str | None) -> None:
        """Mark the arm whose retrievals follow (per-arm gate). THREAD-LOCAL — set
        it in the same thread that performs the arm's retrieval (the within-query
        arm-parallel pool gives each arm its own thread)."""
        self._current_arm.value = arm

    def _get_current_arm(self):
        return getattr(self._current_arm, "value", None)

    def prepare(self, question: str) -> dict:
        """Prime + warm the bridge for the top-level ``question``. Per-query.

        Applies the qtype gate FIRST (per top-level question): when gated out,
        the bridge is suppressed for ALL of this question's retrievals (primary
        + sub-Q + iter), which fall through to dense. Otherwise reflects the
        PRIMARY retrieval's bridge decision (the per-query telemetry row).
        """
        self._current_q = question
        self._bridge_arm_decisions = {}
        if self._per_arm_gates is not None:
            return self._prepare_per_arm(question)
        allow, qtype = self._gate_allows(question)
        self._gate_skip = not allow
        self._gate_qtype = qtype
        if not allow:
            return {"fired": False, "reason": f"qtype_gated:{qtype}",
                    "bridge_passage_id": None, "cost_usd": 0.0, "n_ranked": 0,
                    "bridge_qtype_skipped": True, "bridge_qtype": qtype}
        _idx, info = self._bridge_ranking(question)
        info["bridge_qtype_skipped"] = False
        info["bridge_qtype"] = qtype
        return info

    def _prepare_per_arm(self, question: str) -> dict:
        """Per-arm gate: classify the qtype ONCE, then warm
        the bridge ranking iff AT LEAST ONE arm's gate allows it (so the cache is
        ready for the arms that DO bridge). The serve/dense decision per retrieval
        is made later from current_arm. Records the per-arm decision map."""
        try:
            qtype = classify_query_v2(question)
        except Exception:  # noqa: BLE001 — never break on classifier failure
            qtype = None
        self._gate_qtype = qtype
        self._bridge_arm_decisions = {
            arm: {"gate": g, "allow": self._gate_allows_for(g, qtype),
                  "qtype": qtype}
            for arm, g in self._per_arm_gates.items()
        }
        any_allow = any(d["allow"] for d in self._bridge_arm_decisions.values())
        self._gate_skip = not any_allow
        if not any_allow:
            return {"fired": False, "reason": f"all_arms_qtype_gated:{qtype}",
                    "bridge_passage_id": None, "cost_usd": 0.0, "n_ranked": 0,
                    "bridge_qtype_skipped": True, "bridge_qtype": qtype,
                    "bridge_arm_decisions": dict(self._bridge_arm_decisions),
                    "per_arm_gates": dict(self._per_arm_gates)}
        _idx, info = self._bridge_ranking(question)
        info["bridge_qtype_skipped"] = False
        info["bridge_qtype"] = qtype
        info["bridge_arm_decisions"] = dict(self._bridge_arm_decisions)
        info["per_arm_gates"] = dict(self._per_arm_gates)
        return info

    def _wrapped_retrieve(self, question, entity_seeds=None):
        # The qtype gate (decided per top-level question in prepare) suppresses
        # the bridge for EVERY retrieval of a gated-out question — dense
        # fallthrough regardless of scope. (Per-arm: gate_skip is set only when
        # ALL arms are gated out for this qtype.)
        if self._gate_skip:
            return self._orig_retrieve(question, entity_seeds)
        # With a per-arm map, the gate for THIS retrieval is
        # the active arm's gate (current_arm, thread-local). Unset / unknown arm →
        # the v3bu gate (the shared primary group; v3bu==decompose is enforced).
        if self._per_arm_gates is not None:
            arm = self._get_current_arm()
            gate = (self._per_arm_gates[arm] if arm in self._per_arm_gates
                    else self._per_arm_gates["v3bu"])
            if not self._gate_allows_for(gate, self._gate_qtype):
                return self._orig_retrieve(question, entity_seeds)
        # scope='primary': only the PRIMARY, seed-free retrieval of the current
        # top-level question is bridged. scope='all': EVERY retrieval (incl.
        # sub-Q / seeded iter-refinement) is bridged. Non-bridged paths and any
        # graceful-degrade fall through to the original dense retrieve.
        if self._scope == "all":
            serve = bool(question)
        else:
            serve = (not entity_seeds and question == self._current_q)
        if serve:
            idx, _info = self._bridge_ranking(question)
            if idx:
                top_k = int(getattr(self._pipeline.config, "top_k_chunks", 10)
                            or 10)
                return list(idx[:top_k]), "bridge_substrate", 1.0
        return self._orig_retrieve(question, entity_seeds)

    def stats(self) -> dict:
        return {
            "scope": self._scope,
            "qtype_gate": self._qtype_gate,
            "per_arm_gates": (dict(self._per_arm_gates)
                              if self._per_arm_gates else None),
            "n_bridge_runs": self.n_bridge_runs,
            "n_distinct_queries": len(self._ranking_cache),
            "n_fallback": self.n_fallback,
            "n_cost_capped": self.n_cost_capped,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "max_cost_usd": self.max_cost_usd,
            # Stage failure telemetry.
            "svo_failures": self._svo_failures,
            "entity_failures": self._entity_failures,
            "judge_failures": self._judge_failures,
            "haiku_5xx_count": self._haiku_5xx,
            # ChainFilter behavioural telemetry (None when OFF).
            "chain_filter": (dict(self._chain_filter.counters)
                             if self._chain_filter is not None else None),
        }


class _StubDensePipeline:
    """Minimal pipeline surface for the offline ``--dry-run`` substrate check.

    Exposes exactly what :class:`_BridgeSubstrate` consumes: ``config`` (with
    ``top_k_chunks``), ``query_embedder``, ``chunk_vecs``, ``chunk_ids``,
    ``chunks_by_id`` and a dense ``retrieve``. No corpus, no API.
    """

    class _Cfg:
        top_k_chunks = 5

    _KW_VOCAB = ["alpha", "beta", "gamma", "delta", "gold", "filler"]

    def __init__(self) -> None:
        import numpy as np
        self.config = self._Cfg()
        texts = ["alpha beta", "gamma gold", "delta filler",
                 "filler filler", "beta gamma", "gold delta", "alpha gold"]
        self.chunk_ids = [f"p{i}" for i in range(len(texts))]
        self.chunks_by_id = {
            cid: {"text": t} for cid, t in zip(self.chunk_ids, texts)
        }
        self.chunk_vecs = np.stack([self._embed(t) for t in texts], axis=0)

    def _embed(self, text: str):
        import numpy as np
        idx = {w: i for i, w in enumerate(self._KW_VOCAB)}
        v = np.zeros(len(self._KW_VOCAB), dtype=np.float32)
        for w in text.lower().split():
            if w in idx:
                v[idx[w]] += 1.0
        nrm = float(np.linalg.norm(v)) or 1.0
        return v / nrm

    def query_embedder(self, text: str):
        return self._embed(text)

    def retrieve(self, question, entity_seeds=None):
        import numpy as np
        qv = self._embed(question)
        scores = self.chunk_vecs @ qv
        top = np.argsort(-scores)[: self.config.top_k_chunks]
        return [int(i) for i in top], "dense", float(scores[top[0]]) if len(top) else 0.0


def _dry_run_bridge_substrate(judge_model: str = "claude-haiku-4-5",
                              max_cost_usd: float = 10.0,
                              scope: str = "primary") -> dict:
    """Offline mechanics + paired-comparison check of the bridge substrate.

    Builds a synthetic dense pipeline, installs :class:`_BridgeSubstrate`
    (``require_backend=False`` → Haiku stages degrade with no key), and for 5
    synthetic questions compares the substrate's PRIMARY retrieval order
    against the plain-dense order (the 4-arm + bridge vs 4-arm + dense A/B, at
    the substrate layer). Also verifies the SCOPE rule on the iter-refinement
    (seeded) and sub-question paths: ``primary`` keeps them dense, ``all``
    bridges them too. No corpus, no API, no cost.
    """
    stub = _StubDensePipeline()
    dense_order_of = {}
    questions = ["alpha beta", "gamma gold", "delta", "beta", "gold delta"]
    for q in questions:
        dense_idx, route, _ = stub.retrieve(q)       # plain dense (pre-install)
        dense_order_of[q] = list(dense_idx)

    sub = _BridgeSubstrate(stub, judge_model=judge_model,
                           max_cost_usd=max_cost_usd, require_backend=False,
                           scope=scope)
    ok = True
    n_fired = 0
    n_diverged = 0
    n_seed_bridged = 0
    paired = []
    for q in questions:
        prep = sub.prepare(q)
        bridge_idx, broute, _ = stub.retrieve(q)           # primary, no seeds
        _seed_idx, sroute, _ = stub.retrieve(q, ["alpha"])  # seeded iter-path
        dense_idx = dense_order_of[q]
        if prep["fired"]:
            n_fired += 1
            if broute != "bridge_substrate":
                ok = False
        if sroute == "bridge_substrate":
            n_seed_bridged += 1
        # Scope semantics on the seeded (iter-refinement) path:
        if scope == "all":
            # bridged whenever the primary fired (degraded ranking still ≠ empty)
            if prep["fired"] and sroute != "bridge_substrate":
                ok = False
        else:  # primary: seeded path MUST bypass the substrate
            if sroute != "dense":
                ok = False
        if not isinstance(bridge_idx, list):
            ok = False
        diverged = (list(bridge_idx) != list(dense_idx))
        if diverged:
            n_diverged += 1
        paired.append({
            "question": q, "fired": prep["fired"], "reason": prep["reason"],
            "dense_top": dense_idx, "bridge_top": list(bridge_idx),
            "seed_route": sroute, "diverged_from_dense": diverged,
        })
    # A DISTINCT sub-question (not the prepared primary, no seeds): bridged
    # only under scope='all'.
    sub.prepare("alpha beta")
    _i, subq_route, _ = stub.retrieve("gamma gold")
    subq_bridged = (subq_route == "bridge_substrate")
    if subq_bridged != (scope == "all"):
        ok = False
    return {
        "mode": "dry_run", "scope": scope, "synthetic_queries": len(questions),
        "n_bridge_fired": n_fired, "n_diverged_from_dense": n_diverged,
        "n_seed_path_bridged": n_seed_bridged, "subq_bridged": subq_bridged,
        "substrate_stats": sub.stats(), "paired": paired, "ok": ok,
    }


def _dry_run_iter_ragnatela(kmax: int = 3) -> dict:
    """Offline γ-convergence + pool-safety check for the iter-ragnatela
    loop layered on the bridge substrate (no index/LLM/cost).

    Four synthetic arms (v3bu / decompose / iter / iter_dup_a PDD) whose γ
    rises as the bridge-RESHAPED context arrives each round; one arm is a
    γ-LOW anti-context distractor the pool mutes. Asserts the loop (a) converges
    within ``kmax`` iterations, (b) keeps the pool at exactly 4 arms every
    iteration, and (c) never surfaces a ``bridge`` arm — the bridge is the
    retrieval substrate upstream of the pool, not a 5th arm. This mirrors
    ``tests/iterative_ragnatela/test_pool_safety_invariant.py`` as a runner-
    level smoke so ``--dry-run`` proves the layered runner is fire-ready.
    """
    from mothrag.iterative_ragnatela import (
        ArmAnswer, RagnatelaConfig, RagnatelaOrchestrator,
    )
    from mothrag.iterative_ragnatela.gamma_pooling import normalize_answer

    canonical = tuple(sorted(("v3bu", "decompose", "iter", "iter_dup_a")))
    seen_arm_sets: list[tuple] = []
    pool_sizes: list[int] = []

    def arm_runner(question: str, context: list):
        n = len(context)
        g = min(0.40 + 0.14 * n, 0.95)   # γ rises with accumulated evidence
        answers = [
            ArmAnswer("v3bu", "Paris", g + 0.02),
            ArmAnswer("decompose", "Paris", g),
            ArmAnswer("iter_dup_a", "Paris", g - 0.02),   # PDD 4th arm
            ArmAnswer("iter", "Lyon", 0.20),              # anti-context distractor
        ]
        seen_arm_sets.append(tuple(sorted(a.arm for a in answers)))
        pool_sizes.append(len(answers))
        return answers

    def bridge_retriever(sub_questions: list, context: list):
        """Bridge SUBSTRATE seam: reshapes the next round's context items."""
        rid = len(context)
        return [f"bridge::ev-r{rid}-{i}::{normalize_answer(s)[:24]}"
                for i, s in enumerate(sub_questions)]

    cfg = RagnatelaConfig(max_iterations=kmax)
    result = RagnatelaOrchestrator(
        arm_runner, retriever=bridge_retriever, config=cfg,
    ).run("In which city is the Eiffel Tower located?")

    pool_size_ok = (bool(result.traces)
                    and all(t.n_high + t.n_mid + t.n_low == 4
                            for t in result.traces)
                    and all(s == 4 for s in pool_sizes))
    arm_set_ok = bool(seen_arm_sets) and all(s == canonical for s in seen_arm_sets)
    no_bridge_arm = all("bridge" not in s for s in seen_arm_sets)
    ok = bool(result.iterations_used <= kmax and result.converged
              and result.answer == "Paris" and pool_size_ok and arm_set_ok
              and no_bridge_arm)
    return {
        "mode": "dry_run_iter_ragnatela", "kmax": kmax,
        "iterations_used": result.iterations_used,
        "converged": result.converged, "stop_reason": result.stop_reason,
        "final_gamma": result.final_gamma, "gamma_trace": result.gamma_trace,
        "answer": result.answer,
        "pool_size_4_every_iter": pool_size_ok,
        "arm_set_canonical_every_iter": arm_set_ok,
        "no_bridge_arm_key": no_bridge_arm, "ok": ok,
    }


def _build_chain_filter(args):
    """Construct the ChainFilter v0.1 from CLI flags.

    Returns a ``ChainFilter`` (enabled, WITH a live OpenIE triple extractor) or
    ``None`` when ``--use-chainfilter`` is absent (default → never built → zero
    regression). The ON path wires a real OpenIE-based ``triple_extractor``
    (mirror of the eval OpenIE path) so the filter actually runs end-to-end
    instead of hitting the no-extractor passthrough — i.e. ON is behaviourally
    distinct from OFF (the smoke test can measure a real Δ). The γ scorer + fact
    filter use the deterministic v0.1 defaults; a real γ verifier is a later
    swap-in. ChainFilter is a POST-retrieval reranker — NOT a 5th arm."""
    if not getattr(args, "use_chainfilter", False):
        return None
    from mothrag.retrieval.chain_filter import ChainFilter, ChainFilterConfig
    from mothrag.retrieval.openie import OpenIEClient

    cfg = ChainFilterConfig(
        enabled=True,
        hop_gate_min=int(getattr(args, "chainfilter_hop_min", 2)),
        top_k_out=int(getattr(args, "chainfilter_top_out", 5)),
        # FIX A — chain_deep cohort opt-out (default OFF).
        cohort_gate_skip_chain_deep=bool(
            getattr(args, "use_chainfilter_cohort_gate", False)),
    )
    gmin = getattr(args, "chainfilter_gamma_min", None)
    if gmin is not None:
        cfg.gamma_cfg.gamma_low = float(gmin)

    # Live OpenIE triple extractor (Groq Llama-3.3-70B-versatile). Uses Groq
    # (same Llama-3.3-70B already in the stack for the main reader; no separate
    # TOGETHER_API_KEY). The Groq
    # host is on the OpenIEClient allowlist, so no allow_custom_endpoint needed.
    # The client is built LAZILY on first use so --use-chainfilter is
    # constructible offline (no key needed at build); at fire time the key is set
    # and one client is reused, one .process() call per candidate passage.
    _holder: dict = {}

    def _triple_extractor(text: str):
        if not text:
            return []
        oie = _holder.get("client")
        if oie is None:
            try:
                oie = OpenIEClient(
                    api_key=os.environ.get("GROQ_API_KEY"),
                    base_url="https://api.groq.com/openai/v1",
                    model="llama-3.3-70b-versatile",
                )
            except Exception:  # noqa: BLE001 — no key ⇒ extractor degrades to []
                return []
            _holder["client"] = oie
        try:
            return [list(t) for t in oie.process(text, chunk_id="chainfilter").triples]
        except Exception:  # noqa: BLE001 — one passage's OpenIE failing ≠ abort
            return []

    return ChainFilter(config=cfg, triple_extractor=_triple_extractor)


# ---- Main -----------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    # ---- Same flags as eval_wiki_iterative.py ----
    # NB: --data-dir / --queries / --out are required for a LIVE run but NOT
    # for --dry-run (offline substrate mechanics check); validated below.
    ap.add_argument("--data-dir", required=False, type=Path)
    ap.add_argument("--queries", required=False, type=Path)
    ap.add_argument("--reader-model", default="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    ap.add_argument("--reader-base-url", default="https://api.together.xyz/v1")
    ap.add_argument("--reader-api-key-env", default="TOGETHER_API_KEY")
    ap.add_argument("--embedding", default="bge-base")
    ap.add_argument("--reranker", default="bge-rerank")
    ap.add_argument("--bottom-up-boost", type=float, default=1.0)
    ap.add_argument("--top-k-chunks", type=int, default=10)
    ap.add_argument("--top-k-subq", type=int, default=10,
                    help="top-K passages per sub-Q in decompose arm")
    ap.add_argument("--max-iterations", type=int, default=4)
    ap.add_argument("--top-k-total", type=int, default=15)
    ap.add_argument("--stop-early", action="store_true", default=True)
    ap.add_argument("--no-stop-early", dest="stop_early", action="store_false")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", required=False, type=Path)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    # Lever 5 — final reader override (iter arm only)
    ap.add_argument("--final-reader-model", default=None)
    ap.add_argument("--final-reader-provider", default=None,
                    choices=[None, "openai", "anthropic", "groq", "together", "nvidia"])
    ap.add_argument("--final-reader-base-url", default=None)
    ap.add_argument("--final-reader-api-key-env", default=None)
    ap.add_argument("--final-reader-max-tokens", type=int, default=384)
    # Lever 2 — graph-aware iter retrieval
    ap.add_argument("--use-graph-aware-iter", action="store_true", default=False)
    ap.add_argument("--max-accumulated-entities", type=int, default=32)
    # Lever 4 — faithfulness loop (iter arm only)
    ap.add_argument("--use-faithfulness-loop", action="store_true", default=False)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-provider", default="openai",
                    choices=["openai", "anthropic", "groq", "together", "nvidia", "gemini"])
    ap.add_argument("--judge-api-key-env", default=None)
    ap.add_argument("--judge-min-score", type=float, default=1.0)
    ap.add_argument("--max-judge-iterations", type=int, default=None)
    # γ verifier (iter arm only)
    ap.add_argument("--use-gamma-verifier", action="store_true", default=False)
    ap.add_argument("--gamma-diagnostic-dump", dest="gamma_diagnostic_dump",
                    action="store_true", default=False,
                    help="Dump the final-iteration "
                         "verified proof tree (per-step rule / verifier_status / "
                         "verifier_reason + is_complete + raw) into each "
                         "per_question record. Diagnostic only; default OFF.")
    ap.add_argument("--use-relaxed-object-span", dest="use_relaxed_object_span",
                    action="store_true", default=False,
                    help="Relax the γ-verifier object grounding "
                         "from 'object token in the cited span' to 'object tokens "
                         "covered by the passage'. Fixes the dominant secondary "
                         "γ-invalid mode (descriptive/boolean objects whose words "
                         "aren't in the quoted fragment). Default OFF = legacy.")
    ap.add_argument("--path-serialize", dest="path_serialize",
                    action="store_true", default=False,
                    help="Order the reader's evidence "
                         "along the multi-hop reasoning path (spine = "
                         "accumulated_entities, hop-1/bridge first) instead of "
                         "raw retrieval order, at all reader sites (intermediate "
                         "/ proof-tree / final). Reuses signals already "
                         "extracted (no new LLM calls); graceful fallback to bag "
                         "order when no spine (single-hop). Default OFF = "
                         "byte-identical V4.")
    ap.add_argument("--use-gamma-liberal", action="store_true", default=False)
    ap.add_argument("--gamma-max-retrigger", type=int, default=2)
    ap.add_argument("--gamma-prompt-variant", default="full",
                    choices=["full", "llama"])
    ap.add_argument("--use-gamma-router", action="store_true", default=False)
    # Lever 1C — LLM-NER cache
    ap.add_argument("--ner-cache", default=None)
    ap.add_argument("--ner-build-model", default="gemini-2.5-flash")
    ap.add_argument("--ner-build-provider", default="gemini",
                    choices=["gemini", "openai", "groq", "together"])
    # Reader self-consistency
    ap.add_argument("--reader-n-samples", type=int, default=1)
    ap.add_argument("--reader-temperature", default="0.0")
    # Within-iter C7
    ap.add_argument("--use-c7-iter", action="store_true", default=False)
    ap.add_argument("--c7-iter-trigger", default="gated", choices=["gated", "blanket"])
    ap.add_argument("--c7-iter-embedder", default="gemini-embedding-2")

    # ---- Phase 1 patches (CLI parity with eval_wiki_iterative.py) so
    # route_prospective can reproduce the step3 PROD baselines. Names +
    # IterativeConfig wiring are VERBATIM from eval_wiki_iterative.py.
    #
    # DEFAULTS are matched to NO-REGRESSION for route_prospective, which (unlike
    # eval_wiki's argparse) currently inherits IterativeConfig's dataclass
    # defaults: P1 use_gamma_refuse_loop defaults TRUE (iterative_pipeline.py)
    # → route_prospective already runs P1 ON, and PROD pins P1 ON/LOCKED. So
    # --use-gamma-refuse-loop defaults ON here (with an explicit
    # --disable-gamma-refuse-loop for ablation); all OTHER patch fields
    # dataclass-default False, so they default OFF (verbatim eval_wiki).
    # NB: P4 (abstain filter) bundles with P1 — no standalone toggle; P8
    # (few-shot) fires inside --use-stepchain-parity-composite — no standalone
    # toggle. --use-p4-abstain-filter / --use-p8-few-shot do not exist in
    # eval_wiki_iterative.py and are intentionally NOT fabricated.
    ap.add_argument("--use-gamma-refuse-loop", dest="use_gamma_refuse_loop",
                    action="store_true", default=True,
                    help="P1 — γ refuse triggers broader-query retry "
                         "instead of immediate terminal abstain. Default ON "
                         "(PROD-locked; also the route_prospective legacy "
                         "behaviour). Use "
                         "--disable-gamma-refuse-loop to ablate OFF.")
    ap.add_argument("--disable-gamma-refuse-loop", dest="use_gamma_refuse_loop",
                    action="store_false",
                    help="P1 ablation: force the γ refuse-loop OFF.")
    ap.add_argument("--use-stepchain-parity-composite", action="store_true",
                    default=False,
                    help="Composite of P6-P10 StepChain parity (cap "
                         "raise, entity-seeded query, P8 few-shot system, "
                         "rerank-by-cosine, scorer).")
    ap.add_argument("--use-bug-pattern-wave-a", action="store_true", default=False,
                    help="Bug-pattern composite (P11 γ-cap "
                         "reader fallback, P12 decompose truncate, P13 "
                         "abstain-prior filter, P14 faith-loop second-chance, "
                         "P24 unified abstain markers).")
    ap.add_argument("--use-p11-gamma-cap-fallback", action="store_true",
                    default=False,
                    help="P11 STANDALONE ablation (γ-cap free-text "
                         "reader fallback) WITHOUT the rest of the bug-pattern "
                         "composite. Superseded by --use-bug-pattern-wave-a if both set.")
    ap.add_argument("--disable-p11-gamma-cap-fallback", action="store_true",
                    default=False,
                    help="Force P11 OFF even when --use-bug-pattern-"
                         "wave-a is set ('FULL extras MINUS P11' ablation).")
    ap.add_argument("--composite-max-iterations", type=int, default=None,
                    help="Override IterativeConfig.composite_max_"
                         "iterations (default 5 when --use-stepchain-parity-"
                         "composite set). Pass == --max-iterations to "
                         "neutralize the P6 cap raise.")
    ap.add_argument("--use-p12-decompose-collapse-cap", action="store_true",
                    default=False,
                    help="P12 STANDALONE (>6 sub_qs => truncate to 6 "
                         "instead of collapse to [question]) WITHOUT the "
                         "bug-pattern composite.")
    ap.add_argument("--use-p13-sub-q-abstain-filter", action="store_true",
                    default=False,
                    help="P13 STANDALONE (sub-Q abstain prior filter) "
                         "WITHOUT the bug-pattern composite.")
    ap.add_argument("--use-p15-gamma-gated-naturalize", action="store_true",
                    default=False,
                    help="P15 STANDALONE (gate legacy naturalized-only "
                         "branch on gamma_status=='valid') WITHOUT the "
                         "bug-pattern composite.")
    ap.add_argument("--use-p24-unified-abstain-markers", action="store_true",
                    default=False,
                    help="P24 STANDALONE (unified ABSTAIN_MARKERS "
                         "module) WITHOUT the bug-pattern composite.")

    # ---- New flag ----
    ap.add_argument("--mode", default="v2",
                    choices=["v2", "strict", "ensemble_arbitrate"],
                    help="v2 (default): chain_deep->iter / bridge_entity->decompose / "
                         "semantic_rich->sel_v1 (V3+bu+decompose arbitrated). "
                         "strict: chain_deep->iter / bridge_entity->decompose / "
                         "semantic_rich->V3+bu only (no arbitration). "
                         "ensemble_arbitrate: ALWAYS run all 3 arms "
                         "(V3+bu+decompose+iter) and apply DeterministicArbitrator "
                         "on outputs over the same passages. Higher per-query "
                         "compute; skips sel_v2 routing classifier.")

    # ---- retrieval modality flag ----
    ap.add_argument("--retrieval", default="dense",
                    choices=["dense", "dense_plus_infobox",
                             "router_gated_infobox"],
                    help="Retrieval modality. dense (default): unchanged "
                         "production cosine retrieval. dense_plus_infobox: "
                         "blend dense top-K with structured InfoboxIndex "
                         "triples harvested from the corpus at startup; "
                         "infobox chunks compete with prose on every query. "
                         "router_gated_infobox: same augmentation, but per-"
                         "query the deterministic regex router "
                         "(mothrag.routing.is_entity_attribute_query) gates "
                         "which queries see the infobox chunks -- single-"
                         "clause entity-attribute queries fire infobox; "
                         "multi-hop / chain / comparison queries fall back to "
                         "plain dense. Empirical: unconditional "
                         "dense_plus_infobox helps 2W +5.7pp but hurts HP "
                         "-0.86pp / MQ -12.5pp; the router preserves the 2W "
                         "lift while neutralising the HP/MQ regression.")
    ap.add_argument("--infobox-top-n-boost", type=int, default=3,
                    help="Max number of infobox chunks prepended "
                         "to each arm's retrieval. Default 3.")

    # ---- PAM-lite: continuous-probability router opt-in ----
    ap.add_argument("--router", default="sel_v2",
                    choices=["sel_v2", "pam_lite"],
                    help="Router selection. sel_v2 (default): production "
                         "binary arm_subset (chain_deep/bridge_entity/"
                         "semantic_rich + polar / implicit-multihop / "
                         "gamma overrides). pam_lite: continuous P_arm "
                         "per arm via deterministic sigmoid over the "
                         "linguistic features in "
                         "mothrag.routing.semantic_features; variable-K "
                         "subset (arms with P_arm > --pam-lite-threshold). "
                         "P_arm modulates the DeterministicArbitrator "
                         "score (combined = P_arm x (w_gamma*gamma + "
                         "w_agree*agree + w_faith*faith)). Backward "
                         "compat: sel_v2 default unchanged.")
    ap.add_argument("--pam-lite-threshold", type=float, default=0.3,
                    help="PAM-lite inclusion threshold (default 0.3). "
                         "Ignored when --router=sel_v2.")

    # ---- Mirror arbitrate_post.py PAM-lite arbitrator + ablation flags so
    # route_prospective can drive LIVE eval with the same configuration
    # surface. Re-uses the monkey-patch pattern (commit 007869c).
    ap.add_argument("--arbitrator", default="legacy",
                    choices=["legacy", "pam_lite"],
                    help="legacy (default): existing DeterministicArbitrator / "
                         "_run_ensemble_arbitrate composition. pam_lite: drive "
                         "final pick from PAM-lite P_arm vector via "
                         "arbitrate_pam_lite. Requires "
                         "--router=pam_lite.")
    ap.add_argument("--arbitrator-mode", default="argmax",
                    choices=["argmax", "weighted_mix", "subset"],
                    help="PAM-lite arbitrator mode (only used when "
                         "--arbitrator=pam_lite). argmax (default): pick "
                         "highest P_arm. weighted_mix: weighted vote on "
                         "distinct preds. subset: threshold filter then argmax.")
    ap.add_argument("--disable-cfde114-boost", action="store_true", default=False,
                    help="Ablation: monkey-patch _score_v3bu_p_arm to zero "
                         "the cfde114 v3 comparison_marker boost on "
                         "is_1hop_polar. Mirrors arbitrate_post --disable-"
                         "cfde114-boost.")
    ap.add_argument("--disable-hop-multipliers", action="store_true", default=False,
                    help="Ablation: monkey-patch get_hop_weight to always "
                         "return 1.0 (restores unitary scoring). "
                         "Mirrors arbitrate_post --disable-hop-multipliers.")
    # CLI wiring of mechanism ablation flags (PDD-mechanism A/B keystone).
    ap.add_argument("--tie-break", default="priority",
                    choices=["priority", "lexicographic", "first", "random"],
                    help="Tie-break strategy passed to "
                         "arbitrate_pam_lite_traced. priority (default, "
                         "legacy byte-compat) / lexicographic / first / "
                         "random (seeded).")
    ap.add_argument("--disable-fallback", action="store_true", default=False,
                    help="Ablation: skip all-uncertain + subset-"
                         "empty fallbacks in PAM-lite arbitration.")
    ap.add_argument("--w-agree", type=float, default=0.5,
                    help="DeterministicArbitrator weight: cross-arm "
                         "agreement (default 0.5). Setting to 0 isolates "
                         "PDD (ablation A).")
    ap.add_argument("--w-gamma", type=float, default=1.0,
                    help="DeterministicArbitrator weight: gamma "
                         "verifier (default 1.0).")
    ap.add_argument("--w-faith", type=float, default=0.3,
                    help="DeterministicArbitrator weight: "
                         "faithfulness (default 0.3).")
    ap.add_argument("--dup-random-answer", action="store_true", default=False,
                    help="Ablation B: replace dup-arm prediction "
                         "with a random pick from the other arms in the "
                         "pool. Isolates the cosine-match component of "
                         "the PDD mechanism (without this, dup matches "
                         "its base via cosine = 1.0).")
    ap.add_argument("--use-gamma-aware-pdd", action="store_true", default=False,
                    help="γ-aware PDD "
                         "gating — drop the dup (iter_dup_a / PDD) arm from "
                         "arbitration when the iter arm is γ=valid, so its "
                         "signal-dup no longer double-counts in pairwise "
                         "agreement (effective 3-arm pool). Default OFF = "
                         "byte-identical legacy 4-arm arbitration.")
    ap.add_argument("--use-gamma-aware-bridge-cohort",
                    dest="use_gamma_aware_bridge_cohort",
                    action="store_true", default=False,
                    help="FIX B — γ-aware bridge "
                         "cohort 2-pass (CONSERVATIVE): a first-pass dense iter "
                         "probe yields a γ proxy; skip the bridge ONLY for the "
                         "easiest cohort — γ=valid AND qtype==semantic_rich AND "
                         "hop_count==1 — and reuse the probe iter in the pool. "
                         "Multi-hop / chain_deep / bridge_entity always keep the "
                         "bridge (a broader ~60%% skip lost MQ/2W bulk). Default "
                         "OFF = legacy bridge.prepare() flow.")
    ap.add_argument("--use-adaptive-gamma-retrigger",
                    dest="use_adaptive_gamma_retrigger",
                    action="store_true", default=False,
                    help="FIX C — per-cohort γ-loop "
                         "retrigger cap (CONSERVATIVE): ONLY chain_deep=3 (deep "
                         "chains abandoned too early at the fixed 2); "
                         "bridge_entity=2 / semantic_rich=2 / other=2 stay at "
                         "baseline (an earlier semantic_rich=1 starved the bulk "
                         "cohort, MQ/2W regression). Default OFF = legacy fixed cap.")
    ap.add_argument("--use-faithfulness-gamma-coord",
                    dest="use_faithfulness_gamma_coord",
                    action="store_true", default=False,
                    help="FIX D — "
                         "skip the faithfulness check on TWO safe branches: "
                         "(a) γ=VALID (already γ-verified ⇒ redundant), or "
                         "(b) γ-loop exhausted AND qtype!=chain_deep AND iter>=2 "
                         "(non-chain_deep cap-hit is safe). chain_deep cap-hit "
                         "answers KEEP the gate (recovery cohort). Default OFF = "
                         "legacy faithfulness flow.")
    ap.add_argument("--simulate-n-cap", type=int, default=None,
                    help="Ablation C: cap the agreement denominator "
                         "at this N (was N-1 in the live calc) to isolate "
                         "the denominator effect from the numerator. "
                         "Default None = no cap.")

    # ---- opt-in arm pool (infobox / MothGraphArm / bm25) ----
    ap.add_argument("--arms-pool", default="v3bu,decompose,iter",
                    help="Comma-separated list of arms to include in the "
                         "ensemble_arbitrate composition. Default: "
                         "'v3bu,decompose,iter' (production 3-arm baseline). "
                         "Opt-in additions: 'infobox_arm' (direct "
                         "structured-fact lookup over InfoboxIndex, no LLM), "
                         "'mothgraph_arm' (anchor-driven iterative graph "
                         "traversal over OpenIE-extracted triples with "
                         "gamma-validated paths + stability + soft "
                         "fallback, no LLM in the arm itself), "
                         "'bm25_arm' (sparse keyword retrieval over "
                         "corpus via rank_bm25). Examples: "
                         "'v3bu,decompose,iter,infobox_arm' = 4-arm pool; "
                         "'v3bu,decompose,iter,infobox_arm,mothgraph_arm' = "
                         "5-arm MothRag pool. The legacy 3 arms STAY function-"
                         "based for production parity; opt-in arms are "
                         "class-based and integrated via the Arm Protocol. "
                         "Currently consumed only by --mode=ensemble_arbitrate.")
    ap.add_argument("--mothgraph-max-iters", type=int, default=3,
                    help="MothGraphArm refinement loop cap (default 3).")
    ap.add_argument("--mothgraph-base-depth", type=int, default=2,
                    help="MothGraphArm base traversal depth (default 2; "
                         "may expand to 3 adaptively on high-complexity queries).")
    ap.add_argument("--mothgraph-top-k", type=int, default=8,
                    help="MothGraphArm top-K paths per traversal round (default 8).")
    ap.add_argument("--mothgraph-use-spacy", action="store_true", default=False,
                    help="Use spacy dependency-parse extractor alongside the "
                         "stdlib regex tier when building the graph index. "
                         "Default False (stdlib-only, zero external dependency).")

    # ---- bridge retrieval SUBSTRATE (NOT a 5th arm) ----
    # Drives the unified "4-arm + bridge substrate" run from ONE runner:
    #   --mode ensemble_arbitrate \
    #   --arms-pool v3bu,decompose,iter,iter_dup_a \
    #   --use-bridge-substrate
    # The bridge reshapes each query's PRIMARY retrieval; the arm POOL stays 4
    # (zero pool-safety violation). ``--use-bridge-arm`` is a back-compat alias
    # for the same flag (the "arm" spelling is a known misnomer — it is a
    # substrate).
    ap.add_argument("--use-bridge-substrate", "--use-bridge-arm",
                    dest="use_bridge_substrate", action="store_true",
                    default=False,
                    help="Install the BridgeRAG-Haiku retrieval SUBSTRATE "
                         "upstream of the arm pool. The primary (seed-free) "
                         "retrieval of each top-level question is reshaped by "
                         "the tripartite-judge bridge pipeline "
                         "(gemini-ANN multi-query fusion over the same corpus + "
                         "Haiku SVO/dual-entity/judge stages); sub-Q / iter "
                         "refinements stay dense. NOT a new arm — the pool is "
                         "unchanged. Requires ANTHROPIC_API_KEY for the Haiku "
                         "stages (degrades to dense order without it).")
    ap.add_argument("--bridge-substrate-scope", default="primary",
                    choices=["primary", "all"],
                    help="Bridge substrate reach. 'primary' "
                         "(default): only each top-level "
                         "question's primary seed-free retrieval is bridged; "
                         "sub-Q + iter-refinement stay dense (~1 bridge run/q). "
                         "'all': EVERY retrieval (primary + sub-Q + iter "
                         "refinement) is bridged — the bridge replaces the "
                         "dense retrieval mechanism wholesale (costlier; "
                         "deduped per query string, cost-capped).")
    ap.add_argument("--bridge-substrate-qtype-gate", default="none",
                    choices=["none", "semantic_rich_only",
                             "exclude_bridge_entity"],
                    help="Input-feature (qtype) gate on the "
                         "bridge substrate. 'none' (default): bridge every "
                         "question (current behaviour). 'semantic_rich_only': "
                         "bridge ONLY questions classify_query_v2 labels "
                         "semantic_rich (the cohort the bridge helps cross-DS "
                         "+2..+4pp); chain_deep + bridge_entity fall to dense. "
                         "'exclude_bridge_entity': bridge everything EXCEPT "
                         "bridge_entity (the cohort the bridge hurts -3.7..-14.8"
                         "pp). Anti-leak: question-text feature only, never a "
                         "DS signal.")
    ap.add_argument("--arm-bridge-qtype-gate", default=None,
                    help="PER-ARM bridge qtype gate "
                         "(MUTUALLY EXCLUSIVE with --bridge-substrate-qtype-gate). "
                         "Each arm carries its own gate so the C7 arbitrator can "
                         "pick the bridge-on vs bridge-off arm per query. Form "
                         "(verbatim gate values, no aliases): "
                         "'v3bu:none,decompose:none,iter:exclude_bridge_entity,"
                         "iter_dup_a:exclude_bridge_entity'. Coherence enforced: "
                         "iter_dup_a==iter (dup copies iter), v3bu==decompose "
                         "(share pre-pool retrieval). Anti-leak: arm-name + qtype "
                         "(classify_query_v2) only, never a DS/corpus signal.")
    ap.add_argument("--bridge-judge-model", default="claude-haiku-4-5",
                    help="LLM for the bridge SVO/entity/judge stages "
                         "(default claude-haiku-4-5; the Sonnet-upgrade "
                         "ablation in the decision gate overrides this).")
    ap.add_argument("--bridge-judge-provider", default="anthropic",
                    choices=["anthropic", "gemini"],
                    help="Provider for the bridge SVO/entity/judge stages. "
                         "Default 'anthropic' "
                         "(Haiku, V4 byte-identical); 'gemini' routes "
                         "--bridge-judge-model to GeminiReader (commodity Flash, "
                         "needs GEMINI_API_KEY). Mirrors --judge-provider for "
                         "the gamma-judge.")
    ap.add_argument("--bridge-max-cost-usd", type=float, default=10.0,
                    help="Cross-query TOTAL spend cap for the bridge substrate "
                         "(default $10). Once exceeded, further queries revert "
                         "to the dense substrate (logged in the summary).")
    ap.add_argument("--allow-uncached", action="store_true", default=False,
                    help="Proceed with --use-bridge-"
                         "substrate even if MOTHRAG_GEMINI_CACHE_DIR is unset "
                         "or the gemini corpus cache (chunk_vecs_gemini_doc.npy) "
                         "is missing near --data-dir. Without this, a missing "
                         "cache hard-fails with exit 2 (a live corpus re-embed "
                         "is $50-150 + Gemini 429/503 RPD risk).")
    ap.add_argument("--dry-run", action="store_true", default=False,
                    help="Offline mechanics + paired-comparison check of the "
                         "bridge substrate on 5 synthetic queries (no corpus, "
                         "no API, no cost). Exits 0 on success. Does not need "
                         "--data-dir/--queries/--out. With --use-iter-ragnatela, "
                         "also runs the γ-convergence + pool-safety smoke test.")
    # ---- Iterative Ragnatela γ-feedback loop ----
    ap.add_argument("--use-iter-ragnatela", dest="use_iter_ragnatela",
                    action="store_true", default=False,
                    help="Run the Iterative Ragnatela "
                         "γ-feedback loop over the 4-arm pool (an INTERNAL "
                         "upgrade of the iter machinery, NOT a 5th arm; the pool "
                         "stays v3bu/decompose/iter/iter_dup_a). Layers on top of "
                         "the bridge substrate when --use-bridge-substrate is also "
                         "set (bridge reshapes each round's retrieval; the loop "
                         "γ-pools the arms, generates context-aware sub-questions "
                         "from the uncertain band, re-retrieves, and stops on "
                         "γ-convergence). Offline-validated via --dry-run.")
    ap.add_argument("--iter-ragnatela-kmax", "--kmax", dest="iter_ragnatela_kmax",
                    type=int, default=3,
                    help="Max γ-feedback iterations before the loop stops "
                         "(default 3). Bounds the loop so it can "
                         "never iterate forever; --kmax is a short alias.")
    # ---- ChainFilter v0.1 CLI pre-wire ----
    ap.add_argument("--use-chainfilter", dest="use_chainfilter",
                    action="store_true", default=False,
                    help="Install the ChainFilter v0.1 "
                         "POST-retrieval reranker over the bridge_top100 (γ-band "
                         "weighted fact-coverage + input-feature hop gate). It is "
                         "a FILTER, NOT a 5th arm (pool-safety preserved). Default "
                         "OFF (byte-identical passthrough).")
    ap.add_argument("--use-chainfilter-cohort-gate",
                    dest="use_chainfilter_cohort_gate",
                    action="store_true", default=False,
                    help="FIX A — skip the ChainFilter "
                         "rerank on the chain_deep cohort (extraction-bound: the "
                         "OpenIE triple-extract can discard answer-bearing chunks "
                         "there). Pure passthrough on chain_deep; legacy ChainFilter "
                         "on every other cohort. Default OFF = byte-identical.")
    ap.add_argument("--chainfilter-hop-min", dest="chainfilter_hop_min",
                    type=int, default=2,
                    help="ChainFilter hop-gate threshold (default 2): fire only "
                         "on questions whose input features give hop_count >= this "
                         "(or is_chain_deep). Single-hop questions bypass.")
    ap.add_argument("--chainfilter-gamma-min", dest="chainfilter_gamma_min",
                    type=float, default=None,
                    help="ChainFilter anti-context γ floor (maps to the γ LOW/MID "
                         "band boundary): facts whose γ is below this are LOW "
                         "(anti-context) and excluded from a passage's coverage "
                         "score. Default = the RagnatelaConfig γ_low (0.33).")
    ap.add_argument("--chainfilter-top-out", dest="chainfilter_top_out",
                    type=int, default=5,
                    help="ChainFilter emitted top-K (default 5; the R@5 metric).")
    # ---- M8 specialist slot router (pool-safe) ----
    ap.add_argument("--use-m8-specialists", dest="use_m8_specialists",
                    action="store_true", default=False,
                    help="Install the pool-safe specialist slot "
                         "router: the 'decompose' slot becomes polymorphic and, "
                         "on a comparison / compositional question (input-feature "
                         "classified), is FILLED by CompareArm / DecomposeArm 2.0 "
                         "via SUBSTITUTION — the arm pool stays exactly 4 "
                         "(v3bu / <decompose-slot> / iter / iter_dup_a); a "
                         "non-firing specialist degrades to the generic decompose "
                         "arm. Default OFF (byte-identical to the current pool). "
                         "Anti-leak: question-text features only. (PDD "
                         "cardinality wiring into iter_dup_a is a separate flag, "
                         "next increment.)")

    # ---- Path C retry-on-abstain escalation + Strategy #8/#9 ----
    ap.add_argument("--retry-strategies", default=None,
                    help="Enable retry-on-abstain cascade. Comma-separated names "
                         "or preset alias. Examples: 'default_7', 'sweet_spot', "
                         "'soft_fallback_only', 'default_7,active_gap_query', "
                         "'default_7,active_gap_query,sub_question_reroute'. "
                         "Default: disabled (no retry escalation).")
    ap.add_argument("--retry-mode", default="loop",
                    choices=["loop", "abstention"],
                    help="Retry cascade terminal mode. loop (default): SoftFallback "
                         "guarantees non-empty answer. abstention: terminal abstain "
                         "allowed for KB-audit / gap-discovery deployments.")
    ap.add_argument("--retry-budget-limit", type=int, default=8,
                    help="Cap on LLM calls inside the cascade per query (default 8).")
    ap.add_argument("--sub-question-layers", default="syntactic,spectral,llm",
                    help="Strategy #9 layer selection, comma-separated. "
                         "Default: syntactic,spectral,llm (all 3).")
    ap.add_argument("--sub-question-max-depth", type=int, default=3)
    ap.add_argument("--sub-question-max-sub-questions", type=int, default=6)
    ap.add_argument("--active-gap-max-rounds", type=int, default=3)
    ap.add_argument("--active-gap-max-passages-per-round", type=int, default=5)
    ap.add_argument("--use-spectral", action="store_true", default=False,
                    help="Strategy #9 Layer 2 hint: enable per-aspect "
                         "gamma + L4b + agreement disaggregation (forces "
                         "'spectral' into --sub-question-layers if absent).")

    args = ap.parse_args()

    # ---- Offline bridge-substrate dry-run (no corpus/API).
    if args.dry_run:
        print("\n=== route_prospective bridge-substrate DRY-RUN ===", flush=True)
        all_ok = True
        for scope in ("primary", "all"):
            sim = _dry_run_bridge_substrate(
                judge_model=args.bridge_judge_model,
                max_cost_usd=args.bridge_max_cost_usd,
                scope=scope,
            )
            all_ok = all_ok and sim["ok"]
            print(f"[dry-run] {json.dumps(sim, indent=2)}", flush=True)
            print(f"[dry-run] scope={scope}: "
                  f"{sim['n_bridge_fired']}/{sim['synthetic_queries']} primary "
                  f"bridge-fired | seed-path-bridged={sim['n_seed_path_bridged']} "
                  f"| subq_bridged={sim['subq_bridged']} | "
                  f"diverged={sim['n_diverged_from_dense']} | ok={sim['ok']}",
                  flush=True)
        # Layer the iter-ragnatela γ-convergence + pool-safety smoke test on
        # top of the bridge substrate when requested.
        if args.use_iter_ragnatela:
            rag = _dry_run_iter_ragnatela(kmax=args.iter_ragnatela_kmax)
            all_ok = all_ok and rag["ok"]
            print(f"[dry-run] {json.dumps(rag, indent=2)}", flush=True)
            print(f"[dry-run] iter-ragnatela: converged={rag['converged']} in "
                  f"{rag['iterations_used']} iters (kmax={rag['kmax']}) | "
                  f"gamma_trace={rag['gamma_trace']} | pool_size_4_every_iter="
                  f"{rag['pool_size_4_every_iter']} | no_bridge_arm="
                  f"{rag['no_bridge_arm_key']} | answer={rag['answer']!r} | "
                  f"ok={rag['ok']}", flush=True)
        return 0 if all_ok else 1

    # A LIVE run requires the corpus + queryset + output path.
    for req, name in ((args.data_dir, "--data-dir"),
                      (args.queries, "--queries"), (args.out, "--out")):
        if req is None:
            raise SystemExit(f"{name} is required for a live run "
                             "(use --dry-run for the offline substrate check).")

    # ---- Bridge cache HARD-FAIL (anti-waste). Only when
    # the bridge substrate is active — non-bridge runs are unaffected. A
    # missing gemini corpus cache makes from_corpus re-embed the whole corpus
    # ($50-150 + Gemini RPD death); an unset query cache means no $0 cross-fire.
    if args.use_bridge_substrate and not args.allow_uncached:
        missing = []
        if not os.environ.get("MOTHRAG_GEMINI_CACHE_DIR"):
            missing.append("MOTHRAG_GEMINI_CACHE_DIR (query-embed cache)")
        if str(args.embedding).startswith("gemini"):
            corpus_cache = args.data_dir / "chunk_vecs_gemini_doc.npy"
            if not corpus_cache.exists():
                missing.append(f"gemini corpus cache ({corpus_cache})")
        if missing:
            print("[route_prospective] FATAL: --use-bridge-substrate with "
                  "missing cache(s): " + "; ".join(missing) + ". A live corpus "
                  "re-embed is $50-150 + Gemini 429/503 RPD death; pre-build "
                  "the cache(s) or pass --allow-uncached to proceed anyway "
                  "(expensive).", file=sys.stderr, flush=True)
            raise SystemExit(2)

    # ---- PAM-lite arbitrator validation + ablation monkey-patches.
    # Mirrors arbitrate_post.py (commit 007869c).
    use_pam_lite_arb = (args.arbitrator == "pam_lite")
    if use_pam_lite_arb and args.router != "pam_lite":
        raise SystemExit(
            "--arbitrator=pam_lite requires --router=pam_lite "
            "(arbitrate_pam_lite consumes the PAM-lite P_arm vector).",
        )
    if use_pam_lite_arb:
        print(
            f"[setup] arbitrator=pam_lite (mode={args.arbitrator_mode}) -- "
            f"final pick driven by arbitrate_pam_lite over P_arm.",
            flush=True,
        )

    if args.disable_cfde114_boost:
        import mothrag.core.query_type_classifier as _qtc

        _qtc_sigmoid = _qtc._sigmoid
        _qtc_get_hop_weight = _qtc.get_hop_weight
        _qtc_hop_structure = _qtc._hop_structure

        def _v3bu_no_boost(f):
            hop = _qtc_hop_structure(f)
            # cfde114 boost zeroed; rest mirrors _score_v3bu_p_arm body.
            base = _qtc_sigmoid(
                +0.5 * f.single_entity
                + 0.4 * f.attribute_marker
                + 0.3 * f.single_hop
                - 0.3 * f.multi_hop_marker
                - 0.2 * f.chain_marker
                + 0.2
            )
            return base * _qtc_get_hop_weight("v3bu", hop)

        _qtc._score_v3bu_p_arm = _v3bu_no_boost
        print(
            "[setup] ablation: --disable-cfde114-boost ON "
            "(_score_v3bu_p_arm boost zeroed)",
            flush=True,
        )

    if args.disable_hop_multipliers:
        import mothrag.core.query_type_classifier as _qtc
        _qtc.get_hop_weight = lambda arm, hop: 1.0
        print(
            "[setup] ablation: --disable-hop-multipliers ON "
            "(get_hop_weight returns 1.0)",
            flush=True,
        )

    # Per-query telemetry counters (mirrors arbitrate_post).
    cfde114_fire_count = 0
    hop_multiplier_active_count = 0
    non_unitary_p_arm_count = 0

    # Reader API key
    api_key = os.environ.get(args.reader_api_key_env)
    if not api_key:
        for v in ("TOGETHER_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"):
            if os.environ.get(v):
                api_key = os.environ[v]
                break
    if not api_key:
        raise SystemExit(f"No API key found in env var {args.reader_api_key_env}")

    print(f"[setup] loading corpus from {args.data_dir} (embedding={args.embedding}) ...")
    if "," in args.reader_temperature:
        reader_temp = [float(t.strip()) for t in args.reader_temperature.split(",")]
    else:
        reader_temp = float(args.reader_temperature)

    pipe_cfg = PipelineConfig(
        embedding=args.embedding,
        reranker=args.reranker,
        bottom_up_boost=args.bottom_up_boost,
        top_k_chunks=args.top_k_chunks,
        reader_prompt="v3-think",
        reader_max_tokens=2000,
        reader_n_samples=args.reader_n_samples,
        reader_temperature=reader_temp,
    )
    pipeline = MothRAGPipeline.from_corpus(
        args.data_dir,
        embedding=args.embedding,
        reranker=args.reranker,
        bottom_up_boost=args.bottom_up_boost,
        reader_model=args.reader_model,
        reader_api_key=api_key,
        reader_base_url=args.reader_base_url,
        config=pipe_cfg,
    )
    print(f"[setup] corpus loaded: {len(pipeline.chunk_ids)} chunks, "
          f"{len(pipeline.entities)} entities")

    # NER cache (L1C)
    if args.ner_cache:
        from mothrag.retrieval.ner import load_cache, build_ner_cache
        cache_path = Path(args.ner_cache)
        cache = load_cache(cache_path)
        if not cache:
            print(f"[setup] L1C: NER cache empty, building via "
                  f"{args.ner_build_provider}/{args.ner_build_model}")
            queries_for_cache = (load_queryset(Path(args.queries))[: args.n]
                                 if args.n else load_queryset(Path(args.queries)))
            if args.ner_build_provider == "gemini":
                raise SystemExit("L1C NER cache build with Gemini provider not supported "
                                 "— pre-build offline or use --ner-build-provider {openai|groq|together}")
            ner_client = build_final_reader_client(
                provider=args.ner_build_provider, base_url=None,
                api_key_env=None, model_name=args.ner_build_model,
            )
            cache = build_ner_cache(ner_client, args.ner_build_model,
                                     queries_for_cache, cache_path, save_every=25)
            print(f"[setup] L1C: NER cache built {len(cache)} entries -> {cache_path}")
        else:
            print(f"[setup] L1C: NER cache loaded {len(cache)} entries from {cache_path}")
        pipeline.ner_cache = cache

    final_reader_client = None
    if args.final_reader_model:
        final_reader_client = build_final_reader_client(
            provider=args.final_reader_provider,
            base_url=args.final_reader_base_url,
            api_key_env=args.final_reader_api_key_env,
            model_name=args.final_reader_model,
        )

    judge_client = None
    if args.use_faithfulness_loop:
        if not args.judge_model:
            raise SystemExit("--use-faithfulness-loop requires --judge-model")
        if args.judge_provider == "gemini":
            import google.genai as genai
            key_env = args.judge_api_key_env or "GEMINI_API_KEY"
            judge_api_key = os.environ.get(key_env)
            if not judge_api_key:
                raise SystemExit(f"--judge env var {key_env} is empty")
            judge_client = genai.Client(api_key=judge_api_key)
            print(f"[setup] L4 judge: provider=gemini model={args.judge_model} "
                  f"key_env={key_env}")
        else:
            judge_client = build_final_reader_client(
                provider=args.judge_provider, base_url=None,
                api_key_env=args.judge_api_key_env, model_name=args.judge_model,
            )

    c7_iter_embedder = None
    if args.use_c7_iter:
        c7_iter_embedder = build_c7_iter_embedder(args.c7_iter_embedder)

    iter_cfg = IterativeConfig(
        max_iterations=args.max_iterations,
        top_k_total=args.top_k_total,
        stop_early=args.stop_early,
        final_reader_client=final_reader_client,
        final_reader_model=args.final_reader_model,
        final_reader_max_tokens=args.final_reader_max_tokens,
        use_graph_aware_iter=args.use_graph_aware_iter,
        max_accumulated_entities=args.max_accumulated_entities,
        use_faithfulness_loop=args.use_faithfulness_loop,
        judge_client=judge_client,
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
        judge_min_score=args.judge_min_score,
        max_judge_iterations=args.max_judge_iterations,
        use_gamma_verifier=args.use_gamma_verifier,
        use_gamma_liberal=args.use_gamma_liberal,
        gamma_max_retrigger=args.gamma_max_retrigger,
        # Capture proof trees into per-q (default OFF).
        gamma_diagnostic_dump=getattr(args, "gamma_diagnostic_dump", False),
        # Relax object grounding (default OFF).
        use_relaxed_object_span=getattr(args, "use_relaxed_object_span", False),
        # Hop-ordered reader evidence (default OFF).
        path_serialize=getattr(args, "path_serialize", False),
        # FIX C + FIX D (default OFF).
        use_adaptive_gamma_retrigger=getattr(
            args, "use_adaptive_gamma_retrigger", False),
        use_faithfulness_gamma_coord=getattr(
            args, "use_faithfulness_gamma_coord", False),
        gamma_prompt_variant=args.gamma_prompt_variant,
        use_gamma_router=args.use_gamma_router,
        use_c7_iter=args.use_c7_iter,
        c7_iter_trigger=args.c7_iter_trigger,
        c7_iter_embedder=c7_iter_embedder,
        # ---- Phase 1 patches (VERBATIM eval_wiki_iterative.py
        # IterativeConfig wiring, so route_prospective reproduces step3). ----
        use_gamma_refuse_loop=args.use_gamma_refuse_loop,
        use_stepchain_parity_composite=args.use_stepchain_parity_composite,
        use_bug_pattern_wave_a=args.use_bug_pattern_wave_a,
        use_p11_gamma_cap_fallback=args.use_p11_gamma_cap_fallback,
        disable_p11_gamma_cap_fallback=args.disable_p11_gamma_cap_fallback,
        composite_max_iterations=(args.composite_max_iterations
                                  if args.composite_max_iterations is not None
                                  else 5),
        use_p12_decompose_collapse_cap=args.use_p12_decompose_collapse_cap,
        use_p13_sub_q_abstain_filter=args.use_p13_sub_q_abstain_filter,
        use_p15_gamma_gated_naturalize=args.use_p15_gamma_gated_naturalize,
        use_p24_unified_abstain_markers=args.use_p24_unified_abstain_markers,
    )
    iter_runner = IterativeMothRAG(pipeline, iter_cfg)

    # ---- dense_plus_infobox augmentation ------------------
    # When --retrieval dense_plus_infobox is set, harvest InfoboxTriples
    # from the pipeline's existing chunks and append them as synthetic
    # chunks so the dense retriever surfaces them naturally on
    # entity-attribute questions. No monkey-patching of pipeline.retrieve;
    # the synthetic chunks are first-class members of pipeline.chunk_vecs
    # / chunk_ids / chunks_by_id from this point on.
    #
    # router_gated_infobox: same augmentation, but the
    # _InfoboxGate snapshots the plain pre-augmentation state so it can
    # swap back per-query when the deterministic router decides the
    # query is multi-hop / chain (which dropped F1 -0.86 / -12.5pp on
    # HP / MQ T1).
    infobox_gate: "_InfoboxGate | None" = None
    if args.retrieval == "dense_plus_infobox":
        _augment_pipeline_with_infobox(
            pipeline,
            top_n_boost=args.infobox_top_n_boost,
        )
    elif args.retrieval == "router_gated_infobox":
        infobox_gate = _InfoboxGate(
            pipeline,
            top_n_boost=args.infobox_top_n_boost,
        )

    # ---- build opt-in arm pool members --------------
    arms_pool = _parse_arms_pool(args.arms_pool)
    opt_in_arms: dict = {}
    if "infobox_arm" in arms_pool:
        infobox_arm = _build_infobox_arm_from_pipeline(pipeline)
        if infobox_arm is not None:
            opt_in_arms["infobox_arm"] = infobox_arm
            print(
                f"[setup] arms-pool: infobox_arm wired "
                f"({len(infobox_arm.infobox_index)} triples harvested)."
            )
        else:
            print(
                "[setup] arms-pool: infobox_arm requested but 0 triples "
                "harvestable from corpus; arm omitted from pool."
            )
    if "mothgraph_arm" in arms_pool:
        mothgraph_arm = _build_mothgraph_arm_from_pipeline(
            pipeline,
            reader_model=args.reader_model,
            max_iters=args.mothgraph_max_iters,
            base_depth=args.mothgraph_base_depth,
            top_k=args.mothgraph_top_k,
            use_spacy=args.mothgraph_use_spacy,
        )
        if mothgraph_arm is not None:
            opt_in_arms["mothgraph_arm"] = mothgraph_arm
            print(
                f"[setup] arms-pool: mothgraph_arm wired "
                f"({len(mothgraph_arm.graph_index)} edges over "
                f"{mothgraph_arm.graph_index.n_entities} entities harvested)."
            )
        else:
            print(
                "[setup] arms-pool: mothgraph_arm requested but 0 triples "
                "harvestable from corpus; arm omitted from pool."
            )

    # ---- Install the bridge retrieval substrate. ----
    # Wraps pipeline.retrieve so the primary retrieval feeding ALL arms
    # (v3bu / decompose / iter / iter_dup_a) is bridge-reshaped; the arm pool
    # is NOT touched. The substrate degrades to dense when ANTHROPIC_API_KEY is
    # absent or the cost cap is hit, so it never breaks / overspends a run.
    # ChainFilter v0.1 reranker (default OFF → None → no-op).
    # Built BEFORE the bridge substrate so it can be installed INTO it: when
    # present, _BridgeSubstrate._bridge_ranking runs chain_filter.filter() over
    # each bridge ranking (reconstruct Candidates → filter → remap to indices).
    # A POST-retrieval reranker, NOT a 5th arm.
    chain_filter = _build_chain_filter(args)
    if chain_filter is not None:
        print(f"[setup] ChainFilter v0.1 ACTIVE (--use-chainfilter) — "
              f"hop_min={args.chainfilter_hop_min} top_out={args.chainfilter_top_out}"
              f" γ_low={chain_filter.cfg.gamma_cfg.gamma_low}. Reshapes the bridge "
              f"ranking via OpenIE triples + γ-band fact-coverage; single-hop / "
              f"no-support → passthrough. POST-retrieval reranker, NOT a 5th arm.",
              flush=True)
        if not args.use_bridge_substrate:
            print("[setup] WARNING: --use-chainfilter without --use-bridge-substrate"
                  " — ChainFilter reshapes the BRIDGE ranking; with no bridge it is "
                  "inert this run.", file=sys.stderr, flush=True)

    # Parse + validate the per-arm gate (MUTEX with the uniform
    # --bridge-substrate-qtype-gate). Done before substrate install so a
    # malformed spec fails fast with a clear message (no run).
    per_arm_gates = None
    if getattr(args, "arm_bridge_qtype_gate", None):
        if args.bridge_substrate_qtype_gate != "none":
            raise SystemExit(
                "[setup] --arm-bridge-qtype-gate is mutually exclusive with "
                "--bridge-substrate-qtype-gate (got "
                f"{args.bridge_substrate_qtype_gate!r}); set only one.")
        per_arm_gates = _parse_arm_bridge_gates(args.arm_bridge_qtype_gate)

    bridge_substrate: "_BridgeSubstrate | None" = None
    if args.use_bridge_substrate:
        # The live-backend gate must follow the bridge provider: Gemini needs
        # GEMINI_API_KEY, not ANTHROPIC_API_KEY. A wrong key check would
        # silently run the bridge "degraded(no key)" → n_bridge_runs=0 →
        # silent dense fallback.
        _bridge_key_env = ("GEMINI_API_KEY"
                           if args.bridge_judge_provider == "gemini"
                           else "ANTHROPIC_API_KEY")
        require_backend = bool(os.environ.get(_bridge_key_env))
        bridge_substrate = _BridgeSubstrate(
            pipeline,
            judge_model=args.bridge_judge_model,
            judge_provider=args.bridge_judge_provider,
            max_cost_usd=args.bridge_max_cost_usd,
            require_backend=require_backend,
            scope=args.bridge_substrate_scope,
            qtype_gate=args.bridge_substrate_qtype_gate,
            chain_filter=chain_filter,
            per_arm_gates=per_arm_gates,
        )
        print(f"[setup] bridge SUBSTRATE installed "
              f"(scope={args.bridge_substrate_scope} "
              f"qtype_gate={args.bridge_substrate_qtype_gate} "
              f"per_arm_gates={per_arm_gates} "
              f"judge={args.bridge_judge_model} "
              f"provider={args.bridge_judge_provider} "
              f"max_cost=${args.bridge_max_cost_usd} "
              f"bridge_backend={'live' if require_backend else f'degraded(no {_bridge_key_env})'}). "
              f"Pool stays {len(arms_pool)}-arm; bridge is upstream substrate, "
              f"NOT a candidate.", flush=True)
        if args.bridge_substrate_scope == "all":
            print("[setup] NB scope=all bridges EVERY retrieval (primary + "
                  "sub-Q + iter) — higher cost; watch --bridge-max-cost-usd.",
                  flush=True)
        if not require_backend:
            print("[setup] WARNING: ANTHROPIC_API_KEY not set — bridge LLM "
                  "stages degrade to dense order (the substrate ~= dense). Set "
                  "the key for the real 4-arm + bridge run.", file=sys.stderr,
                  flush=True)
        if per_arm_gates:
            print(f"[setup] PER-ARM bridge gate active: {per_arm_gates} "
                  f"(C7 arbitrator picks the bridge-on vs bridge-off arm per "
                  f"query; iter_dup_a copies iter).", flush=True)
    # Expose the substrate to the arm runners ONLY when a per-arm gate is active,
    # so _mark_arm is a strict no-op in uniform / no-substrate mode.
    _set_active_substrate(bridge_substrate if per_arm_gates else None)

    # Build the pool-safe M8 specialist slot router once
    # (default OFF → None → byte-identical to the generic decompose slot).
    specialist_router = None
    if getattr(args, "use_m8_specialists", False):
        specialist_router = _build_specialist_router(
            pipeline, args.reader_model, args.top_k_subq)
        print("[setup] M8 specialist slot router installed (--use-m8-specialists)"
              " — the polymorphic 'decompose' slot SUBSTITUTES CompareArm / "
              "DecomposeArm 2.0 on their input-feature cohort; the pool stays 4 "
              "arms (specialists are NOT a 5th arm), and a non-firing specialist "
              "degrades to the generic decompose arm.", flush=True)

    queries = load_queryset(args.queries)
    if args.n:
        queries = queries[: args.n]
    print(f"[setup] {len(queries)} queries to process | mode={args.mode} "
          f"| retrieval={args.retrieval}")

    # Resume support
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done_qids: set[str] = set()
    per_q: list[dict] = []
    if args.out.exists():
        try:
            prev = json.loads(args.out.read_text(encoding="utf-8"))
            per_q = prev.get("per_question", [])
            done_qids = {r["qid"] for r in per_q}
            print(f"[setup] resuming: {len(done_qids)} qids already done")
        except json.JSONDecodeError:
            pass

    t0 = time.time()
    for i, q in enumerate(queries):
        qid = str(q.get("id") or q.get("qid") or i)
        if qid in done_qids:
            continue
        question = q.get("question") or q.get("q") or ""
        gold = q.get("answer") or q.get("gold") or ""
        gold_aliases = q.get("answer_aliases") or q.get("gold_aliases") or []
        gold_list = [gold] + list(gold_aliases) if gold else list(gold_aliases)
        gold_doc_ids = q.get("gold_doc_ids") or q.get("supporting_facts") or []
        if isinstance(gold_doc_ids, list) and gold_doc_ids and isinstance(gold_doc_ids[0], (list, tuple)):
            gold_doc_ids = [str(x[0]) for x in gold_doc_ids]
        gold_doc_ids = [str(x) for x in (gold_doc_ids or [])]

        # Classify the cohort ONCE up-front (input-feature label_v2); reused by
        # the FIX B bridge cohort gate here AND the routing/telemetry below.
        feat = classify_with_features(question)
        qtype = feat["label_v2"]

        # ---- Prime the bridge substrate for THIS query.
        # Runs the bridge once; the wrapped pipeline.retrieve then serves the
        # bridge ranking to every arm's primary retrieval. No-op when the
        # substrate is not installed (dense baseline).
        #
        # FIX B — γ-aware bridge cohort 2-pass (default OFF). A first-pass DENSE
        # iter probe (the bridge is not yet warmed, so retrieval is dense) yields
        # a γ proxy; the bridge expansion is skipped ONLY for the EASIEST cohort
        # — γ=valid AND qtype==semantic_rich AND hop_count==1 — and the probe's
        # iter result is REUSED by the pool (no double iter on the skip path).
        # CONSERVATIVE: multi-hop semantic_rich, chain_deep, and bridge_entity
        # ALWAYS keep the bridge (a broader variant skipped ~60% of γ=valid
        # queries and lost MQ/2W on the bulk cohort).
        bridge_prep: dict = {"fired": False, "reason": "off"}
        bridge_skipped_easy_semantic_rich = False
        _precomputed_iter = None
        if bridge_substrate is not None:
            if getattr(args, "use_gamma_aware_bridge_cohort", False):
                _probe = _run_iter(iter_runner, question)   # dense (unwarmed)
                if _bridge_cohort_should_skip(
                        _probe.get("gamma_final_status"), qtype, enabled=True,
                        hop_count=_hop_count_proxy(feat)):
                    bridge_prep = {"fired": False,
                                   "reason": "gamma_aware_bridge_cohort_skip_easy_sr",
                                   "bridge_passage_id": None, "cost_usd": 0.0,
                                   "n_ranked": 0}
                    bridge_skipped_easy_semantic_rich = True
                    _precomputed_iter = _probe              # reuse on the skip path
                else:
                    bridge_prep = bridge_substrate.prepare(question)
                    # bridge fired → pool re-runs iter bridged (probe discarded)
            else:
                bridge_prep = bridge_substrate.prepare(question)
        bridge_active = bool(bridge_prep.get("fired", False))

        # ---- router-gated infobox state-swap (pre-retrieval).
        # When --retrieval router_gated_infobox is set, the gate decides
        # per-query whether the pipeline's chunk surface includes the
        # synthetic infobox chunks. Default-off conservative for multi-hop /
        # chain / comparison queries (HP / MQ regression cohort); fires on
        # single-clause entity-attribute (2W beneficiary cohort).
        infobox_fired: bool = (args.retrieval == "dense_plus_infobox")
        infobox_router_reason: str = (
            "unconditional" if args.retrieval == "dense_plus_infobox"
            else "disabled" if args.retrieval == "dense"
            else "unset"
        )
        if infobox_gate is not None:
            infobox_fired, infobox_router_reason = infobox_gate.decide(question)

        # ---- Routing (unified with arm_subset, mirror of arbitrate_post.py
        # --auto-arm-subset semantics). 4-signal cascade applied prospectively:
        # arm_subset() decides which arms to RUN (not just which to consider
        # posthoc). F1 invariant vs posthoc arbitrate; compute reduced by
        # skipping unselected arms.
        #
        # PATCH B: --mode=ensemble_arbitrate bypasses sel_v2 entirely and
        # runs all 3 arms with DeterministicArbitrator. The retry-cascade
        # wrapper (--retry-strategies) is applied AFTER any routing path
        # below, so v2 / strict / ensemble_arbitrate all benefit from it.
        # ---- (feat / qtype already classified up-front for the FIX B gate.)

        # Per-query telemetry counters (mirrors arbitrate_post).
        # Read-only inspection of the routing surface; does not
        # affect routing decisions.
        try:
            from mothrag.core.query_type_classifier import (
                _hop_structure as _qtc_hopf,
            )
            from mothrag.routing.semantic_features import (
                extract_semantic_features as _qtc_feats,
            )
            _sfeat = _qtc_feats(question)
            _hop = _qtc_hopf(_sfeat)
            if _hop.get("is_1hop_polar") and _sfeat.comparison_marker > 0.0:
                cfde114_fire_count += 1
            if any(v for k, v in _hop.items() if k != "is_general_multihop"):
                hop_multiplier_active_count += 1
            if any(_hop.values()):
                # any hop class fires => non-unitary multiplier applied
                non_unitary_p_arm_count += 1
        except Exception:
            # Defensive: telemetry MUST NOT block the main routing path.
            pass

        # Thread arms_pool so subset metadata reflects sel_v2's actual
        # routing decisions including opt-in arms (infobox_arm,
        # mothgraph_arm). Telemetry / downstream eval scripts that read
        # ``arm_subset`` per query will see real fire rates instead of
        # zero. NB: the v2 dispatch block below still drives execution
        # via the legacy 3-arm paths; opt-in arms only execute under
        # --mode ensemble_arbitrate (where _run_ensemble_arbitrate's
        # opt-in arm loop runs them as composition candidates).
        subset = arm_subset(question, query_features=feat, arms_pool=arms_pool)
        v3bu_in_subset = "v3bu" in subset
        arm_used = ""
        sel_reason = ""
        # Carriers populated by the routing block below so the escalation
        # wrapper has the per-arm outputs when it builds a RetryContext.
        arm_outputs: dict = {}
        # γ-aware gating + PDD cohort-gated: per-q
        # PDD telemetry defaults. Only the ensemble_arbitrate pool arbitrates
        # over the dup arm; every other branch leaves these False (no dup).
        pdd_active = False
        pdd_skipped_chain_deep_valid = False
        pdd_preserved_semantic_rich = False
        # FIX A — ChainFilter cohort-gate per-q telemetry
        # (flag + qtype derived, mirrors the actual ChainFilter cohort decision).
        _cf_on = bool(getattr(args, "use_chainfilter", False))
        chainfilter_skipped_chain_deep = bool(
            _cf_on and getattr(args, "use_chainfilter_cohort_gate", False)
            and qtype == "chain_deep")
        chainfilter_active = bool(_cf_on and not chainfilter_skipped_chain_deep)
        # FIX C/D — iter-side per-q telemetry (filled from
        # the iter arm result; None on branches that don't run the iter arm).
        gamma_retrigger_cap_used = None
        faithfulness_active = None
        faithfulness_skipped_clean_valid = None
        faithfulness_skipped_exhausted_safe = None
        gamma_diagnostic = None  # proof-tree dump
        object_relaxed_match_count = 0  # object grounding match-mode count
        object_exact_match_count = 0
        try:
            if args.mode == "ensemble_arbitrate":
                # Bypass sel_v2; always-3-arms + DeterministicArbitrator.
                arm_used = "ensemble_arbitrate"
                arm = _run_ensemble_arbitrate(
                    pipeline, iter_runner, question,
                    args.reader_model, args.top_k_subq,
                    arms_pool=arms_pool,
                    opt_in_arms=opt_in_arms,
                    specialist_router=specialist_router,
                    router=args.router,
                    pam_lite_threshold=args.pam_lite_threshold,
                    # mechanism ablation pass-through
                    w_gamma=args.w_gamma,
                    w_agree=args.w_agree,
                    w_faith=args.w_faith,
                    simulate_n_cap=args.simulate_n_cap,
                    dup_random_answer=args.dup_random_answer,
                    use_gamma_aware_pdd=args.use_gamma_aware_pdd,
                    qtype=qtype,
                    precomputed_iter=_precomputed_iter,
                )
                pred = arm["pred"]
                pdd_active = arm.get("pdd_active", False)
                pdd_skipped_chain_deep_valid = arm.get("pdd_skipped_chain_deep_valid", False)
                pdd_preserved_semantic_rich = arm.get("pdd_preserved_semantic_rich", False)
                gamma_retrigger_cap_used = arm.get("gamma_retrigger_cap_used")
                faithfulness_active = arm.get("faithfulness_active")
                faithfulness_skipped_clean_valid = arm.get(
                    "faithfulness_skipped_clean_valid")
                faithfulness_skipped_exhausted_safe = arm.get(
                    "faithfulness_skipped_exhausted_safe")
                gamma_diagnostic = arm.get("gamma_diagnostic")
                object_relaxed_match_count = arm.get("object_relaxed_match_count", 0)
                object_exact_match_count = arm.get("object_exact_match_count", 0)
                retrieved = arm["retrieved_chunk_ids"]
                n_calls = arm["n_llm_calls"]
                pt_q = arm["prompt_tokens"]
                ct_q = arm["completion_tokens"]
                lat_q = arm["latency_s"]
                gamma_status = arm.get("gamma_final_status")
                iters_used = arm.get("iterations_used", 0)
                sel_reason = (
                    f"ensemble_arbitrate:{arm.get('selected_arm', '')}:"
                    f"{arm.get('arbitrate_signal', '')}"
                )
                arm_outputs = {
                    "v3bu_pred": arm.get("v3bu_pred"),
                    "dec_pred": arm.get("dec_pred"),
                    "iter_pred": arm.get("iter_pred"),
                    "selected_arm": arm.get("selected_arm"),
                    "arbitrate_signal": arm.get("arbitrate_signal"),
                    "arm_scores": arm.get("arm_scores"),
                }
            elif not v3bu_in_subset:
                # V3+bu excluded by arm_subset (chain_deep / bridge_entity /
                # implicit-multihop signal). Mirror arbitrate_excl_v3bu logic.
                if qtype == "chain_deep":
                    arm_used = "excl_v3bu:chain_deep:iter"
                    arm = _run_iter(iter_runner, question)
                    pred = arm["pred"]
                    retrieved = arm["retrieved_chunk_ids"]
                    n_calls = arm["n_llm_calls"]
                    pt_q = arm["prompt_tokens"]
                    ct_q = arm["completion_tokens"]
                    lat_q = arm["latency_s"]
                    gamma_status = arm.get("gamma_final_status")
                    iters_used = arm.get("iterations_used", 0)
                    sel_reason = "excl_v3bu:chain-deep-iter"
                elif qtype == "bridge_entity":
                    arm_used = "excl_v3bu:bridge:decompose"
                    arm = _run_decompose(pipeline, question, args.reader_model,
                                         args.top_k_subq)
                    pred = arm["pred"]
                    retrieved = arm["retrieved_chunk_ids"]
                    n_calls = arm["n_llm_calls"]
                    pt_q = arm["prompt_tokens"]
                    ct_q = arm["completion_tokens"]
                    lat_q = arm["latency_s"]
                    gamma_status = None
                    iters_used = 0
                    sel_reason = "excl_v3bu:bridge-decompose"
                else:
                    # semantic_rich + implicit-multihop (iter1+iter2 refinement):
                    # arbitrate_excl_v3bu uses iter as primary, decompose backup.
                    # Run both, apply selective_arbitrate via arbitrate_excl_v3bu.
                    arm_used = "excl_v3bu:semrich_multihop"
                    a_de = _run_decompose(pipeline, question, args.reader_model,
                                           args.top_k_subq)
                    a_it = _run_iter(iter_runner, question)
                    chosen, sel_reason = arbitrate_excl_v3bu(
                        a_de["pred"], a_it["pred"], question, v3bu_fallback=None,
                    )
                    pred = chosen
                    retrieved = list(a_de["retrieved_chunk_ids"])
                    seen = set(retrieved)
                    for cid in a_it["retrieved_chunk_ids"]:
                        if cid not in seen:
                            retrieved.append(cid)
                            seen.add(cid)
                    n_calls = a_de["n_llm_calls"] + a_it["n_llm_calls"]
                    pt_q = a_de["prompt_tokens"] + a_it["prompt_tokens"]
                    ct_q = a_de["completion_tokens"] + a_it["completion_tokens"]
                    lat_q = a_de["latency_s"] + a_it["latency_s"]
                    gamma_status = a_it.get("gamma_final_status")
                    iters_used = a_it.get("iterations_used", 0)
            else:
                # V3+bu in subset: either polar-override (re-include over
                # chain_deep/bridge_entity) or plain semantic_rich.
                # Posthoc route_by_query_type_v2 behavior:
                #   chain_deep + polar  → iter primary, V3+bu last-resort
                #   bridge   + polar  → decompose primary, V3+bu last-resort
                #   semantic_rich     → selective_arbitrate(v3bu, decompose)
                if qtype == "chain_deep":
                    arm_used = "polar_override_chain_deep:iter"
                    arm = _run_iter(iter_runner, question)
                    pred = arm["pred"]
                    retrieved = arm["retrieved_chunk_ids"]
                    n_calls = arm["n_llm_calls"]
                    pt_q = arm["prompt_tokens"]
                    ct_q = arm["completion_tokens"]
                    lat_q = arm["latency_s"]
                    gamma_status = arm.get("gamma_final_status")
                    iters_used = arm.get("iterations_used", 0)
                    sel_reason = "router_v2:chain-deep-use-iter"
                elif qtype == "bridge_entity":
                    arm_used = "polar_override_bridge:decompose"
                    arm = _run_decompose(pipeline, question, args.reader_model,
                                         args.top_k_subq)
                    pred = arm["pred"]
                    retrieved = arm["retrieved_chunk_ids"]
                    n_calls = arm["n_llm_calls"]
                    pt_q = arm["prompt_tokens"]
                    ct_q = arm["completion_tokens"]
                    lat_q = arm["latency_s"]
                    gamma_status = None
                    iters_used = 0
                    sel_reason = "router_v2:bridge-force-decompose"
                else:  # semantic_rich plain
                    if args.mode == "strict":
                        arm_used = "v3bu"
                        arm = _run_v3bu(pipeline, question)
                        pred = arm["pred"]
                        retrieved = arm["retrieved_chunk_ids"]
                        n_calls = arm["n_llm_calls"]
                        pt_q = arm["prompt_tokens"]
                        ct_q = arm["completion_tokens"]
                        lat_q = arm["latency_s"]
                        gamma_status = None
                        iters_used = 0
                        sel_reason = "strict:v3bu"
                    else:  # mode v2 → sel_v1 (V3+bu + decompose + arbitrate)
                        arm_used = "sel_v1"
                        a_v3 = _run_v3bu(pipeline, question)
                        a_de = _run_decompose(pipeline, question, args.reader_model,
                                               args.top_k_subq)
                        chosen, sel_reason = selective_arbitrate(
                            a_v3["pred"], a_de["pred"], question)
                        pred = chosen
                        retrieved = list(a_v3["retrieved_chunk_ids"])
                        seen = set(retrieved)
                        for cid in a_de["retrieved_chunk_ids"]:
                            if cid not in seen:
                                retrieved.append(cid)
                                seen.add(cid)
                        n_calls = a_v3["n_llm_calls"] + a_de["n_llm_calls"]
                        pt_q = a_v3["prompt_tokens"] + a_de["prompt_tokens"]
                        ct_q = a_v3["completion_tokens"] + a_de["completion_tokens"]
                        lat_q = a_v3["latency_s"] + a_de["latency_s"]
                        gamma_status = None
                        iters_used = 0

            # ---- v2 mode opt-in arm composition. ----
            # When sel_v2 routed at least one opt-in arm (infobox_arm /
            # mothgraph_arm) into the subset AND that arm is wired in
            # opt_in_arms, RUN it now and arbitrate alongside the legacy
            # v2 pred. Skipped under --mode=ensemble_arbitrate (already
            # composed by _run_ensemble_arbitrate) and --mode=strict (the
            # strict baseline must not be perturbed by side composition).
            if args.mode == "v2" and opt_in_arms:
                opt_in_in_subset = [
                    a for a in subset
                    if a not in ("v3bu", "decompose", "iter")
                    and a in opt_in_arms
                ]
                if opt_in_in_subset:
                    legacy_candidate = {
                        "pred": pred,
                        "retrieved_chunk_ids": list(retrieved),
                        "n_llm_calls": int(n_calls),
                        "prompt_tokens": int(pt_q),
                        "completion_tokens": int(ct_q),
                        "latency_s": float(lat_q),
                    }
                    # Use the legacy arm_used as the candidate key so the
                    # arbitrator can name-tie-break deterministically.
                    # When arm_used carries a compound label (e.g.
                    # 'excl_v3bu:chain_deep:iter'), strip to the trailing
                    # primary arm token for stable naming.
                    legacy_key = arm_used.rsplit(":", 1)[-1] if arm_used else "legacy"
                    if legacy_key not in ("v3bu", "decompose", "iter"):
                        legacy_key = "legacy"
                    v2_candidates: dict[str, dict] = {legacy_key: legacy_candidate}

                    for arm_name in opt_in_in_subset:
                        arm_obj = opt_in_arms.get(arm_name)
                        if arm_obj is None:
                            continue
                        try:
                            if not arm_obj.applicable(question):
                                continue
                        except Exception:  # noqa: BLE001
                            continue
                        try:
                            arm_result = arm_obj.run(
                                question, reader_model=args.reader_model,
                            )
                        except Exception:  # noqa: BLE001
                            continue
                        if not arm_result.pred:
                            continue
                        # Pool-safety axiom: skip fallback-tagged results
                        # (see _run_ensemble_arbitrate opt-in loop for the
                        # full rationale and the empirical MQ F1=1 cohort
                        # regression that motivated it).
                        if arm_result.metadata.get("is_fallback"):
                            continue
                        v2_candidates[arm_name] = {
                            "pred": arm_result.pred,
                            "retrieved_chunk_ids": list(arm_result.retrieved_chunk_ids),
                            "n_llm_calls": int(arm_result.n_llm_calls),
                            "prompt_tokens": int(arm_result.prompt_tokens),
                            "completion_tokens": int(arm_result.completion_tokens),
                            "latency_s": float(arm_result.latency_s),
                        }

                    if len(v2_candidates) >= 2:
                        arb = _arbitrate_candidates(
                            pipeline,
                            candidates=v2_candidates,
                            iter_gamma_status=gamma_status,
                            arm_probabilities=None,  # sel_v2 = binary, no P_arm
                            use_gamma_aware_pdd=args.use_gamma_aware_pdd,
                            qtype=qtype,
                        )
                        pred = arb["pred"]
                        pdd_active = arb.get("pdd_active", False)
                        pdd_skipped_chain_deep_valid = arb.get("pdd_skipped_chain_deep_valid", False)
                        pdd_preserved_semantic_rich = arb.get("pdd_preserved_semantic_rich", False)
                        retrieved = arb["retrieved_chunk_ids"]
                        n_calls = arb["n_llm_calls"]
                        pt_q = arb["prompt_tokens"]
                        ct_q = arb["completion_tokens"]
                        lat_q = arb["latency_s"]
                        sel_reason = (
                            f"{sel_reason}+v2_opt_in:{arb['selected_arm']}:"
                            f"{arb['arbitrate_signal']}"
                        )
                        arm_outputs["v2_opt_in_arms"] = list(v2_candidates.keys())
                        arm_outputs["v2_arbitrate_selected"] = arb["selected_arm"]
                        arm_outputs["v2_arbitrate_signal"] = arb["arbitrate_signal"]
                        arm_outputs["v2_arm_scores"] = arb["arm_scores"]

            # ---- Optional retry-on-abstain escalation wrapper.
            # Fires only when --retry-strategies is set AND the chosen pred
            # carries an abstention signal (gamma_status invalid /
            # uncertainty-template chosen). Default: no-op (zero regression).
            esc_meta: dict = {}
            if args.retry_strategies:
                pred_after, esc_meta = _maybe_run_escalation(
                    pipeline=pipeline, iter_runner=iter_runner,
                    question=question, reader_model=args.reader_model,
                    top_k_subq=args.top_k_subq,
                    pred=pred, gamma_status=gamma_status,
                    arm_outputs=arm_outputs, args=args,
                )
                if pred_after != pred:
                    pred = pred_after

            em, f1 = best_em_f1(pred, gold_list)
            r10 = recall_at_k(retrieved, gold_doc_ids, 10) if gold_doc_ids else float("nan")
            row = {
                "qid": qid,
                "question": question,
                "gold": gold,
                "gold_aliases": gold_aliases,
                "pred": pred,
                "em": em,
                "f1": f1,
                "r_at_10": r10,
                "qtype": qtype,
                "qtype_v1": feat["label"],
                "n_entities": feat["n_entities"],
                "n_relations": feat["n_relations"],
                "has_chain": feat["has_chain"],
                "n_tokens": feat["n_tokens"],
                "np_depth": feat["np_depth"],
                "arm_used": arm_used,
                "sel_reason": sel_reason,
                "n_llm_calls": n_calls,
                "prompt_tokens": pt_q,
                "completion_tokens": ct_q,
                "latency_s": lat_q,
                "iterations_used": iters_used,
                "gamma_final_status": gamma_status,
                # γ-aware gating + PDD cohort-gated: per-q γ-aware PDD decision
                # (3 mutually-exclusive-at-fire counters).
                "pdd_active": pdd_active,
                "pdd_skipped_chain_deep_valid": pdd_skipped_chain_deep_valid,
                "pdd_preserved_semantic_rich": pdd_preserved_semantic_rich,
                # FIX A — ChainFilter cohort gate per-q.
                "chainfilter_active": chainfilter_active,
                "chainfilter_skipped_chain_deep": chainfilter_skipped_chain_deep,
                # FIX B — γ-aware bridge cohort per-q.
                "bridge_active": bridge_active,
                "bridge_skipped_easy_semantic_rich":
                    bridge_skipped_easy_semantic_rich,
                # FIX C — adaptive retrigger cap per-q.
                "gamma_retrigger_cap_used": gamma_retrigger_cap_used,
                # FIX D — faithfulness γ-coord per-q
                # (two safe-skip branches).
                "faithfulness_active": faithfulness_active,
                "faithfulness_skipped_clean_valid":
                    faithfulness_skipped_clean_valid,
                "faithfulness_skipped_exhausted_safe":
                    faithfulness_skipped_exhausted_safe,
                # Final-iter proof tree (None off-flag).
                "gamma_diagnostic": gamma_diagnostic,
                # Object grounding match-mode per-q counts.
                "object_relaxed_match_count": object_relaxed_match_count,
                "object_exact_match_count": object_exact_match_count,
                # Telemetry (zero-cost when flags not set)
                "production_mode": args.mode,
                "ensemble_selected_arm": arm_outputs.get("selected_arm"),
                "ensemble_arbitrate_signal": arm_outputs.get("arbitrate_signal"),
                "ensemble_arm_scores": arm_outputs.get("arm_scores"),
                "escalation_applied": esc_meta.get("escalation_applied"),
                "escalation_recovered_by": esc_meta.get("escalation_recovered_by"),
                "escalation_budget_used": esc_meta.get("escalation_budget_used"),
                "final_answer_confidence": esc_meta.get("final_answer_confidence"),
                "terminal_abstain": esc_meta.get("terminal_abstain"),
                "original_abstention_signal": esc_meta.get("original_abstention_signal"),
                # Per-query router decision telemetry.
                "retrieval_mode": args.retrieval,
                "infobox_fired": infobox_fired,
                "infobox_router_reason": infobox_router_reason,
                # Bridge substrate per-query decision.
                "bridge_substrate": args.use_bridge_substrate,
                "bridge_fired": bridge_prep.get("fired", False),
                "bridge_reason": bridge_prep.get("reason"),
                "bridge_passage_id": bridge_prep.get("bridge_passage_id"),
                # Qtype-gate per-query decision.
                "bridge_qtype_skipped": bridge_prep.get("bridge_qtype_skipped",
                                                        False),
                "bridge_qtype": bridge_prep.get("bridge_qtype"),
                # Per-arm bridge decision map (empty unless a per-arm gate is
                # active).
                "bridge_arm_decisions": bridge_prep.get("bridge_arm_decisions", {}),
            }
            per_q.append(row)
        except Exception as exc:  # noqa: BLE001
            # Keep the last frames so a per-query failure is diagnosable from the
            # JSON alone (the Groq-block + gemini-shape incidents cost hours
            # because this row only carried the bare exception repr).
            tb_full = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            tb_tail = (tb_full if len(tb_full) <= 4000
                       else tb_full[:3000] + "\n...[snip]...\n" + tb_full[-900:])
            per_q.append({
                "qid": qid, "question": question, "gold": gold,
                "pred": "", "em": 0.0, "f1": 0.0,
                "qtype": qtype, "arm_used": arm_used,
                "error": f"{type(exc).__name__}: {exc}",
                "error_tb": tb_tail,
            })
            print(f"[q{i + 1}] ERROR {type(exc).__name__}: {exc}", flush=True)

        if (i + 1) % args.checkpoint_every == 0 or (i + 1) == len(queries):
            elapsed = time.time() - t0
            done = [r for r in per_q if "error" not in r]
            n_done = len(done)
            em_avg = sum(r["em"] for r in done) / max(n_done, 1)
            f1_avg = sum(r["f1"] for r in done) / max(n_done, 1)
            r10s = [r["r_at_10"] for r in done if not (r["r_at_10"] != r["r_at_10"])]
            r10_avg = sum(r10s) / len(r10s) if r10s else float("nan")
            arms = Counter(r["arm_used"] for r in done)
            print(f"[{i+1}/{len(queries)}] EM={em_avg:.3f} F1={f1_avg:.3f} "
                  f"R@10={r10_avg:.3f} arms={dict(arms)} elapsed={elapsed:.0f}s",
                  flush=True)
            _write_partial(args.out, per_q, args, partial=(i + 1 < len(queries)),
                           per_arm_gates=per_arm_gates)

    # Final summary
    done = [r for r in per_q if "error" not in r]
    n_done = len(done)
    em_avg = sum(r["em"] for r in done) / max(n_done, 1)
    f1_avg = sum(r["f1"] for r in done) / max(n_done, 1)
    r10s = [r["r_at_10"] for r in done if not (r["r_at_10"] != r["r_at_10"])]
    r10_avg = sum(r10s) / len(r10s) if r10s else float("nan")
    arm_dist = Counter(r["arm_used"] for r in done)
    qtype_dist = Counter(r["qtype"] for r in done)
    iter_dist = Counter(r["iterations_used"] for r in done)
    total_cost = sum(r["n_llm_calls"] for r in done)
    avg_cost = total_cost / max(n_done, 1)
    print(f"\n=== FINAL n={n_done} EM={em_avg:.3f} F1={f1_avg:.3f} "
          f"R@10={r10_avg:.3f} mode={args.mode} ===")
    print(f"qtype:    {dict(qtype_dist)}")
    print(f"arms:     {dict(arm_dist)}")
    print(f"iter dist:{dict(iter_dist)}")
    print(f"cost:     total_calls={total_cost} avg_calls/q={avg_cost:.2f}")
    if bridge_substrate is not None:
        bs = bridge_substrate.stats()
        n_bridge = sum(1 for r in done if r.get("bridge_fired"))
        print(f"bridge:   substrate ON | fired={n_bridge}/{n_done} q | "
              f"runs={bs['n_bridge_runs']} fallback={bs['n_fallback']} "
              f"cost_capped={bs['n_cost_capped']} "
              f"est_cost=${bs['total_cost_usd']}/{bs['max_cost_usd']}")
    _write_partial(args.out, per_q, args, partial=False,
                   per_arm_gates=per_arm_gates)
    return 0


def _ds_name_from_args(args) -> str:
    """Dataset label from the --data-dir basename.
    Anti-leak: the run is ALREADY single-DS; this is a telemetry LABEL only, never
    a per-DS behaviour switch (one tech, one config cross-DS)."""
    import os
    try:
        base = os.path.basename(str(getattr(args, "data_dir", "")).rstrip("/\\"))
        return base or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _gamma_valid_rate_by(per_q, *, key=None, const=None):
    """γ-valid yield grouped by per-q ``key`` (e.g. ``"qtype"``) OR bucketed under a
    single ``const`` label (e.g. the DS name). Returns
    ``{bucket: {"valid": v, "total": t, "rate": v/t}}``. A row counts as valid iff
    ``gamma_final_status == "valid"``; rows lacking a γ status still count in the
    denominator (so the rate is honest about cascade failures)."""
    from collections import defaultdict
    agg: dict = defaultdict(lambda: [0, 0])  # bucket -> [valid, total]
    for r in per_q:
        bucket = const if const is not None else r.get(key)
        if bucket is None:
            bucket = "unknown"
        agg[bucket][1] += 1
        if r.get("gamma_final_status") == "valid":
            agg[bucket][0] += 1
    return {b: {"valid": v, "total": t, "rate": (v / t if t else 0.0)}
            for b, (v, t) in agg.items()}


def _write_partial(out_path: Path, per_q: list[dict], args, *, partial: bool,
                   per_arm_gates: dict | None = None) -> None:
    # HOTFIX: telemetry counters live in caller frame (main eval loop scope),
    # not _write_partial scope. Use frame introspection to grab real values
    # (NOT zero-fallback). Reversible: a later refactor should pass these as
    # proper kwargs.
    # HOTFIX: per_arm_gates is now an EXPLICIT kwarg (default None) — it was
    # previously referenced at the config-write site without being in scope,
    # which raised NameError at the q20 checkpoint and crashed every Phase 1
    # smoke test with an empty JSON on disk. Passed from both call sites in
    # main(); defaults to None for any other caller.
    import sys as _sys
    _caller_locals = _sys._getframe(1).f_locals
    cfde114_fire_count = _caller_locals.get('cfde114_fire_count', 0)
    hop_multiplier_active_count = _caller_locals.get('hop_multiplier_active_count', 0)
    non_unitary_p_arm_count = _caller_locals.get('non_unitary_p_arm_count', 0)
    # Bridge substrate handle lives in the main loop frame.
    _bridge_sub = _caller_locals.get('bridge_substrate', None)
    done = [r for r in per_q if "error" not in r]
    n_done = len(done)
    em_avg = sum(r["em"] for r in done) / max(n_done, 1)
    f1_avg = sum(r["f1"] for r in done) / max(n_done, 1)
    r10s = [r["r_at_10"] for r in done if not (r["r_at_10"] != r["r_at_10"])]
    r10_avg = sum(r10s) / len(r10s) if r10s else None
    iter_done = [r for r in done if r.get("iterations_used", 0) > 0]
    it_avg = (sum(r["iterations_used"] for r in iter_done) / len(iter_done)
              if iter_done else 0.0)
    arm_dist = Counter(r["arm_used"] for r in done)
    qtype_dist = Counter(r["qtype"] for r in done)
    total_cost = sum(r["n_llm_calls"] for r in done)
    avg_cost = total_cost / max(n_done, 1)
    out = {
        "summary": {
            "n": n_done,
            "em": round(em_avg, 4),
            "f1": round(f1_avg, 4),
            "r_at_10": round(r10_avg, 4) if r10_avg is not None else None,
            "mean_iterations_used": round(it_avg, 3),
            "arm_distribution": dict(arm_dist),
            "qtype_distribution": dict(qtype_dist),
            "total_cost_units": int(total_cost),
            "avg_cost_per_query": round(avg_cost, 3),
            "partial": partial,
            "config": {
                "mode": args.mode,
                "retrieval": args.retrieval,
                "infobox_top_n_boost": args.infobox_top_n_boost,
                "arms_pool": args.arms_pool,
                "mothgraph_max_iters": args.mothgraph_max_iters,
                "mothgraph_base_depth": args.mothgraph_base_depth,
                "mothgraph_top_k": args.mothgraph_top_k,
                "mothgraph_use_spacy": args.mothgraph_use_spacy,
                "router": args.router,
                "pam_lite_threshold": args.pam_lite_threshold,
                "data_dir": str(args.data_dir),
                "queries": str(args.queries),
                "reader_model": args.reader_model,
                "reader_base_url": args.reader_base_url,
                "embedding": args.embedding,
                "reranker": args.reranker,
                "bottom_up_boost": args.bottom_up_boost,
                "top_k_chunks": args.top_k_chunks,
                "top_k_subq": args.top_k_subq,
                "max_iterations": args.max_iterations,
                "top_k_total": args.top_k_total,
                "stop_early": args.stop_early,
                "final_reader_model": args.final_reader_model,
                "final_reader_provider": args.final_reader_provider,
                "final_reader_base_url": args.final_reader_base_url,
                "final_reader_max_tokens": args.final_reader_max_tokens,
                "use_graph_aware_iter": args.use_graph_aware_iter,
                "max_accumulated_entities": args.max_accumulated_entities,
                "use_faithfulness_loop": args.use_faithfulness_loop,
                "judge_model": args.judge_model,
                "judge_provider": args.judge_provider,
                "judge_min_score": args.judge_min_score,
                "max_judge_iterations": args.max_judge_iterations,
                "use_gamma_verifier": args.use_gamma_verifier,
                "use_gamma_liberal": args.use_gamma_liberal,
                "gamma_max_retrigger": args.gamma_max_retrigger,
                "gamma_prompt_variant": args.gamma_prompt_variant,
                "use_gamma_router": args.use_gamma_router,
                "use_c7_iter": args.use_c7_iter,
                "c7_iter_trigger": args.c7_iter_trigger,
                "c7_iter_embedder": args.c7_iter_embedder if args.use_c7_iter else None,
                "ner_cache": args.ner_cache,
                "reader_n_samples": args.reader_n_samples,
                "reader_temperature": args.reader_temperature,
                # Arbitrator + ablation config provenance
                "arbitrator": args.arbitrator,
                "arbitrator_mode": args.arbitrator_mode,
                "disable_cfde114_boost": args.disable_cfde114_boost,
                "disable_hop_multipliers": args.disable_hop_multipliers,
                # γ-aware coordination flags (PDD gating wired; bridge-gating
                # pending architecture decision).
                "use_gamma_aware_pdd": getattr(args, "use_gamma_aware_pdd", False),
                # 4-fix bundle provenance (all default OFF).
                "use_chainfilter_cohort_gate":
                    getattr(args, "use_chainfilter_cohort_gate", False),
                "use_gamma_aware_bridge_cohort":
                    getattr(args, "use_gamma_aware_bridge_cohort", False),
                "use_adaptive_gamma_retrigger":
                    getattr(args, "use_adaptive_gamma_retrigger", False),
                "use_faithfulness_gamma_coord":
                    getattr(args, "use_faithfulness_gamma_coord", False),
                # Bridge substrate provenance.
                "use_bridge_substrate": getattr(args, "use_bridge_substrate", False),
                "bridge_substrate_scope": getattr(args, "bridge_substrate_scope", None),
                "bridge_substrate_qtype_gate": getattr(args, "bridge_substrate_qtype_gate", None),
                # The parsed per-arm gate map (None unless set).
                "arm_bridge_qtype_gate": (per_arm_gates if per_arm_gates else None),
                "bridge_judge_model": getattr(args, "bridge_judge_model", None),
                "bridge_judge_provider": getattr(args, "bridge_judge_provider", None),
                "bridge_max_cost_usd": getattr(args, "bridge_max_cost_usd", None),
                # iter-ragnatela provenance.
                "use_iter_ragnatela": getattr(args, "use_iter_ragnatela", False),
                # ChainFilter v0.1 pre-wire provenance.
                "use_chainfilter": getattr(args, "use_chainfilter", False),
                "chainfilter_hop_min": getattr(args, "chainfilter_hop_min", None),
                "chainfilter_top_out": getattr(args, "chainfilter_top_out", None),
                "iter_ragnatela_kmax": getattr(args, "iter_ragnatela_kmax", None),
                # M8 specialist slot-router provenance.
                "use_m8_specialists": getattr(args, "use_m8_specialists", False),
            },
            # Bridge substrate run stats (None when OFF).
            "bridge_substrate": (_bridge_sub.stats() if _bridge_sub is not None
                                 else None),
            # Telemetry counters mirror arbitrate_post summary (commit 007869c).
            # Useful for cross-script comparison of cfde114/hop firing rates
            # across LIVE vs cached eval paths.
            "telemetry": {
                "cfde114_fire_count": cfde114_fire_count,
                "hop_multiplier_active_count": hop_multiplier_active_count,
                "non_unitary_p_arm_count": non_unitary_p_arm_count,
            },
            # FIX B — aggregate γ-aware PDD counters (sum of the per-q
            # booleans). Cohort analysis reads these summary totals; zero across
            # the board when --use-gamma-aware-pdd is off (legacy) since no per-q
            # counter fires.
            "counters": {
                "pdd_active_total":
                    sum(int(r.get("pdd_active", 0)) for r in per_q),
                "pdd_skipped_chain_deep_valid_total":
                    sum(int(r.get("pdd_skipped_chain_deep_valid", 0)) for r in per_q),
                "pdd_preserved_semantic_rich_total":
                    sum(int(r.get("pdd_preserved_semantic_rich", 0)) for r in per_q),
                # Aggregate counters (sum of per-q booleans; all zero when the
                # respective flag is OFF). FIX C is a cap distribution rather
                # than a sum.
                "chainfilter_active_total":
                    sum(int(r.get("chainfilter_active", 0)) for r in per_q),
                "chainfilter_skipped_chain_deep_total":
                    sum(int(r.get("chainfilter_skipped_chain_deep", 0)) for r in per_q),
                "bridge_active_total":
                    sum(int(r.get("bridge_active", 0)) for r in per_q),
                "bridge_skipped_easy_semantic_rich_total":
                    sum(int(r.get("bridge_skipped_easy_semantic_rich", 0))
                        for r in per_q),
                "faithfulness_active_total":
                    sum(int(r.get("faithfulness_active", 0)) for r in per_q),
                # FIX D — two split safe-skip counters.
                "faithfulness_skipped_clean_valid_total":
                    sum(int(r.get("faithfulness_skipped_clean_valid", 0))
                        for r in per_q),
                "faithfulness_skipped_exhausted_safe_total":
                    sum(int(r.get("faithfulness_skipped_exhausted_safe", 0))
                        for r in per_q),
                "gamma_retrigger_cap_distribution":
                    dict(Counter(r["gamma_retrigger_cap_used"] for r in per_q
                                 if r.get("gamma_retrigger_cap_used") is not None)),
                # γ-valid yield broken down by cohort and by dataset (data-dir
                # proxy) — the diagnostic for the MQ γ-valid-38% cascade
                # root-cause.
                "gamma_valid_rate_per_qtype":
                    _gamma_valid_rate_by(per_q, key="qtype"),
                "gamma_valid_rate_per_DS":
                    _gamma_valid_rate_by(
                        per_q, const=_ds_name_from_args(args)),
                # Object grounding match-mode totals.
                "object_relaxed_match_total":
                    sum(int(r.get("object_relaxed_match_count", 0)) for r in per_q),
                "object_exact_match_total":
                    sum(int(r.get("object_exact_match_count", 0)) for r in per_q),
            },
        },
        "per_question": per_q,
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
