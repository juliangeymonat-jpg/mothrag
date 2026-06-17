# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Single source-of-truth arm execution + pooled arbitration.

Unifies the arm-dispatch + arbitration logic shared by the pip ``api.py``
production path, the eval ``scripts/route_prospective.py`` path, and
``mothrag/eval/pipeline.py`` — so the three cannot drift. Two pieces:

  * :func:`run_arms` — execute the NON-dup arms (parallel via a within-query
    ThreadPoolExecutor, or serial), then materialise dup arms (e.g.
    ``iter_dup_a``, the PDD 4th arm) by COPYING their base arm's already-computed
    result. A dup is NEVER recomputed (per ``mothrag/routing/dup_arm.py``: a dup
    "executes the BASE arm's code path, identical inputs → identical
    predictions"); the eval path does ``dict(a_it)`` and this generalises it.
    Re-running the base would waste a call and, under any reader non-determinism,
    diverge the dup from its base.
  * :func:`arbitrate_pool` — the γ + cross-arm-agreement + P_arm scoring core
    (the same one ``_arbitrate_candidates`` uses): build ``answers`` from the
    results, derive the iter γ signal, compute ``pairwise_agreement`` (threshold
    0.70), and run :class:`DeterministicArbitrator`. Generic over the per-arm
    result type ``T`` via ``pred_of`` (identity for pip ``str`` arms, ``d["pred"]``
    for eval ``dict`` arms).

POOL-SAFETY (N=4 LOCKED): ``run_arms`` clamps the executor to
``min(4, n_real_arms)`` — never more than 4 threads, even if a future pool grows
(oversubscription / cache-thrash guard). Dups add NO thread.

THREAD-SAFETY (verified): the parallel region is
safe because (a) of the canonical 4-arm pool, exactly ONE arm (``iter``) touches
``self.retriever`` / ``self.embedder`` inside the pool — ``v3bu`` / ``decompose``
are reader-only; (b) ``self.config`` and any ``_last_iter_meta`` are written only
OUTSIDE the pool or by that single ``iter`` thread, and the
``ThreadPoolExecutor`` context-manager's ``shutdown(wait=True)`` is a hard
barrier before :func:`run_arms` returns (single-writer/single-reader, no lock).
Any NEW arm that calls ``self.retriever`` / ``self.embedder`` inside the pool, or
any move of config-mutation into the pool, breaks (a)/(b) and needs a lock.

Anti-leak: operates on arm predictions + cross-arm agreement only; no gold/DS.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Generic, Optional, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# LOCKED N=4 pool-safety axiom — the within-query executor ceiling.
POOL_SAFETY_MAX_WORKERS = 4

ArmFn = Callable[[], T]


@dataclass(frozen=True)
class ArmSpec(Generic[T]):
    """One arm to dispatch. ``fn`` is a zero-arg thunk closing over the query
    context (question / passages / q_emb / top_k / reader / retriever).

    A DUP arm (``is_dup=True``, ``dup_of=<base>``) carries ``fn=None`` — it is
    never executed; :func:`run_arms` fills it by copying ``dup_of``'s result.
    """

    name: str
    fn: Optional[ArmFn[T]] = None
    is_dup: bool = False
    dup_of: Optional[str] = None


def run_arms(
    specs: Sequence[ArmSpec[T]],
    *,
    parallel: bool = True,
    max_workers: int = POOL_SAFETY_MAX_WORKERS,
    copy_fn: Optional[Callable[[T], T]] = None,
) -> "dict[str, T]":
    """Execute the real arms then materialise dups by copy. Returns an
    insertion-ordered ``{arm_name: result}`` whose key order is DETERMINISTIC =
    the input ``specs`` order (independent of thread scheduling), so parallel and
    serial yield identical dicts.

    ``copy_fn`` copies a base result for a dup (``dict`` for eval dict-results,
    identity for pip ``str``; an immutable ``str`` dup IS the base object →
    guaranteed identical). A dup whose base did NOT run is SKIPPED (pool-safety:
    "a dup cannot fire when its base didn't").
    """
    _copy = copy_fn or (lambda x: x)
    real = [s for s in specs if not s.is_dup]
    n_real = len(real)
    # POOL-SAFETY: never exceed 4 threads, never exceed the real-arm count.
    workers = min(POOL_SAFETY_MAX_WORKERS, max(1, int(max_workers)), max(1, n_real))

    results: "dict[str, T]" = {}
    if parallel and n_real > 1 and workers > 1:
        # Within-query pool. Deterministic gather: collect by iterating the
        # submitted list IN SPEC ORDER (.result() blocks), NOT by completion.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            submitted = [(s.name, pool.submit(s.fn)) for s in real]  # type: ignore[arg-type]
            for name, fut in submitted:
                results[name] = fut.result()
        # context-manager exit → shutdown(wait=True): hard barrier before return.
    else:
        for s in real:
            results[s.name] = s.fn()  # type: ignore[misc]

    # Phase 2 — materialise dups by COPY of their base (deterministic, no fn call).
    out: "dict[str, T]" = {}
    for s in specs:
        if s.is_dup:
            if s.dup_of in results:
                out[s.name] = _copy(results[s.dup_of])
            # else: base excluded → dup skipped (N=4 axiom).
        elif s.name in results:
            out[s.name] = results[s.name]
    return out


def _is_dup_arm(name: str) -> bool:
    """Canonical dup-arm predicate (mirrors ``routing.dup_arm.is_dup_arm``).

    Lazy import keeps ``arms_runner`` import-light and avoids any cycle; the
    fallback matches the same defensive pattern used in
    ``route_prospective.py`` (``"_dup_" in name``) so a missing module never
    silently misclassifies a dup as a real arm.
    """
    try:
        from mothrag.routing.dup_arm import is_dup_arm
        return bool(is_dup_arm(name))
    except Exception:  # noqa: BLE001 — never let classification break arbitration
        return "_dup_" in name


def gamma_aware_pdd_should_skip(
    results: "dict[str, T]",
    iter_gamma_status: Optional[str],
    qtype: Optional[str] = None,
    *,
    enabled: bool,
) -> bool:
    """Pure predicate for γ-aware PDD gating.

    Returns ``True`` iff the dup (PDD) arms should be DROPPED from this query's
    arbitration pool, i.e. ALL of:
      * the ``--use-gamma-aware-pdd`` flag is on (``enabled``),
      * the iter arm is γ=``"valid"`` (iter confident — its signal-dup
        amplification via pairwise agreement is then redundant and can tip the
        arbitrator toward iter even when another arm holds the correct answer),
      * the query IS ``chain_deep`` — a later evaluation reversed the v1 cohort
        gate: the ensemble vote is
        NOISE on deep multi-hop chains when iter is already γ-confident, so the
        dup is dropped there; but the semantic_rich BULK genuinely benefits from
        PDD (v1 dropped it on ~174/200 semantic_rich queries → MQ -3.6pp / 2W
        -3.1pp), so the dup is PRESERVED on every non-chain_deep cohort. ``qtype``
        is the input-feature label (``classify_query_v2`` / ``label_v2``); ``None``
        (pip path / unclassified) is treated as non-chain_deep ⇒ PDD preserved,
        the safe legacy 4-arm behaviour for callers that don't classify.
      * at least one dup arm is actually present to drop.

    No side effects — both :func:`arbitrate_pool` (to decide the drop) and the
    eval telemetry call site share THIS one predicate so they cannot drift.
    Flag OFF / iter γ≠valid / non-chain_deep ⇒ ``False`` ⇒ legacy 4-arm pool kept.
    """
    if not enabled:
        return False
    if iter_gamma_status != "valid":
        return False
    if qtype != "chain_deep":
        return False
    return any(_is_dup_arm(n) for n in results)


def arbitrate_pool(
    results: "dict[str, T]",
    *,
    pred_of: Callable[[T], str],
    embedder,
    iter_gamma_status: Optional[str] = None,
    arm_probabilities: Optional[dict] = None,
    w_gamma: float = 1.0,
    w_agree: float = 0.5,
    w_faith: float = 0.3,
    simulate_n_cap: Optional[int] = None,
    use_gamma_aware_pdd: bool = False,
    qtype: Optional[str] = None,
):
    """Pooled arbitration over arm results — the shared scoring core.

    Byte-identical to ``scripts/route_prospective.py:_arbitrate_candidates``'s
    core: ``answers`` from ``pred_of``, iter γ signal (invalid/partial/valid →
    0.0/0.5/1.0), ``pairwise_agreement(threshold=0.70)``, then
    :class:`DeterministicArbitrator`. The arbitrator selects argmax(score) with an
    ALPHABETICAL tie-break — INSENSITIVE to ``results`` insertion order — so
    parallel-vs-serial key ordering cannot perturb the selected arm.

    γ-aware PDD gating (default OFF): when
    ``use_gamma_aware_pdd`` is on AND the iter arm is γ=valid, the dup (PDD) arms
    are dropped from the pool BEFORE ``answers`` / agreement are built, so the
    duplicated iter vote no longer double-counts in pairwise agreement (the
    effective pool is the 3 distinct arms). OFF ⇒ unchanged. The γ-verifier
    internals are untouched — this only consumes its ``iter_gamma_status``.
    """
    from mothrag.core.arbitrate import DeterministicArbitrator, pairwise_agreement

    # γ-aware PDD: drop dup arms when iter is confident (single shared
    # predicate) ONLY on chain_deep (ensemble vote is noise there); the semantic_rich
    # bulk keeps the dup (empirically, semantic_rich needs PDD).
    if gamma_aware_pdd_should_skip(results, iter_gamma_status, qtype,
                                   enabled=use_gamma_aware_pdd):
        results = {n: r for n, r in results.items() if not _is_dup_arm(n)}

    answers = {name: (pred_of(r) or "") for name, r in results.items()}

    # γ signal from the iter arm (the only arm exposing it in production).
    gamma_signals: "dict[str, float]" = {}
    if "iter" in results and iter_gamma_status is not None:
        if iter_gamma_status == "invalid":
            gamma_signals["iter"] = 0.0
        elif iter_gamma_status == "partial":
            gamma_signals["iter"] = 0.5
        elif iter_gamma_status == "valid":
            gamma_signals["iter"] = 1.0

    try:
        agreement = pairwise_agreement(answers, embedder=embedder, threshold=0.70)
    except Exception:  # noqa: BLE001
        agreement = {k: 0.0 for k in answers}

    # Ablation parity: cap the agreement denominator.
    if simulate_n_cap is not None and simulate_n_cap >= 1:
        n_others = max(0, len(answers) - 1)
        capped_others = max(0, simulate_n_cap - 1)
        if 0 < capped_others < n_others:
            scale = n_others / float(capped_others)
            agreement = {k: min(1.0, v * scale) for k, v in agreement.items()}

    arbitrator = DeterministicArbitrator(
        w_gamma=w_gamma, w_agree=w_agree, w_faith=w_faith,
    )
    return arbitrator.arbitrate(
        answers=answers,
        gamma_signals=gamma_signals,
        agreement_signals=agreement,
        arm_probabilities=arm_probabilities,
    )


__all__ = ["ArmSpec", "run_arms", "arbitrate_pool", "gamma_aware_pdd_should_skip",
           "POOL_SAFETY_MAX_WORKERS"]
