# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Unified arm runner (run_arms + arbitrate_pool).

Pins the vetted design (all 3 adversarial lenses PASS):
dup = COPY not recompute; parallel ≡ serial (deterministic gather); arbitration
is insertion-order-invariant; pool-safety (dup follows base, N=4 worker clamp);
the shutdown(wait=True) barrier; the --arms-serial escape hatch; generic over
the arm result type (pip str / eval dict).
"""
from __future__ import annotations

import math

import pytest

from mothrag.core import arms_runner as ar
from mothrag.core.arms_runner import ArmSpec, arbitrate_pool, run_arms


# ---- stub embedder for pairwise_agreement (identical text → cosine 1.0) -----

class _StubEmbedder:
    def embed_batch(self, texts):
        def vec(t):
            v = [0.0] * 26
            for ch in (t or "").lower():
                if "a" <= ch <= "z":
                    v[ord(ch) - 97] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            return [x / n for x in v]
        return [vec(t) for t in texts]


def _spec(name, val, *, dup_of=None, counter=None):
    if dup_of is not None:
        return ArmSpec(name=name, fn=None, is_dup=True, dup_of=dup_of)

    def fn():
        if counter is not None:
            counter[name] = counter.get(name, 0) + 1
        return val
    return ArmSpec(name=name, fn=fn)


# ---- 1. dup is a COPY, base run exactly once (pip str + eval dict) -----------

def test_dup_is_copy_not_recompute_pip_str():
    counter: dict = {}
    specs = [_spec("v3bu", "A", counter=counter), _spec("decompose", "B", counter=counter),
             _spec("iter", "Paris", counter=counter),
             _spec("iter_dup_a", None, dup_of="iter")]
    out = run_arms(specs, parallel=False)
    assert counter["iter"] == 1                    # base run ONCE (not recomputed)
    assert out["iter_dup_a"] == out["iter"] == "Paris"
    assert out["iter_dup_a"] is out["iter"]        # immutable str → same object


def test_dup_is_copy_eval_dict_with_metadata_reinject():
    base = {"pred": "Paris", "retrieved_chunk_ids": ["c1"]}
    specs = [ArmSpec("iter", fn=lambda: base),
             ArmSpec("iter_dup_a", fn=None, is_dup=True, dup_of="iter")]

    def _copy_with_meta(d):
        out = dict(d)
        meta = dict(out.get("metadata") or {})
        meta["dup_of"] = "iter"
        meta["dup_arm_id"] = "iter_dup_a"
        out["metadata"] = meta
        return out

    out = run_arms(specs, parallel=False, copy_fn=_copy_with_meta)
    assert out["iter_dup_a"]["pred"] == "Paris"
    assert out["iter_dup_a"] is not out["iter"]              # dict copy, distinct
    assert out["iter_dup_a"]["metadata"]["dup_of"] == "iter"  # metadata re-injected
    assert out["iter_dup_a"]["metadata"]["dup_arm_id"] == "iter_dup_a"


# ---- 2. parallel ≡ serial (identical key order + values) --------------------

def test_parallel_equals_serial():
    specs = [_spec("v3bu", "A"), _spec("decompose", "B"), _spec("iter", "C"),
             _spec("iter_dup_a", None, dup_of="iter")]
    par = run_arms(specs, parallel=True)
    ser = run_arms(specs, parallel=False)
    assert list(par.keys()) == list(ser.keys()) == ["v3bu", "decompose", "iter", "iter_dup_a"]
    assert par == ser


# ---- 3. arbitration is insertion-order-invariant ----------------------------

def test_arbitrate_pool_order_invariance():
    import itertools
    emb = _StubEmbedder()
    base = {"v3bu": "Paris", "decompose": "Paris", "iter": "Lyon", "iter_dup_a": "Lyon"}
    selected, signals = set(), set()
    for perm in itertools.permutations(base.items()):
        results = dict(perm)
        r = arbitrate_pool(results, pred_of=lambda x: x, embedder=emb,
                           iter_gamma_status="valid")
        selected.add(r.selected_arm)
        signals.add(r.arbitrate_signal)
        # arm_scores compared as a sorted tuple → order-independent
        assert tuple(sorted(r.arm_scores.items())) == tuple(sorted(
            arbitrate_pool(base, pred_of=lambda x: x, embedder=emb,
                           iter_gamma_status="valid").arm_scores.items()))
    assert len(selected) == 1 and len(signals) == 1   # identical across all 24 orders


# ---- 4. dup double-counts cross-arm agreement ------

def test_dup_double_counts_agreement():
    from mothrag.core.arbitrate import pairwise_agreement
    answers = {"v3bu": "Apple", "decompose": "Banana",
               "iter": "Paris", "iter_dup_a": "Paris"}
    agr = pairwise_agreement(answers, embedder=_StubEmbedder(), threshold=0.70)
    assert agr["iter"] > 0 and agr["iter_dup_a"] > 0     # mutual cosine=1.0 match
    assert agr["v3bu"] == 0 and agr["decompose"] == 0    # no agreement partners


# ---- 5. pool-safety: a dup whose base was EXCLUDED is skipped ----------------

def test_pool_safety_dup_skipped_when_base_excluded():
    # v3bu excluded by the (simulated) subset → not in specs to run; its dup must
    # NOT appear either.
    specs = [_spec("decompose", "B"), _spec("iter", "C"),
             ArmSpec("v3bu_dup_a", fn=None, is_dup=True, dup_of="v3bu")]
    out = run_arms(specs, parallel=False)
    assert "v3bu" not in out and "v3bu_dup_a" not in out
    assert set(out) == {"decompose", "iter"}


# ---- 6. worker count clamped to min(4, n_real) ------------------------------

def test_max_workers_clamped_to_4(monkeypatch):
    seen = {}
    real_TPE = ar.ThreadPoolExecutor

    class _SpyTPE(real_TPE):
        def __init__(self, *a, max_workers=None, **k):
            seen["max_workers"] = max_workers
            super().__init__(*a, max_workers=max_workers, **k)

    monkeypatch.setattr(ar, "ThreadPoolExecutor", _SpyTPE)
    specs = [_spec(f"a{i}", str(i)) for i in range(5)]   # 5 real arms
    run_arms(specs, parallel=True, max_workers=8)        # request 8
    assert seen["max_workers"] == 4                       # clamped to the N=4 ceiling


# ---- 8. shutdown(wait=True) barrier: iter-thread writes visible after return -

def test_last_iter_meta_barrier_safe():
    state: dict = {}

    def iter_fn():
        state["meta"] = {"hit_cap": False, "steps_used": 2}
        return "C"

    specs = [_spec("v3bu", "A"), _spec("decompose", "B"),
             ArmSpec("iter", fn=iter_fn)]
    run_arms(specs, parallel=True)
    assert state["meta"] == {"hit_cap": False, "steps_used": 2}  # barrier ⇒ visible


# ---- 9. --arms-serial never touches the pool --------------------------------

def test_arms_serial_forces_sequential(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("ThreadPoolExecutor must not be constructed when serial")
    monkeypatch.setattr(ar, "ThreadPoolExecutor", _boom)
    specs = [_spec("v3bu", "A"), _spec("decompose", "B"), _spec("iter", "C")]
    out = run_arms(specs, parallel=False)
    assert out == {"v3bu": "A", "decompose": "B", "iter": "C"}


# ---- 10. generic over result type (pip str + eval dict) ---------------------

def test_generic_over_result_type():
    emb = _StubEmbedder()
    # pip str shape
    str_out = run_arms([_spec("iter", "Paris")], parallel=False)
    r1 = arbitrate_pool(str_out, pred_of=lambda x: x, embedder=emb)
    assert r1.selected_arm == "iter"
    # eval dict shape
    dict_out = run_arms([ArmSpec("iter", fn=lambda: {"pred": "Paris"})], parallel=False,
                        copy_fn=dict)
    r2 = arbitrate_pool(dict_out, pred_of=lambda d: d.get("pred", ""), embedder=emb)
    assert r2.selected_arm == "iter"


# ---- single real arm auto-degrades to serial (no pool) ----------------------

def test_single_arm_no_pool(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("no pool for a single arm")
    monkeypatch.setattr(ar, "ThreadPoolExecutor", _boom)
    out = run_arms([_spec("iter", "C")], parallel=True)   # 1 real arm
    assert out == {"iter": "C"}
