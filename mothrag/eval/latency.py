# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Latency analysis helpers for MothRAG eval JSON files.

Computes p50/p95/p99 latency, prompt/completion token sums, throughput
(q/s, q/h), and a cost estimate based on the configured reader model.

For ensemble runs, also reports parallel (``max(branchA, branchB)``) and
sequential (``branchA + branchB``) latency.
"""

import json
import sys
from pathlib import Path
from statistics import median


# Per-1M-token cost (USD) — public pricing as of 2026-04
COST_PER_1M = {
    "gpt-4o":                       {"in": 2.50,  "out": 10.00},
    "gpt-5.4":                      {"in": 1.25,  "out": 10.00},
    "gpt-5.5":                      {"in": 5.00,  "out": 25.00},
    "claude-sonnet-4-6":            {"in": 3.00,  "out": 15.00},
    "claude-opus-4-7":              {"in": 15.00, "out": 75.00},
    "gemini-2.5-flash":             {"in": 0.075, "out": 0.30},
    "gemini-3.1-pro-preview":       {"in": 1.25,  "out": 5.00},
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"in": 0.60,  "out": 0.60},
    "llama-3.3-70b-versatile":      {"in": 0.60,  "out": 0.60},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"in": 0.20, "out": 0.20},
}


def percentile(sorted_xs: list[float], p: float) -> float:
    if not sorted_xs:
        return 0.0
    idx = int(p / 100.0 * (len(sorted_xs) - 1))
    return sorted_xs[idx]


def estimate_cost(reader_model: str, sum_in_tokens: int, sum_out_tokens: int):
    if reader_model not in COST_PER_1M:
        return None, f"unknown reader_model {reader_model}"
    pricing = COST_PER_1M[reader_model]
    cost_in = sum_in_tokens / 1_000_000 * pricing["in"]
    cost_out = sum_out_tokens / 1_000_000 * pricing["out"]
    return cost_in + cost_out, None


def analyze(in_path: str | Path, out_path: str | Path | None = None) -> dict | None:
    """Compute latency stats for a single eval JSON.

    Writes the summary to ``<in_path>_latency.json`` (or the supplied
    ``out_path``) and returns the computed dict.
    """
    in_path = str(in_path)
    d = json.load(open(in_path))
    summary = d.get("summary", {})
    items = d.get("per_question") or d.get("items") or d.get("results") or []
    if not items:
        print(f"[!] No per-question items in {in_path}", file=sys.stderr)
        return None

    n = len(items)
    latencies = [i.get("latency_s", 0.0) for i in items if i.get("latency_s")]
    prompt_toks = [i.get("prompt_tokens", 0) for i in items]
    completion_toks = [i.get("completion_tokens", 0) for i in items]

    if not latencies:
        print(f"[!] No latency_s field in items of {in_path}", file=sys.stderr)
        return None

    latencies_sorted = sorted(latencies)
    sum_in = sum(prompt_toks)
    sum_out = sum(completion_toks)
    total_latency_s = sum(latencies)

    reader_model = summary.get("reader_model", "unknown")
    cost_total, cost_err = estimate_cost(reader_model, sum_in, sum_out)

    routes: dict[str, list[float]] = {}
    for i in items:
        route = i.get("route") or i.get("pipeline_route") or "default"
        routes.setdefault(route, []).append(i.get("latency_s", 0.0))
    route_p50 = {r: median(sorted(ls)) if ls else 0 for r, ls in routes.items() if ls}

    out = {
        "input_file": in_path,
        "n_questions": n,
        "reader_model": reader_model,
        "latency_s": {
            "mean": sum(latencies) / len(latencies),
            "p50": percentile(latencies_sorted, 50),
            "p95": percentile(latencies_sorted, 95),
            "p99": percentile(latencies_sorted, 99),
            "min": latencies_sorted[0],
            "max": latencies_sorted[-1],
            "total_compute_s": total_latency_s,
        },
        "tokens": {
            "prompt_total": sum_in,
            "completion_total": sum_out,
            "prompt_mean": sum_in / n if n else 0,
            "completion_mean": sum_out / n if n else 0,
        },
        "throughput": {
            "q_per_s": n / total_latency_s if total_latency_s > 0 else 0,
            "q_per_h": (n / total_latency_s * 3600) if total_latency_s > 0 else 0,
        },
        "cost_usd_estimate": cost_total,
        "cost_per_1k_q_estimate": (cost_total * 1000 / n) if (cost_total and n) else None,
        "cost_error": cost_err,
        "route_p50_latency_s": route_p50,
    }

    out_path_str = str(out_path) if out_path else in_path.replace(".json", "_latency.json")
    Path(out_path_str).write_text(json.dumps(out, indent=2))
    return out


def analyze_ensemble(ens_path: str | Path, v3bu_path: str | Path,
                     dec_path: str | Path,
                     out_path: str | Path | None = None) -> dict | None:
    """Compute ensemble latency by combining V3+bu + decompose source files.

    Two modes reported:
      - ``parallel``: ``max(v3bu, decompose)`` per question (concurrent branches)
      - ``sequential``: ``v3bu + decompose`` per question (worst case)
    """
    ens = json.load(open(ens_path))
    v3 = json.load(open(v3bu_path))
    dec = json.load(open(dec_path))

    v3_lat = {it["qid"]: it.get("latency_s", 0.0) for it in v3.get("per_question", [])}
    v3_pin = {it["qid"]: it.get("prompt_tokens", 0) for it in v3.get("per_question", [])}
    v3_pout = {it["qid"]: it.get("completion_tokens", 0) for it in v3.get("per_question", [])}
    dec_lat = {it["qid"]: it.get("latency_s", 0.0) for it in dec.get("per_question", [])}
    dec_pin = {it["qid"]: it.get("prompt_tokens", 0) for it in dec.get("per_question", [])}
    dec_pout = {it["qid"]: it.get("completion_tokens", 0) for it in dec.get("per_question", [])}

    par_lats: list[float] = []
    seq_lats: list[float] = []
    sum_in_total = sum_out_total = 0
    for it in ens.get("per_question", []):
        qid = it["qid"]
        lv = v3_lat.get(qid, 0.0)
        ld = dec_lat.get(qid, 0.0)
        if lv > 0 and ld > 0:
            par_lats.append(max(lv, ld))
            seq_lats.append(lv + ld)
        sum_in_total += v3_pin.get(qid, 0) + dec_pin.get(qid, 0)
        sum_out_total += v3_pout.get(qid, 0) + dec_pout.get(qid, 0)

    if not par_lats:
        print(f"[!] No matching qids with latency in {ens_path}", file=sys.stderr)
        return None

    par_sorted = sorted(par_lats)
    seq_sorted = sorted(seq_lats)
    n = len(par_lats)
    par_total = sum(par_lats)
    seq_total = sum(seq_lats)

    reader = ens.get("summary", {}).get("reader_model") or v3.get("summary", {}).get("reader_model", "unknown")
    cost_total, cost_err = estimate_cost(reader, sum_in_total, sum_out_total)

    out = {
        "input_ensemble": str(ens_path),
        "source_v3bu": str(v3bu_path),
        "source_decompose": str(dec_path),
        "n_questions_matched": n,
        "reader_model": reader,
        "latency_s_parallel": {
            "mean": par_total / n,
            "p50": percentile(par_sorted, 50),
            "p95": percentile(par_sorted, 95),
            "p99": percentile(par_sorted, 99),
            "min": par_sorted[0],
            "max": par_sorted[-1],
            "total_compute_s": par_total,
        },
        "latency_s_sequential": {
            "mean": seq_total / n,
            "p50": percentile(seq_sorted, 50),
            "p95": percentile(seq_sorted, 95),
            "p99": percentile(seq_sorted, 99),
            "min": seq_sorted[0],
            "max": seq_sorted[-1],
            "total_compute_s": seq_total,
        },
        "tokens_total": {
            "prompt_total": sum_in_total,
            "completion_total": sum_out_total,
        },
        "throughput_q_per_h_parallel": (n / par_total * 3600) if par_total > 0 else 0,
        "throughput_q_per_h_sequential": (n / seq_total * 3600) if seq_total > 0 else 0,
        "cost_usd_total_estimate": cost_total,
        "cost_per_1k_q_estimate": (cost_total * 1000 / n) if (cost_total and n) else None,
        "cost_error": cost_err,
        "decisions": ens.get("summary", {}).get("decisions"),
    }

    out_path_str = str(out_path) if out_path else str(ens_path).replace(".json", "_latency_ensemble.json")
    Path(out_path_str).write_text(json.dumps(out, indent=2))
    return out
