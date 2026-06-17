"""C7 applied to Aurora rejected_chains (bimodal-by-construction ensemble).

Design rationale: C7 needs a bimodal ensemble; the spec_inf rejected_chains
schema enforces alternative inference paths → bimodal-by-construction.

For each Aurora-X spec_inf output:
1. Extract chosen answer + N rejected_chain answers (the "chain" field)
2. Embed K = 1+N candidate strings
3. Apply C7 centroid bisection (Method-D auto-phase)
4. Determine: was chosen KEPT (above median) or CANCELLED (below median)?
5. Test: P(EM=0 | chosen cancelled) vs P(EM=0 | chosen kept)
   If C7-cancel-chosen predicts EM=0 → C7 is meta-confidence signal

Hypothesis: C7 cancellation of chosen answer = low-confidence flag.
"""
from __future__ import annotations
import io
import json
import math
import re
import sys
from pathlib import Path
from collections import Counter

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
WO = ROOT.parent.parent
PROJECT = WO.parent.parent
OUT = ROOT / "measurements"
OUT.mkdir(parents=True, exist_ok=True)

PRIOR_FULL = WO / "dir-30y-clean-eval-opus-llama-opust3" / "code" / "measurements" / "clean_eval_full.json"
NEW_MULTITIER = WO / "dir-30y-clean-eval-opus-llama-opust3" / "code" / "measurements" / "clean_eval_multitier.json"

# NOTE: Aurora driver previously imported `semantic_embed_store.semantic_embed_batch`
# from a sibling worker dir (`dir-29c-semantic-embedding-retrieval/code/`). For the
# MothRAG integration we keep this import lazy inside ``main()`` so library users
# can call ``c7_aurora_rejected_chains()`` with their own embedder (e.g. Gemini Emb2).


def vnorm(s):
    s = str(s if s is not None else "").lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def hard_em(p, g):
    return 1 if vnorm(p) == vnorm(g) else 0


def f1_token(p, g):
    p_tok = set(vnorm(p).split())
    g_tok = set(vnorm(g).split())
    if not p_tok or not g_tok:
        return 0.0
    tp = len(p_tok & g_tok)
    if tp == 0:
        return 0.0
    prec = tp / len(p_tok)
    rec = tp / len(g_tok)
    return 2 * prec * rec / (prec + rec)


def soft_em(p, g, threshold=0.7):
    return 1 if f1_token(p, g) >= threshold else 0


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (None, None, None)
    p = k / n
    den = 1 + z*z/n
    centre = (p + z*z/(2*n)) / den
    half = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / den
    return (p, max(0, centre - half), min(1, centre + half))


def normalize_rows(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def parse_rejected_chains(raw):
    """Extract list of {chain, rejection_reason} from raw output."""
    if not raw or "rejected_chains" not in raw:
        return []
    # Find rejected_chains list with brace counting
    idx = raw.find("rejected_chains")
    if idx < 0:
        return []
    # Find first [ after the key
    bracket_start = raw.find("[", idx)
    if bracket_start < 0:
        return []
    # Find matching ]
    depth = 0
    in_str = False
    escape = False
    end = -1
    for i in range(bracket_start, len(raw)):
        c = raw[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return []
    arr_str = raw[bracket_start:end]
    try:
        return json.loads(arr_str)
    except Exception:
        return []


def c7_aurora_rejected_chains(chosen: str, rejected_chains: list[str],
                               embedder, query_embed=None) -> dict:
    """Library wrapper: apply C7 (Method-D auto-phase) to chosen + rejected chains.

    Args:
        chosen: the answer the system selected.
        rejected_chains: list of alternative chain answers (the "chain" field in
            speculative_inference rejected_chains).
        embedder: callable ``embedder(list[str]) -> np.ndarray (K, D)`` — caller
            provides this (e.g. Gemini Embedding 2 batch embed function).
        query_embed: optional pre-computed query embedding. If provided, it
            replaces the centroid for projection (Aurora-spec faithful), else
            uses centroid of candidate embeddings.

    Returns:
        dict with keys:
          - ``chosen_kept`` (bool): True iff the chosen answer was KEPT (above median)
          - ``phases``: phase array (0 or pi) per candidate
          - plus ``c7_apply`` info fields (projections, K, n_keep, n_cancel, etc.)

    Aurora reference: Method-D auto-phase non-monotonic chain filter.
    """
    if not chosen or not rejected_chains:
        return {"chosen_kept": True, "K": 1, "degenerate": True,
                "n_keep": 1, "n_cancel": 0, "phases": [0.0]}
    candidates = [chosen] + list(rejected_chains)
    embeddings = embedder(candidates)
    embeddings = normalize_rows(embeddings)
    if query_embed is not None:
        # Use query embedding as the bisection axis (Aurora-spec variant)
        query_unit = query_embed / (np.linalg.norm(query_embed) + 1e-9)
        projections = embeddings @ query_unit
        median_proj = float(np.median(projections))
        phases = np.where(projections >= median_proj, 0.0, np.pi)
        info = {
            "projections": projections.tolist(),
            "projection_std": float(np.std(projections)),
            "median_projection": median_proj,
            "K": len(candidates),
            "n_keep": int((phases == 0.0).sum()),
            "n_cancel": int((phases == np.pi).sum()),
            "axis": "query",
        }
    else:
        phases, info = c7_apply(embeddings)
        info["axis"] = "centroid"
    chosen_kept = bool(phases[0] == 0.0)
    return {"chosen_kept": chosen_kept, "phases": phases.tolist(), **info}


def c7_apply(embeddings):
    """Method-D auto-phase: phases[k] = 0 if projection >= median else π (cancel)."""
    K = embeddings.shape[0]
    centroid = embeddings.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm < 1e-12 or K < 2:
        return np.zeros(K), {"degenerate": True}
    centroid_unit = centroid / centroid_norm
    projections = embeddings @ centroid_unit
    median_proj = float(np.median(projections))
    phases = np.where(projections >= median_proj, 0.0, np.pi)
    return phases, {
        "projections": projections.tolist(),
        "projection_std": float(np.std(projections)),
        "projection_range": float(projections.max() - projections.min()),
        "median_projection": median_proj,
        "K": K,
        "n_keep": int((phases == 0.0).sum()),
        "n_cancel": int((phases == np.pi).sum()),
    }


def main():
    # Lazy-load Aurora driver dependency (only needed when running CLI)
    DIR29C_CODE = WO / "dir-29c-semantic-embedding-retrieval" / "code"
    sys.path.insert(0, str(DIR29C_CODE))
    from semantic_embed_store import semantic_embed_batch  # noqa: E402

    print("[load] data sources")
    prior = json.loads(PRIOR_FULL.read_text(encoding="utf-8"))
    gt = prior["ground_truth"]
    all_results = dict(prior["results"])
    if NEW_MULTITIER.exists():
        mt = json.loads(NEW_MULTITIER.read_text(encoding="utf-8"))
        all_results.update(mt["results"])
        gt.update(mt.get("ground_truth", {}))
    print(f"  systems: {list(all_results.keys())}")

    # Apply C7 to spec_inf outputs per Aurora system
    print("\n=== C7-on-rejected_chains per Aurora system ===")
    aurora_systems = [s for s in all_results if s.startswith("aurora_")]
    overall_results = {}

    for sys_name in aurora_systems:
        results = all_results[sys_name]
        print(f"\n  --- {sys_name} ---")
        # Collect candidates for embedding
        per_query = []
        all_strings_to_embed = []
        for qid, r in results.items():
            if r.get("tagged_as") != "speculative_inference":
                continue
            chosen = r.get("prediction", "").strip()
            if not chosen:
                continue
            raw = r.get("raw", "") or ""
            rej = parse_rejected_chains(raw)
            if not rej:
                continue
            # Extract rejected chain answers (the "chain" field — these are the alternatives)
            rej_chains = [str(rc.get("chain", "")).strip() for rc in rej if rc.get("chain")]
            rej_chains = [c for c in rej_chains if c]
            if not rej_chains:
                continue
            # Build candidate set: chosen + rejected
            candidates = [chosen] + rej_chains
            start_idx = len(all_strings_to_embed)
            all_strings_to_embed.extend(candidates)
            per_query.append({
                "qid": qid,
                "chosen": chosen,
                "rejected_chains": rej_chains,
                "K": len(candidates),
                "embed_start": start_idx,
                "embed_end": start_idx + len(candidates),
                "gold": gt.get(qid, ""),
            })
        print(f"    spec_inf with rejected_chains: {len(per_query)}")

        if not per_query:
            print(f"    (no spec_inf with rejected_chains; skipping)")
            continue

        # Batch embed all candidates
        print(f"    embedding {len(all_strings_to_embed)} candidate strings...")
        all_embs = semantic_embed_batch(all_strings_to_embed)
        all_n = normalize_rows(all_embs)

        # Apply C7 per query
        K_dist = Counter()
        c7_keeps_chosen = 0
        c7_cancels_chosen = 0
        per_query_c7 = []
        for entry in per_query:
            embs = all_n[entry["embed_start"]:entry["embed_end"]]  # (K, D)
            phases, info = c7_apply(embs)
            chosen_phase = phases[0]  # chosen is at index 0
            K_dist[entry["K"]] += 1
            chosen_kept = (chosen_phase == 0.0)
            if chosen_kept:
                c7_keeps_chosen += 1
            else:
                c7_cancels_chosen += 1
            per_query_c7.append({
                "qid": entry["qid"],
                "K": entry["K"],
                "chosen_kept": chosen_kept,
                "chosen": entry["chosen"],
                "gold": entry["gold"],
                "em": hard_em(entry["chosen"], entry["gold"]),
                "soft_em": soft_em(entry["chosen"], entry["gold"]),
                "f1": f1_token(entry["chosen"], entry["gold"]),
                "projection_std": info.get("projection_std"),
                "projection_range": info.get("projection_range"),
            })

        print(f"    K distribution: {dict(K_dist)}")
        print(f"    chosen-kept: {c7_keeps_chosen}/{len(per_query_c7)}  "
              f"chosen-cancelled: {c7_cancels_chosen}/{len(per_query_c7)}")

        # Conditional metrics: chosen-cancelled (low-confidence flag) vs chosen-kept
        em_kept = [r["em"] for r in per_query_c7 if r["chosen_kept"]]
        em_cancelled = [r["em"] for r in per_query_c7 if not r["chosen_kept"]]
        soft_kept = [r["soft_em"] for r in per_query_c7 if r["chosen_kept"]]
        soft_cancelled = [r["soft_em"] for r in per_query_c7 if not r["chosen_kept"]]
        f1_kept = [r["f1"] for r in per_query_c7 if r["chosen_kept"]]
        f1_cancelled = [r["f1"] for r in per_query_c7 if not r["chosen_kept"]]

        def stats(label, em_list, soft_list, f1_list):
            n = len(em_list)
            if n == 0:
                return {"n": 0}
            em_count = sum(em_list)
            soft_count = sum(soft_list)
            f1_mean = float(np.mean(f1_list))
            em_p, em_lo, em_hi = wilson_ci(em_count, n)
            soft_p, soft_lo, soft_hi = wilson_ci(soft_count, n)
            print(f"      {label}: n={n}  HardEM={em_count}/{n}={em_p*100:.1f}% [{em_lo*100:.1f},{em_hi*100:.1f}]  "
                  f"SoftEM={soft_count}/{n}={soft_p*100:.1f}%  F1={f1_mean:.3f}")
            return {"n": n, "em_count": em_count, "em_rate": em_p, "em_ci": [em_lo, em_hi],
                    "soft_em_count": soft_count, "soft_em_rate": soft_p, "f1_mean": f1_mean}

        print(f"    Conditional metrics:")
        s_kept = stats("CHOSEN-KEPT (high-confidence)", em_kept, soft_kept, f1_kept)
        s_cancelled = stats("CHOSEN-CANCELLED (low-confidence flag)", em_cancelled, soft_cancelled, f1_cancelled)

        # Differential
        if s_kept.get("n", 0) > 0 and s_cancelled.get("n", 0) > 0:
            diff_em = s_kept["em_rate"] - s_cancelled["em_rate"]
            diff_soft = s_kept["soft_em_rate"] - s_cancelled["soft_em_rate"]
            print(f"    DIFFERENTIAL Δ(kept − cancelled): HardEM {diff_em*100:+.1f}pp  SoftEM {diff_soft*100:+.1f}pp")

        overall_results[sys_name] = {
            "n_total_specinf_with_rej": len(per_query_c7),
            "K_distribution": dict(K_dist),
            "chosen_kept_n": c7_keeps_chosen,
            "chosen_cancelled_n": c7_cancels_chosen,
            "stats_kept": s_kept,
            "stats_cancelled": s_cancelled,
            "differential_em_pp": (s_kept["em_rate"] - s_cancelled["em_rate"]) if (s_kept.get("n",0)>0 and s_cancelled.get("n",0)>0) else None,
            "per_query_c7": per_query_c7,
        }

    out_path = OUT / "c7_rejected_chains_results.json"
    out_path.write_text(json.dumps(overall_results, indent=2, default=str), encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
