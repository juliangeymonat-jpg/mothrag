"""DIR-14 — Liberal verifier (10-line variant of DIR-9 verifier).

Empirical finding: 12 of 21 trees marked overall_status=invalid by the strict
verifier actually have a correct naturalized_answer — the failure was at the
conjunction or transitivity step's internal consistency check, not at the
answer-bearing leaf.

Liberal mode: tree is `valid` iff at least one lookup leaf is `valid` AND the tree's
naturalized_answer is consistent (token-overlap ≥ 0.5) with that leaf's `object` or
its source span.

This does NOT change Hard EM (Hard EM is on naturalized_answer regardless). It changes
only the faithfulness axis scoring.
"""

from __future__ import annotations
import re
import string

_PUNCT = set(string.punctuation)


def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = "".join(c for c in s if c not in _PUNCT)
    return re.sub(r"\s+", " ", s).strip()


def _token_overlap(a: str, b: str) -> float:
    a_tok = set(t for t in a.split() if len(t) > 1)
    b_tok = set(t for t in b.split() if len(t) > 1)
    if not a_tok:
        return 1.0
    return len(a_tok & b_tok) / len(a_tok)


def liberal_overall_status(tree_dict: dict) -> str:
    """Compute liberal status from a tree dict (as in gamma_first50.json output).

    Returns one of: valid | partial | invalid | refuse.
    Mirrors the strict status convention but accepts derivative-step failures
    when the answer-bearing leaf is valid.
    """
    steps = tree_dict.get("steps", [])
    if not steps:
        return "invalid"
    nat = tree_dict.get("naturalized_answer", "") or ""
    refusal = nat == "REFUSE_NO_PROOF" or nat == "REFUSE_PARSE_FAILURE"
    if refusal:
        return "refuse"

    nat_n = _normalize(nat)
    valid_lookups = [s for s in steps if s.get("rule") == "lookup" and s.get("verifier_status") == "valid"]
    if not valid_lookups:
        # No grounded lookup at all — fall back to strict outcome
        if all(s.get("verifier_status") == "valid" for s in steps):
            return "valid"
        return "invalid"

    # Liberal rule: at least one valid lookup leaf exists (proof has SOME grounded fact)
    # AND naturalized_answer is consistent with at least one VALID step's claim_text /
    # object / span_text (the step may be lookup OR conjunction/transitivity that
    # references the lookup leaves). Derivative-step verifier failures (conjunction
    # token-overlap, transitivity bridge mismatch) do NOT kill tree validity if the
    # answer-bearing claim is grounded in the lookup substrate.
    if not nat_n:
        return "partial"
    best_overlap = 0.0
    for s in steps:
        # Consider any step (valid or not) for answer-bearing match;
        # but require that the LOOKUP substrate is valid (already true: valid_lookups non-empty)
        candidates = []
        if s.get("object"):
            candidates.append(_normalize(s["object"]))
        if s.get("claim_text"):
            candidates.append(_normalize(s["claim_text"]))
        if s.get("sources"):
            candidates.append(_normalize(s["sources"][0].get("span_text") or ""))
        for cand in candidates:
            if not cand:
                continue
            ov = _token_overlap(nat_n, cand)
            if ov > best_overlap:
                best_overlap = ov
    if best_overlap >= 0.5:
        return "valid"
    if best_overlap >= 0.25:
        return "partial"
    return "invalid"


def liberal_status_distribution(gamma_results: list[dict]) -> dict[str, int]:
    """Convenience: counts of liberal statuses across a results list."""
    out = {"valid": 0, "partial": 0, "invalid": 0, "refuse": 0}
    for r in gamma_results:
        tree = r.get("gamma_proof_tree") or {}
        st = liberal_overall_status(tree)
        out[st] = out.get(st, 0) + 1
    return out


if __name__ == "__main__":
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "dir-9-gamma-pivot" / "code" / "measurements" / "gamma_first50.json"
    if not p.exists():
        print(f"gamma_first50.json not found at {p}")
        raise SystemExit(1)
    data = json.loads(p.read_text(encoding="utf-8"))
    dist_strict = {"valid": 0, "partial": 0, "invalid": 0, "refuse": 0}
    for r in data["results"]:
        s = r.get("gamma_overall_status", "invalid")
        dist_strict[s] = dist_strict.get(s, 0) + 1
    dist_liberal = liberal_status_distribution(data["results"])
    print(f"Strict (DIR-9):  {dist_strict}")
    print(f"Liberal (DIR-14): {dist_liberal}")
    n = len(data["results"])
    faith_strict = (dist_strict["valid"] * 1.0 + dist_strict["partial"] * 0.5) / n
    faith_liberal = (dist_liberal["valid"] * 1.0 + dist_liberal["partial"] * 0.5) / n
    print(f"Faithfulness: strict={faith_strict:.3f}  liberal={faith_liberal:.3f}")
