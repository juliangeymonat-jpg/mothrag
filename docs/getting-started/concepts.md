# Core concepts

MothRAG sits between a corpus and a question. Everything below is **training-free** and runs on commodity LLM APIs — no GPU, no learned weights anywhere.

## 1. Bridge retrieval substrate

Before any reading, retrieval is reshaped by a **bridge substrate**: multi-query ANN fusion re-ranked by an LLM judge conditioned on retrieved "bridge" evidence, applied to every retrieval (primary, sub-question, iterative). A post-retrieval **ChainFilter** then re-scores the ranking by chain density over OpenIE triples, gated only by input features of the question (never dataset identity).

## 2. The arm pool

Several retrieval-and-reading pipelines ("arms") run on the same query; a rule-based classifier picks which subset to run.

| Arm | What it does | When it wins |
|---|---|---|
| **v3bu** | Top-down entity-anchored retrieval + bottom-up boost + single reader call | Short bridge queries (1-hop, "X is Y") |
| **decompose** | Splits the question into sub-questions, retrieves + reads each, synthesizes | Multi-hop with explicit relational structure |
| **iter** | Iterative retrieval refined by accumulated facts, with the γ-verifier checking chain consistency | Deep chains, comparative-selection queries |

**The fourth arm — PDD (Pool-Duplicate Dispatch).** The paper's headline configuration uses a four-arm pool (`arms_pool=4`): a deterministic **copy of the `iter` arm** (`iter_dup_a`) is added as a fourth slot. It runs **no extra inference** — the prediction is identical by construction — but it **double-weights the grounding-checked (γ) voice** in arbitration. (The default pip pool is the three base arms; the four-arm PDD config is the one the paper reports.)

## 3. The classifier — `arm_subset(question)`

A pure-Python rule cascade over query surface features (no learned weights, no training set):

1. A base label (`chain_deep` / `bridge_entity` / `semantic_rich`) from token + entity + relation counts.
2. Polar-comparison override: yes/no "Are X and Y both Z?" → `v3bu` re-included.
3. Implicit-multihop signal: comparative-selection / kinship-possessive → exclude `v3bu`.

It returns the subset to run, e.g. `["v3bu", "decompose", "iter"]` (all three) or `["decompose", "iter"]` (v3bu excluded).

## 4. Deterministic arbitration (no oracle peek)

When the subset has more than one arm, predictions are combined by a fixed-weight score — never by looking at ground truth:

```
score(arm) = P_arm · ( 1.0·γ(arm) + 0.5·agreement(arm) + 0.3·faithfulness(arm) )
```

- **γ — grounding (weight 1.0):** does the answer's proof tree verify against the evidence? A γ-cap fallback fires when grounding can't be satisfied within budget.
- **agreement (0.5):** cross-arm consensus — the signal the PDD duplicate amplifies.
- **faithfulness (0.3):** answer/evidence consistency.

Every term is a pure function of the question and the arms' own predictions — never of any test-set label. That is what makes the routing **production-deployable**.

## 5. The two entry points

- **`mothrag.MothRAG`** — the high-level Python API (this docs site).
- **`scripts/route_prospective.py`** — the paper reproduction runner; it runs only the arms in the chosen subset, saving compute versus running the full pool.

Both use the same `arm_subset()` classifier and the same arbitration; they differ only in *when* the arm-skip decision is applied.

## Next

- [Quickstart](quickstart.md) — end-to-end example.
- [High-level API reference](../guide/api.md) — full method list.
- [The paper (Zenodo)](https://doi.org/10.5281/zenodo.20668567) — full architecture, ablations, and results.
