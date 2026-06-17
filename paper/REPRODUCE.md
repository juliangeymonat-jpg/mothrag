# Reproducing the paper results

This document gives the **verbatim evaluation command** behind every number in the paper, the inputs you need, and the released outputs you can verify against — so you can check our tables without spending anything, or re-run the full evaluation if you want to.

## 0. What is already released (verify without re-running)

[`paper/results/`](results/) contains the per-query outputs behind the paper's tables — six JSONs, `{summary, per_question[1000]}` each:

| File | Table | Config |
|---|---|---|
| `HP_n1000_v4.json` / `2W_n1000_v4.json` / `MQ_n1000_v4.json` | Table 1 + 2 | Full ("premium") configuration |
| `HP_flash_n1000_final.json` / `2W_flash_n1000_final.json` / `MQ_flash_n1000_final.json` | Table 3 | Economy tier (retrieval judge → gemini-2.5-flash) |

Every per-query record carries the question, gold, prediction, EM/F1, arm/arbitration telemetry, and the config echo. Recomputing the summary F1 from `per_question` reproduces the table cells exactly. The economy-tier files were assembled from a 2×500 sharded run plus same-day refires of 92 queries lost to a provider-side spend block (Appendix B of the paper); per-query pairing was verified exact.

## 1. Requirements to re-run

- Python ≥3.10, `pip install -e .[prod]`
- API keys (see `.env.example`): `GROQ_API_KEY` (reader), `GEMINI_API_KEY` (embedder + grounding judge), `ANTHROPIC_API_KEY` (premium retrieval judge only)
- Corpora: HotpotQA / 2WikiMultiHopQA / MuSiQue in HippoRAG2 source convention (per-dataset directory with `hipporag2_source/*.json` question files + chunked corpus). The corpus/index build is a one-time, no-LLM-cost step; see `benchmarks/` for layout.
- Budget: the full 3×1000 premium run measured **$96.98** total (reader $52.64 + retrieval judge $44.34). The economy tier's judge cost is ~15× lower. A `--n 100` smoke is ~1/10th.

## 2. The verbatim evaluation command (full / "premium" configuration)

```bash
python scripts/route_prospective.py \
  --mode ensemble_arbitrate \
  --arms-pool v3bu,decompose,iter,iter_dup_a \
  --use-bridge-substrate \
  --bridge-substrate-scope=all \
  --arm-bridge-qtype-gate=v3bu:none,decompose:none,iter:exclude_bridge_entity,iter_dup_a:exclude_bridge_entity \
  --use-gamma-verifier \
  --use-graph-aware-iter \
  --use-gamma-refuse-loop \
  --use-p11-gamma-cap-fallback \
  --use-stepchain-parity-composite \
  --use-faithfulness-loop \
  --use-c7-iter --c7-iter-trigger gated --c7-iter-embedder gemini-embedding-2 \
  --use-chainfilter \
  --judge-model gemini-2.5-flash --judge-provider gemini \
  --embedding gemini-embedding-2 \
  --reader-model llama-3.3-70b-versatile \
  --reader-base-url https://api.groq.com/openai/v1 \
  --reader-api-key-env GROQ_API_KEY \
  --data-dir <dataset_dir> \
  --queries <dataset_dir>/hipporag2_source/<dataset>.json \
  --n 1000 \
  --out results/<DS>_n1000.json
```

Run once per dataset (`hotpotqa.json`, `2wiki_hotpot_format.json`, `musique_hotpot_format.json`). The retrieval judge defaults to the premium tier (Claude Haiku via `ANTHROPIC_API_KEY`).

## 3. Economy tier (Table 3)

Identical command plus exactly two flags — this is the "one-flag swap" of the paper (one model + its provider switch):

```bash
  --bridge-judge-model gemini-2.5-flash --bridge-judge-provider gemini
```

Pair the outputs per query-ID against the premium run to reproduce the ΔF1 / CI columns.

## 4. Notes

- A single uniform configuration produces every reported number — no per-dataset flags, no per-dataset hyperparameters (anti-leakage guarantee, §3.6 of the paper).
- Reader pricing was pinned at evaluation date (Groq llama-3.3-70b-versatile: $0.59/M prompt, $0.79/M completion); provider prices drift — re-check before comparing costs.
- Hosted-API serving may drift over time (paper Limitations); per-query outputs above are the frozen reference.
