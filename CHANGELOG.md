# Changelog

## 0.5.0 — 2026-06 (first public release)

Public release accompanying the paper *"MOTHRAG: Training-Free Multi-Hop Question Answering at Research-SOTA Parity on Commodity LLM APIs"* (Zenodo DOI [10.5281/zenodo.20668567](https://doi.org/10.5281/zenodo.20668567); arXiv pending).

- Four-arm ensemble pool (direct / decomposition / iterative / Pool-Duplicate Dispatch) with deterministic arbitration (γ 1.0 / agreement 0.5 / faithfulness 0.3).
- Bridge retrieval substrate (multi-query ANN fusion + tripartite LLM judge) with per-arm input-feature gating, plus ChainFilter post-retrieval chain-density re-scoring.
- Grounding (γ) verification with proof-tree-structured answers and γ-cap fallback; iterative re-retrieval driven by grounding failures.
- Premium (Claude Haiku) and economy (gemini-2.5-flash) retrieval-judge tiers — a one-flag cost/quality frontier.
- High-level `MothRAG` API (`from_documents` → `query`) with graceful no-key fallbacks; provider extras for Gemini / OpenAI-compatible readers / FAISS.
- Full paper evaluation harness (`scripts/route_prospective.py`) and released per-query outputs for all reported tables (`paper/results/`, 6×n=1000).
