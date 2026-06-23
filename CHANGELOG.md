# Changelog

## 0.6.0 (2026-06-23)

- Incremental `update` / `delete` on the high-level API: `MothRAG.update(doc_id, text)` supersedes a document in place and `MothRAG.delete(doc_id)` retracts it, each a single embedding pass over the changed document with no index rebuild and no retraining. New `MutableVectorStore` protocol; the default in-memory store implements `delete` / `delete_by_doc` / `upsert`. Clear errors are raised on append-only stores or non-dense retrieval.
- Fix: an empty custom vector store passed to the `MothRAG` constructor is no longer discarded (it has length 0 and was dropped by a truthiness check).

## 0.5.0 — 2026-06 (first public release)

Public release accompanying the paper *"MOTHRAG: Training-Free Multi-Hop Question Answering at Research-SOTA Parity on Commodity LLM APIs"* (Zenodo DOI [10.5281/zenodo.20668567](https://doi.org/10.5281/zenodo.20668567)).

- Four-arm ensemble pool (direct / decomposition / iterative / Pool-Duplicate Dispatch) with deterministic arbitration (γ 1.0 / agreement 0.5 / faithfulness 0.3).
- Bridge retrieval substrate (multi-query ANN fusion + tripartite LLM judge) with per-arm input-feature gating, plus ChainFilter post-retrieval chain-density re-scoring.
- Grounding (γ) verification with proof-tree-structured answers and γ-cap fallback; iterative re-retrieval driven by grounding failures.
- Premium (Claude Haiku) and economy (gemini-2.5-flash) retrieval-judge tiers — a one-flag cost/quality frontier.
- High-level `MothRAG` API (`from_documents` → `query`) with graceful no-key fallbacks; provider extras for Gemini / OpenAI-compatible readers / FAISS.
- Full paper evaluation harness (`scripts/route_prospective.py`) and released per-query outputs for all reported tables (`paper/results/`, 6×n=1000).
