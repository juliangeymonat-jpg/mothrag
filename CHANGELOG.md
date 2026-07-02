# Changelog

## 0.6.1 (2026-07-01)

- Fix: the `mothrag` console script now works. `mothrag = mothrag.cli.main:main` was declared but `mothrag/cli/main.py` did not ship in 0.6.0, so `mothrag --help` crashed with `ModuleNotFoundError`. Ships a real CLI: `mothrag query` (answer a question over `--text` / `--docs` inputs, with `--json`, `--embedder`, `--top-k`, `--production`), `mothrag smoke` (forwards everything, including `-h`, to `mothrag-smoke`), and `mothrag version` / `--version`.
- The echo fallback is now LOUD in the library itself, not just the CLI: `_resolve_default_reader` warns with the exact fix (`pip install 'mothrag[openai]'`) both when no key is set and when a key IS set but the reader SDK is missing, so the README-quickstart path can no longer silently pass chunk-echo off as an LLM answer. The CLI derives its `reader_mode` from the resolved reader instance (ground truth, cannot drift from the library), warns and exits 3 when a real reader returns an empty answer (failed API call), validates `--docs` paths and `--top-k`, and no longer crashes on console encodings that cannot represent the answer.
- The offline hash-fallback embedder now buckets via crc32 instead of the per-process-salted `hash()`, so repeated runs and persisted indexes are actually reproducible.
- CI: GitHub Actions gates on push/PR and on release tags. Both run the same released-artifact acceptance script (build the wheel, install it in a clean venv, run the documented commands from a neutral directory, entry-point and version-sync guards); the release path additionally checks that the tag matches the package version and runs the unit test suite before publishing via PyPI Trusted Publishing (OIDC).

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
