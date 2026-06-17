# Production deployment

The v0.5.0 high-level API (`mothrag.MothRAG`) is a developer-friendly wrapper. The **paper-grade stack** — the four-arm PDD pool, bridge retrieval substrate + ChainFilter, γ-verifier, and faithfulness loop — runs through `scripts/route_prospective.py`.

## Two entry points, same routing decision

| Entry point | When to use | Interface |
|---|---|---|
| `mothrag.MothRAG` (v0.5.0+) | Developer Python API, custom corpora, async/batch | `from_documents → query` |
| `scripts/route_prospective.py` | Paper-grade evaluation, reproducible benchmarks, full safety stack | CLI |

Both share the same `arm_subset()` classifier and the same deterministic arbitration.

## Reproducing the paper headlines

The exact, verified reproduction command — a **single uniform configuration** across all three datasets — lives in **`paper/REPRODUCE.md`** in the repository. Rather than duplicate it here (where it drifts), follow that file. The headline configuration is:

- **Reader:** Groq `llama-3.3-70b-versatile` (`GROQ_API_KEY`).
- **Embedder + grounding judge:** Gemini (`gemini-embedding-2`, `gemini-2.5-flash`) via `GEMINI_API_KEY`.
- **Retrieval judge:** Claude Haiku (premium) via `ANTHROPIC_API_KEY`, or Gemini for the economy tier.
- **Stack:** four-arm pool (`v3bu` / `decompose` / `iter` / `iter_dup_a` = PDD) + bridge substrate + ChainFilter + γ-verifier + faithfulness loop.

Headline F1 (n=1000 per dataset): **HotpotQA 78.1 · 2WikiMultiHopQA 76.3 · MuSiQue 50.5** (avg 68.3). Measured cost ≈ $0.032/query (≈ $0.018/query on the economy tier).

### Pre-requisites
- API keys: `GROQ_API_KEY` (reader) + `GEMINI_API_KEY` (embedder + grounding judge); `ANTHROPIC_API_KEY` for the premium retrieval judge (optional — the economy tier runs the judge on Gemini).
- A pre-built corpus directory (the high-level API auto-chunks raw text; the CLI expects a pre-built corpus — see `paper/REPRODUCE.md` for inputs).

## High-level API vs paper-grade CLI

| Capability | `MothRAG` (Python API) | `route_prospective.py` (CLI) |
|---|---|---|
| Auto-chunking from raw text / strings | yes | no (expects a pre-built corpus) |
| Custom embedder / reader plug-in | yes | partial (CLI flags) |
| Async / batch query | yes | n/a (batch over a query file) |
| Four-arm PDD pool | single-arm read in v0.5.0 | yes |
| Bridge substrate + ChainFilter | not in v0.5.0 | yes |
| γ-verifier + faithfulness loop | not in v0.5.0 | yes |

**v0.5.x roadmap:** bring the full safety stack into the `MothRAG` class behind a `production=True` flag.
