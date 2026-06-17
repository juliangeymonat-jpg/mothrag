# MOTHRAG

> **Training-free multi-hop question answering at research-SOTA parity — on commodity LLM APIs alone.**

MOTHRAG answers questions that require chaining evidence across many documents. It matches the accuracy of the best *published research* systems on the standard multi-hop benchmarks using **only commodity LLM APIs** — no GPU, no self-hosted model, no constrained-decoding stack — at a measured ~$0.03 per query, with an inspectable proof tree per answer. Nothing is trained; there are no learned components anywhere.

## Results (paper, n=1000 per dataset, Llama-3.3-70B reader, single uniform configuration)

| System | Deployment profile | HotpotQA | 2WikiMultiHopQA | MuSiQue | AVG |
|---|---|---|---|---|---|
| HippoRAG 2 (as published) | offline OpenIE graph + NV-Embed-v2 | 75.5 | 71.0 | 48.6 | 65.0 |
| NeocorRAG (as published) | GPU-bound constrained decoding + NV-Embed-v2 | **78.3** | 76.1 | **52.6** | **69.0** |
| **MOTHRAG (ours)** | **commodity APIs only** | 78.1 | **76.3** | 50.5 | 68.3 |

F1; competitor numbers as published (same reader class). MOTHRAG reaches the **highest average F1 among commercially-deployable frameworks** — within **0.7 points** of the GPU-bound research state of the art (parity on HotpotQA at −0.2, an edge on 2WikiMultiHopQA at +0.2, an honest gap on MuSiQue at −2.1). Measured cost **$0.032/query**; a documented economy tier runs at **≈$0.018/query** at parity on HotpotQA/2WikiMultiHopQA.

## How it works

A query-feature-only classifier (`arm_subset()`, **zero ground-truth access**) decides per query which arms to run, over a four-arm pool — `v3bu` (single-shot retrieve-and-read), `decompose` (sub-question chaining), `iter` (γ-verified iterative retrieval), and **`iter_dup_a` = PDD** (a deterministic duplicate of `iter` that double-weights the grounding-checked voice). Retrieval is reshaped by a bridge substrate + ChainFilter; outputs are combined by deterministic arbitration. See [Concepts](getting-started/concepts.md).

## 30-second start

```python
from mothrag import MothRAG

rag = MothRAG.from_documents([
    "The Eiffel Tower is located in Paris.",
    "Python was created by Guido van Rossum in 1991.",
])

result = rag.query("Who created Python?")
print(result.answer)
```

→ [Full quickstart](getting-started/quickstart.md)

## Why MOTHRAG?

- **One line to start**: `MothRAG.from_documents(...)` auto-loads, chunks, embeds, indexes.
- **Deployable**: every component is a commodity pay-per-call API — no GPU, no non-commercial models.
- **Training-free**: no learned weights anywhere; routing is a pure function of the query.
- **Honest scope**: negative results disclosed; the MuSiQue gap stated plainly.
- **Apache 2.0**: open code, no closed components.

## Install

```bash
pip install mothrag                # core + offline fallbacks
pip install "mothrag[prod]"        # full production stack (Gemini + Groq Llama + sentence-transformers)
```

## Links

- [Quickstart](getting-started/quickstart.md) · [Installation](getting-started/installation.md) · [Concepts](getting-started/concepts.md)
- [High-level API](guide/api.md) · [Loaders](guide/loaders.md) · [Production deployment](guide/production.md)
- [The paper (Zenodo)](https://doi.org/10.5281/zenodo.20668567)
