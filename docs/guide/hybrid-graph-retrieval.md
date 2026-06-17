# Hybrid graph retrieval (HippoRAG2 backend)

MothRag's retrieval base is pluggable via the `retrieval=` keyword on the [`MothRAG`](api.md) constructor. The default `retrieval="dense"` preserves the v0.5.0 alpha behaviour (embedder + in-memory cosine index). The `retrieval="hybrid_graph"` option swaps in the OSU-NLP-Group HippoRAG2 SDK ([github.com/OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG), Apache 2.0) for personalized-PageRank dense + graph fusion, while leaving the V3+bu / decompose / iter arms and arbitration logic untouched.

## Install

```bash
pip install 'mothrag[hybrid-graph]'    # pulls hipporag>=1.0
```

## Quickstart

```python
from mothrag import MothRAG

rag = MothRAG.from_documents(
    "docs/",                                    # or list[str], or list[Document]
    production=True,
    retrieval="hybrid_graph",
    retrieval_config={
        "save_dir": "/path/to/hipporag-cache",  # graph artifacts (reused on re-run)
        "llm_model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "embedding_model_name": "gemini-embedding-001",
    },
)

qr = rag.query("Multi-hop question about the corpus?")
print(qr.answer)
print(qr.metadata["retrieval"])  # "hybrid_graph"
print(qr.retrieved_chunks)
```

## What changes vs the dense baseline

| Layer | `retrieval="dense"` | `retrieval="hybrid_graph"` |
|---|---|---|
| Indexing | Embedder → in-memory cosine index | HippoRAG2 builds a passage-entity graph + dense index (under `save_dir`) |
| Retrieval | top-K cosine over embeddings | dense top-K + personalized PageRank over the passage-entity graph, fused per HippoRAG2's protocol |
| Embedder | MothRag's configured embedder | MothRag's embedder is exposed for cross-arm signals; HippoRAG2 uses its own dense leg keyed by `embedding_model_name` |
| Arms | V3+bu + decompose + iter (unchanged) | V3+bu + decompose + iter (unchanged) |
| Arbitration | `arbitrate_with_c7` / `arbitrate_excl_v3bu` (unchanged) | `arbitrate_with_c7` / `arbitrate_excl_v3bu` (unchanged) |
| Reader | MothRag's configured reader | MothRag's configured reader |

The architectural invariant is **same arms, same arbitration, different retrieval base.** Any F1 delta between the two configurations attributes cleanly to the retrieval layer.

## Configuration reference

`retrieval_config` is forwarded as kwargs to `HybridGraphRetriever.__init__`:

| Key | Default | Meaning |
|---|---|---|
| `save_dir` | `.mothrag-hipporag-cache` | Local directory used as HippoRAG2's `save_dir` for graph artifacts. Pre-built graphs (released by the HippoRAG2 authors alongside the ICML 2025 paper) can be placed here for reuse — HippoRAG2 detects existing artifacts and skips the build step. |
| `llm_model_name` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Entity-extraction LLM identifier passed through to HippoRAG2. |
| `embedding_model_name` | `gemini-embedding-001` | Dense-leg embedding model identifier. Match this to MothRag's `embedder` for cross-arm signal parity. |
| `hipporag_config` | `{}` | Dict merged into HippoRAG2's `BaseConfig` for full SDK access (`synonymy_edge_topk`, `passage_node_weight`, `damping`, etc.). |
| `embedder`, `reader_llm` | auto-wired | Retained for introspection; the wrapper does not pass them to HippoRAG2 directly. |

## Graph cache reuse

HippoRAG2 graph construction is the expensive step (tens of minutes to hours, scaling with corpus size and entity-extraction LLM throughput). Two practical patterns:

1. **Reuse the HippoRAG2 authors' published graphs** for the standard multi-hop QA benchmarks (HotpotQA / 2WikiMultiHopQA / MuSiQue). Place them under your chosen `save_dir`; HippoRAG2 detects existing artifacts and skips the build step.

2. **Build once locally and re-use** — the first `MothRAG.from_documents(...)` call against a fresh `save_dir` triggers a one-shot graph build; subsequent runs (per-query smoke, ablation sweeps, re-runs after a reader change) all reuse the cached graph.

The wrapper does no caching of its own; it delegates everything to HippoRAG2's `save_dir` mechanism.

## Evaluation example

A representative evaluation setup — MuSiQue with a Llama-3.3-70B reader at n = 200:

```python
from mothrag import MothRAG
from mothrag.embedders import GeminiEmbedder
from mothrag.readers import GroqReader   # any Llama-3.3-70B OpenAI-compatible wrapper works

rag = MothRAG.from_documents(
    musique_corpus_path,                  # path or list of documents
    embedder=GeminiEmbedder(model="gemini-embedding-001"),
    reader=GroqReader(model="llama-3.3-70b-versatile"),
    production=True,
    retrieval="hybrid_graph",
    retrieval_config={
        "save_dir": "/path/to/musique-hipporag-graph",   # pre-built or empty
        "llm_model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "embedding_model_name": "gemini-embedding-001",
    },
)

# Iterate the n=200 MQ T1 split exactly as in the B-mode baseline harness.
for q in musique_t1_split_200:
    qr = rag.query(q.question)
    log(qr.answer, qr.metadata["retrieval"], qr.metadata.get("arbitrate_reason"))
```

Comparison is against the B-mode dense baseline MothRag MQ T1 F1 (= 0.4298). Any other configuration delta (embedder, reader, sample, scorer, bootstrap protocol) should match the baseline harness exactly so the F1 delta attributes to retrieval base only.

## Backward compatibility

- `MothRAG()` with no `retrieval=` keyword defaults to `"dense"` and is byte-identical to v0.5.0 alpha.
- Existing `vector_db=` kwarg still works; it backs the default `DenseRetriever`.
- All other constructor keywords (`embedder`, `reader`, `production`, free-form `**config`) are untouched.

## Troubleshooting

- **`ImportError: HybridGraphRetriever requires the HippoRAG2 SDK`** — install via `pip install mothrag[hybrid-graph]`.
- **`RuntimeError: Unsupported hipporag BaseConfig kwargs`** — the wrapper targets `hipporag>=1.0`; if your installed SDK has a shifted signature, pass an explicit `retrieval_config={"hipporag_config": {...}}` dict matching your SDK version, or pin the extra (`pip install 'mothrag[hybrid-graph]==<matching-version>'`).
- **Graph build hangs / OOMs** — entity extraction is HippoRAG2-side; consult their setup guide. The wrapper does not introduce additional cost beyond what HippoRAG2 itself requires.

## See also

- [`mothrag/core/retrieval/protocol.py`](../../mothrag/core/retrieval/protocol.py) — the `Retriever` Protocol.
- [`mothrag/core/retrieval/dense.py`](../../mothrag/core/retrieval/dense.py) — `DenseRetriever` adapter.
- [`mothrag/core/retrieval/hybrid_graph.py`](../../mothrag/core/retrieval/hybrid_graph.py) — `HybridGraphRetriever` source.
- [API reference](api.md) — full `MothRAG` constructor signature.
- HippoRAG2 paper + SDK: <https://github.com/OSU-NLP-Group/HippoRAG>.
