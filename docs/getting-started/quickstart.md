# Quickstart (5 minutes)

This guide shows you how to take a folder of text documents and turn it into a question-answering system in five minutes, with zero configuration.

## 1. Install

```bash
pip install mothrag
```

You can run MothRag with **zero API keys** — the package ships with deterministic offline fallbacks (hash-bucket embedder + echo reader). For paper-quality answers, install the production stack and set one API key:

```bash
pip install "mothrag[prod]"
export GROQ_API_KEY=...       # the paper's reader (Llama-3.3-70B); TOGETHER_API_KEY also works
export GEMINI_API_KEY=...     # optional, upgrades embedder + grounding judge
```

## 2. One-line corpus ingestion

```python
from mothrag import MothRAG

# Option A — list of strings
rag = MothRAG.from_documents([
    "The Eiffel Tower is in Paris, completed in 1889.",
    "Python was created by Guido van Rossum in 1991.",
    "Memento (2000) was directed by Christopher Nolan.",
])

# Option B — folder of .txt / .md / .json files
rag = MothRAG.from_documents("examples/sample_corpus/")
```

`from_documents` accepts:

- A path to a single file (`.txt`, `.md`, `.json`, `.jsonl`)
- A path to a directory (recursively walks supported extensions)
- A list of strings (each becomes a `Document`)
- A list of `Document` objects (preserves metadata)

## 3. Ask questions

```python
result = rag.query("Who created Python?")
print(result.answer)
# → "Python was created by Guido van Rossum in 1991."

print(result.retrieved_chunks)   # supporting passages
print(result.arm_subset)         # routing decision (e.g. ['v3bu', 'decompose', 'iter'])
```

The return value is a [`QueryResult`](../guide/api.md#queryresult) with provenance: the answer string, the retrieved chunks, the arm subset chosen by the classifier, and a metadata dict.

## 4. Batch and async

```python
# Parallel batched query
results = rag.batch_query(["Q1?", "Q2?", "Q3?"], max_workers=4)

# Async (single-question)
import asyncio
answer = asyncio.run(rag.aquery("Q?"))
```

## 5. Customizing (optional)

```python
from mothrag import MothRAG
from mothrag.core._api_adapters import OpenAICompatibleReader

reader = OpenAICompatibleReader(
    model="gpt-4o",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
)

rag = MothRAG.from_documents("./docs", reader=reader, top_k_chunks=15)
```

## Next steps

- [Installation reference](installation.md)
- [Core concepts](concepts.md): arms, classifier, arbitrate
- [Production deployment](../guide/production.md) — reproducing the paper stack
- [The paper (Zenodo)](https://doi.org/10.5281/zenodo.20668567)
