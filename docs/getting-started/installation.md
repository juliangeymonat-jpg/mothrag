# Installation

## Requirements

- Python 3.10, 3.11, 3.12, or 3.13
- 4 GB+ RAM (more for large corpora)
- Optional: GPU (only required if you supply your own embedder/reader that uses one)

## Standard install

```bash
pip install mothrag
```

This installs the **minimal core**: numpy, requests, typing-extensions, python-dotenv, tqdm. The high-level API works out of the box with offline fallbacks (hash-bucket embedder, echo reader). Good for development, smoke tests, learning.

## Production stack

```bash
pip install "mothrag[prod]"
```

Adds: sentence-transformers, openai client, google-genai (Gemini Embedding 2), and the retrieval extras (scikit-learn, networkx, rank-bm25). This reproduces the **MOTHRAG** paper headline stack.

## Granular extras

| Extra | What it adds | When to use |
|---|---|---|
| `gemini` | google-genai | Gemini Embedding 2 (best embedder) |
| `sentence-transformers` | sentence-transformers + transformers | Sentence-Transformers MiniLM-L6 (default fallback) |
| `openai` | openai SDK | OpenAI / Together / Groq / OpenRouter / Anthropic readers |
| `retrieval` | scikit-learn, networkx, rank-bm25 | NER cache, BM25 retrieval, graph utilities |
| `dev` | pytest, ruff, build, mkdocs-material | Development & docs |
| `prod` | gemini + sentence-transformers + openai + retrieval | Production stack one-shot |
| `all` | prod + dev | Everything |

Example — Gemini embeddings only, no LLM reader (offline echo):

```bash
pip install "mothrag[gemini]"
export GEMINI_API_KEY=...
```

## API keys

Set as environment variables; MothRag auto-detects them on instantiation.

| Variable | Use |
|---|---|
| `GROQ_API_KEY` | Llama-3.3-70B via Groq — the paper's reader (recommended) |
| `GEMINI_API_KEY` | Gemini Embedding 2 (embedder) + grounding judge |
| `ANTHROPIC_API_KEY` | Claude Haiku — premium retrieval judge (optional) |
| `TOGETHER_API_KEY` | Llama-3.3-70B via Together AI (alternative reader) |
| `OPENAI_API_KEY` | GPT-4o, GPT-5.x via OpenAI (alternative reader) |

Without keys, MothRag still imports and runs end-to-end via offline fallbacks. The first time you instantiate without keys, you'll see a `WARNING` log line indicating which fallback was used.

## Windows note

If you see `UnicodeEncodeError` from CLI scripts using γ (gamma) glyphs in help text:

```bash
set PYTHONIOENCODING=utf-8
```

This affects only the `--help` console rendering; the runtime path is unaffected.

## Verifying

```python
import mothrag
print(mothrag.__version__)        # → "0.5.0"

from mothrag import MothRAG
rag = MothRAG.from_documents(["Hello world."])
print(rag.query("Who is mentioned?").answer)
```
