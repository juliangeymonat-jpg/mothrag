# High-level API reference

## `MothRAG`

```python
from mothrag import MothRAG
```

### Constructor

```python
MothRAG(
    embedder: Embedder | str | None = None,
    reader: Reader | None = None,
    vector_db: VectorStore | None = None,
    **config,
)
```

`embedder` accepts an `Embedder` instance, a string spec, or `None`. Any backend passed as `None` is resolved from the auto-default chain:

- **embedder**: `VERTEX_AI_PROJECT` / `GOOGLE_CLOUD_PROJECT` → Vertex AI text-embedding-005 (Vertex does not expose gemini-embedding-2 yet) → `GEMINI_API_KEY` / `GOOGLE_API_KEY` → Gemini Studio gemini-embedding-2 → sentence-transformers MiniLM-L6 → hash fallback.
- **reader**: `GROQ_API_KEY` (the paper's reader) / `TOGETHER_API_KEY` → Llama-3.3-70B → echo fallback.
- **vector_db**: in-memory cosine-similarity store.

When `embedder` is a string spec it is dispatched via `_resolve_embedder_spec`. Format `"backend"` or `"backend:model"`. Recognized backends: `vertex`, `gemini`, `openai`, `cohere`, `st` (= `sentence-transformers`), `hash`. Examples:

```python
MothRAG(embedder="vertex")                          # default Vertex model in GDPR region (text-embedding-005)
MothRAG(embedder="vertex:text-embedding-005")       # explicit (default) Vertex 768-d model
MothRAG(embedder="gemini:gemini-embedding-2")       # Gemini Studio production model (3072-d)
MothRAG(embedder="gemini:text-embedding-005")       # Studio 768-d alternative
MothRAG(embedder="openai:text-embedding-3-small")
MothRAG(embedder="hash")                            # offline / smoke
```

See [Vertex AI setup](vertex-setup.md) for GCP project, IAM, and region configuration.

Recognized `config` keys (with defaults):

| Key | Default | Meaning |
|---|---|---|
| `top_k_chunks` | 10 | top-K retrieved per query |
| `chunk_size_tokens` | 400 | target chunk size (whitespace tokens) |
| `chunk_overlap_tokens` | 50 | overlap between adjacent chunks |
| `use_router` | True | apply `arm_subset()` classifier for routing provenance |

### `MothRAG.from_documents(source, **kwargs)`

Classmethod constructor that also ingests. `source` can be:

- `str | Path` — single file or directory (auto-dispatched via [`mothrag.loaders.auto_load`](loaders.md)).
- `list[str]` — raw text snippets (each becomes a Document).
- `list[Document]` — pre-built documents.

### `query(question: str, **kwargs) -> QueryResult`

End-to-end: embed → retrieve → route → read.

### `aquery(question: str, **kwargs) -> QueryResult`

Async wrapper around `query`. Useful inside `asyncio.gather` for concurrent queries.

### `batch_query(questions, max_workers=4, **kwargs) -> list[QueryResult]`

Parallel batched query via thread pool.

### `ingest(source)`

Incremental ingestion — add more documents to the existing index without rebuilding.

---

## Data classes

### `Document`

```python
@dataclass
class Document:
    text: str
    metadata: dict = {}
```

### `Chunk`

```python
@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_id: str
    metadata: dict
    embedding: list[float] | None
```

### `QueryResult`

```python
@dataclass
class QueryResult:
    answer: str
    retrieved_chunks: list[Chunk]
    arm_used: str
    arm_subset: list[str]
    confidence: float | None
    metadata: dict
```

---

## Protocols (for custom backends)

```python
class Embedder(Protocol):
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...

class Reader(Protocol):
    def read(self, question: str, passages: Sequence[str]) -> str: ...

class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk]) -> None: ...
    def retrieve(self, query_embedding: list[float], top_k: int = 10) -> list[Chunk]: ...
    def __len__(self) -> int: ...
```

Implement any of these (the others fall back to defaults) and pass via constructor.
