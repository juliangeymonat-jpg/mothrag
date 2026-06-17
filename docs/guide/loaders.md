# Document loaders

The `mothrag.loaders` package converts files to `Document` objects via `auto_load`. v0.5.0 ships two loaders.

## `auto_load(source)`

```python
from mothrag.loaders import auto_load

docs = auto_load("path/to/file.txt")       # single file
docs = auto_load("path/to/folder/")        # directory (recursive)
docs = auto_load(["a.txt", "b.json"])      # list of paths
```

Dispatch by extension; concatenates results.

## Built-in loaders

### `load_text` — `.txt`, `.md`, `.markdown`

Reads the entire file as a single `Document`. Metadata: `source`, `filename`, `format`, `size_bytes`.

```python
from mothrag.loaders import load_text
docs = load_text("README.md")
```

### `load_json` — `.json`, `.jsonl`

Supports four input shapes:

- list of strings → one `Document` per string
- list of objects → each object becomes a `Document`; the loader scans keys `text`, `content`, `body`, `passage`, `answer` (first hit wins) and treats the rest as metadata
- single object → one `Document` (same key lookup)
- `.jsonl` → each line is parsed as one of the above

```python
from mothrag.loaders import load_json
docs = load_json("corpus.jsonl")
```

## Deferred formats

PDF (`.pdf`), HTML (`.html`/`.htm`), DOCX (`.docx`) raise `NotImplementedError` with an explicit message pointing at the v0.5.0 workaround: pre-extract text externally and pass strings via `MothRAG.from_documents([...])`.

Native support planned for v0.5.1.
