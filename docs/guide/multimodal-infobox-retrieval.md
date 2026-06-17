# Multi-modal retrieval — dense + structured infobox

`retrieval="dense_plus_infobox"` blends the v0.5.0 alpha dense
retriever with a structured `(subject, attribute, value)` triple index
harvested from the same chunks at ingest time. For entity-attribute
questions — "When was X born?", "Who is Y's spouse?", "What is the
capital of Z?" — the infobox lookup surfaces a near-perfect precision
hit; for open-ended Wh-questions, the dense recall is preserved
unchanged.

## When to enable

Enable `retrieval="dense_plus_infobox"` when **both** are true:

1. The corpus contains structured entity pages with **either** Wikipedia
   `{{Infobox ...}}` templates **or** prose patterns the conservative
   natural-language fact extractor recognises (biographical / capital-of
   / nationality patterns).
2. The workload includes a meaningful share of entity-attribute
   questions. In the failure-pathway analysis, ~11% of failures are
   *plain-attribute* lookups where dense retrieval misses the canonical
   fact.

Skip it when the corpus is purely conversational prose or the workload
is reasoning-heavy with little entity-attribute structure — the
infobox blend is zero-cost on misses but still consumes a few extra
chunks in the top-K when it fires.

## Minimal example

```python
from mothrag import MothRAG

rag = MothRAG.from_documents(
    "wiki_corpus/",
    production=True,
    retrieval="dense_plus_infobox",
    top_k_chunks=10,
)

qr = rag.query("When was Albert Einstein born?")
print(qr.answer)
# The first chunk in qr.retrieved_chunks is the infobox hit:
#   chunk_id="infobox:albert_einstein:born"
#   text="Albert Einstein -- born: 14 March 1879"
```

## Configuration

`retrieval_config={...}` accepts:

| key                    | type                   | default | meaning                                                        |
|------------------------|------------------------|--------:|----------------------------------------------------------------|
| `infobox_top_n_boost`  | `int`                  |     `3` | Max number of infobox chunks prepended to dense top-K          |
| `hint_extractor`       | `Callable[str, list]`  |  `None` | Override the deterministic question-side hint parser           |
| `seed_triples`         | `Iterable[InfoboxTriple]` |  `()` | Pre-fed external triples (Wikidata, enterprise KG, ...)         |

```python
from mothrag.core.retrieval import InfoboxTriple

rag = MothRAG.from_documents(
    "docs/",
    retrieval="dense_plus_infobox",
    retrieval_config={
        "infobox_top_n_boost": 2,
        "seed_triples": [
            InfoboxTriple("Ada Lovelace", "born", "10 December 1815"),
            InfoboxTriple("Ada Lovelace", "occupation", "mathematician"),
        ],
    },
)
```

## How extraction works

At `ingest()` time, each chunk is scanned by two complementary
extractors. Both are best-effort and emit zero triples on chunks
without recognisable structured patterns.

### `extract_wikitext_infobox(text, *, subject=None, source_chunk_id="")`

Parses canonical Wikipedia `{{Infobox ...}}` templates. Robust to
template-name variation; cleans `[[wiki|links]]` and strips nested
`{{template|args}}` invocations. Defaults to `confidence=1.0`.

```python
from mothrag.core.retrieval import extract_wikitext_infobox

triples = extract_wikitext_infobox("""
{{Infobox person
| name = Marie Curie
| birth_date = 7 November 1867
| nationality = [[Poland|Polish]]
}}
""")
# triples[0]  -> InfoboxTriple(subject="Marie Curie",
#                              attribute="born",  # birth_date aliased
#                              value="7 November 1867", ...)
```

### `extract_natural_facts(text, *, source_chunk_id="")`

Conservative regex patterns for biographical / geographical claims
("X was born in Y", "the capital of Z is W", ...). Each pattern is
high-precision; ambiguous sentences refuse rather than guess.
Defaults to `confidence=0.7` (lower than wikitext to reflect prose
ambiguity).

## Attribute alias map

The index normalises common synonyms to canonical attribute keys so
question-side lookups don't require knowing the corpus's exact column
names. The built-in map handles:

| canonical key  | aliases collapsed                                        |
|----------------|----------------------------------------------------------|
| `born`         | `date_of_birth`, `birth_date`, `born_date`               |
| `died`         | `date_of_death`, `death_date`                            |
| `birthplace`   | `place_of_birth`, `birth_place`                          |
| `deathplace`   | `place_of_death`, `death_place`                          |
| `occupation`   | `profession`                                             |
| `nationality`  | `citizenship`                                            |
| `spouse`       | `spouses`, `spouse_s`, `partner`                         |
| `parents`      | `parent`                                                 |
| `headquarters` | `headquartered`, `headquartered_in`, `hq`                |
| `capital`      | `capital_city`                                           |

Extend with `index.register_attribute_alias(alias, canonical)` for
domain-specific keys.

## Question-side hint extraction

The deterministic `extract_question_hints(question)` parser
recognises:

- `When/On what date was X born?` → `(X, born)`
- `Where was X born?` → `(X, birthplace)`
- `When did X die?` → `(X, died)`
- `Where did X die?` → `(X, deathplace)`
- `What is the capital of X?` → `(X, capital)`
- `Who is X's {spouse|wife|husband|partner}?` → `(X, spouse)`
- `Who is the {spouse|father|director|author|...} of X?` → `(X, ...)`
- `What is X's {nationality|occupation|religion}?` → `(X, ...)`

For more complex questions, pass `hint_extractor=your_llm_ner_fn`
that returns `list[(subject, attribute)]`. Failures in the custom
extractor fall back silently to the deterministic parser.

## Composition with other retrieval modes

Only one retrieval mode is active per MothRAG instance. If you need
graph fusion **and** infobox lookups, you can build a custom Retriever
that wraps `HybridGraphRetriever` as the dense leg of
`MultiModalRetriever`:

```python
from mothrag import MothRAG
from mothrag.core.retrieval import (
    HybridGraphRetriever, MultiModalRetriever, InfoboxIndex,
    build_infobox_index_from_chunks,
)

# Build the graph retriever and pre-built infobox index separately.
graph = HybridGraphRetriever(embedder=..., reader_llm=...,
                              save_dir=".hipporag-cache")
infobox_idx = build_infobox_index_from_chunks(my_chunks)

# Compose:
custom = MultiModalRetriever(dense=graph, infobox_index=infobox_idx)

rag = MothRAG.from_documents(
    "docs/",
    retrieval="dense",          # bypassed by `retriever=`
    retriever=custom,
)
```

## Limitations

- **Pattern recall**: the natural-language extractor only covers
  ~10 sentence templates. Domain-specific phrases (e.g. legal /
  medical) need either a wider pattern set or an LLM-NER pass that
  emits `InfoboxTriple` records into `retrieval_config['seed_triples']`.
- **Synthetic chunks vs source chunks**: when `chunk_provider` is
  unset and the triple has no `source_chunk_id`, the retriever emits
  a one-line synthetic chunk (`"Subject -- attribute: value"`). The
  reader sees the fact but no surrounding context. For paper-grade
  workloads, ensure the underlying corpus is indexed so triples can
  hydrate to their source chunks.
- **No reranking**: blended results are a prepend, not a re-rank. If
  multiple high-confidence infobox triples match the same question,
  they all appear in `top_n_boost` order before the dense top-K.

## File layout

| File                                            | Role                                               |
|-------------------------------------------------|----------------------------------------------------|
| `mothrag/core/retrieval/infobox.py`             | `InfoboxIndex`, `InfoboxTriple`, extractors        |
| `mothrag/core/retrieval/multimodal.py`          | `MultiModalRetriever`, `extract_question_hints`    |
| `mothrag/core/api.py`                           | Dispatch `retrieval="dense_plus_infobox"`          |
| `tests/test_infobox_retrieval.py`               | 32-case unit + integration suite                   |
| `docs/guide/multimodal-infobox-retrieval.md`    | This guide                                         |
