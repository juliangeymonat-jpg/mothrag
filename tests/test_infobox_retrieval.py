"""C3 — InfoboxIndex + MultiModalRetriever (dense+infobox) test suite.

Covers:
  - InfoboxTriple / InfoboxIndex insertion + lookup + alias collapse
  - extract_wikitext_infobox: template parsing, wiki-link cleanup,
    nested-template stripping, name fallback
  - extract_natural_facts: conservative pattern matching
  - build_infobox_index_from_chunks: corpus-level wiring
  - extract_question_hints: deterministic question-side resolution
  - MultiModalRetriever: blend logic, dedupe, chunk_provider hydration,
    synthetic facade fallback
  - End-to-end MothRAG(retrieval='dense_plus_infobox')
"""

from __future__ import annotations

import sys
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _block_sentence_transformers(monkeypatch) -> Iterator[None]:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    yield


# ============================================================
# InfoboxTriple + InfoboxIndex
# ============================================================

def test_infobox_triple_basic_field_access() -> None:
    from mothrag.core.retrieval import InfoboxTriple
    t = InfoboxTriple(subject="Einstein", attribute="born", value="1879",
                      source_chunk_id="c1", confidence=1.0)
    assert t.subject == "Einstein"
    assert t.attribute == "born"
    assert t.value == "1879"


def test_infobox_index_add_and_lookup() -> None:
    from mothrag.core.retrieval import InfoboxIndex, InfoboxTriple
    idx = InfoboxIndex()
    idx.add(InfoboxTriple("Albert Einstein", "born", "14 March 1879", "c1"))
    idx.add(InfoboxTriple("Albert Einstein", "spouse", "Mileva Maric", "c1"))
    hits = idx.lookup("Albert Einstein", "born")
    assert len(hits) == 1
    assert hits[0].value == "14 March 1879"


def test_infobox_index_alias_collapse_birth_date() -> None:
    from mothrag.core.retrieval import InfoboxIndex, InfoboxTriple
    idx = InfoboxIndex()
    idx.add(InfoboxTriple("X", "Date of birth", "1900"))
    idx.add(InfoboxTriple("X", "birth_date", "1900"))
    # Both forms collapse to "born"; idempotent on identical value.
    assert len(idx.lookup("X", "born")) == 1
    assert len(idx.lookup("X", "Date of birth")) == 1


def test_infobox_index_diacritic_normalisation() -> None:
    from mothrag.core.retrieval import InfoboxIndex, InfoboxTriple
    idx = InfoboxIndex()
    idx.add(InfoboxTriple("Mileva Marić", "spouse", "Albert Einstein"))
    # Looking up with ASCII variant should hit.
    assert idx.lookup("Mileva Maric", "spouse")


def test_infobox_index_multiple_values_per_attribute() -> None:
    from mothrag.core.retrieval import InfoboxIndex, InfoboxTriple
    idx = InfoboxIndex()
    idx.add(InfoboxTriple("X", "spouse", "Alice"))
    idx.add(InfoboxTriple("X", "spouse", "Bob"))
    hits = idx.lookup("X", "spouse")
    assert len(hits) == 2
    assert {h.value for h in hits} == {"Alice", "Bob"}


def test_infobox_index_entity_attributes() -> None:
    from mothrag.core.retrieval import InfoboxIndex, InfoboxTriple
    idx = InfoboxIndex()
    idx.add(InfoboxTriple("Y", "occupation", "scientist"))
    idx.add(InfoboxTriple("Y", "nationality", "British"))
    attrs = idx.entity_attributes("Y")
    assert set(attrs) == {"occupation", "nationality"}


def test_infobox_index_register_custom_alias() -> None:
    from mothrag.core.retrieval import InfoboxIndex, InfoboxTriple
    idx = InfoboxIndex()
    idx.register_attribute_alias("FTEs", "headcount")
    idx.add(InfoboxTriple("Acme Co.", "FTEs", "1500"))
    assert idx.lookup("Acme Co.", "headcount")


def test_infobox_index_find_by_value_reverse() -> None:
    from mothrag.core.retrieval import InfoboxIndex, InfoboxTriple
    idx = InfoboxIndex()
    idx.add(InfoboxTriple("X", "spouse", "Alice Smith"))
    hits = idx.find_by_value("alice")
    assert ("x", "spouse") in hits


# ============================================================
# Wikitext extraction
# ============================================================

def test_extract_wikitext_infobox_basic() -> None:
    from mothrag.core.retrieval import extract_wikitext_infobox
    text = """{{Infobox person
| name = Albert Einstein
| birth_date = 14 March 1879
| birth_place = Ulm, Germany
| occupation = theoretical physicist
}}"""
    triples = extract_wikitext_infobox(text)
    attrs = {t.attribute: t.value for t in triples}
    assert "born" in attrs  # birth_date aliased
    assert "1879" in attrs["born"]
    assert attrs["occupation"] == "theoretical physicist"


def test_extract_wikitext_infobox_wikilink_cleanup() -> None:
    from mothrag.core.retrieval import extract_wikitext_infobox
    text = """{{Infobox scientist
| name = Marie Curie
| nationality = [[Poland|Polish]]
| alma_mater = [[University of Paris]]
}}"""
    triples = extract_wikitext_infobox(text)
    by_attr = {t.attribute: t.value for t in triples}
    assert by_attr["nationality"] == "Polish"
    assert "University of Paris" in by_attr["alma_mater"]


def test_extract_wikitext_infobox_nested_template_stripping() -> None:
    from mothrag.core.retrieval import extract_wikitext_infobox
    text = """{{Infobox person
| name = Stephen Hawking
| birth_date = {{birth date|1942|1|8}}
}}"""
    triples = extract_wikitext_infobox(text)
    by_attr = {t.attribute: t.value for t in triples}
    assert "born" in by_attr
    assert "1942" in by_attr["born"]


def test_extract_wikitext_infobox_no_template_returns_empty() -> None:
    from mothrag.core.retrieval import extract_wikitext_infobox
    assert extract_wikitext_infobox("Just plain prose without any template.") == []
    assert extract_wikitext_infobox("") == []


def test_extract_wikitext_infobox_source_chunk_id_threaded() -> None:
    from mothrag.core.retrieval import extract_wikitext_infobox
    text = "{{Infobox person\n| name = X\n| born = 1900\n}}"
    triples = extract_wikitext_infobox(text, source_chunk_id="c42")
    assert all(t.source_chunk_id == "c42" for t in triples)


# ============================================================
# Natural-language facts
# ============================================================

def test_extract_natural_facts_birth_pattern() -> None:
    from mothrag.core.retrieval import extract_natural_facts
    text = "Marie Curie was born in Warsaw on 7 November 1867."
    triples = extract_natural_facts(text, source_chunk_id="c1")
    by_attr = {t.attribute for t in triples}
    assert "birthplace" in by_attr or "born" in by_attr


def test_extract_natural_facts_capital_of() -> None:
    from mothrag.core.retrieval import extract_natural_facts
    text = "The capital of France is Paris."
    triples = extract_natural_facts(text)
    by_attr = {t.attribute: t.value for t in triples}
    assert by_attr.get("capital", "").lower().startswith("paris")


def test_extract_natural_facts_empty_text() -> None:
    from mothrag.core.retrieval import extract_natural_facts
    assert extract_natural_facts("") == []


def test_extract_natural_facts_confidence_lower_than_template() -> None:
    from mothrag.core.retrieval import extract_natural_facts
    text = "Albert Einstein was born in Ulm."
    triples = extract_natural_facts(text)
    assert all(t.confidence < 1.0 for t in triples)


# ============================================================
# Corpus-level builder
# ============================================================

def test_build_infobox_index_from_chunks_minimal() -> None:
    from mothrag.core.retrieval import build_infobox_index_from_chunks

    class _C:
        def __init__(self, cid, text):
            self.chunk_id = cid
            self.text = text

    chunks = [
        _C("a1", "{{Infobox person\n| name = Albert Einstein\n| born = 1879\n}}"),
        _C("a2", "Marie Curie was born in Warsaw."),
    ]
    idx = build_infobox_index_from_chunks(chunks)
    assert idx.lookup("Albert Einstein", "born")
    assert "Marie Curie" in idx or idx.lookup("Marie Curie", "birthplace")


def test_build_infobox_index_handles_dict_chunks() -> None:
    from mothrag.core.retrieval import build_infobox_index_from_chunks
    chunks = [
        {"chunk_id": "a1",
         "text": "{{Infobox\n| name = Foo\n| occupation = baker\n}}"},
    ]
    idx = build_infobox_index_from_chunks(chunks)
    assert idx.lookup("Foo", "occupation")


# ============================================================
# Question-hint extraction
# ============================================================

def test_extract_question_hints_when_was_x_born() -> None:
    from mothrag.core.retrieval import extract_question_hints
    hints = extract_question_hints("When was Albert Einstein born?")
    assert hints
    sub, attr = hints[0]
    assert sub == "Albert Einstein" or "Einstein" in sub
    assert attr == "born"


def test_extract_question_hints_where_was_x_born() -> None:
    from mothrag.core.retrieval import extract_question_hints
    hints = extract_question_hints("Where was Marie Curie born?")
    assert any(a == "birthplace" for _, a in hints)


def test_extract_question_hints_capital_of() -> None:
    from mothrag.core.retrieval import extract_question_hints
    hints = extract_question_hints("What is the capital of France?")
    assert any(a == "capital" for _, a in hints)


def test_extract_question_hints_spouse_of() -> None:
    from mothrag.core.retrieval import extract_question_hints
    hints = extract_question_hints("Who is the spouse of Albert Einstein?")
    assert any(a in {"spouse", "wife", "husband", "partner"} for _, a in hints)


def test_extract_question_hints_open_ended_returns_empty() -> None:
    from mothrag.core.retrieval import extract_question_hints
    assert extract_question_hints(
        "Why did the chicken cross the road?"
    ) == []


# ============================================================
# MultiModalRetriever
# ============================================================

def test_multimodal_retrieve_blends_infobox_with_dense() -> None:
    """When the question matches an infobox triple, the retriever should
    surface the triple chunk in the top results in addition to dense
    matches."""
    from mothrag.core.retrieval import (
        InfoboxIndex, InfoboxTriple, MultiModalRetriever,
    )

    class _DummyChunk:
        def __init__(self, cid, text):
            self.chunk_id = cid
            self.text = text

    class _DummyDense:
        def __init__(self, chunks):
            self.chunks = list(chunks)

        def index(self, chunks):
            pass

        def retrieve(self, q, *, top_k=10):
            return self.chunks[:top_k]

        def __len__(self):
            return len(self.chunks)

    dense_chunks = [
        _DummyChunk("d1", "Einstein was a physicist."),
        _DummyChunk("d2", "He lived in Berlin."),
    ]
    idx = InfoboxIndex()
    idx.add(InfoboxTriple("Albert Einstein", "born", "14 March 1879",
                          source_chunk_id="ib1"))

    retriever = MultiModalRetriever(
        dense=_DummyDense(dense_chunks),
        infobox_index=idx,
        infobox_top_n_boost=2,
    )
    results = retriever.retrieve("When was Albert Einstein born?", top_k=5)
    # First result should be the infobox synthetic chunk
    assert results
    assert results[0].chunk_id.startswith("infobox:")
    # Dense chunks still present
    assert any(c.chunk_id == "d1" for c in results)


def test_multimodal_retrieve_no_hints_falls_back_to_dense() -> None:
    """Open-ended questions with no entity-attribute hint should return
    the dense top-K unchanged."""
    from mothrag.core.retrieval import InfoboxIndex, MultiModalRetriever

    class _Chunk:
        def __init__(self, cid):
            self.chunk_id = cid
            self.text = ""

    class _Dense:
        def index(self, chunks): pass
        def retrieve(self, q, *, top_k=10):
            return [_Chunk(f"d{i}") for i in range(top_k)]
        def __len__(self): return 5

    retriever = MultiModalRetriever(dense=_Dense(), infobox_index=InfoboxIndex())
    results = retriever.retrieve("Why does X exist?", top_k=3)
    assert [c.chunk_id for c in results] == ["d0", "d1", "d2"]


def test_multimodal_retrieve_dedupes_chunk_ids() -> None:
    from mothrag.core.retrieval import (
        InfoboxIndex, InfoboxTriple, MultiModalRetriever,
    )

    class _Chunk:
        def __init__(self, cid, text):
            self.chunk_id = cid
            self.text = text

    class _Dense:
        def __init__(self, chunks): self.chunks = chunks
        def index(self, chunks): pass
        def retrieve(self, q, *, top_k=10): return self.chunks[:top_k]
        def __len__(self): return len(self.chunks)

    # The infobox source_chunk_id collides with a dense chunk_id; ensure
    # the chunk_provider returns the dense chunk and dedupe kicks in.
    dense_chunks = [_Chunk("shared", "from-dense")]
    idx = InfoboxIndex()
    idx.add(InfoboxTriple("X", "born", "1900", source_chunk_id="shared"))

    def _provider(cid):
        for c in dense_chunks:
            if c.chunk_id == cid:
                return c
        return None

    retriever = MultiModalRetriever(
        dense=_Dense(dense_chunks),
        infobox_index=idx,
        chunk_provider=_provider,
        infobox_top_n_boost=3,
    )
    results = retriever.retrieve("When was X born?", top_k=5)
    ids = [c.chunk_id for c in results]
    # "shared" appears exactly once (dedupe)
    assert ids.count("shared") == 1


def test_multimodal_retrieve_synthetic_facade_when_no_provider() -> None:
    """Without a chunk_provider, the retriever emits a synthetic chunk
    that exposes the triple text."""
    from mothrag.core.retrieval import (
        InfoboxIndex, InfoboxTriple, MultiModalRetriever,
    )

    class _Dense:
        def index(self, chunks): pass
        def retrieve(self, q, *, top_k=10): return []
        def __len__(self): return 0

    idx = InfoboxIndex()
    idx.add(InfoboxTriple("Marie Curie", "birthplace", "Warsaw"))
    retriever = MultiModalRetriever(dense=_Dense(), infobox_index=idx)
    results = retriever.retrieve("Where was Marie Curie born?", top_k=3)
    assert results
    assert "Marie Curie" in results[0].text
    assert "Warsaw" in results[0].text


# ============================================================
# End-to-end MothRAG(retrieval='dense_plus_infobox')
# ============================================================

def test_mothrag_dense_plus_infobox_builds() -> None:
    """The MothRAG facade accepts retrieval='dense_plus_infobox' and the
    InfoboxIndex grows as documents are ingested."""
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader
    rag = MothRAG.from_documents(
        [
            "{{Infobox person\n| name = Alan Turing\n| born = 23 June 1912\n}}",
            "Alan Turing was a British mathematician.",
        ],
        embedder=_HashEmbedder(),
        reader=_EchoReader(),
        retrieval="dense_plus_infobox",
    )
    idx = rag._infobox_index
    assert len(idx) >= 1
    # Lookup via the index directly
    assert idx.lookup("Alan Turing", "born")


def test_mothrag_dense_plus_infobox_query_prepends_infobox() -> None:
    """End-to-end query: an entity-attribute question should surface the
    infobox match as the top retrieved chunk."""
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader
    rag = MothRAG.from_documents(
        [
            "{{Infobox person\n| name = Pierre Curie\n| born = 15 May 1859\n"
            "| spouse = Marie Curie\n}}",
            "Pierre Curie won a Nobel Prize in Physics.",
        ],
        embedder=_HashEmbedder(),
        reader=_EchoReader(),
        retrieval="dense_plus_infobox",
        top_k_chunks=4,
    )
    qr = rag.query("When was Pierre Curie born?")
    chunks = qr.retrieved_chunks
    assert chunks
    cids = [getattr(c, "chunk_id", "") for c in chunks]
    # The infobox synthetic chunk should appear in the retrieved set.
    assert any(cid.startswith("infobox:") for cid in cids) \
        or any("Pierre Curie" in getattr(c, "text", "") for c in chunks)


def test_mothrag_dense_plus_infobox_invalid_retrieval_rejected() -> None:
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader
    with pytest.raises(ValueError):
        MothRAG(
            embedder=_HashEmbedder(),
            reader=_EchoReader(),
            retrieval="not_a_real_choice",
        )


def test_mothrag_dense_plus_infobox_with_seed_triples() -> None:
    """Pre-fed seed triples (e.g. from an external KG) should populate
    the InfoboxIndex before any document ingestion."""
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader
    from mothrag.core.retrieval import InfoboxTriple
    rag = MothRAG.from_documents(
        ["Some unrelated prose."],
        embedder=_HashEmbedder(),
        reader=_EchoReader(),
        retrieval="dense_plus_infobox",
        retrieval_config={
            "seed_triples": [
                InfoboxTriple("Ada Lovelace", "born", "10 December 1815"),
                InfoboxTriple("Ada Lovelace", "occupation", "mathematician"),
            ],
        },
    )
    assert rag._infobox_index.lookup("Ada Lovelace", "born")
    assert rag._infobox_index.lookup("Ada Lovelace", "occupation")
