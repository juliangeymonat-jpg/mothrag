# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""HybridGraphRetriever -- wraps the OSU-NLP-Group HippoRAG2 SDK
(Apache 2.0, github.com/OSU-NLP-Group/HippoRAG) as a MothRag
:class:`mothrag.core.retrieval.Retriever`.

The wrapper drives HippoRAG2's personalized-PageRank-over-passage-entity-
graph retrieval pipeline while leaving MothRag's three reading arms
(V3+bu / decompose / iter) and arbitration logic completely untouched.
Same embedder + reader + arbitrator as the dense baseline; only the
retrieval base changes.

Design notes
------------

The wrapper is intentionally thin: it constructs a HippoRAG2 instance
configured to match the MothRag embedder choice (so the dense-leg of the
hybrid retrieval shares the same vector space as the rest of the
pipeline), calls ``hipporag.index(docs=passages)`` once during MothRag
ingestion, and calls ``hipporag.retrieve(queries=[question],
num_to_retrieve=top_k)`` per query. The result list is unwrapped to a
sequence of :class:`mothrag.core.api.Chunk` instances so MothRag arms
consume it identically to the dense path.

The HippoRAG2 dependency is optional. Install via
``pip install mothrag[hybrid-graph]``. The wrapper lazy-imports
``hipporag`` only at construction time so importing
:mod:`mothrag.core.retrieval` stays cheap on minimal installs.

Graph cache
-----------

HippoRAG2 graph construction is the expensive step (~tens of minutes
to hours depending on corpus size + LLM throughput). The wrapper
delegates caching to HippoRAG2 via its ``save_dir`` config: if the
directory already contains a built graph, HippoRAG2 reuses it; otherwise
it builds and saves. The wrapper exposes the ``save_dir`` kwarg so
deployments can point at a pre-built graph published alongside the
HippoRAG2 paper (recommended for the standard multi-hop QA benchmarks).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


# Default HippoRAG2 model identifiers chosen for cross-comparison parity
# with the MothRag baseline (Llama-3.3-70B reader + Gemini-Embedding-001
# at retrieval time). Override via constructor kwargs if the deployment
# uses a different upstream stack.
_DEFAULT_LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
_DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
_DEFAULT_SAVE_DIR_TEMPLATE = ".mothrag-hipporag-cache"


class HybridGraphRetriever:
    """HippoRAG2 dense+graph retrieval wrapper.

    Parameters
    ----------
    save_dir
        Local directory used as HippoRAG2's ``save_dir`` for graph
        artifacts. If the directory already contains a pre-built graph,
        HippoRAG2 reuses it; otherwise the graph is built once on the
        first ``index(...)`` call and re-used on subsequent runs.
        Defaults to ``.mothrag-hipporag-cache`` under the current
        working directory.
    llm_model_name
        Identifier passed to HippoRAG2 for the entity-extraction LLM.
        Defaults to ``meta-llama/Llama-3.3-70B-Instruct-Turbo`` for
        cross-comparison parity with the MothRag baseline reader.
    embedding_model_name
        Identifier passed to HippoRAG2 for the dense retrieval embedder.
        Defaults to ``gemini-embedding-001`` for parity with the MothRag
        baseline embedder.
    hipporag_config
        Optional dict merged into HippoRAG2's ``GlobalConfig`` so
        deployments can override the full HippoRAG2 surface
        (``synonymy_edge_topk``, ``passage_node_weight``, ``damping``,
        etc.) without touching the wrapper.
    embedder, reader_llm
        Optional MothRag embedder / reader instances retained for
        introspection; the wrapper does NOT pass them to HippoRAG2
        directly (HippoRAG2 instantiates its own SDK clients via model
        names) -- they are surfaced here so MothRAG.from_documents can
        plumb the same MothRag-side defaults transparently.
    """

    name = "hybrid_graph"

    def __init__(
        self,
        *,
        save_dir: str | Path | None = None,
        llm_model_name: str = _DEFAULT_LLM_MODEL,
        embedding_model_name: str = _DEFAULT_EMBEDDING_MODEL,
        hipporag_config: dict | None = None,
        embedder: Any = None,
        reader_llm: Any = None,
    ) -> None:
        try:
            import hipporag  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "HybridGraphRetriever requires the HippoRAG2 SDK. "
                "Install via `pip install mothrag[hybrid-graph]` or "
                "`pip install hipporag`."
            ) from e

        from hipporag import HippoRAG
        from hipporag.utils.config_utils import BaseConfig

        self.save_dir = Path(save_dir or _DEFAULT_SAVE_DIR_TEMPLATE).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.embedder = embedder
        self.reader_llm = reader_llm

        cfg_kwargs: dict[str, Any] = {
            "save_dir": str(self.save_dir),
            "llm_model_name": llm_model_name,
            "embedding_model_name": embedding_model_name,
        }
        if hipporag_config:
            cfg_kwargs.update(hipporag_config)
        try:
            global_config = BaseConfig(**cfg_kwargs)
        except TypeError:
            # HippoRAG2's BaseConfig signature has shifted across releases;
            # surface a clear error so callers can pin the right SDK
            # version via the [hybrid-graph] extra.
            raise RuntimeError(
                "Unsupported hipporag BaseConfig kwargs. The MothRag "
                "hybrid-graph wrapper targets HippoRAG2 >= 1.0; update "
                "`mothrag[hybrid-graph]` or pass an explicit "
                "`hipporag_config={}` dict matching your SDK version."
            )
        self._hipporag = HippoRAG(global_config=global_config)
        self._chunks_by_passage: dict[str, Any] = {}
        self._n_indexed = 0

    def index(self, chunks: Sequence) -> None:
        """Build (or reuse) the HippoRAG2 graph over the chunk corpus.

        The HippoRAG2 SDK is responsible for the heavy graph construction
        work; the wrapper passes the chunk texts through unchanged and
        retains a passage-text -> Chunk lookup table so retrieval can
        reconstruct full :class:`mothrag.core.api.Chunk` objects from
        HippoRAG2's text-only return values.
        """
        chunks = list(chunks)
        if not chunks:
            return
        passages = []
        for c in chunks:
            text = getattr(c, "text", None) or ""
            if not text:
                continue
            passages.append(text)
            self._chunks_by_passage[text] = c

        # HippoRAG2's index API has had two signatures across releases;
        # we try the modern keyword form first then fall back.
        try:
            self._hipporag.index(docs=passages)
        except TypeError:
            self._hipporag.index(passages)
        self._n_indexed += len(passages)

    def retrieve(self, question: str, *, top_k: int = 10) -> list:
        """Run HippoRAG2 dense + PPR retrieval, return MothRag Chunks."""
        from mothrag.core.api import Chunk

        if not self._n_indexed:
            return []
        try:
            results = self._hipporag.retrieve(
                queries=[question], num_to_retrieve=top_k,
            )
        except TypeError:
            results = self._hipporag.retrieve([question], top_k)
        out: list[Chunk] = []
        # HippoRAG2's retrieve returns either a list[QuerySolution]-like
        # objects or a list[list[str]] depending on SDK version. Both
        # shapes are unwrapped here.
        if not results:
            return out
        first = results[0]
        ranked_passages = self._extract_ranked_passages(first)
        for text in ranked_passages[:top_k]:
            chunk = self._chunks_by_passage.get(text)
            if chunk is None:
                chunk = Chunk(text=text, doc_id="", chunk_id="",
                              metadata={"source": "hipporag_passage"})
            out.append(chunk)
        return out

    def __len__(self) -> int:
        return self._n_indexed

    @staticmethod
    def _extract_ranked_passages(payload: Any) -> list[str]:
        """Best-effort unwrap of HippoRAG2's per-query result shape.

        Modern releases return a ``QuerySolution`` dataclass with a
        ``docs`` (or ``retrieved_docs``) attribute carrying the ranked
        passage texts; older releases return a bare ``list[str]``.
        """
        if isinstance(payload, list):
            return [p if isinstance(p, str) else getattr(p, "text", str(p))
                    for p in payload]
        for attr in ("docs", "retrieved_docs", "passages", "results"):
            cand = getattr(payload, attr, None)
            if isinstance(cand, list):
                return [p if isinstance(p, str) else getattr(p, "text", str(p))
                        for p in cand]
        return []


__all__ = ["HybridGraphRetriever"]
