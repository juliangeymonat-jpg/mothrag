# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""End-to-end MothRAG retrieval + reader pipeline.

Programmatic API used by the public ``mothrag.simple.run`` entry point and by
the CLI eval script. Encapsulates the V3+bu retrieval recipe (cosine + BM25
candidate set with safety net, cross-encoder rerank, optional bottom-up
boost) and the V3-think reader prompt with optional self-consistency voting.

Usage::

    from mothrag.eval.pipeline import MothRAGPipeline
    pipe = MothRAGPipeline.from_corpus(
        data_dir="data_wiki_hotpot500",
        embedding="st-mini",
        reader_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    )
    answer, info = pipe.answer("What movie did the wife of Inception's director star in?")
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from mothrag.core.anchor import Anchor
from mothrag.core.mothrag import (
    EntryPointClassifier,
    ContextGraphBuilder,
    NavigationPolicyHeuristic,
    build_anchor_registry,
)
from mothrag.core.symbolic_memory import SymbolicMemoryStore
from mothrag.eval.metrics import normalize_answer
from mothrag.utils.url_safety import validate_base_url


# ---- Reader prompts (V1, V2, V3-think, V3-think-cite) ----

READER_SYSTEM_V1 = """You are a question answering assistant. Answer the user's question using ONLY the provided passages. Be concise: respond with the exact phrase or short answer (2-10 words). Do not explain or add commentary."""

READER_SYSTEM_V2 = """You are a HotpotQA-style answer extractor. Extract the EXACT answer from the passages.

STRICT FORMAT RULES (failure to follow = wrong answer):
1. Reply with ONLY the answer text. No prefix like "Yes,", "The answer is", "Sure,". No trailing period.
2. Yes/no question: reply EXACTLY "yes" or "no" (lowercase, nothing else).
3. "Who" / "Which" entity question: reply with the SHORTEST entity name that answers.
4. "How many" / "When" / "Where" question: reply with the number / year / place ONLY.
5. If passages contain multiple candidate answers, pick the one matching ALL constraints.
6. If passages don't contain the answer, reply: "Not in passages"."""

READER_SYSTEM_V3_THINK = """You are a HotpotQA-style multi-hop reasoner. You will:

1. SCAN the passages and IDENTIFY which one(s) contain facts relevant to the question.
2. EXTRACT the relevant facts (cite passage number).
3. COMBINE the facts if the question is multi-hop.
4. WRITE the final answer.

ANSWER FORMAT (HotpotQA dataset style — match the gold's natural style):
- Yes/no question: EXACTLY "yes" or "no" (lowercase, nothing else).
- For all other questions, COPY the answer span VERBATIM from the most relevant passage. Do NOT abbreviate, paraphrase, or shorten.
- The answer is usually 1-10 words. Pick the contiguous span that most directly answers the question.
- No prefix ("The answer is", "Yes,"), no trailing period, no quotes, no parenthetical role, no commentary.
- If passages don't contain the answer, write "Not in passages".

OUTPUT FORMAT (mandatory — labels exactly as below):
Reasoning:
- Step 1 (identify): <which passages [N] are relevant and why>
- Step 2 (extract): <the key fact span(s), quoting verbatim>
- Step 3 (combine): <if multi-hop, how facts combine; else "single-hop">
Final answer: <verbatim span from passage, or yes/no>"""

READER_SYSTEM_V3_THINK_CITE = READER_SYSTEM_V3_THINK + """

CITATION REQUIREMENT (mandatory — for grounding eval):
After Step 3, you MUST include a 'Cited passages:' line listing the passage IDs you used to derive the answer in [N1, N2, ...] format. Cite ONLY passages whose content you used; do NOT cite passages you did not use."""


def _is_openai_reasoning_model(model: str) -> bool:
    m = model.lower()
    return any(p in m for p in ["gpt-5.4", "gpt-5.5", "gpt-5-pro", "o1", "o3", "o4"])


def _is_anthropic_thinking_model(model: str) -> bool:
    m = model.lower()
    return any(p in m for p in ["claude-opus-4-6", "claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"])


def _reader_kwargs(model: str, messages, max_tokens: int, temperature: float):
    if _is_openai_reasoning_model(model):
        return dict(model=model, messages=messages, max_completion_tokens=max_tokens)
    if _is_anthropic_thinking_model(model):
        return dict(model=model, messages=messages, max_tokens=max_tokens)
    return dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)


def make_reader_messages(question: str, passages: list[str],
                         prompt_version: str = "v3-think") -> list[dict]:
    ctx = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    if prompt_version == "v3-think-cite":
        system = READER_SYSTEM_V3_THINK_CITE
        user_msg = f"Passages:\n{ctx}\n\nQuestion: {question}\n\nReasoning:"
    elif prompt_version == "v3-think":
        system = READER_SYSTEM_V3_THINK
        user_msg = f"Passages:\n{ctx}\n\nQuestion: {question}\n\nReasoning:"
    elif prompt_version == "v2":
        system = READER_SYSTEM_V2
        user_msg = f"Passages:\n{ctx}\n\nQuestion: {question}\nA:"
    else:
        system = READER_SYSTEM_V1
        user_msg = f"Passages:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]


def parse_reader_output(raw: str, prompt_version: str = "v3-think") -> str:
    """Extract the final answer string from raw reader output."""
    if prompt_version not in ("v3-think", "v3-think-cite"):
        return raw.strip()
    for marker in ("Final answer:", "Final Answer:", "FINAL ANSWER:", "final answer:"):
        idx = raw.find(marker)
        if idx >= 0:
            ans = raw[idx + len(marker):].strip()
            return ans.split("\n")[0].strip().rstrip(".").strip()
    lines = [ln.strip() for ln in raw.strip().split("\n") if ln.strip()]
    return (lines[-1] if lines else "").rstrip(".").strip()


def call_reader_with_usage(client, model: str, messages: list[dict],
                           max_tokens: int = 64, temperature: float = 0.0):
    """Returns ``(text, usage)`` where ``usage = {prompt_tokens, completion_tokens, latency_s}``."""
    t0 = time.time()
    resp = client.chat.completions.create(**_reader_kwargs(model, messages, max_tokens, temperature))
    latency_s = time.time() - t0
    text = resp.choices[0].message.content.strip()
    pt = ct = 0
    u = getattr(resp, "usage", None)
    if u is not None:
        pt = getattr(u, "prompt_tokens", 0) or 0
        ct = getattr(u, "completion_tokens", 0) or 0
    return text, {"prompt_tokens": int(pt), "completion_tokens": int(ct),
                  "latency_s": float(latency_s)}


def call_reader_majority_with_usage(client, model: str, messages: list[dict],
                                    max_tokens: int, n_samples: int,
                                    temperature, reader_prompt: str):
    """K=N self-consistency: call reader N times, majority-vote on parsed answer.

    ``temperature`` may be a float (single-temp Wang 2022 standard) OR a list
    of floats (temp-spread variant — e.g. ``[0.0, 0.3, 0.5, 0.7, 0.9]``). When
    list, ``n_samples`` is overridden by ``len(temperature)``; each sample uses
    one temperature value.
    """
    # Support temp-spread: list of temperatures
    if isinstance(temperature, (list, tuple)):
        temps = list(temperature)
        n_samples = len(temps)
    else:
        temps = [float(temperature)] * n_samples

    if n_samples <= 1:
        return call_reader_with_usage(client, model, messages, max_tokens,
                                       temps[0] if temps else float(temperature))
    raws = []
    parseds = []
    total_pt = total_ct = 0
    total_lat = 0.0
    for t in temps:
        try:
            text, usage = call_reader_with_usage(client, model, messages, max_tokens, t)
            raws.append(text)
            parseds.append(parse_reader_output(text, reader_prompt))
            total_pt += usage["prompt_tokens"]
            total_ct += usage["completion_tokens"]
            total_lat += usage["latency_s"]
        except Exception:  # noqa: BLE001
            raws.append("")
            parseds.append("")
    normalized = [normalize_answer(p) for p in parseds]
    counter = Counter([n for n in normalized if n])
    agg_usage = {"prompt_tokens": total_pt, "completion_tokens": total_ct, "latency_s": total_lat}
    if not counter:
        return (raws[0] if raws else ""), agg_usage
    winning_norm, _ = counter.most_common(1)[0]
    for raw, norm in zip(raws, normalized):
        if norm == winning_norm:
            return raw, agg_usage
    return raws[0], agg_usage


# ---- Pipeline configuration + result types ----

@dataclass
class PipelineConfig:
    """Knobs for :class:`MothRAGPipeline`. Defaults match the paper's
    "default" config (V3+bu, BGE cross-encoder rerank, V3-think reader)."""

    embedding: str = "st-mini"
    reranker: str = "bge-rerank"  # "bge-rerank" | "bm25" | "none"
    bottom_up_boost: float = 1.0
    bottom_up_max_hops: int = 2
    top_k_chunks: int = 10
    top_k_candidates: int = 30
    reader_prompt: str = "v3-think"
    reader_max_tokens: int = 2000
    reader_temperature: float | list[float] = 0.0  # float OR list[float] for temp-spread SC
    reader_n_samples: int = 1
    no_safety_net: bool = False


@dataclass
class AnswerInfo:
    answer: str
    raw: str = ""
    passages: list[str] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    route: str = ""
    top1_conf: float = 0.0
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---- Main pipeline ----

def _tokenize(text: str) -> list[str]:
    import re
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", text)]


class MothRAGPipeline:
    """Self-contained retrieval + reader pipeline.

    Construct via :meth:`from_corpus` (which handles loading + indexing) or by
    passing pre-built components.
    """

    def __init__(self, *,
                 entities: list[dict],
                 edges: list[dict],
                 chunks: list[dict],
                 chunk_vecs: np.ndarray,
                 chunk_ids: list[str],
                 chunks_by_id: dict,
                 chunks_per_entity: dict[str, list[int]],
                 entities_by_id: dict,
                 plugin,
                 anchors: list[Anchor],
                 classifier: EntryPointClassifier,
                 builder: ContextGraphBuilder,
                 policy: NavigationPolicyHeuristic,
                 bm25,
                 cross_encoder,
                 sym_store: Optional[SymbolicMemoryStore],
                 query_embedder,
                 reader_client,
                 reader_model: str,
                 config: PipelineConfig):
        self.entities = entities
        self.edges = edges
        self.chunks = chunks
        self.chunk_vecs = chunk_vecs
        self.chunk_ids = chunk_ids
        self.chunks_by_id = chunks_by_id
        self.chunks_per_entity = chunks_per_entity
        self.entities_by_id = entities_by_id
        self.plugin = plugin
        self.anchors = anchors
        self.anchors_by_id = {a.anchor_id: a for a in anchors}
        self.classifier = classifier
        self.builder = builder
        self.policy = policy
        self.bm25 = bm25
        self.cross_encoder = cross_encoder
        self.sym_store = sym_store
        self.query_embedder = query_embedder
        self.reader_client = reader_client
        self.reader_model = reader_model
        self.config = config
        # LLM-NER cache for query entity linking. When set,
        # `_link_entities()` routes through link_query_entities_with_cache;
        # else falls back to plugin.link_query_entities (exact-match baseline).
        self.ner_cache: Optional[dict[str, list[str]]] = None

    def _link_entities(self, question: str) -> list[str]:
        """NER-cache-aware entity linker — uses NER cache if available."""
        if self.ner_cache is not None:
            from mothrag.retrieval.ner import link_query_entities_with_cache
            return link_query_entities_with_cache(
                question, self.entities_by_id, self.ner_cache,
                fallback_plugin=self.plugin,
            )
        return self.plugin.link_query_entities(question, self.entities_by_id)

    # ---- Construction ----

    @classmethod
    def from_corpus(cls,
                    data_dir: str | Path,
                    *,
                    plugin=None,
                    embedding: str = "st-mini",
                    reranker: str = "bge-rerank",
                    bottom_up_boost: float = 1.0,
                    reader_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    reader_api_key: Optional[str] = None,
                    reader_base_url: str = "https://api.together.xyz/v1",
                    allow_custom_endpoint: bool = False,
                    config: Optional[PipelineConfig] = None) -> "MothRAGPipeline":
        """Load a preprocessed corpus and build all retrieval components.

        ``data_dir`` must contain ``entities.json``, ``edges.json``, ``chunks.jsonl``
        in the format produced by :mod:`mothrag.data.preprocess_wikipedia`.
        """
        from rank_bm25 import BM25Okapi

        from mothrag.retrieval.embeddings import (
            SentenceTransformerEmbedder, CrossEncoderReranker,
        )

        config = config or PipelineConfig(
            embedding=embedding, reranker=reranker, bottom_up_boost=bottom_up_boost,
        )

        if plugin is None:
            from mothrag.plugins.wikipedia import WikipediaDomainPlugin
            plugin = WikipediaDomainPlugin()

        data_dir = Path(data_dir)
        entities = json.loads((data_dir / "entities.json").read_text(encoding="utf-8"))
        edges = json.loads((data_dir / "edges.json").read_text(encoding="utf-8"))
        chunks = []
        with (data_dir / "chunks.jsonl").open(encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))

        texts = [c["text"] for c in chunks]
        # PROD corpus embedder is gemini-embedding-2.
        # Accept the canonical model id + short aliases so callers can pass the
        # exact PROD string ``gemini-embedding-2`` (not just the legacy
        # ``gemini`` shorthand) and get the SAME corpus path — no workaround.
        if config.embedding in ("gemini", "gemini-2", "gemini-embedding-2"):
            # Gemini Embedding 2 path: load pre-computed chunk vecs cache + build
            # an API-backed query embedder (RETRIEVAL_QUERY task_type).
            from google import genai
            from google.genai import types as gtypes
            import time as _time

            gemini_key = os.getenv("GEMINI_API_KEY")
            if not gemini_key:
                raise ValueError("GEMINI_API_KEY required for embedding='gemini'")
            gemini_client = genai.Client(api_key=gemini_key)

            def _gemini_is_transient(e):
                msg = str(e)
                for tok in ("429", "RESOURCE_EXHAUSTED", "500", "502", "503", "504",
                            "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED",
                            "timeout", "Timeout", "Connection", "ConnectionError"):
                    if tok in msg:
                        return True
                return isinstance(e, (ConnectionError, TimeoutError))

            def _gemini_embed_one(text, task_type, max_attempts=8):
                for attempt in range(max_attempts):
                    try:
                        resp = gemini_client.models.embed_content(
                            model="gemini-embedding-2", contents=[text],
                            config=gtypes.EmbedContentConfig(task_type=task_type),
                        )
                        v = np.array(resp.embeddings[0].values, dtype=np.float32)
                        n = np.linalg.norm(v)
                        return (v / n).astype(np.float32) if n > 0 else v
                    except Exception as e:
                        if _gemini_is_transient(e) and attempt < max_attempts - 1:
                            _time.sleep(min(64, 4 * (2 ** attempt)))
                        else:
                            raise

            cache_path = data_dir / "chunk_vecs_gemini_doc.npy"
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Gemini chunk cache not found at {cache_path}. "
                    "Generate via eval_wiki_async.py --embedding gemini first.")
            chunk_vecs = np.load(cache_path)
            if chunk_vecs.shape[0] != len(texts):
                raise ValueError(
                    f"Gemini cache shape {chunk_vecs.shape} != chunks {len(texts)}")
            print(f"[gemini] loaded chunk vecs cache {chunk_vecs.shape} from {cache_path}")

            class _GeminiEmbedderShim:
                """Drop-in for SentenceTransformerEmbedder.encode() interface."""
                def encode(self, text, **_kw):
                    if isinstance(text, list):
                        return np.stack([_gemini_embed_one(t, "RETRIEVAL_QUERY")
                                         for t in text])
                    return _gemini_embed_one(text, "RETRIEVAL_QUERY")
                def __call__(self, text):
                    return self.encode(text)

            embedder = _GeminiEmbedderShim()
        else:
            embedder = SentenceTransformerEmbedder(config.embedding)
            chunk_vecs = embedder.encode(texts, batch_size=32, show_progress_bar=True)

        chunks_by_id = {c["id"]: c for c in chunks}
        chunk_ids = [c["id"] for c in chunks]
        chunks_per_entity: dict[str, list[int]] = defaultdict(list)
        for i, cid in enumerate(chunk_ids):
            c = chunks_by_id[cid]
            if "entity_id" in c:
                chunks_per_entity[c["entity_id"]].append(i)
            for m in c.get("mentions", []):
                chunks_per_entity[m].append(i)
        entities_by_id = {e["id"]: e for e in entities}

        anchors = build_anchor_registry(entities, edges, chunk_vecs, chunks_by_id,
                                         chunk_ids, embedder, plugin)
        classifier = EntryPointClassifier(anchors, plugin, entities_by_id=entities_by_id)
        builder = ContextGraphBuilder({a.anchor_id: a for a in anchors},
                                       chunks_per_entity, chunk_vecs)
        policy = NavigationPolicyHeuristic(plugin)
        bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])

        cross_encoder = None
        if config.reranker == "bge-rerank":
            cross_encoder = CrossEncoderReranker("bge-rerank")

        sym_store = None
        if config.bottom_up_boost > 0:
            sym_store = SymbolicMemoryStore.from_edges(edges)

        from openai import OpenAI

        api_key = reader_api_key or os.getenv("TOGETHER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Reader API key not provided. Set TOGETHER_API_KEY or pass reader_api_key=...")
        # Allowlist guard: prevents arbitrary user-supplied base_url from
        # exfiltrating the reader API key. Default endpoint is allowlisted.
        validated_base_url = validate_base_url(
            reader_base_url, allow_custom_endpoint=allow_custom_endpoint,
        )
        reader_client = OpenAI(api_key=api_key, base_url=validated_base_url)

        return cls(
            entities=entities, edges=edges, chunks=chunks,
            chunk_vecs=chunk_vecs, chunk_ids=chunk_ids,
            chunks_by_id=chunks_by_id, chunks_per_entity=chunks_per_entity,
            entities_by_id=entities_by_id, plugin=plugin,
            anchors=anchors, classifier=classifier, builder=builder, policy=policy,
            bm25=bm25, cross_encoder=cross_encoder, sym_store=sym_store,
            query_embedder=embedder, reader_client=reader_client,
            reader_model=reader_model, config=config,
        )

    # ---- Retrieval ----

    def retrieve(self, question: str,
                 entity_seeds: Optional[list[str]] = None
                 ) -> tuple[list[int], str, float]:
        """Return ``(top_chunk_indices, route, top1_conf)`` for a question.

        Graph-aware iter: when ``entity_seeds`` is provided, those
        entity IDs are unioned with the question-linked entities for the
        bottom-up neighborhood boost, and their direct chunks are added to the
        candidate pool. This lets an iterative caller propagate accumulated
        entities across iterations to drive multi-hop retrieval.
        """
        cfg = self.config
        qv = self.query_embedder(question)
        top3 = self.classifier.classify(qv, q_text=question, top_k=3)
        chosen, route = self.policy.decide(top3, self.anchors_by_id, self.builder)

        candidate_indices: set[int] = set()
        for anc in chosen:
            candidate_indices.update(self.builder.build(anc))

        # Surface entity_seeds chunks into the candidate pool too
        if entity_seeds:
            for eid in entity_seeds:
                candidate_indices.update(self.chunks_per_entity.get(eid, []))

        if self.policy.needs_hop_expansion(question):
            seeds = self._link_entities(question)
            for s in seeds:
                candidate_indices.update(self.chunks_per_entity.get(s, []))

        flat_scores = self.chunk_vecs @ qv
        flat_top = np.argsort(-flat_scores)[:30]
        if not cfg.no_safety_net:
            candidate_indices.update(flat_top.tolist())
        elif not candidate_indices:
            candidate_indices.update(range(min(10, len(self.chunk_ids))))

        candidate_list = list(candidate_indices)
        cand_arr = self.chunk_vecs[candidate_list]
        cosine_scores = cand_arr @ qv
        order = np.argsort(-cosine_scores)[: cfg.top_k_candidates]
        scored_idx = [candidate_list[i] for i in order]

        if cfg.reranker == "bge-rerank" and self.cross_encoder is not None:
            pairs = [(question, self.chunks_by_id[self.chunk_ids[ci]]["text"])
                     for ci in scored_idx]
            ce_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
            if self.sym_store is not None and cfg.bottom_up_boost > 0:
                q_entities = self._link_entities(question)
                # Union question-linked entities with caller-provided seeds
                if entity_seeds:
                    q_entities = list(set(q_entities) | set(entity_seeds))
                neighborhood: set[str] = set()
                for qe in q_entities:
                    if self.sym_store.has_entity(qe):
                        for tgt, _, _ in self.sym_store.lookup_neighbors(
                                qe, max_hops=cfg.bottom_up_max_hops, top_k=200):
                            neighborhood.add(tgt)
                boosts = []
                for ci in scored_idx:
                    chunk = self.chunks_by_id[self.chunk_ids[ci]]
                    eid = chunk.get("entity_id")
                    boosts.append(cfg.bottom_up_boost if eid in neighborhood else 0.0)
                ce_scores = ce_scores + np.array(boosts, dtype=ce_scores.dtype)
            cand_ranked = sorted(zip(scored_idx, ce_scores), key=lambda x: -x[1])[: cfg.top_k_chunks]
            top_chunks_idx = [c for c, _ in cand_ranked]
        elif cfg.reranker == "bm25":
            bm25_scores = self.bm25.get_scores(_tokenize(question))
            cand_bm25 = sorted([(ci, bm25_scores[ci]) for ci in scored_idx],
                               key=lambda x: -x[1])[: cfg.top_k_chunks]
            top_chunks_idx = [c for c, _ in cand_bm25]
        else:
            top_chunks_idx = scored_idx[: cfg.top_k_chunks]

        top1_conf = top3[0][1] if top3 else 0.0
        return top_chunks_idx, route, float(top1_conf)

    # ---- Reader ----

    def read(self, question: str, passages: list[str]) -> tuple[str, str, dict]:
        """Call the reader on ``passages``. Returns ``(answer, raw, usage)``."""
        cfg = self.config
        msgs = make_reader_messages(question, passages, cfg.reader_prompt)
        raw, usage = call_reader_majority_with_usage(
            self.reader_client, self.reader_model, msgs,
            max_tokens=cfg.reader_max_tokens,
            n_samples=cfg.reader_n_samples,
            temperature=cfg.reader_temperature,
            reader_prompt=cfg.reader_prompt,
        )
        return parse_reader_output(raw, cfg.reader_prompt), raw, usage

    # ---- End-to-end ----

    def answer(self, question: str) -> AnswerInfo:
        """Run the full pipeline on a single question."""
        top_chunks_idx, route, top1_conf = self.retrieve(question)
        passages = [self.chunks_by_id[self.chunk_ids[ci]]["text"] for ci in top_chunks_idx]
        retrieved_ids = [self.chunk_ids[ci] for ci in top_chunks_idx]
        ans, raw, usage = self.read(question, passages)
        return AnswerInfo(
            answer=ans, raw=raw, passages=passages,
            retrieved_chunk_ids=retrieved_ids, route=route, top1_conf=top1_conf,
            latency_s=usage["latency_s"], prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
        )
