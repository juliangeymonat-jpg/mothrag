# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Plug-and-play MothRAG API.

20-line user code::

    from mothrag import simple
    results = simple.run(
        questions=["What movie did the wife of Inception's director star in?"],
        corpus_path="hotpotqa-mini",
        reader="llama-3.3-70b",
        config="default",
    )
    print(results[0].answer)            # "Memento"
    print(results[0].confidence)        # 0.87
    print(results[0].retrieved_chunks)  # supporting passages

Available reader aliases:

  - ``llama-3.3-70b``       -> Together AI Llama-3.3-70B-Instruct-Turbo
  - ``llama-4-scout-17b``   -> Together AI Llama-4-Scout-17B-Instruct
  - ``gpt-4o``              -> OpenAI GPT-4o
  - ``claude-sonnet-4-6``   -> Anthropic via OpenAI-compatible proxy

You can also pass the full provider model id and a ``base_url`` if you have
a custom endpoint.

Available configs (see :class:`mothrag.eval.pipeline.PipelineConfig`):

  - ``default``      = V3+bu + BGE cross-encoder rerank + V3-think reader
  - ``fast``         = top-k cosine + BM25 rerank + V1 reader (no cross-encoder)
  - ``high-quality`` = ``default`` + K=3 self-consistency
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mothrag.eval.pipeline import MothRAGPipeline, PipelineConfig
from mothrag.utils.url_safety import validate_base_url


READER_ALIASES = {
    "llama-3.3-70b": ("meta-llama/Llama-3.3-70B-Instruct-Turbo",
                       "https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "llama-4-scout-17b": ("meta-llama/llama-4-scout-17b-16e-instruct",
                           "https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "gpt-4o": ("gpt-4o", None, "OPENAI_API_KEY"),
    "gpt-5.4": ("gpt-5.4", None, "OPENAI_API_KEY"),
    "claude-sonnet-4-6": ("claude-sonnet-4-6",
                           "https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY"),
}


CONFIG_PRESETS: dict[str, dict] = {
    "default": dict(
        embedding="st-mini", reranker="bge-rerank", bottom_up_boost=1.0,
        reader_prompt="v3-think", reader_max_tokens=2000, reader_n_samples=1,
        top_k_chunks=10,
    ),
    "fast": dict(
        embedding="st-mini", reranker="bm25", bottom_up_boost=0.0,
        reader_prompt="v1", reader_max_tokens=64, reader_n_samples=1,
        top_k_chunks=5,
    ),
    "high-quality": dict(
        embedding="st-base", reranker="bge-rerank", bottom_up_boost=1.0,
        reader_prompt="v3-think-cite", reader_max_tokens=2000, reader_n_samples=3,
        reader_temperature=0.7, top_k_chunks=10,
    ),
}


# Pre-built mini corpora hosted alongside the package release.
# (To be populated post-paper. Until then, users supply their own preprocessed corpus.)
CORPUS_ALIASES: dict[str, str] = {
    # "hotpotqa-mini": "https://huggingface.co/datasets/juliangeymonat-jpg/mothrag-hotpotqa-mini",
}


@dataclass
class SimpleResult:
    """Result returned by :func:`run` for a single question."""
    question: str
    answer: str
    confidence: float = 0.0
    retrieved_chunks: list[str] = field(default_factory=list)
    route: str = ""
    latency_s: float = 0.0
    raw: str = ""
    faithfulness: Optional[float] = None  # populated only if compute_faithfulness=True


def _resolve_reader(reader: str,
                    api_key: Optional[str],
                    base_url: Optional[str],
                    *,
                    allow_custom_endpoint: bool = False) -> tuple[str, str, str]:
    if reader in READER_ALIASES:
        model, alias_url, env_var = READER_ALIASES[reader]
    else:
        model = reader
        alias_url = None
        env_var = "OPENAI_API_KEY"
    resolved_url = base_url or alias_url or "https://api.together.xyz/v1"
    # Allowlist guard: prevents arbitrary user-supplied base_url from
    # exfiltrating the reader API key. Built-in alias URLs are allowlisted,
    # so this is a no-op for default usage.
    validate_base_url(resolved_url, allow_custom_endpoint=allow_custom_endpoint)
    return model, resolved_url, env_var


def _resolve_corpus(corpus_path: str | Path) -> Path:
    if str(corpus_path) in CORPUS_ALIASES:
        raise NotImplementedError(
            f"Corpus alias '{corpus_path}' is not yet bundled. "
            "Please point to a local directory containing entities.json, edges.json, chunks.jsonl."
        )
    p = Path(corpus_path)
    if not p.exists():
        raise FileNotFoundError(f"Corpus directory not found: {p.resolve()}")
    for required in ("entities.json", "edges.json", "chunks.jsonl"):
        if not (p / required).exists():
            raise FileNotFoundError(f"Missing {required} in corpus directory {p}")
    return p


def _resolve_config(config: str | dict | PipelineConfig) -> PipelineConfig:
    if isinstance(config, PipelineConfig):
        return config
    if isinstance(config, dict):
        return PipelineConfig(**config)
    if config in CONFIG_PRESETS:
        return PipelineConfig(**CONFIG_PRESETS[config])
    raise ValueError(f"Unknown config '{config}'. Choose from {list(CONFIG_PRESETS)} or pass dict / PipelineConfig.")


def run(questions: list[str], *,
        corpus_path: str | Path,
        reader: str = "llama-3.3-70b",
        config: str | dict | PipelineConfig = "default",
        reader_api_key: Optional[str] = None,
        reader_base_url: Optional[str] = None,
        allow_custom_endpoint: bool = False,
        compute_faithfulness: bool = False,
        faithfulness_judge: Optional[str] = None) -> list[SimpleResult]:
    """Run MothRAG end-to-end on a list of questions.

    Args:
        questions: Free-text questions.
        corpus_path: Directory containing ``entities.json``, ``edges.json``,
            ``chunks.jsonl`` produced by ``mothrag.data.preprocess_wikipedia``.
        reader: Reader alias (see :data:`READER_ALIASES`) or a raw provider model id.
        config: Preset name (``default`` / ``fast`` / ``high-quality``) or a
            :class:`PipelineConfig` instance / dict of overrides.
        reader_api_key, reader_base_url: Override env-based credentials.
        allow_custom_endpoint: Set to ``True`` to permit a ``reader_base_url``
            outside the MothRAG allowlist (self-hosted vLLM, internal proxy,
            etc.). A warning is logged when used. Default ``False`` raises
            :class:`ValueError` on non-allowlisted hosts to prevent
            accidental API-key exfiltration.
        compute_faithfulness: Whether to score faithfulness via a judge call.
        faithfulness_judge: Override judge model (default: same as reader).

    Returns:
        One :class:`SimpleResult` per input question, in the same order.
    """
    cfg = _resolve_config(config)
    corpus_dir = _resolve_corpus(corpus_path)
    reader_model, reader_url, _ = _resolve_reader(
        reader, reader_api_key, reader_base_url,
        allow_custom_endpoint=allow_custom_endpoint,
    )

    pipe = MothRAGPipeline.from_corpus(
        data_dir=corpus_dir,
        embedding=cfg.embedding,
        reranker=cfg.reranker,
        bottom_up_boost=cfg.bottom_up_boost,
        reader_model=reader_model,
        reader_api_key=reader_api_key,
        reader_base_url=reader_url,
        allow_custom_endpoint=allow_custom_endpoint,
        config=cfg,
    )

    results: list[SimpleResult] = []
    for q in questions:
        info = pipe.answer(q)
        results.append(SimpleResult(
            question=q,
            answer=info.answer,
            confidence=info.top1_conf,
            retrieved_chunks=info.passages,
            route=info.route,
            latency_s=info.latency_s,
            raw=info.raw,
        ))

    if compute_faithfulness:
        from mothrag.eval.faithfulness import faithfulness_score
        judge_model = faithfulness_judge or reader_model
        for r in results:
            score, _ = faithfulness_score(
                pipe.reader_client, judge_model,
                r.question, r.retrieved_chunks, r.answer,
            )
            r.faithfulness = score

    return results
