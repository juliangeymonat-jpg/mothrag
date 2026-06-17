# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""BottomUpBoostStrategy (#5) — NER-augmented re-retrieval + re-read.

Cost: 0 LLM calls (uses existing NER cache / OpenIE primitives and the
existing reader). Extracts entities from the question + retrieved passages,
augments the query, re-retrieves, and re-runs the iter arm.

NER / OpenIE primitives live in :mod:`mothrag.retrieval.ner` and
:mod:`mothrag.retrieval.openie`; both are optional extras
(``mothrag[retrieval]``) so this strategy degrades cleanly when they're
not installed.
"""

from __future__ import annotations

import logging
import re

from mothrag.core.retry.protocol import RetryContext

logger = logging.getLogger(__name__)


# Naive fallback entity extractor (Capitalised noun phrases). Used when the
# `mothrag[retrieval]` NER stack is not installed. Good enough to surface
# obvious bridge entities without dragging in spaCy / transformers.
_NAIVE_NP_RE = re.compile(r"\b(?:[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+){0,3})\b")


class BottomUpBoostStrategy:
    """Re-retrieve with entity-augmented query when γ flagged a missing entity."""

    name = "bottom_up_boost"
    cost_estimate = 0

    def applicable(self, ctx: RetryContext) -> bool:
        if ctx.run_arm_iter is None or ctx.embedder is None or ctx.vector_db is None:
            return False
        if ctx.abstention_signal not in (
            "gamma_refuse", "iter_abstain", "h4_refuse", "empty_answer"
        ):
            return False
        return bool(ctx.passages)

    def try_recover(self, ctx: RetryContext) -> str | None:
        entities = self._extract_entities(ctx)
        if not entities:
            return None
        augmented = f"{ctx.question} {' '.join(entities)}"
        try:
            aug_emb = ctx.embedder.embed_batch([augmented])[0]
            new_chunks = ctx.vector_db.retrieve(aug_emb, top_k=ctx.top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("bottom_up_boost re-retrieve failed: %s", exc)
            return None
        new_passages = [c.text for c in new_chunks]
        if not new_passages:
            return None
        try:
            answer = ctx.run_arm_iter(
                question=ctx.question,
                passages=new_passages,
                q_emb=aug_emb,
                top_k=ctx.top_k,
                max_steps=ctx.config.get("max_iter_steps", 3),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("bottom_up_boost iter re-run failed: %s", exc)
            return None
        if answer:
            return answer
        return None

    @staticmethod
    def _extract_entities(ctx: RetryContext) -> list[str]:
        """Try mothrag.retrieval.ner if installed; else naive capitalised-NP."""
        try:
            from mothrag.retrieval.ner import link_query_entities_with_cache
        except Exception:
            link_query_entities_with_cache = None

        if link_query_entities_with_cache is not None:
            try:
                hits = link_query_entities_with_cache(ctx.question, cache=None)
                ents = [h.get("entity") or h.get("text") for h in hits or [] if h]
                ents = [e for e in ents if e]
                if ents:
                    return ents
            except Exception:
                pass

        # Naive fallback: capitalised NPs from question + first 3 passages.
        text = ctx.question + " " + " ".join(ctx.passages[:3])
        cands = set(_NAIVE_NP_RE.findall(text))
        # Drop entities already in the question to add information.
        already = set(_NAIVE_NP_RE.findall(ctx.question))
        new = [c for c in cands if c not in already]
        return new[:5]


__all__ = ["BottomUpBoostStrategy"]
