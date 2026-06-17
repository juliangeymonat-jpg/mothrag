# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""InfoboxArm (C3.6) -- direct structured-fact lookup arm.

Wraps the existing :class:`mothrag.core.retrieval.InfoboxIndex` as a
first-class arm in the ``--arms-pool`` composition. Unlike the dense /
graph / sparse retrieval arms (which surface PASSAGES that a reader
LLM then synthesises into an answer), InfoboxArm performs DIRECT
structured-fact lookup: when the question matches a canonical
entity-attribute hint shape ("When was X born?", "Who is X's spouse?",
"What is the capital of X?"), the arm extracts the ``(subject,
attribute)`` pair, queries the :class:`InfoboxIndex`, and returns the
matching triple's ``value`` as the answer with NO LLM call.

Cost: 0 LLM calls per invocation. Latency dominated by the regex
hint extraction + dict lookup -- microseconds in practice.

Composition vs C3 / C3.5:

- :data:`retrieval='dense_plus_infobox'` (C3) and the router-gated
  variant (C3.5) BLEND infobox chunks into the DENSE INDEX so the
  dense arms (V3+bu / decompose / iter) implicitly see them at
  retrieval time. The infobox chunks compete with prose passages on
  cosine similarity.
- InfoboxArm (C3.6) is the ORTHOGONAL composition: a dedicated arm
  whose entire purpose is structured-fact dispatch. The arm sees ONLY
  the infobox index, never prose. Sel_v2 arbitration then composes
  InfoboxArm's direct-lookup answer with the LLM-mediated answers
  from the other arms; agreement reinforces, disagreement triggers
  the arbitrator's standard weighting.

Inclusion in the arm pool (sel_v2 trigger):

- InfoboxArm is applicable when the question matches one of the
  hint patterns in :func:`mothrag.core.retrieval.extract_question_hints`
  (the same deterministic regex set used by C3 / C3.5). When the
  hint set is empty, the arm declines via :meth:`applicable` and
  sel_v2 excludes it from the per-query subset -- zero cost on
  non-entity-attribute questions.

Caller responsibility:

- The caller must pass a pre-built :class:`InfoboxIndex` at
  construction time. This is the same index built by
  ``scripts/route_prospective.py:_augment_pipeline_with_infobox``;
  the InfoboxArm reuses it without re-harvesting.
"""

from __future__ import annotations

import time
from typing import Any

from mothrag.arms.base import ArmResult


class InfoboxArm:
    """Direct structured-fact lookup arm.

    Parameters
    ----------
    infobox_index
        Pre-built :class:`mothrag.core.retrieval.InfoboxIndex`. Must
        carry triples harvested from the same corpus the other arms
        retrieve over (otherwise the InfoboxArm answers will not
        align with the dense passages in arbitration).
    hint_extractor
        Optional callable ``(question) -> list[(subject, attribute)]``.
        Defaults to
        :func:`mothrag.core.retrieval.extract_question_hints` (the
        deterministic regex extractor shared with C3 / C3.5).
    """

    name = "infobox_arm"

    def __init__(
        self,
        infobox_index,
        *,
        hint_extractor=None,
    ) -> None:
        self.infobox_index = infobox_index
        self._hint_extractor = hint_extractor

    def _hints(self, question: str):
        if self._hint_extractor is not None:
            return self._hint_extractor(question)
        from mothrag.core.retrieval import extract_question_hints
        return extract_question_hints(question)

    def applicable(self, question: str) -> bool:
        if not question or not question.strip():
            return False
        try:
            hints = self._hints(question)
        except Exception:  # noqa: BLE001
            return False
        return bool(hints)

    def run(self, question: str, **ctx: Any) -> ArmResult:  # noqa: ARG002
        """Run direct structured-fact lookup.

        Algorithm:
          1. Extract ``(subject, attribute)`` hints via the configured
             :func:`extract_question_hints`.
          2. For each hint, query :meth:`InfoboxIndex.lookup`.
          3. Pick the highest-confidence triple value as the answer.
          4. When no hint resolves to a triple, return ``pred=""``
             (other arms cover via arbitration).

        Latency telemetry is recorded; n_llm_calls / prompt_tokens /
        completion_tokens are zero (no reader invocation).
        """
        t0 = time.time()
        if not question or not question.strip():
            return ArmResult(pred="", latency_s=time.time() - t0)
        try:
            hints = self._hints(question)
        except Exception as exc:  # noqa: BLE001
            return ArmResult(
                pred="",
                metadata={"error": f"hint_extractor: {type(exc).__name__}: {exc}"},
                latency_s=time.time() - t0,
            )
        if not hints:
            return ArmResult(pred="", latency_s=time.time() - t0,
                              metadata={"hints": []})

        best_triple = None
        best_conf = -1.0
        matched_hint = None
        for subj, attr in hints:
            try:
                results = self.infobox_index.lookup(subj, attr)
            except Exception:  # noqa: BLE001
                continue
            for t in results:
                if t.confidence > best_conf:
                    best_conf = t.confidence
                    best_triple = t
                    matched_hint = (subj, attr)

        latency = time.time() - t0
        if best_triple is None:
            return ArmResult(
                pred="", latency_s=latency,
                metadata={"hints": list(hints), "match": None},
            )

        retrieved_ids = []
        if getattr(best_triple, "source_chunk_id", ""):
            retrieved_ids.append(best_triple.source_chunk_id)

        return ArmResult(
            pred=str(best_triple.value),
            retrieved_chunk_ids=retrieved_ids,
            latency_s=latency,
            metadata={
                "hints": list(hints),
                "match_hint": matched_hint,
                "match_subject": best_triple.subject,
                "match_attribute": best_triple.attribute,
                "match_confidence": best_triple.confidence,
            },
        )


__all__ = ["InfoboxArm"]
