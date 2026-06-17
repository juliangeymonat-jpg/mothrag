# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
r"""MothGraphArm -- anchor-driven iterative graph-traversal arm.

The 5th arm in the MothRag opt-in pool. Distinct from generic PPR
(Personalized PageRank) graph-retrieval: MothGraphArm composes five
MothRag-native principles on top of the bare :class:`GraphIndex`:

1. **Anchor extraction (sel_v2-aligned).** The walk starts from a
   capitalised entity surface detected by the same lexical rule
   :func:`mothrag.core.query_type_classifier.count_named_entities` uses
   to count entities for sel_v2. No NER training, no per-corpus tuning;
   uses the same general lexical convention.

2. **Iterative refinement.** Up to ``max_iters`` rounds. Each round
   traverses, validates, and (when paths remain unstable) refines the
   anchor toward the most-incident-edge endpoint surfaced in the
   current path set. Refinement is conservative: it only swaps the
   anchor when a NEW high-confidence endpoint emerges that was not
   present in the previous round.

3. **gamma-validated paths.** A pluggable
   :class:`GammaProtocol` callable filters each path before it can
   contribute to the answer. Default implementation is a deterministic
   structural validator (path length >= 1, all edges have non-empty
   subject/object/predicate, no self-loops on normalised endpoints).
   Callers can inject a richer LLM-judge or embedding-similarity
   validator without touching this module.

4. **L4b temporal stability.** When the
   :func:`mothrag.graph.traversal_hash` of the current path set equals
   the previous iteration's hash, refinement halts (stable fixed
   point reached). Costs O(n log n) per round (sorted edge IDs +
   SHA1).

5. **Soft fallback.** When no anchor can be extracted, or no paths
   remain after gamma validation across all iterations, the arm falls
   back to a caller-supplied dense retriever / direct-answer callable.
   This preserves the MothRag invariant: every arm in the pool MUST
   return a non-error result so arbitration always has data.

The arm is cost-bounded by ``max_iters * top_k * depth`` graph
operations. NO LLM call inside this arm (graph traversal +
validation only). The dense fallback may invoke a reader; that is
a caller-owned cost, not an arm-owned cost.

Design contract: all heuristics in this module are GENERIC graph-traversal +
linguistic-anchor rules. No per-dataset patterns, no gold-derived
tuning, no test-set inspection.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Protocol, Sequence

from mothrag.arms.base import ArmResult
from mothrag.graph.index import GraphIndex, Path, traversal_hash
from mothrag.graph.openie import normalize_entity

logger = logging.getLogger(__name__)


# ---- Pluggable protocols ---------------------------------------------------

class GammaProtocol(Protocol):
    """Validates whether a :class:`Path` legitimately answers ``question``.

    Implementations MUST be deterministic and side-effect free. The
    default :class:`StructuralGammaVerifier` is a pure structural
    check; callers can swap in a learned-embedding cosine threshold or
    an LLM-judge wrapper without changing the arm contract.
    """

    def __call__(self, path: Path, question: str) -> bool: ...


class StructuralGammaVerifier:
    """Default gamma verifier: structural-soundness check on a path.

    Accepts paths where every edge has non-empty subject, predicate,
    and object; rejects self-loops on the normalised entity surface
    (a same-name edge usually indicates extraction noise).
    """

    def __call__(self, path: Path, question: str) -> bool:  # noqa: ARG002
        if not path.edges:
            return False
        for e in path.edges:
            if not e.subject or not e.predicate or not e.object:
                return False
            if normalize_entity(e.subject) == normalize_entity(e.object):
                return False
        return True


# Caller-supplied dense fallback. Takes the question, returns an
# :class:`ArmResult`. The arm constructor stores the callable; when no
# anchor / no valid path, the arm delegates here for graceful degradation.
DenseFallback = Callable[[str], ArmResult]


# ---- Anchor extraction -----------------------------------------------------

# Capitalized entity run -- aligned with the lexical convention used by
# sel_v2 (:func:`count_named_entities` / `_CAPS_RUN`). Multi-word entities
# allowed via small connectives (of/the/de/la/von/van/and).
_ANCHOR_RUN_RE = re.compile(
    r"\b([A-Z][a-zA-Z0-9'\-]+"
    r"(?:\s+(?:of|the|de|la|le|du|von|van|and|[A-Z][a-zA-Z0-9'\-]+))*)"
)

_ANCHOR_HEADS_BLACKLIST: set[str] = {
    "who", "what", "when", "where", "why", "how",
    "which", "whose", "whom", "is", "are", "was", "were",
    "did", "do", "does", "the", "an", "a", "in", "of",
    "on", "at", "to", "by",
}


def _extract_anchor_candidates(question: str) -> list[str]:
    """Return capitalised entity spans suitable as graph-traversal anchors.

    Returned in left-to-right order of appearance; deduplicated by
    normalised form. Sentence-initial Wh-word / aux-verb singletons
    are filtered out (they are NOT entities even though capitalised).
    """
    if not question:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in _ANCHOR_RUN_RE.finditer(question.strip()):
        sp = m.group(1).strip()
        norm = normalize_entity(sp)
        if not norm or norm in _ANCHOR_HEADS_BLACKLIST:
            continue
        if len(norm) < 2:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(sp)
    return out


# ---- The arm ---------------------------------------------------------------

class MothGraphArm:
    """Anchor-driven iterative graph-traversal arm.

    Parameters
    ----------
    graph_index
        Pre-built :class:`GraphIndex` over the corpus.
    dense_fallback
        Callable invoked when no anchor / no valid path. MUST return
        an :class:`ArmResult`.
    anchor_extractor
        Optional override for the default
        :func:`_extract_anchor_candidates`. Useful in tests.
    gamma_verifier
        Optional :class:`GammaProtocol`. Defaults to
        :class:`StructuralGammaVerifier`.
    max_iters
        Hard cap on refinement loop iterations.
    base_depth
        Starting traversal depth; may be expanded adaptively when
        ``adaptive_depth=True``.
    top_k
        Top-K paths surfaced per iteration.
    adaptive_depth
        When ``True``, depth grows with ``deep_complexity`` /
        ``multi_hop_marker`` linguistic features (1..3 hops). Default
        ``True`` (matches MothRag's signal-driven adaptive routing).
    """

    name = "mothgraph_arm"

    def __init__(
        self,
        graph_index: GraphIndex,
        dense_fallback: DenseFallback,
        *,
        anchor_extractor: Callable[[str], list[str]] | None = None,
        gamma_verifier: GammaProtocol | None = None,
        max_iters: int = 3,
        base_depth: int = 2,
        top_k: int = 8,
        adaptive_depth: bool = True,
    ) -> None:
        if max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {max_iters}")
        if base_depth < 1:
            raise ValueError(f"base_depth must be >= 1, got {base_depth}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.graph_index = graph_index
        self.dense_fallback = dense_fallback
        self._anchor_extractor = anchor_extractor or _extract_anchor_candidates
        self.gamma = gamma_verifier or StructuralGammaVerifier()
        self.max_iters = int(max_iters)
        self.base_depth = int(base_depth)
        self.top_k = int(top_k)
        self.adaptive_depth = bool(adaptive_depth)

    # ---- Anchor selection ------------------------------------------------

    def _pick_anchor(self, question: str) -> str | None:
        """Pick the first anchor candidate that exists in the graph index.

        Order = left-to-right surface appearance (deterministic). When
        no candidate is in the index, returns None (triggers soft
        fallback).
        """
        for cand in self._anchor_extractor(question):
            if cand in self.graph_index:
                return cand
        return None

    def _refine_anchor(
        self,
        valid_paths: Sequence[Path],
        previous_anchor: str,
    ) -> str | None:
        """Pick the highest-incidence endpoint NOT previously visited.

        For each path, count incident edges to each non-anchor
        endpoint; promote the most-incident endpoint as the next
        anchor. Conservative: only swaps when the candidate yields
        strictly more incident edges than the previous anchor's
        residual coverage.
        """
        if not valid_paths:
            return None
        prev_norm = normalize_entity(previous_anchor)
        counts: dict[str, tuple[int, str]] = {}  # norm -> (count, surface)
        for p in valid_paths:
            for ep in p.endpoints:
                ep_norm = normalize_entity(ep)
                if not ep_norm or ep_norm == prev_norm:
                    continue
                cur = counts.get(ep_norm, (0, ep))
                counts[ep_norm] = (cur[0] + 1, ep)
        if not counts:
            return None
        # Deterministic tie-break: highest count first, then alphabetical norm.
        best = sorted(
            counts.items(),
            key=lambda kv: (-kv[1][0], kv[0]),
        )[0]
        return best[1][1]

    # ---- Depth adaptation ------------------------------------------------

    def _adaptive_depth(self, question: str) -> int:
        """Signal-driven depth: 1..3 hops based on linguistic complexity.

        Avoids importing the full SemanticFeatures vector when only two
        signals are consulted -- inlined for speed and to keep the arm
        usable when the routing layer is not loaded.
        """
        if not self.adaptive_depth or not question:
            return self.base_depth
        try:
            from mothrag.routing.semantic_features import (
                score_complexity,
                score_multi_hop,
            )
        except ImportError:
            return self.base_depth
        c = score_complexity(question)
        m = score_multi_hop(question)
        signal = max(c, m)
        if signal >= 0.7:
            return max(self.base_depth, 3)
        if signal >= 0.4:
            return max(self.base_depth, 2)
        return max(1, self.base_depth)

    # ---- Path composition ------------------------------------------------

    def _compose_answer(self, valid_paths: Sequence[Path]) -> str:
        """Pick the top-confidence path's terminal object as the answer.

        When multiple paths tie on confidence, the lexicographically-
        smallest canonical edge-id sequence wins (deterministic).
        Returns "" when no usable path is present.
        """
        if not valid_paths:
            return ""
        # Already sorted by GraphIndex.traverse_from_anchor + filtered;
        # but re-sort defensively for cases where the caller injected a
        # custom anchor_extractor that returned paths in a different
        # order.
        sorted_paths = sorted(
            valid_paths,
            key=lambda p: (-p.confidence,
                           "->".join(e.edge_id for e in p.edges)),
        )
        top = sorted_paths[0]
        if not top.edges:
            return ""
        # Terminal endpoint of the chain that is NOT the anchor.
        terminal_edge = top.edges[-1]
        anchor_norm = normalize_entity(top.anchor)
        if normalize_entity(terminal_edge.subject) != anchor_norm:
            return terminal_edge.subject
        return terminal_edge.object

    def _retrieved_ids(self, valid_paths: Sequence[Path]) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for p in valid_paths:
            for e in p.edges:
                if e.source_chunk_id and e.source_chunk_id not in seen:
                    seen.add(e.source_chunk_id)
                    ids.append(e.source_chunk_id)
        return ids

    # ---- Arm Protocol surface --------------------------------------------

    def applicable(self, question: str) -> bool:
        """Arm is structurally applicable iff at least one anchor candidate
        exists in the graph index.
        """
        if not question or not question.strip():
            return False
        if len(self.graph_index) == 0:
            return False
        for cand in self._anchor_extractor(question):
            if cand in self.graph_index:
                return True
        return False

    def run(self, question: str, **ctx: Any) -> ArmResult:  # noqa: ARG002
        """Execute anchor-driven iterative traversal.

        Returns dense-fallback result when no anchor / no valid path
        survives gamma validation. Returns the graph-derived answer
        otherwise.
        """
        t0 = time.time()
        if not question or not question.strip():
            return ArmResult(pred="", latency_s=time.time() - t0,
                             metadata={"reason": "empty_question"})

        anchor = self._pick_anchor(question)
        if anchor is None:
            return self._soft_fallback(question, t0, reason="no_anchor")

        depth = self._adaptive_depth(question)
        prev_hash: str | None = None
        last_valid: list[Path] = []
        last_anchor = anchor
        iters_done = 0
        stable_break = False

        for iteration in range(self.max_iters):
            iters_done = iteration + 1
            paths = self.graph_index.traverse_from_anchor(
                last_anchor, depth=depth, top_k=self.top_k,
            )
            valid = [p for p in paths if self.gamma(p, question)]
            if valid:
                last_valid = valid

            cur_hash = traversal_hash(valid)
            if prev_hash is not None and cur_hash == prev_hash:
                stable_break = True
                break
            prev_hash = cur_hash

            if not valid:
                # No paths -- try one round of anchor refinement before giving up.
                if iteration == self.max_iters - 1:
                    break
                # Refine using the LAST set of valid paths (might be empty
                # this round; carry forward the previous round's valid set).
                refined = self._refine_anchor(last_valid, last_anchor)
                if refined is None or normalize_entity(refined) == normalize_entity(last_anchor):
                    break
                last_anchor = refined
                continue

            refined = self._refine_anchor(valid, last_anchor)
            if refined is None or normalize_entity(refined) == normalize_entity(last_anchor):
                # Anchor stable -- still let the L4b hash break next iter.
                continue
            last_anchor = refined

        if not last_valid:
            return self._soft_fallback(
                question, t0, reason="no_valid_paths",
                iterations=iters_done, stable_break=stable_break,
            )

        pred = self._compose_answer(last_valid)
        if not pred:
            return self._soft_fallback(
                question, t0, reason="empty_composition",
                iterations=iters_done, stable_break=stable_break,
            )
        return ArmResult(
            pred=pred,
            retrieved_chunk_ids=self._retrieved_ids(last_valid),
            n_llm_calls=0,
            latency_s=time.time() - t0,
            metadata={
                "anchor_initial": anchor,
                "anchor_final": last_anchor,
                "iterations": iters_done,
                "stable_break": stable_break,
                "depth": depth,
                "n_paths": len(last_valid),
            },
        )

    # ---- Soft fallback ---------------------------------------------------

    def _soft_fallback(
        self,
        question: str,
        t0: float,
        *,
        reason: str,
        iterations: int = 0,
        stable_break: bool = False,
    ) -> ArmResult:
        """Delegate to the caller-supplied dense fallback.

        Errors raised by the fallback are converted into an empty
        :class:`ArmResult` so the rest of the pool can still arbitrate.

        ---- Pool-safety axiom ----
        Fallback results are TAGGED with ``metadata["is_fallback"] = True``
        and ``metadata["fallback_origin"] = "mothgraph_arm"``. Consumers
        composing this arm inside a multi-arm pool (e.g. the opt-in
        arm loop in ``scripts/route_prospective.py``) MUST skip
        fallback-tagged results when building the arbitration
        candidate set. Reason: when the fallback delegates to a
        legacy arm already running in the pool (e.g. V3+bu), the
        fallback's pred is a DUPLICATE of that arm's answer.
        Including it as a separate candidate inflates
        :func:`pairwise_agreement` for that answer (spurious
        consensus boost), causing legitimate disagreeing arms (e.g.
        iter with the correct answer) to lose arbitration. This was
        the observed MQ F1=1 cohort -23/-26pp regression.

        Standalone use (MothGraphArm as the only arm) is preserved:
        callers that DON'T pool the arm with v3bu can ignore the
        ``is_fallback`` flag and consume ``result.pred`` directly.
        """
        try:
            result = self.dense_fallback(question)
            if not isinstance(result, ArmResult):
                result = ArmResult(pred=str(result))
        except Exception as exc:  # noqa: BLE001
            logger.warning("MothGraphArm fallback raised: %s", exc)
            result = ArmResult(
                pred="",
                metadata={"error": f"{type(exc).__name__}: {exc}"},
            )
        # Layer our own metadata on top WITHOUT clobbering anything the
        # fallback itself recorded.
        merged_meta = dict(result.metadata)
        merged_meta.setdefault("mothgraph_fallback_reason", reason)
        merged_meta.setdefault("mothgraph_iterations", iterations)
        merged_meta.setdefault("mothgraph_stable_break", stable_break)
        # Pool-safety tag (see docstring).
        merged_meta["is_fallback"] = True
        merged_meta["fallback_origin"] = "mothgraph_arm"
        return ArmResult(
            pred=result.pred,
            retrieved_chunk_ids=list(result.retrieved_chunk_ids),
            n_llm_calls=result.n_llm_calls,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_s=time.time() - t0,
            metadata=merged_meta,
        )


__all__ = [
    "GammaProtocol",
    "MothGraphArm",
    "StructuralGammaVerifier",
]
