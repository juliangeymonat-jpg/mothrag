# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""ActiveGapQueryStrategy (#8) -- active-learning RAG recovery via reader
self-introspection with sel_v2-respecting re-dispatch.

Cost: 1 LLM call per round (reader formulates the gap query) +
1 reader call per round (re-read with augmented context).
Worst-case 2 * ``max_rounds`` LLM calls; default ``max_rounds=3``
caps the worst case at 6 reader invocations.

The strategy is best understood as **re-running MothRag from a wider
starting context**: the original question is unchanged, but the
retrieved passages grow round-by-round as the reader articulates and
the retriever fetches successive fact gaps. The re-read at the end of
each round goes through the same arm that the production sel_v2
classifier picked a priori for the original question -- the strategy
does NOT force "use a different arm". When ``ctx.arm_subset`` carries
the a priori sel_v2 choice (e.g. ``["iter"]`` for chain-deep questions),
the re-read invokes the iter arm; when ``arm_subset`` is empty or the
chosen arm runner is not wired, the strategy falls back to v3bu
(cheapest single-shot) and then iter, but this is a *runner-availability*
fallback, not a sel_v2 override.

Inspiration -- the strategy stitches together three published primitives:

- **Self-Ask** (Press et al. 2022,
  https://arxiv.org/abs/2210.03350): the reader is prompted to
  articulate which fact it needs before answering.
- **ReAct** (Yao et al. 2022, https://arxiv.org/abs/2210.03629):
  reasoning + retrieval are interleaved with a fact-gap step explicit.
- **CRAG** (Yan et al. 2024, https://arxiv.org/abs/2401.15884):
  retrieval-quality assessment triggers a corrective re-retrieve.

The implementation here is intentionally lightweight: the reader is
the LLM (we do not bring in a separate planner model), retrieval uses
the configured ``RetryContext.vector_db`` + embedder (no new index),
and the loop is bounded by ``max_rounds`` so worst-case cost is
deterministic.
"""

from __future__ import annotations

import logging

from mothrag.core.retry.protocol import RetryContext

logger = logging.getLogger(__name__)


# Self-introspection prompt: the reader articulates the missing fact.
# Kept terse so an echo-style reader still produces something parseable
# in offline tests; LLM readers produce richer text.
_GAP_PROMPT_TEMPLATE = (
    "Given the question: {question}\n\n"
    "And these retrieved passages:\n{passages}\n\n"
    "Identify ONE specific factual gap that prevents answering the "
    "question. Output ONLY a single short search-engine-style query "
    "(under 12 words) that would retrieve the missing fact. No preamble, "
    "no explanation.\n\n"
    "GAP QUERY:"
)


def _is_uncertain(text: str) -> bool:
    if not text:
        return True
    t = text.lower().strip()
    return t in (
        "not in passages", "unknown", "no answer", "none",
        "i don't know", "i do not know", "n/a",
    )


class ActiveGapQueryStrategy:
    """Reader self-introspects -> targeted retrieve -> re-read.

    Loops ``max_rounds`` times before deferring to the next strategy in
    the cascade. Each round:

      1. Build a self-introspection prompt over (question, current passages).
      2. LLM call: reader produces a targeted "gap query".
      3. Re-retrieve top-K passages for the gap query via the
         :class:`RetryContext` retriever / embedder + vector_db.
      4. Merge the new passages with the existing context, deduping
         by chunk text.
      5. Re-read via ``run_arm_v3bu`` (the simplest single-shot arm).
      6. If the new answer is non-empty + non-uncertain, return it.
         Else iterate.

    Returns ``None`` if all rounds fail to recover -- the cascade
    continues to the next strategy.
    """

    name = "active_gap_query"
    cost_estimate = 2  # worst-case per call: gap-query LLM + re-read LLM

    def __init__(
        self,
        *,
        max_rounds: int = 3,
        max_passages_per_round: int = 5,
    ) -> None:
        self.max_rounds = int(max_rounds)
        self.max_passages_per_round = int(max_passages_per_round)

    def applicable(self, ctx: RetryContext) -> bool:
        # Needs the reader for self-introspection and either an embedder
        # + vector_db OR an explicit run_arm_v3bu for the final re-read.
        if ctx.reader is None:
            return False
        if ctx.run_arm_v3bu is None and ctx.run_arm_iter is None:
            return False
        # Fire on signals where targeted gap discovery is plausibly useful.
        if ctx.abstention_signal not in (
            "gamma_refuse", "iter_abstain", "h4_refuse",
            "cross_arm_disagree", "empty_answer",
        ):
            return False
        # Precision gate (empirical finding): decline
        # when the original chosen pred is substantive. Replacing a
        # substantive partial-correct pred with an alternative on
        # gamma=invalid caused -13 to -23pp F1 regression in the
        # 0<F1<0.3 mid-range cohort. Restrict #8 to the genuine
        # "empty / refuse-template" abstention cohort where the
        # alternative actually has a chance to improve over the chosen.
        from mothrag.core.retry.strategies.suppress_gate import (
            pred_has_substance,
        )
        if pred_has_substance(ctx.chosen):
            return False
        return True

    def try_recover(self, ctx: RetryContext) -> str | None:
        # Allow per-call override of max_rounds + max_passages_per_round
        # via ctx.config["active_gap_max_rounds"] and
        # ctx.config["active_gap_max_passages_per_round"].
        max_rounds = int(ctx.config.get("active_gap_max_rounds", self.max_rounds))
        max_passages = int(
            ctx.config.get(
                "active_gap_max_passages_per_round", self.max_passages_per_round,
            )
        )

        # Each round costs up to ``cost_estimate`` LLM calls; bail when
        # the cascade budget cannot absorb at least one more round.
        if not ctx.spend(self.cost_estimate):
            logger.debug(
                "active_gap_query: cascade budget exhausted; skipping.",
            )
            return None

        # Working context: start from the initial passages, accumulate
        # newly retrieved ones.
        cur_passages: list[str] = list(ctx.passages)
        seen_passages: set[str] = set(cur_passages)

        for round_idx in range(max_rounds):
            # Step 1+2: reader articulates the gap as a search-engine
            # query.
            try:
                gap_query = self._formulate_gap_query(
                    ctx, cur_passages, round_idx,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "active_gap_query: self-introspection failed (%s); "
                    "deferring to next strategy.", exc,
                )
                return None
            if not gap_query:
                logger.debug(
                    "active_gap_query: empty gap query in round %d; "
                    "stopping.", round_idx,
                )
                break

            # Step 3: targeted re-retrieve for the gap.
            new_passages = self._retrieve_for_gap(
                ctx, gap_query, max_passages=max_passages,
            )
            new_unique = [p for p in new_passages if p not in seen_passages]
            if not new_unique:
                # Retrieval yielded only already-seen passages -- no new
                # information. Move to next round (the gap query may
                # converge on a different formulation next iteration);
                # if we're already on the last round, defer to cascade.
                if round_idx + 1 >= max_rounds:
                    break
                continue
            seen_passages.update(new_unique)
            cur_passages = cur_passages + new_unique[:max_passages]

            # Step 5: re-read with the augmented context.
            answer = self._reread(ctx, cur_passages)
            if answer and not _is_uncertain(answer):
                return answer

            # Iterate: charge another round against the budget. If
            # the budget no longer allows a full round, bail cleanly.
            if round_idx + 1 < max_rounds:
                if not ctx.spend(self.cost_estimate):
                    logger.debug(
                        "active_gap_query: cascade budget exhausted after "
                        "round %d; deferring.", round_idx + 1,
                    )
                    return None
        return None

    # ------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------

    def _formulate_gap_query(
        self,
        ctx: RetryContext,
        cur_passages: list[str],
        round_idx: int,  # noqa: ARG002 (kept for logging hooks)
    ) -> str:
        # Cap the passage payload so the prompt stays inside the reader's
        # context limit on most LLMs (rough char budget per passage).
        excerpt = "\n".join(p[:400] for p in cur_passages[:5])
        prompt = _GAP_PROMPT_TEMPLATE.format(
            question=ctx.question, passages=excerpt,
        )
        raw = ctx.reader.read(prompt, cur_passages[:5])
        if not raw:
            return ""
        # Take the first non-empty line and trim labels like "GAP QUERY:"
        first = next(
            (line.strip() for line in raw.splitlines() if line.strip()),
            "",
        )
        for prefix in ("GAP QUERY:", "Query:", "Search:"):
            if first.upper().startswith(prefix.upper()):
                first = first[len(prefix):].strip()
        # Refuse a no-op rephrase (gap query identical to original).
        if first.strip().lower() == ctx.question.strip().lower():
            return ""
        return first

    def _retrieve_for_gap(
        self, ctx: RetryContext, gap_query: str, *, max_passages: int,
    ) -> list[str]:
        if ctx.embedder is None or ctx.vector_db is None:
            return []
        try:
            q_emb = ctx.embedder.embed_batch([gap_query])[0]
            chunks = ctx.vector_db.retrieve(q_emb, top_k=max_passages)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "active_gap_query: retrieval for gap query failed: %s", exc,
            )
            return []
        return [c.text for c in chunks if getattr(c, "text", None)]

    def _reread(
        self, ctx: RetryContext, cur_passages: list[str],
    ) -> str:
        """Re-read with the augmented context via the sel_v2 a priori arm.

        Respects the production sel_v2 dispatch: when ``ctx.arm_subset``
        names the arm that the classifier picked for this question, the
        re-read goes through that arm. Only when the requested runner is
        not wired do we fall back through v3bu (cheapest single-shot)
        then iter -- this is a runner-availability fallback, not a
        sel_v2 override. The "same arm as the failing path" outcome is
        the intended behaviour for spectral re-entry: the question
        type is unchanged, so sel_v2's a priori choice is unchanged.
        """
        runner = self._select_runner_via_sel_v2(ctx)
        if runner is None:
            return ""
        try:
            if runner is ctx.run_arm_iter:
                # iter signature requires q_emb + top_k; reuse the
                # existing ctx.q_emb (the original question's embedding)
                # since the re-read is over the augmented passage list,
                # not a new question.
                return runner(
                    question=ctx.question,
                    passages=cur_passages,
                    q_emb=ctx.q_emb,
                    top_k=ctx.top_k,
                )
            return runner(question=ctx.question, passages=cur_passages)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "active_gap_query: re-read arm runner failed: %s", exc,
            )
            return ""

    @staticmethod
    def _select_runner_via_sel_v2(ctx: RetryContext):
        """Pick the re-read runner respecting sel_v2's a priori choice.

        Order of preference:
          1. ``ctx.arm_subset[0]`` (the production sel_v2 a priori
             choice for this question), if the corresponding runner
             is wired in ``ctx``.
          2. v3bu (cheapest single-shot fallback).
          3. iter (only when neither sel_v2's choice nor v3bu wired).

        Returns ``None`` when no arm runner is available at all.
        """
        arm_subset = getattr(ctx, "arm_subset", None) or []
        runners_by_name = {
            "v3bu": ctx.run_arm_v3bu,
            "decompose": ctx.run_arm_decompose,
            "iter": ctx.run_arm_iter,
        }
        if arm_subset:
            sel_v2_choice = arm_subset[0]
            chosen = runners_by_name.get(sel_v2_choice)
            if chosen is not None:
                return chosen
        # Runner-availability fallback chain.
        return ctx.run_arm_v3bu or ctx.run_arm_iter


__all__ = ["ActiveGapQueryStrategy"]
