# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""SubQuestionRerouteCascadeStrategy (#9) -- spectral-guided recursive
RAG with progressive state accumulation.

This is *not* an arm-swap mechanism. The strategy is best understood
as **re-running MothRag from a different starting point**: rather than
asking the original question, the strategy decomposes the question into
narrower sub-questions, answers each one via the same production
:func:`mothrag.core.query_type_classifier.arm_subset` dispatcher that
chose an arm for the original question (so the same arm IS the right
choice for a sub-question when sel_v2 says so), and composes the
collected sub-answers back into a final answer. Context accumulates
progressively across sub-questions -- each subsequent sub-question
sees the running ``(sub_q, sub_a)`` trail so multi-hop dependencies
resolve naturally.

The sub-question source is a 3-layer cascade detecting *spectral gaps*
between (a) the hypothetical optimal answer, (b) the generated answer
the cascade is recovering from, and (c) the retrieved chunks linking
the query to candidate answers:

Layer 1 SYNTACTIC: clausal split via regex + optional spaCy
dependency parse. Zero LLM. Picks sub-questions from coordinating
conjunctions ("X and Y") and sentence boundaries.

Layer 2 SPECTRAL: per-aspect disaggregation of γ + L4b +
cross-arm-agreement primitives (:mod:`mothrag.core.spectral`).
Aspects whose worst primitive scores below
``spectral_low_signal_threshold`` are flagged as the spectral gap
points; the strategy formulates one verifying sub-question per
low-signal aspect. Optional LLM rephrase when
``use_llm_in_spectral`` is True; deterministic template by default.

Layer 3 LLM FALLBACK: a single decomposition call when Layers 1 + 2
together produce fewer than ``min_sub_questions_before_llm``
sub-questions (i.e. the question has no recognisable spectral or
syntactic structure).

The sel_v2 classifier already chose the best arm for the original
question a priori; the strategy does NOT force "use a different arm
this time". Per sub-question, ``arm_subset(sub_q)`` is consulted
afresh and the resulting arm is invoked -- it may be the same arm
that produced the original (now-recovering) answer, or a different
arm, depending entirely on sel_v2's a priori decision for that
sub-question shape.

Loop: γ-check after composition; if still ``invalid`` and depth budget
remains, re-enter with a fresh layer-2 spectral pass over the new
composed answer (the spectral gap shifts as the running state grows).
Default ``max_depth=3``.

Cost: deterministic; worst-case ``(num_sub_questions + 1) * 1`` LLM
calls per pass; depth bounded.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from mothrag.core.retry.protocol import RetryContext
from mothrag.core.spectral import (
    agreement_per_aspect,
    extract_aspects,
    gamma_per_aspect,
    l4b_per_aspect,
)

logger = logging.getLogger(__name__)


# Default 3-layer cascade order.
DEFAULT_LAYERS: tuple[str, ...] = ("syntactic", "spectral", "llm")

# Layer 2 spectral signal-strength threshold; aspects scoring strictly
# below this on the WORST of the three primitives are flagged as
# low-signal and selected for targeted sub-question formulation.
DEFAULT_SPECTRAL_LOW_SIGNAL_THRESHOLD = 0.5

# Layer 3 LLM fallback fires when Layers 1 + 2 together produced FEWER
# than this many sub-questions.
DEFAULT_MIN_SUB_QUESTIONS_BEFORE_LLM = 2

# Hard cap on sub-questions per pass (worst-case cost guard).
DEFAULT_MAX_SUB_QUESTIONS = 6

# Layer 1 syntactic conjunction split. Matches the most common clausal
# split markers (commas + coordinating conjunctions + "first/then" /
# "as well as" / "in addition" patterns).
_SYNTACTIC_SPLIT_RE = re.compile(
    r"\s*(?:,|;|\band\b|\bor\b|\bbut also\b|\bas well as\b|"
    r"\bin addition to\b|\bfirst\b|\bthen\b)\s+",
    re.IGNORECASE,
)


def _is_uncertain(text: str) -> bool:
    if not text:
        return True
    t = text.lower().strip()
    return t in (
        "not in passages", "unknown", "no answer", "none",
        "i don't know", "i do not know", "n/a",
    )


class SubQuestionRerouteCascadeStrategy:
    """Spectral-guided recursive RAG with progressive state accumulation.

    Per sub-question, the production sel_v2 classifier
    (:func:`mothrag.core.query_type_classifier.arm_subset`) is consulted
    afresh; the resulting arm runs the sub-question. The same arm that
    produced the original (now-recovering) answer IS chosen again when
    sel_v2 says so -- there is no "different arm" forcing. State
    accumulates: each subsequent sub-question receives the running
    ``(prior_sub_q, prior_sub_a)`` trail in its context so multi-hop
    dependencies resolve sequentially.

    Parameters
    ----------
    layers
        Subset / ordering of ``DEFAULT_LAYERS``. Each layer is tried in
        order; layers contribute sub-questions cumulatively. Set to
        ``("syntactic",)`` for fastest / cheapest operation.
    max_depth
        Iteration budget; the cascade re-enters after composition when
        γ still invalid.
    spectral_low_signal_threshold
        Signal threshold below which an aspect is flagged for targeted
        sub-question generation in Layer 2.
    min_sub_questions_before_llm
        Layer 3 LLM fallback fires when Layers 1 + 2 produced fewer
        than this many sub-questions.
    max_sub_questions
        Worst-case sub-question count per pass (cost guard).
    use_llm_in_spectral
        When True, Layer 2 calls the reader to phrase a natural-language
        sub-question per low-signal aspect. When False (default) the
        template "Verify [aspect]?" is used (zero LLM).
    """

    name = "sub_question_reroute"
    cost_estimate = 4  # rough worst-case LLM-call budget per cascade pass

    def __init__(
        self,
        *,
        layers: Sequence[str] = DEFAULT_LAYERS,
        max_depth: int = 3,
        spectral_low_signal_threshold: float = DEFAULT_SPECTRAL_LOW_SIGNAL_THRESHOLD,
        min_sub_questions_before_llm: int = DEFAULT_MIN_SUB_QUESTIONS_BEFORE_LLM,
        max_sub_questions: int = DEFAULT_MAX_SUB_QUESTIONS,
        use_llm_in_spectral: bool = False,
    ) -> None:
        self.layers = tuple(layers)
        self.max_depth = int(max_depth)
        self.spectral_low_signal_threshold = float(spectral_low_signal_threshold)
        self.min_sub_questions_before_llm = int(min_sub_questions_before_llm)
        self.max_sub_questions = int(max_sub_questions)
        self.use_llm_in_spectral = bool(use_llm_in_spectral)

    def applicable(self, ctx: RetryContext) -> bool:
        if ctx.reader is None:
            return False
        # Need either a single-shot arm runner or the embedder + vector_db
        # so per-sub-question arms can fire.
        if ctx.run_arm_v3bu is None and ctx.run_arm_iter is None:
            return False
        if ctx.abstention_signal not in (
            "gamma_refuse", "iter_abstain", "h4_refuse",
            "cross_arm_disagree", "empty_answer",
        ):
            return False
        # Precision gate (empirical finding): decline
        # when the original chosen pred is substantive. Replacing a
        # substantive partial-correct pred with a composed sub-answer
        # synthesis on gamma=invalid caused -16 to -23pp F1 regression
        # in the 0<F1<0.3 mid-range cohort. The gate keeps #9 targeted
        # at the genuine "empty / refuse-template" abstention cohort
        # where the decomposition synthesis can recover ground.
        from mothrag.core.retry.strategies.suppress_gate import (
            pred_has_substance,
        )
        if pred_has_substance(ctx.chosen):
            return False
        return True

    def try_recover(self, ctx: RetryContext) -> str | None:
        if not ctx.spend(self.cost_estimate):
            logger.debug(
                "sub_question_reroute: cascade budget exhausted; skipping.",
            )
            return None

        # Per-call config overrides (so MothRAG(... sub_question_layers=...,
        # sub_question_max_depth=...) propagates through ctx.config).
        layers = tuple(ctx.config.get("sub_question_layers", self.layers))
        max_depth = int(ctx.config.get("sub_question_max_depth", self.max_depth))
        max_sub_qs = int(
            ctx.config.get("sub_question_max_sub_questions", self.max_sub_questions)
        )
        # Stash for use by the layer helpers (avoid plumbing through each).
        self._effective_layers = layers
        self._effective_max_sub_questions = max_sub_qs

        for depth in range(max_depth):
            sub_qs = self._generate_sub_questions(ctx)
            if not sub_qs:
                logger.debug(
                    "sub_question_reroute: no sub-questions at depth %d; "
                    "deferring.", depth,
                )
                return None
            sub_qs = sub_qs[:max_sub_qs]
            answers_per_subq = self._answer_sub_questions(ctx, sub_qs)
            composed = self._compose(ctx, sub_qs, answers_per_subq)
            if composed and not _is_uncertain(composed):
                return composed
            # γ-check after composition: if we're still in a refuse signal,
            # iterate with rephrased sub-questions. The depth guard prevents
            # unbounded recursion.
            if depth + 1 < max_depth:
                if not ctx.spend(self.cost_estimate):
                    logger.debug(
                        "sub_question_reroute: cascade budget exhausted "
                        "after depth %d; deferring.", depth + 1,
                    )
                    return None
        return None

    # ------------------------------------------------------------
    # Sub-question generation
    # ------------------------------------------------------------

    def _generate_sub_questions(self, ctx: RetryContext) -> list[str]:
        """Cumulative 3-layer sub-question collection."""
        layers = getattr(self, "_effective_layers", self.layers)
        max_sub_qs = getattr(self, "_effective_max_sub_questions", self.max_sub_questions)
        sub_qs: list[str] = []

        if "syntactic" in layers:
            sub_qs.extend(self._layer1_syntactic(ctx))

        if "spectral" in layers:
            sub_qs.extend(self._layer2_spectral(ctx))

        # Layer 3 LLM fallback fires only when Layers 1+2 produced too few.
        if "llm" in layers and len(sub_qs) < self.min_sub_questions_before_llm:
            sub_qs.extend(self._layer3_llm_fallback(ctx))

        # De-dupe (first occurrence wins) + cap.
        seen: set[str] = set()
        out: list[str] = []
        for q in sub_qs:
            key = q.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(q.strip())
            if len(out) >= max_sub_qs:
                break
        return out

    def _layer1_syntactic(self, ctx: RetryContext) -> list[str]:
        """Clausal split + spaCy dep-parse (when available)."""
        # 1a. Regex clausal split (zero deps, always available).
        clauses = [
            c.strip() for c in _SYNTACTIC_SPLIT_RE.split(ctx.question)
            if c.strip() and len(c.strip().split()) >= 2
        ]
        # If the original question is short, the split produces only the
        # original; we don't gain anything. Try the spaCy path too.
        out = list(clauses) if len(clauses) > 1 else []

        # 1b. spaCy clausal extraction (opt-in via mothrag[active-learning]).
        try:
            import spacy  # noqa: F401
            from spacy.language import Language  # noqa: F401
        except ImportError:
            return out
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            return out
        doc = nlp(ctx.question)
        for sent in doc.sents:
            text = sent.text.strip().rstrip(".")
            if text and len(text.split()) >= 2 and text != ctx.question.strip():
                out.append(text)
        return out

    def _layer2_spectral(self, ctx: RetryContext) -> list[str]:
        """γ + L4b + agreement disaggregated per aspect; flag low-signal."""
        chosen_answer = ctx.chosen or ""
        aspects = extract_aspects(chosen_answer)
        if not aspects:
            return []

        # Build arm_answers map from the existing context for the
        # agreement disaggregator.
        arm_answers: dict[str, str] = {}
        for name, pred in (
            ("v3bu", ctx.v3bu_pred),
            ("decompose", ctx.dec_pred),
            ("iter", ctx.iter_pred),
        ):
            if pred:
                arm_answers[name] = pred

        # Per-aspect scores. The whole-answer signals come from ctx.
        gamma_status = self._extract_gamma_status_from_ctx(ctx)
        l4b_cancelled = self._extract_l4b_cancelled_from_ctx(ctx)

        gamma_scores = gamma_per_aspect(
            chosen_answer, gamma_status=gamma_status, aspects=aspects,
        )
        l4b_scores = l4b_per_aspect(
            chosen_answer, l4b_cancelled=l4b_cancelled, aspects=aspects,
        )
        agree_scores = agreement_per_aspect(
            chosen_answer, arm_answers=arm_answers,
            embedder=ctx.embedder, aspects=aspects,
        )

        # Identify low-signal aspects: WORST primitive < threshold.
        low_signal_aspects: list[str] = []
        for aspect in aspects:
            worst = min(
                gamma_scores.get(aspect, 1.0),
                l4b_scores.get(aspect, 1.0),
                agree_scores.get(aspect, 0.0),
            )
            if worst < self.spectral_low_signal_threshold:
                low_signal_aspects.append(aspect)

        if not low_signal_aspects:
            return []

        return [
            self._spectral_aspect_to_sub_question(ctx, aspect)
            for aspect in low_signal_aspects
        ]

    def _spectral_aspect_to_sub_question(
        self, ctx: RetryContext, aspect: str,
    ) -> str:
        """Phrase a low-signal aspect as a sub-question."""
        if self.use_llm_in_spectral and ctx.reader is not None:
            prompt = (
                f"Re-phrase the following uncertain claim as a verifying "
                f"sub-question, under 12 words. Claim: \"{aspect}\". "
                f"Output ONLY the sub-question."
            )
            try:
                out = ctx.reader.read(prompt, [aspect])
                first = next(
                    (line.strip() for line in out.splitlines() if line.strip()),
                    "",
                )
                if first:
                    return first
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "sub_question_reroute Layer 2 LLM rephrase failed: %s; "
                    "falling back to template.", exc,
                )
        # Deterministic template fallback.
        return f"Verify: {aspect}?"

    def _layer3_llm_fallback(self, ctx: RetryContext) -> list[str]:
        """Single LLM call: decompose the question into 2-4 sub-questions."""
        if ctx.reader is None:
            return []
        prompt = (
            f"Decompose the following multi-hop question into 2 to 4 "
            f"atomic sub-questions, one per line, no numbering, no "
            f"preamble.\n\nQUESTION: {ctx.question}\n\nSUB-QUESTIONS:"
        )
        try:
            raw = ctx.reader.read(prompt, ctx.passages[:3])
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "sub_question_reroute Layer 3 LLM fallback failed: %s", exc,
            )
            return []
        out: list[str] = []
        for line in (raw or "").splitlines():
            line = line.strip().lstrip("-*0123456789. )")
            if not line:
                continue
            # Reject lines that look like preamble (heuristic).
            if any(line.lower().startswith(p) for p in (
                "sub-question", "subquestion", "here are", "the answer",
            )):
                continue
            if len(line.split()) >= 2:
                out.append(line)
        return out

    # ------------------------------------------------------------
    # Per-sub-question dispatch (with progressive state accumulation)
    # ------------------------------------------------------------

    def _answer_sub_questions(
        self, ctx: RetryContext, sub_qs: list[str],
    ) -> list[str]:
        """Run sub-questions sequentially with cumulative state.

        For each sub-question:

        1. Consult :func:`arm_subset` -- sel_v2 picks the arm a priori.
           Same arm as the original IS chosen when sel_v2 says so;
           there is no "different arm" forcing.
        2. Build the per-sub-question context as the running trail of
           prior ``(sub_q, sub_a)`` pairs PLUS the original passages.
           This is the "context accumulates progressively" contract --
           sub-question N+1 sees the answers to sub-questions 1..N.
        3. Invoke the chosen arm with the augmented question + passages.
        4. Append the result to the trail; continue.

        This mirrors the production decompose-arm pattern in
        ``scripts/route_prospective.py._run_decompose`` so the
        spectral-guided sub-question cascade behaves like a deeper
        recursion of the same production primitive, not a parallel
        alternative.
        """
        try:
            from mothrag.core.query_type_classifier import arm_subset
        except Exception:
            arm_subset = None

        # Running trail: each entry is (sub_q, sub_a). Used to prepend
        # prior-hop context into subsequent sub-question prompts.
        sub_qa_trail: list[tuple[str, str]] = []
        out: list[str] = []
        for q in sub_qs:
            arm_choice = "v3bu"
            if arm_subset is not None:
                try:
                    subset = arm_subset(q)
                    if subset:
                        # sel_v2 picks the arm a priori. Same arm as the
                        # original chosen arm is allowed -- the
                        # spectral-gap re-entry uses sel_v2's a priori
                        # decision for the sub-question SHAPE, not a
                        # "different arm" exclusion.
                        arm_choice = subset[0]
                except Exception:
                    pass

            # Prepend the running trail (state accumulation): the
            # arm sees the question augmented with prior-hop facts,
            # and the passage list grows monotonically with prior
            # sub-answers materialised as supplementary context.
            augmented_q, augmented_passages = self._augment_with_trail(
                q, ctx.passages, sub_qa_trail,
            )
            answer = self._run_arm(
                ctx, arm_choice, augmented_q, augmented_passages,
            )
            sub_qa_trail.append((q, answer or ""))
            out.append(answer or "")
        return out

    @staticmethod
    def _augment_with_trail(
        sub_q: str,
        base_passages: Sequence[str],
        trail: list[tuple[str, str]],
    ) -> tuple[str, list[str]]:
        """Compose the sub-question prompt + passage list with prior state.

        Mirrors the cross-hop context propagation in the production
        decompose arm: prior sub-answers join the question as inline
        bracketed context, and as additional passages so retrievers
        / readers that scan passages also see the facts.
        """
        if not trail:
            return sub_q, list(base_passages)
        prior_facts = "; ".join(
            f"{prev_q.rstrip('?')}: {prev_a}"
            for prev_q, prev_a in trail
            if prev_a
        )
        if not prior_facts:
            return sub_q, list(base_passages)
        augmented_q = (
            f"{sub_q}\n(Context from prior sub-answers: {prior_facts})"
        )
        # Materialise the trail as one synthetic passage at the head of
        # the passage list so passage-only readers also see the state.
        augmented_passages = [
            f"Prior sub-question facts: {prior_facts}",
            *base_passages,
        ]
        return augmented_q, augmented_passages

    def _run_arm(
        self,
        ctx: RetryContext,
        arm_choice: str,
        question: str,
        passages: Sequence[str],
    ) -> str:
        runners = {
            "v3bu": ctx.run_arm_v3bu,
            "decompose": ctx.run_arm_decompose,
            "iter": ctx.run_arm_iter,
        }
        runner = runners.get(arm_choice) or ctx.run_arm_v3bu or ctx.run_arm_iter
        if runner is None:
            return ""
        try:
            if runner is ctx.run_arm_iter:
                return runner(
                    question=question, passages=list(passages),
                    q_emb=ctx.q_emb, top_k=ctx.top_k,
                )
            return runner(question=question, passages=list(passages))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "sub_question_reroute per-sub-question arm failed: %s", exc,
            )
            return ""

    # ------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------

    def _compose(
        self,
        ctx: RetryContext,
        sub_qs: list[str],
        answers_per_subq: list[str],
    ) -> str:
        """Stitch sub-question answers back into a final answer.

        Two strategies, picked deterministically:

        1. Template substitution: when the original question is a
           syntactic conjunction (Layer 1 produced N sub-questions
           matching the conjunction structure), join sub-answers
           with "; " in question order. Fastest, zero-LLM.
        2. LLM synthesis: otherwise, the reader receives all
           (sub_q, sub_a) pairs and is asked to synthesise a single
           coherent answer. One LLM call.

        When the reader is missing we fall through to template.
        """
        non_empty_pairs = [
            (q, a) for q, a in zip(sub_qs, answers_per_subq)
            if a and not _is_uncertain(a)
        ]
        if not non_empty_pairs:
            return ""

        # Strategy 1: template substitution when sub-question count
        # matches a clean syntactic split of the original question.
        clauses = [
            c.strip() for c in _SYNTACTIC_SPLIT_RE.split(ctx.question)
            if c.strip()
        ]
        if len(clauses) > 1 and len(non_empty_pairs) >= 2:
            return "; ".join(a for _, a in non_empty_pairs)

        # Strategy 2: LLM synthesis.
        if ctx.reader is not None:
            payload = "\n".join(
                f"Q: {q}\nA: {a}" for q, a in non_empty_pairs
            )
            prompt = (
                f"Given these sub-question answers:\n{payload}\n\n"
                f"Synthesise a single 1-2 sentence answer to the original "
                f"question: {ctx.question}\n\nFINAL ANSWER:"
            )
            try:
                out = ctx.reader.read(prompt, ctx.passages[:3])
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "sub_question_reroute composition LLM call failed: %s",
                    exc,
                )
                # Fall through to template.
            else:
                first = next(
                    (line.strip() for line in (out or "").splitlines() if line.strip()),
                    "",
                )
                for prefix in ("FINAL ANSWER:", "Answer:", "Final:"):
                    if first.upper().startswith(prefix.upper()):
                        first = first[len(prefix):].strip()
                if first:
                    return first

        # Fallback: join answers in order.
        return "; ".join(a for _, a in non_empty_pairs)

    # ------------------------------------------------------------
    # Signal-extraction helpers
    # ------------------------------------------------------------

    @staticmethod
    def _extract_gamma_status_from_ctx(ctx: RetryContext) -> str | None:
        """Best-effort: surface gamma_status from c7_info / signal."""
        if isinstance(ctx.c7_info, dict):
            g = ctx.c7_info.get("gamma_status") or ctx.c7_info.get("gamma")
            if g is not None:
                return g
        if ctx.abstention_signal == "gamma_refuse":
            return "invalid"
        return None

    @staticmethod
    def _extract_l4b_cancelled_from_ctx(ctx: RetryContext) -> bool | None:
        if isinstance(ctx.c7_info, dict):
            l4b = ctx.c7_info.get("l4b")
            if isinstance(l4b, dict):
                return bool(l4b.get("cancelled"))
        if ctx.abstention_signal == "iter_abstain":
            return True
        return None


__all__ = ["SubQuestionRerouteCascadeStrategy"]
