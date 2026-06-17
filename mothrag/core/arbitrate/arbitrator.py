# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""DeterministicArbitrator — post-hoc arm selection by weighted signal sum.

Score per arm:

    score(arm) = w_gamma * gamma_valid(arm)
               + w_agree * cross_arm_agreement(arm)
               + w_faith * faith(arm)

Inputs are all in ``[0, 1]``; weights are constants with defaults
``w_gamma=1.0, w_agree=0.5, w_faith=0.3``. The arbitrator picks the
highest-scoring arm with a non-empty answer and reports which component
dominated the selection via :attr:`ArbitrateResult.arbitrate_signal`.

The arbitrator is **training-free** by construction: no learned weights,
no fitted thresholds, no per-dataset calibration. All defaults are
sensible-default constants exposed in the public API so deployments can
override per-application without re-deriving them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping

logger = logging.getLogger(__name__)


# Default weights -- intentionally constants, not learned.
DEFAULT_WEIGHTS = {
    "gamma": 1.0,
    "agree": 0.5,
    "faith": 0.3,
}


# Recognised values for ArbitrateResult.arbitrate_signal.
ARBITRATE_SIGNALS = (
    "consensus",   # cross-arm agreement dominated the decision
    "gamma",       # gamma validity dominated
    "faith",       # faithfulness dominated
    "fallback",    # none of the three components meaningfully fired; arbitrary pick
)


@dataclass
class ArbitrateResult:
    """Outcome of :meth:`DeterministicArbitrator.arbitrate`."""

    selected_arm: str
    answer: str
    arm_scores: dict[str, float] = field(default_factory=dict)
    arbitrate_signal: str = "fallback"
    component_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)
    weights_used: dict[str, float] = field(default_factory=dict)


def _is_uncertain(text: str) -> bool:
    if not text:
        return True
    t = text.lower().strip()
    return t in ("not in passages", "unknown", "no answer", "none", "i don't know")


class DeterministicArbitrator:
    """Score arm outputs by weighted signal sum; pick the winner.

    Parameters
    ----------
    w_gamma, w_agree, w_faith
        Non-negative weights. Defaults match :data:`DEFAULT_WEIGHTS`.
    """

    def __init__(
        self,
        w_gamma: float = DEFAULT_WEIGHTS["gamma"],
        w_agree: float = DEFAULT_WEIGHTS["agree"],
        w_faith: float = DEFAULT_WEIGHTS["faith"],
    ) -> None:
        for label, w in (("w_gamma", w_gamma), ("w_agree", w_agree), ("w_faith", w_faith)):
            if w < 0:
                raise ValueError(f"DeterministicArbitrator: {label} must be >= 0, got {w}.")
        self.w_gamma = float(w_gamma)
        self.w_agree = float(w_agree)
        self.w_faith = float(w_faith)

    @property
    def weights(self) -> dict[str, float]:
        return {"gamma": self.w_gamma, "agree": self.w_agree, "faith": self.w_faith}

    def arbitrate(
        self,
        answers: Mapping[str, str],
        *,
        gamma_signals: Mapping[str, float] | None = None,
        agreement_signals: Mapping[str, float] | None = None,
        faith_signals: Mapping[str, float] | None = None,
        arm_probabilities: Mapping[str, float] | None = None,
    ) -> ArbitrateResult:
        """Score every arm and return the winner.

        Parameters
        ----------
        answers
            ``{arm_name: answer_text}``. Empty / uncertainty-template
            answers (see :func:`_is_uncertain`) score zero on every
            component and are only selected as the last-resort fallback.
        gamma_signals, agreement_signals, faith_signals
            Optional ``{arm_name: signal_in_[0,1]}`` dicts. Missing arms
            and missing dicts default to 1.0 for gamma / faith (so a
            production deployment without these signals does not
            artificially down-weight any arm) and 0.0 for agreement (so
            a missing cross-arm agreement signal does not falsely
            promote a single-arm answer).
        arm_probabilities
            PAM-lite extension: optional ``{arm_name: P_arm_in_[0,1]}``
            from :func:`mothrag.core.query_type_classifier.arm_subset_pam_lite`.
            When supplied, the arm's final score is multiplied by
            ``P_arm`` so the router's per-arm prior modulates the
            signal-based score (``combined = P_arm * (w_gamma * gamma +
            w_agree * agreement + w_faith * faith)``). Missing arms or
            missing dict default to 1.0 (no down-weighting -- preserves
            sel_v2 baseline behaviour byte-for-byte).
        """
        if not answers:
            return ArbitrateResult(
                selected_arm="",
                answer="",
                arm_scores={},
                arbitrate_signal="fallback",
                weights_used=self.weights,
            )

        gamma_signals = dict(gamma_signals or {})
        agreement_signals = dict(agreement_signals or {})
        faith_signals = dict(faith_signals or {})
        arm_probabilities = dict(arm_probabilities or {})

        scores: dict[str, float] = {}
        breakdown: dict[str, dict[str, float]] = {}

        for arm, text in answers.items():
            if _is_uncertain(text):
                scores[arm] = 0.0
                breakdown[arm] = {"gamma": 0.0, "agree": 0.0, "faith": 0.0}
                continue
            g = _clamp(gamma_signals.get(arm, 1.0))
            a = _clamp(agreement_signals.get(arm, 0.0))
            f = _clamp(faith_signals.get(arm, 1.0))
            p_arm = _clamp(arm_probabilities.get(arm, 1.0))
            raw = self.w_gamma * g + self.w_agree * a + self.w_faith * f
            score = p_arm * raw
            scores[arm] = score
            breakdown[arm] = {
                "gamma": p_arm * self.w_gamma * g,
                "agree": p_arm * self.w_agree * a,
                "faith": p_arm * self.w_faith * f,
            }

        # Pick the highest-scoring non-uncertain arm.
        non_uncertain = {arm: s for arm, s in scores.items()
                        if not _is_uncertain(answers[arm])}
        if non_uncertain:
            max_score = max(non_uncertain.values())
            # Tie-break: alphabetical name order (deterministic, no surprises).
            winners = sorted(arm for arm, s in non_uncertain.items() if s == max_score)
            selected = winners[0]
            signal = self._dominant_component(breakdown[selected], max_score)
        else:
            # Last-resort fallback: longest non-empty answer; else empty.
            non_empty = [(arm, txt) for arm, txt in answers.items() if txt]
            if non_empty:
                non_empty.sort(key=lambda pair: (-len(pair[1]), pair[0]))
                selected = non_empty[0][0]
            else:
                selected = sorted(answers.keys())[0]
            signal = "fallback"

        return ArbitrateResult(
            selected_arm=selected,
            answer=answers[selected],
            arm_scores=scores,
            arbitrate_signal=signal,
            component_breakdown=breakdown,
            weights_used=self.weights,
        )

    @staticmethod
    def _dominant_component(breakdown: dict[str, float], total: float) -> str:
        """Pick the component contributing the most to a winning score.

        Returns one of :data:`ARBITRATE_SIGNALS`. If every component
        contributed zero, returns ``"fallback"`` (the arm was chosen by
        tie-break, not by signal).
        """
        if total <= 0:
            return "fallback"
        # Order: agree > gamma > faith if tied -- agreement is the most
        # informative signal when it fires (it requires multiple arms to
        # concur), gamma is binary-ish, faith is a soft regulariser.
        ranked = sorted(
            breakdown.items(),
            key=lambda kv: (-kv[1], _component_priority(kv[0])),
        )
        top_name, top_value = ranked[0]
        if top_value <= 0:
            return "fallback"
        name_to_signal = {"gamma": "gamma", "agree": "consensus", "faith": "faith"}
        return name_to_signal.get(top_name, "fallback")


def _component_priority(name: str) -> int:
    # Smaller = higher priority for tie-breaking in _dominant_component.
    return {"agree": 0, "gamma": 1, "faith": 2}.get(name, 3)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return lo
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


__all__ = [
    "ArbitrateResult",
    "DeterministicArbitrator",
    "DEFAULT_WEIGHTS",
    "ARBITRATE_SIGNALS",
]
