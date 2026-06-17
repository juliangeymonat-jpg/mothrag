# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""ArbitratorV2 -- pool-safe composition algebra for MOTHRAG.

Wraps :class:`DeterministicArbitrator` with two additions:

1. **Pool-safety axiom enforcement.** A registered arm that does NOT
   fire (declined via :meth:`Arm.applicable`, returned empty
   ``pred``, or returned a fallback-tagged result) contributes ZERO
   weight to the composition. The axiom -- formalised after an observed
   MQ F1=1 cohort -23/-26pp regression -- is:

       F1(pool ∪ {X}, C) ≡ F1(pool, C)    when    fire(X, C) = 0

   ArbitratorV2 enforces this STRUCTURALLY: the
   :meth:`arbitrate_pool` method takes a ``pool_results`` mapping of
   ``arm_name -> ArmResult`` (or ``None`` for un-fired arms) plus an
   optional ``arm_applicability`` mapping, and filters to fired arms
   BEFORE delegating to :class:`DeterministicArbitrator`. This makes
   the axiom an API contract, not a fragile downstream filter in
   ``route_prospective.py``.

2. **Pluggable agreement strategy.** Per the agreement-strategy design
   intent, ``agreement`` is no longer hard-wired to pairwise cosine.
   The ``agreement_strategy`` parameter selects between:

     - ``"pairwise"`` (default): existing :func:`pairwise_agreement`
       cosine-over-embeddings (Path B baseline; byte-identical to
       :class:`DeterministicArbitrator` semantics).
     - ``"chain"``: per-pair chain-agreement signal (placeholder hook
       for the chain-agreement implementation; raises
       ``NotImplementedError`` with a pointer until the chain
       extractor is wired).
     - Custom callable ``(answers: Mapping[str, str], **ctx) ->
       Mapping[str, float]``: caller injects domain-specific
       agreement.

   The pluggability lets the architectural axiom (pool-safety) stay
   stable while the agreement signal evolves independently.

Backward compat
---------------

When ``arbitrate_pool`` is called with the legacy 3-arm pool ({v3bu,
decompose, iter}) and ``agreement_strategy="pairwise"`` (default),
the output is BYTE-IDENTICAL to
:meth:`DeterministicArbitrator.arbitrate` for the same inputs. The
new class is additive; existing callers do NOT need to migrate.

Design contract: no per-dataset tuning, no learned weights. All
composition rules are deterministic constants + pluggable strategies.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Protocol

from mothrag.arms.base import ArmResult
from mothrag.core.arbitrate.arbitrator import (
    ArbitrateResult,
    DEFAULT_WEIGHTS,
    DeterministicArbitrator,
)

logger = logging.getLogger(__name__)


# ---- Agreement-strategy interface ------------------------------------------

class AgreementStrategy(Protocol):
    """Callable computing ``{arm_name: agreement_in_[0,1]}`` over answers.

    The default implementation is :func:`pairwise_agreement`. Custom
    strategies (e.g. chain-agreement, entity-overlap) implement the
    same signature.
    """

    def __call__(
        self,
        answers: Mapping[str, str],
        **ctx: Any,
    ) -> Mapping[str, float]: ...


def _pairwise_agreement_adapter(
    answers: Mapping[str, str],
    *,
    embedder=None,
    threshold: float = 0.70,
    **_: Any,
) -> Mapping[str, float]:
    """Adapter: wraps :func:`pairwise_agreement` to the strategy signature.

    Required because :func:`pairwise_agreement` has a fixed positional
    + kwarg signature; :class:`AgreementStrategy` accepts ``**ctx``.
    """
    from mothrag.core.arbitrate import pairwise_agreement
    if embedder is None:
        return {k: 0.0 for k in answers}
    try:
        return pairwise_agreement(answers, embedder=embedder, threshold=threshold)
    except Exception:  # noqa: BLE001
        logger.warning("pairwise_agreement failed; defaulting to 0.0")
        return {k: 0.0 for k in answers}


def _chain_agreement_placeholder(
    answers: Mapping[str, str], **ctx: Any,
) -> Mapping[str, float]:
    """Chain-agreement strategy placeholder.

    The chain-agreement signal measures whether one arm's answer can
    be DERIVED from another arm's intermediate reasoning trace (e.g.
    the decompose arm's sub-question chain), rather than just
    matching the final string surface. Requires per-arm trace
    extractors that the legacy pipelines do not yet expose.

    Raises :class:`NotImplementedError` to make the hook explicit;
    callers wanting chain-agreement today must inject a custom
    callable via ``agreement_strategy=...``.
    """
    raise NotImplementedError(
        "chain agreement requires per-arm reasoning-trace extraction "
        "(future). Inject a custom AgreementStrategy "
        "callable until the trace API is wired."
    )


_AGREEMENT_STRATEGIES: dict[str, AgreementStrategy] = {
    "pairwise": _pairwise_agreement_adapter,
    "chain": _chain_agreement_placeholder,
}


# ---- Pool-safety helpers ---------------------------------------------------

def _is_fired(
    result: ArmResult | None,
    applicable: bool | None,
) -> bool:
    """Pool-safety predicate: did arm X actually contribute?

    An arm is "fired" iff:
      - applicable is True (or not provided), AND
      - the result exists, AND
      - the result has a non-empty pred, AND
      - the result is NOT tagged ``is_fallback`` in metadata.

    Mirrors the opt-in arm loop's filter in
    ``scripts/route_prospective.py``; centralised here so the
    contract holds across all consumers.
    """
    if applicable is False:
        return False
    if result is None:
        return False
    if not result.pred:
        return False
    if result.metadata.get("is_fallback"):
        return False
    return True


# ---- ArbitratorV2 ----------------------------------------------------------

class ArbitratorV2:
    """Pool-safe arbitrator with pluggable agreement strategy.

    Parameters
    ----------
    w_gamma, w_agree, w_faith
        Weights, identical to :class:`DeterministicArbitrator`
        defaults (1.0 / 0.5 / 0.3). Non-learned; sensible
        deployment-overridable constants.
    agreement_strategy
        ``"pairwise"`` (default), ``"chain"`` (placeholder), or a
        custom callable matching :class:`AgreementStrategy`.

    Use ``arbitrate_pool`` for pool-safe arbitration. Use the legacy
    ``arbitrate`` (inherited from :class:`DeterministicArbitrator`)
    for the byte-identical baseline path.
    """

    def __init__(
        self,
        w_gamma: float = DEFAULT_WEIGHTS["gamma"],
        w_agree: float = DEFAULT_WEIGHTS["agree"],
        w_faith: float = DEFAULT_WEIGHTS["faith"],
        agreement_strategy: AgreementStrategy | str = "pairwise",
    ) -> None:
        self._base = DeterministicArbitrator(
            w_gamma=w_gamma, w_agree=w_agree, w_faith=w_faith,
        )
        if isinstance(agreement_strategy, str):
            if agreement_strategy not in _AGREEMENT_STRATEGIES:
                raise ValueError(
                    f"unknown agreement_strategy {agreement_strategy!r}; "
                    f"choices: {sorted(_AGREEMENT_STRATEGIES)} or callable"
                )
            self._agreement_strategy: AgreementStrategy = (
                _AGREEMENT_STRATEGIES[agreement_strategy]
            )
            self._agreement_strategy_name = agreement_strategy
        else:
            self._agreement_strategy = agreement_strategy
            self._agreement_strategy_name = "custom"

    @property
    def weights(self) -> dict[str, float]:
        return self._base.weights

    @property
    def agreement_strategy_name(self) -> str:
        return self._agreement_strategy_name

    # ---- The pool-safe entry point ----------------------------------------

    def arbitrate_pool(
        self,
        pool_results: Mapping[str, ArmResult | None],
        *,
        arm_applicability: Mapping[str, bool] | None = None,
        gamma_signals: Mapping[str, float] | None = None,
        faith_signals: Mapping[str, float] | None = None,
        arm_probabilities: Mapping[str, float] | None = None,
        agreement_ctx: Mapping[str, Any] | None = None,
    ) -> ArbitrateResult:
        """Pool-safe arbitration over the FIRED subset of a pool.

        Parameters
        ----------
        pool_results
            ``{arm_name: ArmResult or None}``. ``None`` means the arm
            did not run at all. ``ArmResult`` with empty ``pred`` OR
            ``metadata["is_fallback"]`` means the arm declined or
            fell back; either way it does not contribute weight.
        arm_applicability
            Optional ``{arm_name: bool}`` snapshot of ``Arm.applicable``
            from before run-time. Used to short-circuit the firing
            predicate (an arm with ``applicable=False`` is treated as
            not fired regardless of any result content).
        gamma_signals, faith_signals, arm_probabilities
            Same semantics as :meth:`DeterministicArbitrator.arbitrate`.
            Restricted to the fired-arm subset internally.
        agreement_ctx
            Extra keyword args forwarded to the agreement strategy
            callable (e.g. ``embedder=...`` for pairwise cosine).

        Pool-safety invariant
        ---------------------
        For any arm ``X`` with ``fire(X) = False``, omitting ``X``
        from ``pool_results`` (or passing ``pool_results[X] = None``)
        produces a BYTE-IDENTICAL :class:`ArbitrateResult` to the
        version where ``X`` is included as a non-firing entry. This
        is the math statement
        ``F1(pool ∪ {X}, C) ≡ F1(pool, C)  when  fire(X, C) = 0``
        in code form.
        """
        applicability = dict(arm_applicability or {})
        # Filter to fired arms only.
        answers: dict[str, str] = {}
        for arm_name, result in pool_results.items():
            applicable = applicability.get(arm_name, True)
            if not _is_fired(result, applicable):
                continue
            answers[arm_name] = result.pred  # type: ignore[union-attr]

        if not answers:
            return ArbitrateResult(
                selected_arm="",
                answer="",
                arm_scores={},
                arbitrate_signal="fallback",
                weights_used=self.weights,
            )

        # Restrict signal dicts to fired arms.
        fired_set = set(answers.keys())
        g_filtered = {k: v for k, v in (gamma_signals or {}).items()
                      if k in fired_set}
        f_filtered = {k: v for k, v in (faith_signals or {}).items()
                      if k in fired_set}
        p_filtered = {k: v for k, v in (arm_probabilities or {}).items()
                      if k in fired_set}

        # Compute agreement via the pluggable strategy.
        ctx = dict(agreement_ctx or {})
        try:
            agreement = self._agreement_strategy(answers, **ctx)
        except NotImplementedError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning(
                "agreement_strategy %r failed; defaulting to 0.0 per arm",
                self._agreement_strategy_name,
            )
            agreement = {k: 0.0 for k in answers}

        # Delegate to the base arbitrator over the fired subset.
        return self._base.arbitrate(
            answers=answers,
            gamma_signals=g_filtered,
            agreement_signals=agreement,
            faith_signals=f_filtered,
            arm_probabilities=p_filtered,
        )

    # ---- Legacy-compatible passthrough ------------------------------------

    def arbitrate(
        self,
        answers: Mapping[str, str],
        *,
        gamma_signals: Mapping[str, float] | None = None,
        agreement_signals: Mapping[str, float] | None = None,
        faith_signals: Mapping[str, float] | None = None,
        arm_probabilities: Mapping[str, float] | None = None,
    ) -> ArbitrateResult:
        """Byte-identical legacy passthrough to
        :meth:`DeterministicArbitrator.arbitrate`.

        Provided so existing callers can drop-in switch to
        :class:`ArbitratorV2` without code changes; opt into
        pool-safety semantics by switching to :meth:`arbitrate_pool`.
        """
        return self._base.arbitrate(
            answers=answers,
            gamma_signals=gamma_signals,
            agreement_signals=agreement_signals,
            faith_signals=faith_signals,
            arm_probabilities=arm_probabilities,
        )


__all__ = [
    "AgreementStrategy",
    "ArbitratorV2",
]
