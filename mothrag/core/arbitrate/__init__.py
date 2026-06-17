# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Deterministic post-hoc arbitration for the ensemble-arbitrate mode.

Public surface:

- :class:`DeterministicArbitrator` — scores arm outputs as
  ``w_gamma * γ_valid + w_agree * cross_arm_agreement + w_faith * faith``
  and returns the highest-scoring arm.
- :class:`ArbitrateResult` — selected arm + per-arm scores + arbitrate_signal.
- :func:`pairwise_agreement` — cosine-based cross-arm agreement helper.

The arbitrator is deliberately training-free: weights are constants (with
sensible defaults), all signals are bounded to ``[0, 1]``, and the
``arbitrate_signal`` enumerates which component dominated the decision.
"""

from __future__ import annotations

from mothrag.core.arbitrate.arbitrator import (
    ArbitrateResult,
    DeterministicArbitrator,
    DEFAULT_WEIGHTS,
    ARBITRATE_SIGNALS,
)
from mothrag.core.arbitrate.arbitrator_v2 import (
    AgreementStrategy,
    ArbitratorV2,
)
from mothrag.core.arbitrate.signals import pairwise_agreement
from mothrag.core.arbitrate.pam_lite_arbitrator import arbitrate_pam_lite
from mothrag.core.arbitrate.gamma_l4b_andgate import (
    gamma_l4b_andgate_decision,
    gamma_l4b_andgate_diagnostic,
)
from mothrag.core.arbitrate.refuse_abstention import (
    refuse_abstention_trigger,
    refuse_abstention_dispatch,
    RefuseDispatch,
    RefuseTrigger,
)
from mothrag.core.arbitrate.pam_lite_arbitrator import (
    arbitrate_pam_lite_traced,
    PamLiteDiagnostic,
    TieBreakStrategy,
)
from mothrag.core.arbitrate.pam_lite_mechanism import (
    trace_pam_lite_mechanism,
    MechanismTrace,
    AGREEMENT_THRESHOLD_DEFAULT,
)
from mothrag.core.arbitrate.signal_dup import (
    DupSignalResult,
    dup_signal_into_aggregator,
    pdd_lift_predicted,
    apply_cardinality_average,
    apply_fixed_weighted_sum,
    AGGREGATOR_NORMALIZATION_TABLE,
)

__all__ = [
    "AgreementStrategy",
    "ArbitrateResult",
    "ArbitratorV2",
    "DeterministicArbitrator",
    "DEFAULT_WEIGHTS",
    "ARBITRATE_SIGNALS",
    "pairwise_agreement",
    "arbitrate_pam_lite",
    "arbitrate_pam_lite_traced",
    "PamLiteDiagnostic",
    "TieBreakStrategy",
    "trace_pam_lite_mechanism",
    "MechanismTrace",
    "AGREEMENT_THRESHOLD_DEFAULT",
    "DupSignalResult",
    "dup_signal_into_aggregator",
    "pdd_lift_predicted",
    "apply_cardinality_average",
    "apply_fixed_weighted_sum",
    "AGGREGATOR_NORMALIZATION_TABLE",
    "gamma_l4b_andgate_decision",
    "gamma_l4b_andgate_diagnostic",
    "refuse_abstention_trigger",
    "refuse_abstention_dispatch",
    "RefuseDispatch",
    "RefuseTrigger",
]
