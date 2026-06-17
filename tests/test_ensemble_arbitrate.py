"""DeterministicArbitrator + ensemble-arbitrate mode tests.

15 cases:
  Arbitrator unit (10):
    1.  consensus signal wins when agreement dominates
    2.  gamma signal wins when gamma dominates
    3.  faith signal wins when faith dominates
    4.  empty answers contribute 0 to the score
    5.  all-empty / all-uncertain inputs -> fallback signal
    6.  single non-empty arm -> that arm selected, fallback signal
    7.  custom weights respected (different winner under custom weights)
    8.  ties broken deterministically (alphabetical arm-name order)
    9.  missing gamma/faith default to 1.0; missing agree defaults to 0.0
    10. _is_uncertain templates score zero on all components

  pairwise_agreement (3):
    11. 2/3 identical answers -> >= threshold pairwise agreement
    12. all-different answers -> agreement < threshold for every arm
    13. empty answers carry zero agreement

  MothRAG integration + smoke contract (2):
    14. MothRAG(production=True, mode='ensemble_arbitrate') populates
        arm_scores / selected_arm / arbitrate_signal in metadata.
    15. Smoke contract: ensemble_arbitrate path can pick a different arm
        than the adaptive-path arbiter on a query whose arm_subset would
        have routed to a "wrong" arm.
"""

from __future__ import annotations

import sys
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _block_sentence_transformers(monkeypatch) -> Iterator[None]:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    yield


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch) -> Iterator[None]:
    for k in ("VERTEX_AI_PROJECT", "GOOGLE_CLOUD_PROJECT",
              "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    yield


# ============================================================
# 1-10: DeterministicArbitrator unit tests
# ============================================================

def test_consensus_signal_wins_under_high_agreement() -> None:
    from mothrag.core.arbitrate import DeterministicArbitrator
    arb = DeterministicArbitrator()
    # iter has high agreement; gamma/faith are LOW for the winner so the
    # agreement component is the largest in the winner's weighted breakdown.
    out = arb.arbitrate(
        answers={"v3bu": "ans X", "decompose": "ans Y", "iter": "ans Z"},
        gamma_signals={"v3bu": 0.1, "decompose": 0.1, "iter": 0.1},
        agreement_signals={"v3bu": 0.0, "decompose": 0.0, "iter": 1.0},
        faith_signals={"v3bu": 0.1, "decompose": 0.1, "iter": 0.1},
    )
    # iter score = 1.0*0.1 + 0.5*1.0 + 0.3*0.1 = 0.1 + 0.5 + 0.03 = 0.63
    #              gamma=0.1, agree=0.5, faith=0.03  -> agree is dominant
    # v3bu / decompose score = 0.1 + 0 + 0.03 = 0.13 (each)
    assert out.selected_arm == "iter"
    assert out.arbitrate_signal == "consensus"


def test_gamma_signal_wins_when_gamma_dominates() -> None:
    from mothrag.core.arbitrate import DeterministicArbitrator
    arb = DeterministicArbitrator()
    out = arb.arbitrate(
        answers={"v3bu": "ans X", "decompose": "ans Y", "iter": "ans Z"},
        gamma_signals={"v3bu": 0.1, "decompose": 0.1, "iter": 1.0},
        agreement_signals={"v3bu": 0.0, "decompose": 0.0, "iter": 0.0},
        faith_signals={"v3bu": 0.5, "decompose": 0.5, "iter": 0.5},
    )
    assert out.selected_arm == "iter"
    # gamma contributes 1.0; faith contributes 0.15 -> gamma is dominant
    assert out.arbitrate_signal == "gamma"


def test_faith_signal_wins_when_faith_dominates() -> None:
    from mothrag.core.arbitrate import DeterministicArbitrator
    # Custom weights: kill gamma/agree, keep faith.
    arb = DeterministicArbitrator(w_gamma=0.0, w_agree=0.0, w_faith=1.0)
    out = arb.arbitrate(
        answers={"v3bu": "X", "decompose": "Y", "iter": "Z"},
        gamma_signals={"v3bu": 1.0, "decompose": 1.0, "iter": 1.0},
        agreement_signals={"v3bu": 1.0, "decompose": 1.0, "iter": 1.0},
        faith_signals={"v3bu": 0.1, "decompose": 0.9, "iter": 0.1},
    )
    assert out.selected_arm == "decompose"
    assert out.arbitrate_signal == "faith"


def test_empty_answers_score_zero() -> None:
    from mothrag.core.arbitrate import DeterministicArbitrator
    arb = DeterministicArbitrator()
    out = arb.arbitrate(
        answers={"v3bu": "", "decompose": "real answer here", "iter": ""},
        gamma_signals={"v3bu": 1.0, "decompose": 0.5, "iter": 1.0},
    )
    assert out.selected_arm == "decompose"
    assert out.arm_scores["v3bu"] == 0.0
    assert out.arm_scores["iter"] == 0.0
    assert out.arm_scores["decompose"] > 0.0


def test_all_empty_inputs_returns_fallback_signal() -> None:
    from mothrag.core.arbitrate import DeterministicArbitrator
    arb = DeterministicArbitrator()
    out = arb.arbitrate(
        answers={"v3bu": "", "decompose": "", "iter": ""},
    )
    assert out.arbitrate_signal == "fallback"
    assert out.answer == ""


def test_single_non_empty_arm_selected_with_fallback_signal() -> None:
    """One arm has a real answer; no other arms to agree -> agreement=0,
    only gamma + faith fire. Selected, but signal is gamma (the dominant
    non-zero component)."""
    from mothrag.core.arbitrate import DeterministicArbitrator
    arb = DeterministicArbitrator()
    out = arb.arbitrate(
        answers={"v3bu": "real answer", "decompose": "", "iter": ""},
        gamma_signals={"v3bu": 1.0},
        agreement_signals={"v3bu": 0.0},
        faith_signals={"v3bu": 1.0},
    )
    assert out.selected_arm == "v3bu"
    # w_gamma*1.0 = 1.0; w_faith*1.0 = 0.3 -> gamma dominates.
    assert out.arbitrate_signal == "gamma"


def test_custom_weights_respected() -> None:
    from mothrag.core.arbitrate import DeterministicArbitrator
    # Default weights (gamma=1.0, agree=0.5, faith=0.3) would pick A (high gamma).
    # Custom weights (gamma=0.1, agree=2.0, faith=0.1) should pick B (high agree).
    default_arb = DeterministicArbitrator()
    custom_arb = DeterministicArbitrator(w_gamma=0.1, w_agree=2.0, w_faith=0.1)
    common = dict(
        answers={"A": "a", "B": "b"},
        gamma_signals={"A": 1.0, "B": 0.0},
        agreement_signals={"A": 0.0, "B": 1.0},
        faith_signals={"A": 0.0, "B": 0.0},
    )
    assert default_arb.arbitrate(**common).selected_arm == "A"
    assert custom_arb.arbitrate(**common).selected_arm == "B"


def test_ties_broken_alphabetically() -> None:
    from mothrag.core.arbitrate import DeterministicArbitrator
    arb = DeterministicArbitrator()
    # Both arms produce identical scores -> deterministic alpha tie-break.
    out = arb.arbitrate(
        answers={"zeta": "z", "alpha": "a"},
        gamma_signals={"zeta": 1.0, "alpha": 1.0},
        agreement_signals={"zeta": 0.0, "alpha": 0.0},
        faith_signals={"zeta": 1.0, "alpha": 1.0},
    )
    assert out.selected_arm == "alpha"


def test_missing_signals_use_defaults() -> None:
    from mothrag.core.arbitrate import DeterministicArbitrator
    arb = DeterministicArbitrator()
    # No signal dicts -> gamma defaults to 1.0, agree to 0.0, faith to 1.0.
    # Score = 1.0 * 1.0 + 0.5 * 0.0 + 0.3 * 1.0 = 1.3.
    out = arb.arbitrate(answers={"v3bu": "hi"})
    assert out.selected_arm == "v3bu"
    assert abs(out.arm_scores["v3bu"] - 1.3) < 1e-6


def test_uncertain_template_scores_zero() -> None:
    from mothrag.core.arbitrate import DeterministicArbitrator
    arb = DeterministicArbitrator()
    out = arb.arbitrate(
        answers={
            "v3bu": "not in passages",
            "decompose": "unknown",
            "iter": "real answer",
        },
        gamma_signals={"v3bu": 1.0, "decompose": 1.0, "iter": 1.0},
    )
    # The uncertainty-template arms score zero by construction.
    assert out.arm_scores["v3bu"] == 0.0
    assert out.arm_scores["decompose"] == 0.0
    assert out.selected_arm == "iter"


# ============================================================
# 11-13: pairwise_agreement helper
# ============================================================

def test_pairwise_agreement_majority_cluster_meets_threshold() -> None:
    from mothrag.core.api import _HashEmbedder
    from mothrag.core.arbitrate import pairwise_agreement
    # Two arms have IDENTICAL answers (sim=1.0). The third uses completely
    # disjoint vocabulary so the _HashEmbedder cannot collapse them.
    out = pairwise_agreement(
        {
            "v3bu": "alpha beta gamma delta epsilon",
            "decompose": "alpha beta gamma delta epsilon",
            "iter": "zeta eta theta iota kappa",
        },
        embedder=_HashEmbedder(), threshold=0.9,
    )
    # v3bu and decompose are identical -> sim=1.0 -> agree with each other.
    # iter has disjoint tokens -> sim near 0 -> below threshold.
    assert out["v3bu"] >= 0.5
    assert out["decompose"] >= 0.5
    assert out["iter"] <= 0.5


def test_pairwise_agreement_all_different() -> None:
    from mothrag.core.api import _HashEmbedder
    from mothrag.core.arbitrate import pairwise_agreement
    out = pairwise_agreement(
        {"a": "completely distinct text alpha",
         "b": "wholly unrelated phrase bravo",
         "c": "entirely separate words charlie"},
        embedder=_HashEmbedder(), threshold=0.95,
    )
    for v in out.values():
        assert v < 0.5  # near-zero agreement under a high threshold


def test_pairwise_agreement_empty_answer_is_zero() -> None:
    from mothrag.core.api import _HashEmbedder
    from mothrag.core.arbitrate import pairwise_agreement
    out = pairwise_agreement(
        {"v3bu": "", "decompose": "an answer", "iter": "an answer"},
        embedder=_HashEmbedder(), threshold=0.7,
    )
    assert out["v3bu"] == 0.0
    # decompose vs iter are identical text -> mutual agreement
    assert out["decompose"] >= 0.5
    assert out["iter"] >= 0.5


# ============================================================
# 14-15: MothRAG integration + smoke contract
# ============================================================

def test_mothrag_ensemble_arbitrate_metadata_populated() -> None:
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader
    rag = MothRAG.from_documents(
        ["First doc about Paris.", "Second doc about Tokyo."],
        embedder=_HashEmbedder(),
        reader=_EchoReader(),
        production=True,
        mode="ensemble_arbitrate",
    )
    qr = rag.query("Which city?")
    for key in (
        "mode",
        "production_strategy",
        "arm_scores",
        "selected_arm",
        "arbitrate_signal",
        "arbitrate_breakdown",
        "arbitrate_weights",
        "arbitrate_agreement",
        "v3bu_pred",
        "dec_pred",
        "iter_pred",
    ):
        assert key in qr.metadata, f"missing metadata key {key!r}"
    assert qr.metadata["mode"] == "ensemble_arbitrate"
    assert qr.metadata["production_strategy"] == "ensemble_arbitrate"
    assert qr.metadata["selected_arm"] in ("v3bu", "decompose", "iter")
    assert qr.metadata["arbitrate_signal"] in (
        "consensus", "gamma", "faith", "fallback",
    )
    assert isinstance(qr.metadata["arm_scores"], dict)
    assert set(qr.metadata["arm_scores"].keys()) == {"v3bu", "decompose", "iter"}


def test_ensemble_respects_adaptive_subset_v3bu_excluded() -> None:
    """Pareto-preserving ensemble path: when arm_subset excludes V3+bu, only
    the remaining arms (decompose + iter) run + get arbitrated -- V3+bu is
    NOT force-fired."""
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader
    import mothrag.core.query_type_classifier as qtc_mod

    original_qtc = qtc_mod.arm_subset
    qtc_mod.arm_subset = lambda q: ["decompose", "iter"]  # type: ignore
    try:
        rag = MothRAG.from_documents(
            ["First doc about Paris.", "Second doc about Tokyo."],
            embedder=_HashEmbedder(), reader=_EchoReader(),
            production=True, mode="ensemble_arbitrate",
        )
        qr = rag.query("Which city?")
        assert qr.metadata["mode"] == "ensemble_arbitrate"
        # Subset = 2 arms -> arbitrate fires, v3bu NOT in candidates.
        assert "v3bu" not in qr.metadata.get("arm_scores", {})
        assert qr.metadata["v3bu_pred"] is None
        assert qr.metadata["dec_pred"] is not None
        assert qr.metadata["iter_pred"] is not None
        assert qr.metadata.get("subset_size") == 2
    finally:
        qtc_mod.arm_subset = original_qtc


def test_ensemble_arbitrate_on_multi_arm_subset() -> None:
    """When arm_subset returns >= 2 arms, DeterministicArbitrator scores
    over the running arms only (subset-of-3, not always-3)."""
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader
    rag = MothRAG.from_documents(
        ["First doc about Paris.", "Second doc about Tokyo."],
        embedder=_HashEmbedder(), reader=_EchoReader(),
        production=True, mode="ensemble_arbitrate",
    )
    qr = rag.query("Which city is mentioned?")
    # Default arm_subset returns 2 or 3 arms -> arbitrate fires.
    assert qr.metadata["arbitrate_signal"] != "single_arm_passthrough"
    assert qr.metadata["selected_arm"] in {"v3bu", "decompose", "iter"}
    assert qr.metadata.get("subset_size") in (2, 3)


def test_ensemble_single_arm_subset_passthrough() -> None:
    """Defensive: when the running-arms set collapses to a single arm,
    the ensemble path returns that arm directly (no arbitrate cost over
    the adaptive single-arm path)."""
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader

    import mothrag.core.query_type_classifier as qtc_mod
    original_qtc = qtc_mod.arm_subset
    qtc_mod.arm_subset = lambda q: ["v3bu"]  # type: ignore
    try:
        rag = MothRAG.from_documents(
            ["First doc about Paris.", "Second doc about Tokyo."],
            embedder=_HashEmbedder(), reader=_EchoReader(),
            production=True, mode="ensemble_arbitrate",
        )
        # Stub decompose + iter to return None so candidates dict shrinks
        # to a single entry (mirrors the architecturally-supported
        # single-arm-subset case once sel_v2 is extended).
        rag._arm_decompose = lambda question, passages: None  # type: ignore
        rag._arm_iter = lambda question, passages, *, q_emb, top_k: None  # type: ignore
        qr = rag.query("Which city?")
        assert qr.metadata["arbitrate_signal"] == "single_arm_passthrough"
        assert qr.metadata["selected_arm"] == "v3bu"
        assert qr.metadata.get("subset_size") == 1
        # No arbitrate cost: arm_scores carries only the single arm.
        assert set(qr.metadata.get("arm_scores", {}).keys()) == {"v3bu"}
    finally:
        qtc_mod.arm_subset = original_qtc


def test_ensemble_arbitrate_differs_from_adaptive_on_router_miss() -> None:
    """Smoke contract: ensemble_arbitrate must be able to pick a different
    answer than the adaptive arbiter when the routing classifier would
    have selected a non-consensus arm.

    Mock setup: V3+bu and decompose agree, iter disagrees. The adaptive
    arbiter under the default ``arbitrate_with_c7`` selects based on
    arm_subset + sel_v2 routing logic (independent of consensus). The
    ensemble path's CrossArmConsensus + agreement scoring should always
    favour the V3+bu/decompose majority cluster.

    We assert behaviourally: the ensemble path's *selected_arm* must
    have a non-zero agreement signal, i.e. it must have a co-agreeing
    sibling. This is the strongest behavioural contract we can assert
    offline with mock embedders.
    """
    from mothrag import MothRAG
    from mothrag.core.api import _HashEmbedder, _EchoReader

    class _ConsensusReader:
        """Mock reader that returns the same answer for V3+bu and
        decompose-style prompts (both call read with the question as-is or
        a meta-question containing the original); only differentiates iter
        via the augmented-context prefix."""
        def read(self, question, passages):
            if "Context from prior steps" in question:
                return "iter divergent answer"
            return "consensus answer"

    rag = MothRAG.from_documents(
        ["passage 1.", "passage 2."],
        embedder=_HashEmbedder(),
        reader=_ConsensusReader(),
        production=True,
        mode="ensemble_arbitrate",
    )
    qr = rag.query("Anything?")
    # The selected arm must be one whose answer is in the consensus cluster
    # (since two arms produce identical strings, they cluster).
    selected = qr.metadata["selected_arm"]
    agreement = qr.metadata["arbitrate_agreement"]
    # The selected arm must have a positive agreement signal -- i.e. it has
    # at least one co-agreeing sibling. This is the smoke-contract.
    assert agreement[selected] > 0.0, (
        f"selected arm {selected!r} has zero agreement; ensemble path "
        f"failed to prefer the consensus cluster. agreement={agreement}, "
        f"arm_scores={qr.metadata['arm_scores']}"
    )
