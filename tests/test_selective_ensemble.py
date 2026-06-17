# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for selective-ensemble arbitration rules."""

from mothrag.core.selective_ensemble import (
    selective_arbitrate, is_uncertain, is_chain_pattern,
    normalize_answer, em_score, f1_score,
    arbitrate_with_c7,
)


def test_normalize_strips_articles_and_punct():
    assert normalize_answer("The Movie!") == "movie"
    assert normalize_answer("a Boy") == "boy"


def test_em_and_f1():
    assert em_score("Boston", "Boston") == 1.0
    assert em_score("Boston", "Boston, MA") == 0.0
    f = f1_score("Pavel Sergeyevich Alexandrov", "Pavel Alexandrov")
    assert 0.5 < f < 1.0


def test_is_uncertain():
    assert is_uncertain("")
    assert is_uncertain("Not in passages")
    assert is_uncertain("unknown")
    assert not is_uncertain("Boston")


def test_chain_pattern_detection():
    assert is_chain_pattern("Who is the co-commentator for X?")
    assert is_chain_pattern("Who was the predecessor of Carol?")
    assert is_chain_pattern("The role was succeeded by another actor")
    assert not is_chain_pattern("Who wrote 1984?")


def test_arbitrate_uncertain_falls_back_to_decompose():
    final, reason = selective_arbitrate("Not in passages", "Memento", question="What movie?")
    assert final == "Memento"
    assert "uncertain" in reason


def test_arbitrate_agreement():
    final, reason = selective_arbitrate("Boston", "boston", question="Where?")
    assert final == "Boston"
    assert reason == "agree"


def test_arbitrate_chain_pattern_picks_decompose():
    final, reason = selective_arbitrate(
        "Bob", "Alice", question="Who is the predecessor of Carol?",
    )
    assert final == "Alice"
    assert "chain-pattern" in reason


def test_arbitrate_overlap_prefers_longer():
    final, reason = selective_arbitrate(
        "Pavel Sergeyevich Alexandrov", "Pavel Alexandrov",
        question="Who?",
    )
    assert final == "Pavel Sergeyevich Alexandrov"
    assert "longer" in reason


# ---- Aurora L6 C7 cancellation at ENS arbitrate layer ----

def _orthogonal_embedder(strings):
    """4-D one-hot embedder for deterministic C7 testing."""
    import numpy as np
    rows = []
    for i, _ in enumerate(strings):
        v = np.zeros(4)
        v[i % 4] = 1.0
        rows.append(v)
    return np.array(rows)


def test_c7_disabled_returns_none_info():
    chosen, reason, c7 = arbitrate_with_c7(
        "Boston", "boston", "Where?",
    )
    assert chosen == "Boston"
    assert reason == "agree"
    assert c7 is None


def test_c7_gated_skips_when_gamma_valid():
    chosen, reason, c7 = arbitrate_with_c7(
        "Tarantino", "Spielberg", "Who directed Pulp Fiction?",
        use_c7=True, c7_trigger="gated", gamma_status="valid",
        embedder=_orthogonal_embedder,
    )
    # gated + γ valid → C7 skipped
    assert c7 is None
    assert chosen in ("Tarantino", "Spielberg")


def test_c7_gated_runs_on_gamma_partial():
    chosen, reason, c7 = arbitrate_with_c7(
        "Tarantino", "Spielberg", "Who directed Pulp Fiction?",
        use_c7=True, c7_trigger="gated", gamma_status="partial",
        embedder=_orthogonal_embedder,
    )
    assert c7 is not None
    assert "chosen_kept" in c7
    # K = chosen + 1 unchosen (the loser of arbitrate)
    assert c7["K"] == 2


def test_c7_blanket_runs_always():
    chosen, reason, c7 = arbitrate_with_c7(
        "Tarantino", "Spielberg", "Who directed Pulp Fiction?",
        use_c7=True, c7_trigger="blanket", gamma_status="valid",
        embedder=_orthogonal_embedder,
    )
    assert c7 is not None
    assert "chosen_kept" in c7


def test_c7_three_arm_router_v2_collects_all_unchosen():
    chosen, reason, c7 = arbitrate_with_c7(
        v3bu_pred="Bob", dec_pred="Alice", iter_pred="Carol",
        question="Who is the predecessor of Carol?",
        use_router_v2=True,
        use_c7=True, c7_trigger="blanket",
        embedder=_orthogonal_embedder,
    )
    # router_v2 chain-pattern → iter or dec wins; the other 2 are rejected_chains
    assert c7 is not None
    # K = chosen + 2 unchosen distinct candidates
    assert c7["K"] == 3


def test_c7_returns_none_when_no_distinct_rejected_chains():
    # All arms agree → no rejected_chains → C7 returns None
    chosen, reason, c7 = arbitrate_with_c7(
        v3bu_pred="Boston", dec_pred="boston", question="Where?",
        use_c7=True, c7_trigger="blanket",
        embedder=_orthogonal_embedder,
    )
    assert chosen == "Boston"
    assert c7 is None  # nothing to cancel against


def test_c7_no_embedder_is_noop():
    # Master switch on but no embedder → silent no-op (no crash)
    chosen, reason, c7 = arbitrate_with_c7(
        "Tarantino", "Spielberg", "Who directed Pulp Fiction?",
        use_c7=True, c7_trigger="blanket",
        embedder=None,
    )
    assert c7 is None
