"""Tests for mothrag.core.spectral (per-aspect γ / L4b / agreement)."""

from __future__ import annotations

import sys
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _block_sentence_transformers(monkeypatch) -> Iterator[None]:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    yield


# ============================================================
# extract_aspects (naive backend)
# ============================================================

def test_extract_aspects_naive_capitalised_nps() -> None:
    from mothrag.core.spectral import extract_aspects
    out = extract_aspects("Paris is the capital of France and home to Eiffel Tower.")
    # Naive extractor catches Capitalised NPs.
    assert "Paris" in out
    assert any("France" in a for a in out)
    assert any("Eiffel" in a for a in out)


def test_extract_aspects_naive_numeric_quantities() -> None:
    from mothrag.core.spectral import extract_aspects
    out = extract_aspects("The population is 2.1M and the area is 105 km.")
    joined = " ".join(out)
    assert "2.1M" in joined or "2.1" in joined
    assert "105" in joined or "105 km" in joined


def test_extract_aspects_empty_input_returns_empty() -> None:
    from mothrag.core.spectral import extract_aspects
    assert extract_aspects("") == []
    assert extract_aspects("   ") == []


def test_extract_aspects_max_aspects_respected() -> None:
    from mothrag.core.spectral import extract_aspects
    # Many capitalised NPs in input; cap at 3.
    text = "Paris London Tokyo Rome Berlin Madrid"
    out = extract_aspects(text, max_aspects=3)
    assert len(out) <= 3


def test_extract_aspects_dedupes_first_occurrence_wins() -> None:
    from mothrag.core.spectral import extract_aspects
    text = "Paris is in France. Paris has Eiffel Tower."
    out = extract_aspects(text)
    paris_count = sum(1 for a in out if a == "Paris")
    assert paris_count == 1


# ============================================================
# gamma_per_aspect
# ============================================================

def test_gamma_per_aspect_status_invalid_broadcasts_zero() -> None:
    from mothrag.core.spectral import gamma_per_aspect
    out = gamma_per_aspect(
        "Paris is the capital of France.",
        gamma_status="invalid",
    )
    assert out
    for score in out.values():
        assert score == 0.0


def test_gamma_per_aspect_status_partial_broadcasts_half() -> None:
    from mothrag.core.spectral import gamma_per_aspect
    out = gamma_per_aspect(
        "Paris is the capital of France.",
        gamma_status="partial",
    )
    for score in out.values():
        assert score == 0.5


def test_gamma_per_aspect_missing_status_defaults_to_one() -> None:
    from mothrag.core.spectral import gamma_per_aspect
    out = gamma_per_aspect("Paris is the capital of France.")
    for score in out.values():
        assert score == 1.0


def test_gamma_per_aspect_numeric_override_clamped() -> None:
    from mothrag.core.spectral import gamma_per_aspect
    out = gamma_per_aspect(
        "Paris is the capital of France.",
        gamma_status=2.5,  # outside [0, 1]
    )
    for score in out.values():
        assert score == 1.0  # clamped


def test_gamma_per_aspect_per_aspect_overrides_respected() -> None:
    from mothrag.core.spectral import gamma_per_aspect, extract_aspects
    text = "Paris is the capital of France."
    aspects = extract_aspects(text)
    paris = next(a for a in aspects if "Paris" in a)
    out = gamma_per_aspect(
        text, gamma_status="valid",
        per_aspect_overrides={paris: 0.1},
    )
    assert out[paris] == 0.1


# ============================================================
# l4b_per_aspect
# ============================================================

def test_l4b_per_aspect_cancelled_true_broadcasts_zero() -> None:
    from mothrag.core.spectral import l4b_per_aspect
    out = l4b_per_aspect(
        "Paris is the capital of France.",
        l4b_cancelled=True,
    )
    for score in out.values():
        assert score == 0.0


def test_l4b_per_aspect_explicit_score_overrides_cancelled_flag() -> None:
    from mothrag.core.spectral import l4b_per_aspect
    out = l4b_per_aspect(
        "Paris is the capital of France.",
        l4b_cancelled=True,
        l4b_score=0.7,
    )
    for score in out.values():
        assert score == pytest.approx(0.7, rel=1e-6)


# ============================================================
# agreement_per_aspect
# ============================================================

def test_agreement_per_aspect_no_embedder_returns_zeros() -> None:
    from mothrag.core.spectral import agreement_per_aspect
    out = agreement_per_aspect(
        "Paris is the capital of France.",
        arm_answers={"v3bu": "Paris.", "iter": "Paris."},
    )
    for score in out.values():
        assert score == 0.0


def test_agreement_per_aspect_single_arm_returns_zeros() -> None:
    from mothrag.core.spectral import agreement_per_aspect
    from mothrag.core.api import _HashEmbedder
    out = agreement_per_aspect(
        "Paris is the capital of France.",
        arm_answers={"v3bu": "Paris is the capital of France."},
        embedder=_HashEmbedder(),
    )
    for score in out.values():
        assert score == 0.0


def test_agreement_per_aspect_identical_other_arm_drives_high_scores() -> None:
    from mothrag.core.spectral import agreement_per_aspect
    from mothrag.core.api import _HashEmbedder
    answer = "Paris is the capital of France."
    out = agreement_per_aspect(
        answer,
        arm_answers={
            "v3bu": answer,         # identical answer
            "decompose": "Paris is the capital of France.",  # identical text
            "iter": "Tokyo is in Japan.",  # disjoint
        },
        embedder=_HashEmbedder(),
        threshold=0.6,
    )
    # At least one aspect should agree with v3bu / decompose (identical
    # vectors under HashEmbedder) -> score > 0.
    assert max(out.values()) > 0.0
