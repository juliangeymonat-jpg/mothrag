# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Path-serialize + bridge-judge provider switch — unit contracts.

Both features are flag-gated and MUST be byte-identical to the prior release
when OFF. These
tests pin (a) the path-serialize reordering + default-OFF identity and (b) the
bridge-judge provider switch routing, with NO network calls.
"""
from __future__ import annotations

from mothrag.eval.iterative_pipeline import (
    _intermediate_user_msg, _passage_text, _path_serialize_order,
)
from mothrag.retrieval.bridge_haiku.bridge_arm import BridgeArm
from mothrag.retrieval.bridge_haiku.tripartite_judge import TripartiteJudge
from mothrag.retrieval.bridge_haiku.types import BridgeConfig


# ----------------------- path-serialize ordering ------------------------ #
def test_reorder_follows_spine_hop_order():
    spine = ["Marie Curie", "Pierre Curie", "Nobel Prize"]
    passages = [
        "An off-topic passage about geography.",
        "The Nobel Prize in Physics 1903 was shared.",
        "Pierre Curie was a French physicist.",
        "Marie Curie discovered polonium.",
    ]
    out = _path_serialize_order(passages, spine)
    assert out[0].startswith("Marie Curie")        # hop-1 first
    assert out[1].startswith("Pierre Curie")        # hop-2
    assert out[2].startswith("The Nobel Prize")     # hop-3
    assert out[3].startswith("An off-topic")        # no-match sinks last
    assert sorted(out) == sorted(passages)          # no drop / dup


def test_reorder_is_stable_on_ties():
    spine = ["x"]
    passages = ["x first", "x second", "no match", "x third"]
    out = _path_serialize_order(passages, spine)
    assert out == ["x first", "x second", "x third", "no match"]


def test_reorder_graceful_no_op():
    p = ["a", "b"]
    assert _path_serialize_order(p, []) == p          # no spine
    assert _path_serialize_order(p, [None, ""]) == p  # empty spine
    assert _path_serialize_order(["only"], ["a"]) == ["only"]  # <2 passages


def test_reorder_handles_dict_passages():
    spine = ["Marie", "Pierre"]
    dpass = [{"text": "random"}, {"text": "Pierre taught"}, {"text": "Marie won"}]
    out = _path_serialize_order(dpass, spine)
    assert [d["text"] for d in out] == ["Marie won", "Pierre taught", "random"]


def test_passage_text_extraction():
    assert _passage_text("plain") == "plain"
    assert _passage_text({"text": "t"}) == "t"
    assert _passage_text({"content": "c"}) == "c"


def test_intermediate_msg_off_is_byte_identical_to_legacy():
    passages = ["p one", "p two"]
    off = _intermediate_user_msg("Q", passages, 1, 4, ["fact a"])
    ctx = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    legacy = ("Original question: Q\nIteration: 1/4\n"
              "Facts extracted in previous iterations:\n- fact a\n\n"
              f"Current retrieved passages:\n{ctx}\n\n"
              "Now perform EXTRACT, INTEGRATE, ASSESS, OUTPUT.")
    assert off == legacy


def test_intermediate_msg_on_reorders_and_relabels():
    spine = ["Marie"]
    passages = ["off topic", "Marie won"]
    on = _intermediate_user_msg("Q", passages, 1, 4, [],
                                path_serialize=True, spine_entities=spine)
    assert "[1] Marie won" in on
    assert "ordered along the reasoning path" in on


def test_intermediate_msg_on_without_spine_falls_back():
    passages = ["a", "b"]
    on = _intermediate_user_msg("Q", passages, 1, 4, [],
                                path_serialize=True, spine_entities=[])
    # no spine → behaves like OFF (legacy label, original order)
    assert "Current retrieved passages:" in on
    assert "[1] a" in on


# ----------------------- bridge-judge provider --------------------------- #
def test_bridge_judge_provider_defaults_anthropic():
    j = TripartiteJudge(require_backend=False)
    assert j._provider == "anthropic"
    assert BridgeConfig().judge_provider == "anthropic"


def test_bridge_judge_provider_gemini_routes():
    j = TripartiteJudge(provider="gemini", require_backend=False)
    assert j._provider == "gemini"


def test_bridge_arm_threads_provider_to_all_three_stages():
    cfg = BridgeConfig(judge_provider="gemini")
    arm = BridgeArm(lambda q, k: [], config=cfg, require_backend=False)
    assert arm.svo._provider == "gemini"
    assert arm.entities._provider == "gemini"
    assert arm.judge._provider == "gemini"


def test_bridge_arm_default_provider_anthropic():
    arm = BridgeArm(lambda q, k: [], config=BridgeConfig(), require_backend=False)
    assert arm.judge._provider == "anthropic"
