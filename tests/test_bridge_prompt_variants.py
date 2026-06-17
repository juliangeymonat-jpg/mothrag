# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Bridge SVO/entity/judge Haiku prompt v2 (opt-in).

The v2 prompts are tightened + few-shot variants of the validated v1
prompts. v1 stays the DEFAULT (no regression to the validated config); v2 keeps
the SAME output contract (parse functions unchanged) so the only thing to
verify offline is: the variant is selectable + wired, both variants parse
robustly, and the default is v1. The quality comparison between variants is
left to a separate evaluation.
"""
from __future__ import annotations

import pytest

from mothrag.retrieval.bridge_haiku import (
    BridgeArm,
    BridgeConfig,
    Candidate,
)
from mothrag.retrieval.bridge_haiku import entity_extractor as ent_mod
from mothrag.retrieval.bridge_haiku import svo_generator as svo_mod
from mothrag.retrieval.bridge_haiku import tripartite_judge as judge_mod
from mothrag.retrieval.bridge_haiku.entity_extractor import (
    DualEntityExtractor,
    parse_entity_response,
)
from mothrag.retrieval.bridge_haiku.svo_generator import (
    SVOQueryGenerator,
    parse_svo_response,
)
from mothrag.retrieval.bridge_haiku.tripartite_judge import (
    TripartiteJudge,
    parse_judge_scores,
)


# ---- default is the validated v1 -------------------------------------------

def test_default_prompt_variant_is_v1():
    assert BridgeConfig().prompt_variant == "v1"
    assert SVOQueryGenerator(require_backend=False).prompt_variant == "v1"
    assert DualEntityExtractor(require_backend=False).prompt_variant == "v1"
    assert TripartiteJudge(require_backend=False).prompt_variant == "v1"


# ---- _select_prompt picks the right variant (and falls back to v1) ---------

@pytest.mark.parametrize("mod", [svo_mod, ent_mod, judge_mod])
def test_select_prompt_variants(mod):
    assert mod._select_prompt("v1") == mod._PROMPTS["v1"]
    assert mod._select_prompt("v2") == mod._PROMPTS["v2"]
    assert mod._select_prompt("bogus") == mod._PROMPTS["v1"]   # safe fallback
    assert mod._PROMPTS["v1"] != mod._PROMPTS["v2"]            # genuinely different


# ---- v2 templates render (no KeyError on the format fields) ----------------

def test_v2_templates_render():
    s_sys, s_usr = svo_mod._select_prompt("v2")
    assert s_usr.format(question="q", bridge="b", n=3)
    e_sys, e_usr = ent_mod._select_prompt("v2")
    assert e_usr.format(question="q", bridge="b")
    j_sys, j_usr = judge_mod._select_prompt("v2")
    assert j_usr.format(question="q", bridge="b", e1="x", e2="y", candidates="[1] c", n=1)


# ---- the stages actually USE the selected variant's system prompt ----------

def test_svo_stage_uses_selected_system():
    gen = SVOQueryGenerator(prompt_variant="v2", require_backend=False)
    seen = {}

    def _cap(system, user, **k):
        seen["system"] = system
        return '["alpha beta", "gamma delta"]', 1, 1

    gen._call = _cap
    gen.generate("question", "bridge", n=2)
    assert seen["system"] == svo_mod._SYSTEM_V2


def test_judge_stage_uses_selected_system():
    j = TripartiteJudge(prompt_variant="v2", require_backend=False)
    seen = {}

    def _cap(system, user, **k):
        seen["system"] = system
        return "[9, 0]", 1, 1

    j._call = _cap
    j.score("q", "b", "e1", "e2", ["c1", "c2"])
    assert seen["system"] == judge_mod._SYSTEM_V2


# ---- both variants keep the SAME output contract (parse unchanged) ---------

def test_parse_contract_stable_across_variants():
    # canned outputs an LLM might return under either prompt — both parse.
    assert parse_svo_response('["a phrase", "b phrase"]', max_n=2) == ["a phrase", "b phrase"]
    assert parse_svo_response('```json\n["x"]\n```', max_n=2) == ["x"]   # fenced
    assert parse_entity_response('{"e1": "Shin Sang-ok", "e2": "country"}') == \
        ("Shin Sang-ok", "country")
    assert parse_judge_scores("[9, 4, 0]", n=3) == [9.0, 4.0, 0.0]
    assert parse_judge_scores("[9, 4]", n=3) == [9.0, 4.0, 5.0]          # padded


# ---- BridgeArm wires the variant from BridgeConfig to all 3 stages ---------

def test_bridge_arm_wires_prompt_variant():
    def ann(q, k):
        return [Candidate(f"p{i}", f"t{i}", 1.0) for i in range(k)]

    arm = BridgeArm(ann, config=BridgeConfig(prompt_variant="v2"),
                    require_backend=False)
    assert arm.svo.prompt_variant == "v2"
    assert arm.entities.prompt_variant == "v2"
    assert arm.judge.prompt_variant == "v2"

    arm_v1 = BridgeArm(ann, config=BridgeConfig(), require_backend=False)
    assert arm_v1.svo.prompt_variant == "v1"   # default unchanged


# ---- determinism: same input → same parsed output (T=0 pipeline) -----------

def test_stage_pipeline_is_deterministic():
    gen = SVOQueryGenerator(prompt_variant="v2", require_backend=False)
    gen._call = lambda system, user, **k: ('["alpha", "beta"]', 1, 1)
    out1, *_ = gen.generate("q", "b", n=2)
    out2, *_ = gen.generate("q", "b", n=2)
    assert out1 == out2 == ["alpha", "beta"]
