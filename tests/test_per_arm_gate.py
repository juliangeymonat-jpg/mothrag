# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Per-arm bridge qtype gate.

Per-arm gate assignment (BUILD-time, anti-leak: arm-name + qtype only, NEVER a
dataset/corpus signal). iter_dup_a copies iter (dup); in pool mode v3bu and
decompose share the pre-pool retrieval — both coherence constraints are enforced
by the parser.
"""
from __future__ import annotations

import importlib.util as _u
import inspect
import json
import os
import pathlib
import sys
import threading

import pytest

_RP_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "route_prospective.py"


def _load_rp():
    spec = _u.spec_from_file_location("route_prospective_mod", _RP_PATH)
    mod = _u.module_from_spec(spec)
    _saved = sys.argv
    sys.argv = ["route_prospective"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = _saved
    return mod


rp = _load_rp()
_SRC = _RP_PATH.read_text(encoding="utf-8")

_HAPPY = ("v3bu:none,decompose:none,"
          "iter:exclude_bridge_entity,iter_dup_a:exclude_bridge_entity")


# --------------------------------------------------------------------------- #
# parser — happy + malformed + coherence
# --------------------------------------------------------------------------- #
def test_parser_happy_path():
    assert rp._parse_arm_bridge_gates(_HAPPY) == {
        "v3bu": "none", "decompose": "none",
        "iter": "exclude_bridge_entity", "iter_dup_a": "exclude_bridge_entity"}


@pytest.mark.parametrize("spec,needle", [
    ("decompose:none,iter:none,iter_dup_a:none", "missing arm"),           # missing v3bu
    ("v3bu:none,decompose:none,iter:exclbe,iter_dup_a:exclbe", "unknown gate"),  # alias rejected
    ("foo:none,decompose:none,iter:none,iter_dup_a:none", "unknown arm"),  # bad arm
    ("v3bu:none,v3bu:none,iter:none,iter_dup_a:none", "duplicate"),        # duplicate
    ("v3bu", "must be 'arm:gate'"),                                        # no colon
])
def test_parser_rejects_malformed(spec, needle):
    with pytest.raises(ValueError) as e:
        rp._parse_arm_bridge_gates(spec)
    assert needle in str(e.value)


def test_parser_enforces_iter_dup_a_equals_iter():
    with pytest.raises(ValueError) as e:
        rp._parse_arm_bridge_gates(
            "v3bu:none,decompose:none,iter:none,iter_dup_a:exclude_bridge_entity")
    assert "iter_dup_a is a duplicate of iter" in str(e.value)


def test_parser_enforces_v3bu_equals_decompose():
    with pytest.raises(ValueError) as e:
        rp._parse_arm_bridge_gates(
            "v3bu:none,decompose:exclude_bridge_entity,iter:none,iter_dup_a:none")
    assert "v3bu and decompose share" in str(e.value)


def test_no_exclbe_alias_strict_verbatim():
    # 'exclbe' must NOT be accepted — only the verbatim value.
    with pytest.raises(ValueError):
        rp._parse_arm_bridge_gates(
            "v3bu:none,decompose:none,iter:exclbe,iter_dup_a:exclbe")
    assert "exclbe" not in rp._ARM_GATE_VALUES


# --------------------------------------------------------------------------- #
# gate decision matrix (3 gate values x qtypes)
# --------------------------------------------------------------------------- #
def test_gate_allows_for_matrix():
    G = rp._BridgeSubstrate._gate_allows_for
    assert all(G("none", qt) for qt in ("semantic_rich", "bridge_entity", "chain_deep", None))
    assert G("semantic_rich_only", "semantic_rich") is True
    assert G("semantic_rich_only", "bridge_entity") is False
    assert G("semantic_rich_only", "chain_deep") is False
    assert G("exclude_bridge_entity", "bridge_entity") is False
    assert G("exclude_bridge_entity", "semantic_rich") is True
    assert G("exclude_bridge_entity", "chain_deep") is True
    # classifier failure (None qtype) fails OPEN to the bridge for every gate
    assert all(G(g, None) for g in rp._ARM_GATE_VALUES)


# --------------------------------------------------------------------------- #
# substrate per-arm behaviour (the 4-arm x gate integration)
# --------------------------------------------------------------------------- #
def _substrate(per_arm_gates, qtype, monkeypatch):
    monkeypatch.setattr(rp, "classify_query_v2", lambda q: qtype)
    stub = rp._StubDensePipeline()
    return rp._BridgeSubstrate(
        stub, judge_model="claude-haiku-4-5", max_cost_usd=10.0,
        require_backend=False, scope="all", per_arm_gates=per_arm_gates), stub


def test_per_arm_serves_bridge_for_allowed_dense_for_gated(monkeypatch):
    gates = rp._parse_arm_bridge_gates(_HAPPY)
    sub, stub = _substrate(gates, "bridge_entity", monkeypatch)
    prep = sub.prepare("alpha beta")
    assert prep["fired"] is True                      # warmed (v3bu/decompose allow)
    # v3bu / decompose (gate none) -> bridge; iter / iter_dup_a (exclbe) -> dense
    routes = {}
    for arm in ("v3bu", "decompose", "iter", "iter_dup_a"):
        sub.set_current_arm(arm)
        _, routes[arm], _ = stub.retrieve("alpha beta")
    assert routes["v3bu"] == "bridge_substrate"
    assert routes["decompose"] == "bridge_substrate"
    assert routes["iter"] == "dense"
    assert routes["iter_dup_a"] == "dense"


def test_per_arm_decisions_recorded_for_telemetry(monkeypatch):
    gates = rp._parse_arm_bridge_gates(_HAPPY)
    sub, _ = _substrate(gates, "bridge_entity", monkeypatch)
    prep = sub.prepare("alpha beta")
    d = prep["bridge_arm_decisions"]
    assert d["v3bu"]["allow"] is True and d["v3bu"]["gate"] == "none"
    assert d["iter"]["allow"] is False and d["iter"]["gate"] == "exclude_bridge_entity"
    assert all(v["qtype"] == "bridge_entity" for v in d.values())
    assert sub.stats()["per_arm_gates"] == gates


def test_all_arms_gated_falls_fully_dense(monkeypatch):
    # every arm = semantic_rich_only, query is bridge_entity → no arm bridges.
    gates = rp._parse_arm_bridge_gates(
        "v3bu:semantic_rich_only,decompose:semantic_rich_only,"
        "iter:semantic_rich_only,iter_dup_a:semantic_rich_only")
    sub, stub = _substrate(gates, "bridge_entity", monkeypatch)
    prep = sub.prepare("alpha beta")
    assert prep["fired"] is False
    sub.set_current_arm("v3bu")
    _, route, _ = stub.retrieve("alpha beta")
    assert route == "dense"


def test_current_arm_is_thread_local_no_race(monkeypatch):
    gates = rp._parse_arm_bridge_gates(_HAPPY)
    sub, stub = _substrate(gates, "bridge_entity", monkeypatch)
    sub.prepare("alpha beta")
    out: dict = {}

    def worker(arm):
        sub.set_current_arm(arm)
        import time
        time.sleep(0.02)                              # force interleave
        _, route, _ = stub.retrieve("alpha beta")
        out[arm] = route

    ts = [threading.Thread(target=worker, args=(a,)) for a in ("v3bu", "iter")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert out["v3bu"] == "bridge_substrate"          # each thread saw its own arm
    assert out["iter"] == "dense"


def test_uniform_mode_unaffected_default_off(monkeypatch):
    # per_arm_gates=None → the existing uniform path; current_arm ignored.
    monkeypatch.setattr(rp, "classify_query_v2", lambda q: "bridge_entity")
    stub = rp._StubDensePipeline()
    sub = rp._BridgeSubstrate(stub, judge_model="claude-haiku-4-5",
                              max_cost_usd=10.0, require_backend=False,
                              scope="all", qtype_gate="none")
    assert sub.stats()["per_arm_gates"] is None
    sub.prepare("alpha beta")
    sub.set_current_arm("iter")                        # ignored in uniform mode
    _, route, _ = stub.retrieve("alpha beta")
    assert route == "bridge_substrate"                # gate=none → always bridge


# --------------------------------------------------------------------------- #
# pool-safety + anti-leak + CLI wiring (source-level guards)
# --------------------------------------------------------------------------- #
def test_pool_safety_no_fifth_arm():
    # the per-arm gate is on the SUBSTRATE; it must not add an arm to the pool.
    assert "_ARM_GATE_ARMS" in _SRC
    # the canonical pool stays the 4 PDD arms (used by the dry-run arm check).
    assert 'sorted(("v3bu", "decompose", "iter", "iter_dup_a"))' in _SRC


def test_anti_leak_gate_uses_qtype_not_dataset():
    # the per-arm gate selector path must classify on the QUESTION (input-feature),
    # never read a dataset / corpus VARIABLE. We check for leak-y code accesses
    # (not the bare word "dataset", which appears in the anti-leak docstrings).
    start = _SRC.index("def _prepare_per_arm")
    end = _SRC.index("def _wrapped_retrieve")
    gate_code = _SRC[start:end]
    assert "classify_query_v2" in gate_code            # input-feature only
    for leak in ("args.dataset", "dataset_name", "corpus_name", "ds_name",
                 ".dataset", "gold"):
        assert leak not in gate_code, f"anti-leak: {leak!r} reached the gate path"
    # the parser is keyed by arm name + gate value only — no dataset access.
    p_start = _SRC.index("def _parse_arm_bridge_gates")
    p_end = _SRC.index("class _BridgeSubstrate")
    p_code = _SRC[p_start:p_end]
    for leak in ("args.dataset", "dataset_name", "corpus_name", ".dataset"):
        assert leak not in p_code, f"anti-leak: {leak!r} in the parser"


def test_cli_flag_registered_and_mutex_enforced():
    assert '"--arm-bridge-qtype-gate"' in _SRC
    # mutually exclusive with the uniform gate (hard fail, no fire).
    assert "mutually exclusive with" in _SRC
    assert "args.bridge_substrate_qtype_gate != \"none\"" in _SRC


def test_telemetry_config_and_per_q_wired():
    assert '"arm_bridge_qtype_gate": (per_arm_gates if per_arm_gates else None)' in _SRC
    assert '"bridge_arm_decisions": bridge_prep.get("bridge_arm_decisions", {})' in _SRC


def test_pip_config_key_accepted_noop():
    from mothrag.core.api import _DEFAULTS
    assert _DEFAULTS["arm_bridge_qtype_gate"] is None


# --------------------------------------------------------------------------- #
# HOTFIX: _write_partial per_arm_gates scoping (regression for the q20 crash)
# --------------------------------------------------------------------------- #
class _AnyArgs:
    """args stub: any attribute not explicitly set reads back as None.

    `_write_partial` serialises ~50 `args.*` config fields; the test only cares
    about the per-arm gate map (passed as an explicit kwarg, NOT via args), so
    every other field is allowed to default to None and serialise cleanly.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, name):  # only hit when normal lookup fails
        return None


def _one_per_q(qtype="bridge_entity", with_decisions=True):
    r = {
        "qid": "q0", "em": 1.0, "f1": 0.5, "r_at_10": 0.5,
        "iterations_used": 1, "arm_used": "iter", "qtype": qtype,
        "n_llm_calls": 3,
    }
    if with_decisions:
        r["bridge_arm_decisions"] = {
            "v3bu": {"allow": True, "gate": "none", "qtype": qtype},
            "decompose": {"allow": True, "gate": "none", "qtype": qtype},
            "iter": {"allow": False, "gate": "exclude_bridge_entity",
                     "qtype": qtype},
            "iter_dup_a": {"allow": False, "gate": "exclude_bridge_entity",
                           "qtype": qtype},
        }
    return r


def test_write_partial_with_per_arm_gates_active(tmp_path):
    # Reproduces the shipped q20 crash: _write_partial referenced per_arm_gates
    # without it being in scope → NameError, empty JSON on disk. Now it is an
    # explicit kwarg; the parsed map must round-trip into summary.config.
    gates = rp._parse_arm_bridge_gates(_HAPPY)
    out = tmp_path / "partial.json"
    per_q = [_one_per_q()]
    # partial=True mirrors the q20 checkpoint write that was crashing.
    rp._write_partial(out, per_q, _AnyArgs(), partial=True, per_arm_gates=gates)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["config"]["arm_bridge_qtype_gate"] == gates
    # per-question bridge_arm_decisions survive the checkpoint write.
    assert data["per_question"][0]["bridge_arm_decisions"]["iter"]["gate"] \
        == "exclude_bridge_entity"
    assert data["summary"]["config"]["arm_bridge_qtype_gate"]["v3bu"] == "none"


def test_write_partial_per_arm_gate_off_persists_none(tmp_path):
    # Negative / default path: gate OFF → no kwarg → config persists None (the
    # inactive sentinel; see test_telemetry_config_and_per_q_wired which pins
    # the source to `... if per_arm_gates else None`). NEVER a populated dict.
    out = tmp_path / "partial_off.json"
    rp._write_partial(out, [_one_per_q(with_decisions=False)], _AnyArgs(),
                      partial=False)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["config"]["arm_bridge_qtype_gate"] is None


# --------------------------------------------------------------------------- #
# ChainFilter OpenIE off Together -> Groq.
# Same Llama-3.3-70B already in the stack for the main reader; no separate
# TOGETHER_API_KEY. api.groq.com is on the OpenIEClient allowlist, so the
# migration needs no allow_custom_endpoint.
# --------------------------------------------------------------------------- #
def _chainfilter_args():
    # explicit ChainFilter attrs: _build_chain_filter does int(getattr(...)) so
    # the values must be real ints (_AnyArgs would hand back None -> TypeError).
    return _AnyArgs(use_chainfilter=True, chainfilter_hop_min=2,
                    chainfilter_top_out=5, chainfilter_gamma_min=None)


def test_openie_uses_groq_endpoint(monkeypatch):
    # The lazily-built OpenIE client must be constructed against the Groq
    # endpoint + model — never the Together default.
    import mothrag.retrieval.openie as _oie_mod
    captured: dict = {}

    class _FakeResult:
        triples = [["Paris", "capital of", "France"]]

    class _FakeOIE:
        def __init__(self, **kw):
            captured.update(kw)

        def process(self, text, chunk_id="anon"):
            return _FakeResult()

    # `from mothrag.retrieval.openie import OpenIEClient` resolves the attribute
    # at call time, so patching the module attribute reaches the local import.
    monkeypatch.setattr(_oie_mod, "OpenIEClient", _FakeOIE)
    cf = rp._build_chain_filter(_chainfilter_args())
    facts = cf.triple_extractor("Paris is the capital of France.")
    assert captured["base_url"] == "https://api.groq.com/openai/v1"
    assert captured["model"] == "llama-3.3-70b-versatile"
    assert "api_key" in captured                       # key IS passed (from GROQ)
    assert facts and facts[0] == ["Paris", "capital of", "France"]


def test_openie_no_together_dependency():
    # Source-level guarantee: the ChainFilter builder *accesses* GROQ_API_KEY and
    # never *accesses* TOGETHER_API_KEY. We check the env-access patterns (not the
    # bare token, which legitimately appears in the migration comment).
    src = inspect.getsource(rp._build_chain_filter)
    for access in ('os.environ.get("TOGETHER_API_KEY")',
                   "os.environ.get('TOGETHER_API_KEY')",
                   'os.environ["TOGETHER_API_KEY"]',
                   "os.getenv(\"TOGETHER_API_KEY\")",
                   "os.getenv('TOGETHER_API_KEY')"):
        assert access not in src, f"residual Together access: {access}"
    assert 'os.environ.get("GROQ_API_KEY")' in src
    assert "https://api.groq.com/openai/v1" in src
    assert "llama-3.3-70b-versatile" in src


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"),
                    reason="GROQ_API_KEY not set — live Groq sanity skipped")
def test_chainfilter_triple_extractor_sanity():
    # Live $0.01 sanity: a one-line fact must yield >=1 triple from Groq Llama.
    from mothrag.retrieval.openie import OpenIEClient
    oie = OpenIEClient(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
    )
    res = oie.process("Paris is the capital of France.")
    assert len(res.triples) > 0
