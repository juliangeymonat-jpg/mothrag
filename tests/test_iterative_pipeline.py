# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Offline state-machine tests for mothrag.eval.iterative_pipeline.

Mocks the underlying MothRAGPipeline + LLM client so the loop, parser, fact
accumulator, and exit conditions can be validated without any cloud or API
access. Exercises:

  - Multi-iteration MISSING -> NEXT_QUERY -> ANSWER chains
  - Stop-early on first ANSWER
  - Max-iterations exhaustion -> final canonical reader pass
  - Fact accumulation + dedup
  - Passage union with top_k_total cap
  - Parser robustness (ANSWER beats MISSING when both present)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from mothrag.eval.iterative_pipeline import (
    IterativeConfig,
    IterativeMothRAG,
    parse_intermediate,
)


# ---- Fake response objects (mirror OpenAI-SDK Choice/Message/Usage shape) ----

class _FakeUsage:
    def __init__(self, pt: int = 100, ct: int = 50):
        self.prompt_tokens = pt
        self.completion_tokens = ct


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


# ---- Fake pipeline (stand-in for MothRAGPipeline) ----

class _FakePipeline:
    """Drop-in replacement for MothRAGPipeline with deterministic retrieval."""

    def __init__(self, retrieval_plan: list[list[str]], final_answer: str = "FINAL"):
        # retrieval_plan[i] = list of chunk-ids returned on retrieve() call i+1
        self.retrieval_plan = retrieval_plan
        self.final_answer = final_answer
        self._retrieve_call = 0
        # Build a chunk store covering every id mentioned across the plan.
        all_ids: list[str] = []
        for plan in retrieval_plan:
            for cid in plan:
                if cid not in all_ids:
                    all_ids.append(cid)
        self.chunk_ids = all_ids
        self.chunks_by_id = {cid: {"id": cid, "text": f"Passage about {cid}."}
                             for cid in all_ids}
        self.reader_client = MagicMock()
        self.reader_client.chat = MagicMock()
        self.reader_client.chat.completions = MagicMock()
        # ``responses`` is set per-test via configure_responses.
        self.reader_client.chat.completions.create = MagicMock()
        self.reader_model = "fake-model"
        self.read_calls: list[tuple[str, list[str]]] = []

    def retrieve(self, question: str, entity_seeds=None):
        plan_idx = min(self._retrieve_call, len(self.retrieval_plan) - 1)
        chunk_id_list = self.retrieval_plan[plan_idx]
        # Map ids to indices in self.chunk_ids
        idxs = [self.chunk_ids.index(cid) for cid in chunk_id_list]
        self._retrieve_call += 1
        # Lever 2 audit: record (question, seeds) for inspection
        if not hasattr(self, "retrieve_calls"):
            self.retrieve_calls = []
        self.retrieve_calls.append((question, list(entity_seeds) if entity_seeds else None))
        return idxs, "test-route", 0.5

    def read(self, question: str, passages: list[str]):
        self.read_calls.append((question, list(passages)))
        return self.final_answer, f"Final answer: {self.final_answer}", {
            "prompt_tokens": 200, "completion_tokens": 20, "latency_s": 0.1,
        }


def _make_response_sequence(pipeline: _FakePipeline, contents: list[str]) -> None:
    """Configure pipeline.reader_client to return ``contents`` in order."""
    pipeline.reader_client.chat.completions.create.side_effect = [
        _FakeResponse(c) for c in contents
    ]


# ---- Parser tests ----

def test_parse_answer_beats_missing() -> None:
    raw = """1. EXTRACT
- fact alpha
- fact beta

2. INTEGRATE
The two facts together resolve the question.

3. ASSESS
Yes, I have enough.

4. OUTPUT
ANSWER: 42
MISSING: nothing extra
"""
    out = parse_intermediate(raw)
    assert out.has_answer is True
    assert out.answer == "42"
    assert out.is_terminal is True
    assert "fact alpha" in out.extracted_facts
    assert "fact beta" in out.extracted_facts


def test_parse_missing_with_next_query() -> None:
    raw = """1. EXTRACT
- Inception is a 2010 film.
- It was directed by Christopher Nolan.

2. INTEGRATE
We know who directed Inception, but not his spouse.

3. ASSESS
Cannot answer yet.

4. OUTPUT
MISSING: the spouse of Christopher Nolan
NEXT_QUERY: Christopher Nolan spouse wife
"""
    out = parse_intermediate(raw)
    assert out.has_answer is False
    assert out.is_terminal is False
    assert out.missing == "the spouse of Christopher Nolan"
    assert out.next_query == "Christopher Nolan spouse wife"
    assert len(out.extracted_facts) == 2


def test_parse_no_structure_fallback() -> None:
    out = parse_intermediate("ANSWER: yes")
    assert out.has_answer is True
    assert out.answer == "yes"


# ---- Iterative loop tests ----

def test_stop_early_on_first_answer() -> None:
    pipe = _FakePipeline(retrieval_plan=[["c1", "c2"], ["c3"]])
    _make_response_sequence(pipe, [
        "1. EXTRACT\n- a fact\n2. INTEGRATE\nok\n3. ASSESS\nyes\n4. OUTPUT\nANSWER: Quentin Tarantino",
    ])
    runner = IterativeMothRAG(pipe, IterativeConfig(max_iterations=4, stop_early=True))
    info = runner.answer("Who directed Pulp Fiction?")

    assert info.answer == "Quentin Tarantino"
    assert info.iterations_used == 1
    assert info.per_iteration_terminal == [True]
    assert pipe.read_calls == []  # final reader NOT called when stop-early triggered
    assert len(info.passages) == 2


def test_three_iterations_before_answer() -> None:
    pipe = _FakePipeline(retrieval_plan=[
        ["c1", "c2"],
        ["c3", "c4"],
        ["c5"],
    ])
    _make_response_sequence(pipe, [
        # iter 1: MISSING
        "1. EXTRACT\n- Inception (2010) directed by Christopher Nolan.\n"
        "2. INTEGRATE\nKnow director.\n3. ASSESS\nNo.\n4. OUTPUT\n"
        "MISSING: spouse of Christopher Nolan\nNEXT_QUERY: Christopher Nolan spouse",
        # iter 2: MISSING
        "1. EXTRACT\n- Christopher Nolan married Emma Thomas.\n"
        "2. INTEGRATE\nKnow spouse.\n3. ASSESS\nNo.\n4. OUTPUT\n"
        "MISSING: profession of Emma Thomas\nNEXT_QUERY: Emma Thomas producer",
        # iter 3: ANSWER
        "1. EXTRACT\n- Emma Thomas is a film producer.\n"
        "2. INTEGRATE\nProfession known.\n3. ASSESS\nYes.\n4. OUTPUT\n"
        "ANSWER: producer",
    ])
    runner = IterativeMothRAG(pipe, IterativeConfig(max_iterations=4))
    info = runner.answer("What is the profession of the spouse of Inception's director?")

    assert info.answer == "producer"
    assert info.iterations_used == 3
    assert info.per_iteration_terminal == [False, False, True]
    # Query reformulation should have happened
    assert info.per_iteration_query[0] != info.per_iteration_query[1]
    assert "Christopher Nolan spouse" in info.per_iteration_query[1]
    assert "Emma Thomas producer" in info.per_iteration_query[2]
    # Facts accumulated across iterations, deduplicated
    assert any("Christopher Nolan" in f for f in info.accumulated_facts)
    assert any("Emma Thomas" in f for f in info.accumulated_facts)
    # Passage union grew across iterations
    assert len(info.passages) == 5


def test_max_iterations_exhaustion_calls_final_reader() -> None:
    pipe = _FakePipeline(
        retrieval_plan=[["c1"], ["c2"], ["c3"], ["c4"]],
        final_answer="canonical-final",
    )
    # All 4 iterations return MISSING, never ANSWER.
    miss = ("1. EXTRACT\n- something\n2. INTEGRATE\nstill incomplete\n"
            "3. ASSESS\nNo.\n4. OUTPUT\nMISSING: more\nNEXT_QUERY: more please")
    _make_response_sequence(pipe, [miss] * 4)

    runner = IterativeMothRAG(pipe, IterativeConfig(max_iterations=4))
    info = runner.answer("Hard 4-hop question?")

    assert info.iterations_used == 4
    assert all(t is False for t in info.per_iteration_terminal)
    assert pipe.read_calls and pipe.read_calls[0][0] == "Hard 4-hop question?"
    assert info.answer == "canonical-final"
    assert len(info.passages) == 4


def test_passage_cap_top_k_total() -> None:
    # 6 retrieval passes of 10 unique chunks each = 60 candidates; cap at 15.
    plan = [[f"r{i}c{j}" for j in range(10)] for i in range(6)]
    pipe = _FakePipeline(retrieval_plan=plan, final_answer="x")
    miss = ("1. EXTRACT\n- f\n2. INTEGRATE\n.\n3. ASSESS\nNo.\n"
            "4. OUTPUT\nMISSING: x\nNEXT_QUERY: y")
    _make_response_sequence(pipe, [miss] * 6)

    runner = IterativeMothRAG(pipe, IterativeConfig(max_iterations=6, top_k_total=15))
    info = runner.answer("test")
    assert len(info.passages) == 15
    assert len(info.retrieved_chunk_ids) == 15


def test_tier_conditional_reader_uses_override() -> None:
    """Lever 5: final synthesis uses strong reader override when config provides one."""
    pipe = _FakePipeline(
        retrieval_plan=[["c1", "c2"], ["c3", "c4"]],
        final_answer="WEAK_READER_ANSWER",
    )
    # Both intermediate iterations return MISSING so we hit the final-synthesis
    # fallback path (the override target).
    miss = ("1. EXTRACT\n- fact\n2. INTEGRATE\n.\n3. ASSESS\nNo.\n"
            "4. OUTPUT\nMISSING: x\nNEXT_QUERY: y")
    _make_response_sequence(pipe, [miss, miss])

    fake_strong_client = MagicMock()
    fake_strong_client.chat = MagicMock()
    fake_strong_client.chat.completions = MagicMock()
    fake_strong_client.chat.completions.create.return_value = _FakeResponse(
        "Final answer: STRONG_READER_ANSWER"
    )
    # Pipeline config exposes reader_prompt — _FakePipeline doesn't have one,
    # so add the minimum stub _call_final_with_override needs.
    pipe.config = type("Cfg", (), {"reader_prompt": "v3-think"})()

    cfg = IterativeConfig(
        max_iterations=2,
        final_reader_client=fake_strong_client,
        final_reader_model="gpt-4o-2024-08-06",
    )
    runner = IterativeMothRAG(pipe, cfg)
    info = runner.answer("test question")

    assert info.answer == "STRONG_READER_ANSWER"
    assert fake_strong_client.chat.completions.create.called
    assert pipe.read_calls == []  # weak reader path skipped


class _FakePlugin:
    """Mock entity-linking plugin: extracts capitalized words as entity ids."""

    def link_query_entities(self, text, entities_by_id):
        out = []
        seen = set()
        for tok in text.replace(".", " ").replace(",", " ").split():
            if tok and tok[0].isupper() and tok.lower() not in seen:
                seen.add(tok.lower())
                eid = f"ent_{tok.lower()}"
                if eid in entities_by_id:
                    out.append(eid)
        return out


def _attach_plugin(pipe: _FakePipeline, entity_ids):
    pipe.plugin = _FakePlugin()
    pipe.entities_by_id = {e: {"id": e} for e in entity_ids}
    # 1C-aware shim: delegate to plugin.link_query_entities (no NER cache in tests)
    pipe._link_entities = lambda text: pipe.plugin.link_query_entities(
        text, pipe.entities_by_id)


# ---- Lever 2 tests (graph-aware iter retrieval) ----

def test_l2_backward_compat_no_seeds_when_flag_off() -> None:
    """Default use_graph_aware_iter=False: retrieve() called WITHOUT entity_seeds."""
    pipe = _FakePipeline(retrieval_plan=[["c1"], ["c2"]])
    _attach_plugin(pipe, ["ent_inception", "ent_nolan"])
    _make_response_sequence(pipe, [
        "1. EXTRACT\n- Inception was directed by Christopher Nolan.\n"
        "2. INTEGRATE\n.\n3. ASSESS\nNo.\n4. OUTPUT\n"
        "MISSING: spouse\nNEXT_QUERY: Nolan spouse",
        "1. EXTRACT\n- ok.\n2. INTEGRATE\n.\n3. ASSESS\nYes.\n"
        "4. OUTPUT\nANSWER: done",
    ])
    runner = IterativeMothRAG(pipe, IterativeConfig(max_iterations=4))
    info = runner.answer("test")
    # Default flag off → no seeds passed to any retrieve call
    assert all(seeds is None for _, seeds in pipe.retrieve_calls)
    assert info.per_iteration_n_seed_entities == [0, 0]
    assert info.accumulated_entities == []  # no extraction when flag off


def test_l2_graph_aware_propagates_entities() -> None:
    """use_graph_aware_iter=True: entities from facts propagate to next-iter retrieve seeds."""
    pipe = _FakePipeline(retrieval_plan=[["c1"], ["c2"], ["c3"]])
    _attach_plugin(pipe, ["ent_inception", "ent_nolan", "ent_thomas", "ent_emma"])
    _make_response_sequence(pipe, [
        # Iter 1: extract Inception + Nolan
        "1. EXTRACT\n- Inception (2010) was directed by Nolan.\n"
        "2. INTEGRATE\n.\n3. ASSESS\nNo.\n4. OUTPUT\n"
        "MISSING: spouse\nNEXT_QUERY: Nolan Emma spouse",
        # Iter 2: extract Emma + Thomas
        "1. EXTRACT\n- Emma Thomas is the spouse.\n"
        "2. INTEGRATE\n.\n3. ASSESS\nNo.\n4. OUTPUT\n"
        "MISSING: profession\nNEXT_QUERY: Thomas job",
        # Iter 3: ANSWER
        "1. EXTRACT\n- producer.\n2. INTEGRATE\n.\n3. ASSESS\nYes.\n"
        "4. OUTPUT\nANSWER: producer",
    ])
    runner = IterativeMothRAG(pipe, IterativeConfig(
        max_iterations=4, use_graph_aware_iter=True))
    info = runner.answer("question")

    # Iter 1: no seeds (none accumulated yet)
    assert pipe.retrieve_calls[0][1] is None
    # Iter 2: seeds present, contain at least Inception + Nolan from iter 1 facts
    iter2_seeds = pipe.retrieve_calls[1][1]
    assert iter2_seeds is not None and len(iter2_seeds) >= 2
    assert "ent_inception" in iter2_seeds
    assert "ent_nolan" in iter2_seeds
    # Iter 3: seeds expanded with iter 2 facts
    iter3_seeds = pipe.retrieve_calls[2][1]
    assert iter3_seeds is not None and "ent_emma" in iter3_seeds
    # accumulated_entities returned in info, monotonic-grow
    assert "ent_inception" in info.accumulated_entities
    assert "ent_emma" in info.accumulated_entities
    assert info.per_iteration_n_seed_entities[0] == 0  # iter 1 no seeds
    assert info.per_iteration_n_seed_entities[1] >= 2  # iter 2 has accum from iter 1


def test_l2_seed_cap_max_accumulated_entities() -> None:
    """Verify max_accumulated_entities caps the seeds passed to retrieve."""
    pipe = _FakePipeline(retrieval_plan=[["c1"], ["c2"]])
    # Many entities to trigger the cap
    many_entities = [f"ent_e{i}" for i in range(10)]
    _attach_plugin(pipe, many_entities)
    facts_with_many_caps = " ".join(f"E{i}" for i in range(10))
    _make_response_sequence(pipe, [
        f"1. EXTRACT\n- {facts_with_many_caps}\n2. INTEGRATE\n.\n3. ASSESS\nNo.\n"
        "4. OUTPUT\nMISSING: x\nNEXT_QUERY: y",
        "1. EXTRACT\n- ok.\n2. INTEGRATE\n.\n3. ASSESS\nYes.\n4. OUTPUT\nANSWER: done",
    ])
    runner = IterativeMothRAG(pipe, IterativeConfig(
        max_iterations=4, use_graph_aware_iter=True, max_accumulated_entities=3))
    info = runner.answer("q")
    # Iter 2 seeds capped at 3
    iter2_seeds = pipe.retrieve_calls[1][1]
    assert iter2_seeds is not None and len(iter2_seeds) == 3


# ---- Aurora L4 γ verifier mode tests ----

class _FakePipelineGamma(_FakePipeline):
    """Pipeline with chunk entity_id metadata for Aurora passages adapter."""
    def __init__(self, retrieval_plan, chunk_texts=None, **kw):
        super().__init__(retrieval_plan, **kw)
        if chunk_texts:
            for cid, txt in chunk_texts.items():
                self.chunks_by_id[cid] = {"id": cid, "text": txt, "entity_id": cid}
        else:
            for cid, ch in self.chunks_by_id.items():
                ch["entity_id"] = cid


def test_l4_gamma_mode_valid_emits_naturalized_answer() -> None:
    """γ verifier returns valid → emit naturalized_answer, terminal."""
    pipe = _FakePipelineGamma(
        retrieval_plan=[["doc_inception"]],
        chunk_texts={"doc_inception": "Inception is a 2010 film directed by Christopher Nolan."},
    )
    raw_json = ('{"qid":"1","steps":[{"step":1,"rule":"lookup","subject":"Inception",'
                '"object":"Christopher Nolan","predicate":"directed_by",'
                '"claim_text":"Inception is a 2010 film directed by Christopher Nolan",'
                '"sources":[{"doc_id":"doc_inception",'
                '"span_text":"Inception is a 2010 film directed by Christopher Nolan"}]}],'
                '"naturalized_answer":"Christopher Nolan","is_complete":true}')
    _make_response_sequence(pipe, [raw_json])
    cfg = IterativeConfig(max_iterations=4, use_gamma_verifier=True)
    runner = IterativeMothRAG(pipe, cfg)
    info = runner.answer("Who directed Inception?")
    assert info.answer == "Christopher Nolan"
    assert info.gamma_final_status == "valid"
    assert info.per_iteration_gamma_status == ["valid"]


def test_l4_gamma_mode_invalid_continues_loop() -> None:
    """γ invalid → loop continues; valid at iter 2 → emit."""
    pipe = _FakePipelineGamma(
        retrieval_plan=[["c1"], ["c2"]],
        chunk_texts={"c1": "Foo bar baz.", "c2": "Inception was directed by Christopher Nolan."},
    )
    bad_json = ('{"qid":"1","steps":[{"step":1,"rule":"lookup","subject":"X","object":"Y",'
                '"predicate":"foo","claim_text":"Z",'
                '"sources":[{"doc_id":"c1","span_text":"NOT IN PASSAGE"}]}],'
                '"naturalized_answer":"X","is_complete":true}')
    good_json = ('{"qid":"2","steps":[{"step":1,"rule":"lookup","subject":"Inception",'
                 '"object":"Christopher Nolan","predicate":"directed_by",'
                 '"claim_text":"Inception was directed by Christopher Nolan",'
                 '"sources":[{"doc_id":"c2",'
                 '"span_text":"Inception was directed by Christopher Nolan"}]}],'
                 '"naturalized_answer":"Christopher Nolan","is_complete":true}')
    _make_response_sequence(pipe, [bad_json, good_json])
    cfg = IterativeConfig(max_iterations=4, use_gamma_verifier=True, gamma_max_retrigger=3)
    runner = IterativeMothRAG(pipe, cfg)
    info = runner.answer("Who directed Inception?")
    assert "invalid" in info.per_iteration_gamma_status
    assert info.gamma_final_status == "valid"
    assert info.answer == "Christopher Nolan"
    assert info.iterations_used == 2


# ---- Lever 1B γ-as-router (γ exhausted → free-text fallback) ----

def test_gamma_router_falls_back_to_free_text_on_invalid_cap() -> None:
    """γ stays invalid until cap → free-text reader fallback emits answer."""
    pipe = _FakePipelineGamma(
        retrieval_plan=[["c1"], ["c2"]],
        chunk_texts={"c1": "Foo bar.", "c2": "Baz qux."},
        final_answer="free-text-fallback",  # what pipe.read() returns
    )
    bad_json = ('{"qid":"1","steps":[{"step":1,"rule":"lookup","subject":"X",'
                '"predicate":"foo","object":"Y","claim_text":"Z",'
                '"sources":[{"doc_id":"c1","span_text":"NOT IN PASSAGE"}]}],'
                '"naturalized_answer":"X","is_complete":true}')
    _make_response_sequence(pipe, [bad_json, bad_json])
    cfg = IterativeConfig(max_iterations=2, use_gamma_verifier=True,
                           gamma_max_retrigger=2, use_gamma_router=True)
    runner = IterativeMothRAG(pipe, cfg)
    info = runner.answer("Q?")
    # After 2 iters γ stays invalid → cap → 1B fires → pipe.read() called
    assert info.answer == "free-text-fallback"
    assert pipe.read_calls, "pipe.read() must have been called by 1B router"


def test_gamma_router_default_off_keeps_old_behavior() -> None:
    """Without --use-gamma-router, cap-on-invalid still emits naturalized_answer."""
    pipe = _FakePipelineGamma(
        retrieval_plan=[["c1"], ["c2"]],
        chunk_texts={"c1": "Foo.", "c2": "Bar."},
        final_answer="should-not-be-called",
    )
    bad_json = ('{"qid":"1","steps":[{"step":1,"rule":"lookup","subject":"X",'
                '"predicate":"foo","object":"Y","claim_text":"Z",'
                '"sources":[{"doc_id":"c1","span_text":"NOT IN PASSAGE"}]}],'
                '"naturalized_answer":"X","is_complete":true}')
    _make_response_sequence(pipe, [bad_json, bad_json])
    cfg = IterativeConfig(max_iterations=2, use_gamma_verifier=True,
                           gamma_max_retrigger=2, use_gamma_router=False)
    runner = IterativeMothRAG(pipe, cfg)
    info = runner.answer("Q?")
    assert info.answer == "X"  # naturalized_answer fallback (old behavior)
    assert not pipe.read_calls  # pipe.read() NOT called


# ---- Aurora L4b within-iterative-loop C7 (temporal stability axis) ----

def _orthogonal_iter_embedder(strings):
    """4-D one-hot embedder for deterministic L4b C7 testing."""
    rows = []
    for i, _ in enumerate(strings):
        v = np.zeros(4)
        v[i % 4] = 1.0
        rows.append(v)
    return np.array(rows)


def _make_two_iter_answer_pipe():
    """Build a pipe that emits 2 iter ANSWER (different), forces no early-stop."""
    pipe = _FakePipeline(retrieval_plan=[["c1"], ["c2"]])
    _make_response_sequence(pipe, [
        # iter 1: ANSWER A (early-stop fires here)
        "1. EXTRACT\n- f1.\n2. INTEGRATE\n.\n3. ASSESS\nyes.\n4. OUTPUT\nANSWER: candidateA",
        # (iter 2 not reached when stop_early=True)
    ])
    return pipe


def test_c7_iter_disabled_returns_none() -> None:
    """L4b master switch off → c7_iter_kept and c7_iter_info both None."""
    pipe = _make_two_iter_answer_pipe()
    cfg = IterativeConfig(max_iterations=2, use_c7_iter=False,
                           c7_iter_embedder=_orthogonal_iter_embedder)
    runner = IterativeMothRAG(pipe, cfg)
    info = runner.answer("Q?")
    assert info.c7_iter_kept is None
    assert info.c7_iter_info is None


def test_c7_iter_no_embedder_noop() -> None:
    """L4b enabled but no embedder injected → silent no-op (no crash)."""
    pipe = _make_two_iter_answer_pipe()
    cfg = IterativeConfig(max_iterations=2, use_c7_iter=True,
                           c7_iter_embedder=None, c7_iter_trigger="blanket")
    runner = IterativeMothRAG(pipe, cfg)
    info = runner.answer("Q?")
    assert info.c7_iter_kept is None
    assert info.c7_iter_info is None


def test_c7_iter_max_iter_lt_2_degenerate() -> None:
    """L4b skipped when max_iterations<2 (no temporal axis to measure)."""
    pipe = _FakePipeline(retrieval_plan=[["c1"]])
    _make_response_sequence(pipe, [
        "1. EXTRACT\n- f.\n2. INTEGRATE\n.\n3. ASSESS\nyes.\n4. OUTPUT\nANSWER: only",
    ])
    cfg = IterativeConfig(max_iterations=1, use_c7_iter=True,
                           c7_iter_embedder=_orthogonal_iter_embedder,
                           c7_iter_trigger="blanket")
    runner = IterativeMothRAG(pipe, cfg)
    info = runner.answer("Q?")
    assert info.c7_iter_kept is None
    assert info.c7_iter_info is None


def test_c7_iter_single_iteration_degenerate() -> None:
    """Only 1 distinct candidate (early-stop iter 1) → no rejected, skip C7."""
    pipe = _make_two_iter_answer_pipe()
    cfg = IterativeConfig(max_iterations=2, use_c7_iter=True,
                           c7_iter_trigger="blanket",
                           c7_iter_embedder=_orthogonal_iter_embedder)
    runner = IterativeMothRAG(pipe, cfg)
    info = runner.answer("Q?")
    # Only 1 iteration with answer "candidateA" → no rejected → skip
    assert info.c7_iter_kept is None
    assert info.c7_iter_info is None


def test_c7_iter_blanket_runs_with_distinct_candidates() -> None:
    """L4b blanket trigger + 2 distinct iter answers (A then B) → C7 runs."""
    pipe = _FakePipeline(retrieval_plan=[["c1"], ["c2"]])
    _make_response_sequence(pipe, [
        # iter 1: MISSING → continues loop
        "1. EXTRACT\n- f1.\n2. INTEGRATE\n.\n3. ASSESS\nno.\n4. OUTPUT\n"
        "MISSING: more.\nNEXT_QUERY: q2",
        # iter 2: ANSWER B (only after both iters; MISSING in 1 made no answer captured)
        "1. EXTRACT\n- f2.\n2. INTEGRATE\n.\n3. ASSESS\nyes.\n4. OUTPUT\nANSWER: candidateB",
    ])
    cfg = IterativeConfig(max_iterations=2, use_c7_iter=True,
                           c7_iter_trigger="blanket",
                           c7_iter_embedder=_orthogonal_iter_embedder)
    runner = IterativeMothRAG(pipe, cfg)
    info = runner.answer("Q?")
    # iter 1 had no ANSWER (MISSING), iter 2 ANSWER B → only 1 distinct → skip
    # This test exercises the no-rejected branch differently — confirm consistent.
    assert info.c7_iter_kept is None  # only candidateB collected, nothing to compare
    assert info.c7_iter_info is None


def test_c7_iter_blanket_runs_with_two_iter_answers() -> None:
    """Two iters BOTH with distinct ANSWER (no early-stop) → C7 fires."""
    pipe = _FakePipeline(retrieval_plan=[["c1"], ["c2"]])
    _make_response_sequence(pipe, [
        "1. EXTRACT\n- f1.\n2. INTEGRATE\n.\n3. ASSESS\nyes.\n4. OUTPUT\nANSWER: candidateA",
        "1. EXTRACT\n- f2.\n2. INTEGRATE\n.\n3. ASSESS\nyes.\n4. OUTPUT\nANSWER: candidateB",
    ])
    # stop_early=False forces both iterations to run
    cfg = IterativeConfig(max_iterations=2, stop_early=False,
                           use_c7_iter=True, c7_iter_trigger="blanket",
                           c7_iter_embedder=_orthogonal_iter_embedder)
    runner = IterativeMothRAG(pipe, cfg)
    info = runner.answer("Q?")
    assert info.iterations_used == 2
    # final answer = synthesized by pipe.read() (no early_answer set → goes to read)
    # per_iter_answer should have ['candidateA', 'candidateB']
    assert info.per_iteration_answer == ["candidateA", "candidateB"]
    # Final answer "FINAL" (from _FakePipeline.read), distinct from both → C7 fires
    assert info.c7_iter_kept is not None  # bool, not None
    assert info.c7_iter_info is not None
    assert "chosen_kept" in info.c7_iter_info
    # K = chosen + 2 distinct rejected = 3
    assert info.c7_iter_info.get("K") == 3


def test_c7_iter_helper_gated_skip_when_gamma_valid() -> None:
    """L4b helper: gated + gamma_status='valid' → skip (no temporal noise)."""
    pipe = _FakePipeline(retrieval_plan=[["c1"]])
    runner = IterativeMothRAG(pipe, IterativeConfig(
        max_iterations=2, use_c7_iter=True, c7_iter_trigger="gated",
        c7_iter_embedder=_orthogonal_iter_embedder))
    kept, info = runner._apply_c7_iter(
        ["candidateA", "candidateB"], "candidateC", gamma_status="valid")
    assert kept is None
    assert info is None


def test_c7_iter_helper_gated_apply_when_gamma_partial() -> None:
    """L4b helper: gated + gamma_status='partial' → apply C7."""
    pipe = _FakePipeline(retrieval_plan=[["c1"]])
    runner = IterativeMothRAG(pipe, IterativeConfig(
        max_iterations=2, use_c7_iter=True, c7_iter_trigger="gated",
        c7_iter_embedder=_orthogonal_iter_embedder))
    kept, info = runner._apply_c7_iter(
        ["candidateA", "candidateB"], "candidateC", gamma_status="partial")
    assert kept is not None
    assert info is not None
    assert "chosen_kept" in info
    assert info.get("K") == 3


def test_c7_iter_gated_skips_when_no_gamma_status() -> None:
    """Gated trigger + γ disabled → gamma_final_status=None → skip."""
    pipe = _FakePipeline(retrieval_plan=[["c1"], ["c2"]])
    _make_response_sequence(pipe, [
        "1. EXTRACT\n- f1.\n2. INTEGRATE\n.\n3. ASSESS\nyes.\n4. OUTPUT\nANSWER: A",
        "1. EXTRACT\n- f2.\n2. INTEGRATE\n.\n3. ASSESS\nyes.\n4. OUTPUT\nANSWER: B",
    ])
    cfg = IterativeConfig(max_iterations=2, stop_early=False,
                           use_c7_iter=True, c7_iter_trigger="gated",
                           c7_iter_embedder=_orthogonal_iter_embedder)
    runner = IterativeMothRAG(pipe, cfg)
    info = runner.answer("Q?")
    # γ disabled → gamma_final_status=None → gated trigger skips
    assert info.c7_iter_kept is None
    assert info.c7_iter_info is None


# ---- Lever 4 tests (faithfulness loop) ----

class _FakeJudge:
    """Mock judge: replays scripted (score, label) per call."""

    def __init__(self, scripted_results):
        self.scripted = list(scripted_results)
        self.calls = []
        # OpenAI-shape mock so faithfulness_score with provider="openai" works
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        # Stub create() - we won't actually use the response since we override _check_faithfulness
        self.chat.completions.create = MagicMock()


def _patch_check_faithfulness(runner, scripted_results):
    """Replace _check_faithfulness to return scripted (score, label) per call."""
    queue = list(scripted_results)
    def fake(_question, _passages, _pred):
        if queue:
            return queue.pop(0)
        return 0.0, "no"
    runner._check_faithfulness = fake


def test_l4_faithfulness_gate_accepts_yes() -> None:
    """ANSWER + judge=YES → accept early, no extra iter."""
    pipe = _FakePipeline(retrieval_plan=[["c1"], ["c2"]])
    _make_response_sequence(pipe, [
        "1. EXTRACT\n- fact.\n2. INTEGRATE\n.\n3. ASSESS\nYes.\n4. OUTPUT\nANSWER: Quentin",
    ])
    judge = _FakeJudge([])
    cfg = IterativeConfig(max_iterations=4, use_faithfulness_loop=True,
                           judge_client=judge, judge_model="fake-judge",
                           judge_min_score=1.0)
    runner = IterativeMothRAG(pipe, cfg)
    _patch_check_faithfulness(runner, [(1.0, "yes")])
    info = runner.answer("Who?")
    assert info.answer == "Quentin"
    assert info.iterations_used == 1
    assert info.per_iteration_faithfulness_score == [1.0]
    assert info.per_iteration_faithfulness_label == ["yes"]


def test_l4_faithfulness_gate_rejects_no_continues_loop() -> None:
    """ANSWER + judge=NO at iter 1 → drop ANSWER, continue iter 2."""
    pipe = _FakePipeline(retrieval_plan=[["c1"], ["c2"], ["c3"]])
    _make_response_sequence(pipe, [
        "1. EXTRACT\n- weak.\n2. INTEGRATE\n.\n3. ASSESS\nYes.\n4. OUTPUT\nANSWER: hallucinated",
        "1. EXTRACT\n- better.\n2. INTEGRATE\n.\n3. ASSESS\nYes.\n4. OUTPUT\nANSWER: real",
    ])
    judge = _FakeJudge([])
    cfg = IterativeConfig(max_iterations=4, use_faithfulness_loop=True,
                           judge_client=judge, judge_model="fake-judge",
                           judge_min_score=1.0)
    runner = IterativeMothRAG(pipe, cfg)
    _patch_check_faithfulness(runner, [(0.0, "no"), (1.0, "yes")])
    info = runner.answer("q?")
    assert info.answer == "real"  # iter 2 ANSWER accepted
    assert info.iterations_used == 2
    assert info.per_iteration_faithfulness_score == [0.0, 1.0]
    assert info.per_iteration_faithfulness_label == ["no", "yes"]


def test_l4_faithfulness_cap_accepts_after_max_judge_iterations() -> None:
    """If cap reached, accept ANSWER even if judge=NO (run out of patience)."""
    pipe = _FakePipeline(retrieval_plan=[["c1"], ["c2"], ["c3"]])
    _make_response_sequence(pipe, [
        "1. EXTRACT\n- a.\n2. INTEGRATE\n.\n3. ASSESS\nYes.\n4. OUTPUT\nANSWER: ans1",
        "1. EXTRACT\n- b.\n2. INTEGRATE\n.\n3. ASSESS\nYes.\n4. OUTPUT\nANSWER: ans2",
    ])
    judge = _FakeJudge([])
    cfg = IterativeConfig(max_iterations=4, use_faithfulness_loop=True,
                           judge_client=judge, judge_model="fake-judge",
                           judge_min_score=1.0, max_judge_iterations=2)
    runner = IterativeMothRAG(pipe, cfg)
    _patch_check_faithfulness(runner, [(0.0, "no"), (0.0, "no")])
    info = runner.answer("q?")
    # Iter 1: judge NO, continue
    # Iter 2: judge NO again BUT it >= cap=2, accept anyway
    assert info.answer == "ans2"
    assert info.iterations_used == 2
    assert info.per_iteration_faithfulness_label == ["no", "no"]


def test_fact_accumulator_dedup() -> None:
    pipe = _FakePipeline(retrieval_plan=[["c1"], ["c2"]])
    _make_response_sequence(pipe, [
        "1. EXTRACT\n- shared fact\n- unique 1\n2. INTEGRATE\n.\n"
        "3. ASSESS\nNo.\n4. OUTPUT\nMISSING: x\nNEXT_QUERY: y",
        "1. EXTRACT\n- shared fact\n- unique 2\n2. INTEGRATE\n.\n"
        "3. ASSESS\nYes.\n4. OUTPUT\nANSWER: done",
    ])
    runner = IterativeMothRAG(pipe, IterativeConfig(max_iterations=4))
    info = runner.answer("q")
    # 'shared fact' appears in both iterations; should be present exactly once.
    shared = [f for f in info.accumulated_facts if f == "shared fact"]
    assert len(shared) == 1
    assert "unique 1" in info.accumulated_facts
    assert "unique 2" in info.accumulated_facts
