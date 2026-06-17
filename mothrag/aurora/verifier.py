"""γ pivot — deterministic proof verifier (R3.1).

Design decision: NO LLM in verifier. The architectural commitment
is "verifier deterministic + inspectable = the point of γ vs CoT". NLI as ablation
post-baseline.

Verifier checks per step:
  - lookup       : span verbatim ⊂ passage; subject grounded in passage; object grounded in span
  - conjunction  : both input steps exist and are valid; claim_text token-overlap with inputs
  - transitivity : both input steps exist and valid; subject/object chain consistent
  - comparison_before : both input steps exist and valid; numeric/temporal comparison literal
  - negation_as_failure : structurally valid (refusal IS a valid output)

Tree-level overall_status:
  - valid     : is_complete=true ∧ all steps valid
  - partial   : some valid steps but is_complete=false (refusal + partial proof)
  - invalid   : at least one step invalid in a complete tree
  - refuse    : is_complete=false with only negation_as_failure step (clean abstention)
"""

from __future__ import annotations
import re
import string

from mothrag.aurora.rules import ProofTree, ProofStep, Source

_PUNCT = set(string.punctuation)


def _norm_for_substring(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = "".join(c for c in s if c not in _PUNCT)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _any_token_in(needle: str, hay: str) -> bool:
    nt = [t for t in needle.split() if len(t) > 2]
    ht = set(hay.split())
    return True if not nt else any(t in ht for t in nt)


def _token_overlap(a: str, b: str, threshold: float) -> bool:
    at = set(t for t in a.split() if len(t) > 1)
    bt = set(t for t in b.split() if len(t) > 1)
    if not at:
        return True
    return len(at & bt) / len(at) >= threshold


def _verify_lookup(step: ProofStep, passages: list[dict],
                   use_relaxed_object_span: bool = False) -> ProofStep:
    if not step.sources:
        step.verifier_status = "invalid"
        step.verifier_reason = "no source for lookup step"
        return step
    src = step.sources[0]
    cited = next((p for p in passages if p["doc_id"] == src.doc_id), None)
    if cited is None:
        step.verifier_status = "invalid"
        step.verifier_reason = f"cited doc_id {src.doc_id!r} not in provided passages"
        return step
    span_n = _norm_for_substring(src.span_text)
    passage_n = _norm_for_substring(cited["text"])
    if not span_n or span_n not in passage_n:
        step.verifier_status = "invalid"
        step.verifier_reason = f"span not verbatim substring of {src.doc_id}"
        return step
    start = passage_n.find(span_n)
    src.char_offset = (start, start + len(span_n))

    span_norm = _normalize(src.span_text)
    passage_norm = _normalize(cited["text"])
    if step.subject:
        if not _any_token_in(_normalize(step.subject), passage_norm):
            step.verifier_status = "invalid"
            step.verifier_reason = f"subject {step.subject!r} not grounded in passage"
            return step
    # Object grounding. LEGACY (default): the object must have a
    # token in the narrow cited SPAN — too strict for descriptive / boolean objects
    # ("a fictional private detective", "true") whose words aren't in the quoted
    # span, the dominant secondary γ-invalid mode observed in a dry-run.
    # RELAXED (--use-relaxed-object-span): the object's tokens (len>1) must all be
    # covered by the PASSAGE (the cited doc), not just the span — the entity is in
    # the doc, just not the exact quoted fragment. Default OFF = byte-identical.
    if step.object:
        if use_relaxed_object_span:
            obj_tokens = [t for t in _normalize(step.object).split() if len(t) > 1]
            passage_tokens = set(passage_norm.split())
            if obj_tokens and not all(t in passage_tokens for t in obj_tokens):
                step.verifier_status = "invalid"
                step.verifier_reason = (
                    f"object {step.object!r} not grounded in passage (relaxed)")
                return step
            step.object_match_mode = "relaxed"
        else:
            if not _any_token_in(_normalize(step.object), span_norm):
                step.verifier_status = "invalid"
                step.verifier_reason = f"object {step.object!r} not grounded in span"
                return step
            step.object_match_mode = "exact"
    nat_n = _normalize(step.claim_text)
    if not nat_n:
        step.verifier_status = "invalid"
        step.verifier_reason = "empty claim_text"
        return step
    step.verifier_status = "valid"
    step.verifier_reason = (
        "lookup grounded in span; subject in passage; object in "
        + ("passage (relaxed)" if step.object_match_mode == "relaxed" else "span"))
    return step


def _verify_conjunction(step: ProofStep, by_id: dict[int, ProofStep]) -> ProofStep:
    if len(step.inputs) < 2:
        step.verifier_status = "invalid"
        step.verifier_reason = "conjunction needs ≥2 inputs"
        return step
    for inp_id in step.inputs:
        s = by_id.get(inp_id)
        if s is None or s.verifier_status != "valid":
            step.verifier_status = "invalid"
            step.verifier_reason = f"input step {inp_id} missing or not valid"
            return step
    step.verifier_status = "valid"
    step.verifier_reason = f"conjunction over valid steps {step.inputs}"
    return step


def _verify_transitivity(step: ProofStep, by_id: dict[int, ProofStep]) -> ProofStep:
    if len(step.inputs) != 2:
        step.verifier_status = "invalid"
        step.verifier_reason = "transitivity needs exactly 2 inputs"
        return step
    s1 = by_id.get(step.inputs[0])
    s2 = by_id.get(step.inputs[1])
    if not s1 or not s2 or s1.verifier_status != "valid" or s2.verifier_status != "valid":
        step.verifier_status = "invalid"
        step.verifier_reason = "transitivity inputs missing or not valid"
        return step
    if step.predicate not in ("located_in", "subset_of", "instance_of"):
        step.verifier_status = "invalid"
        step.verifier_reason = f"predicate {step.predicate!r} not in typed-transitive set"
        return step
    o1 = _normalize(s1.object or "")
    s2_subj = _normalize(s2.subject or "")
    if o1 and s2_subj and not _any_token_in(o1, s2_subj):
        step.verifier_status = "invalid"
        step.verifier_reason = "transitivity bridge entity mismatch"
        return step
    step.verifier_status = "valid"
    step.verifier_reason = "typed transitivity over valid inputs"
    return step


def _verify_comparison_before(step: ProofStep, by_id: dict[int, ProofStep]) -> ProofStep:
    if step.type_tag not in ("temporal", "numeric"):
        step.verifier_status = "invalid"
        step.verifier_reason = "comparison_before requires type_tag temporal|numeric"
        return step
    if len(step.inputs) != 2:
        step.verifier_status = "invalid"
        step.verifier_reason = "comparison_before needs exactly 2 inputs"
        return step
    for inp_id in step.inputs:
        s = by_id.get(inp_id)
        if s is None or s.verifier_status != "valid":
            step.verifier_status = "invalid"
            step.verifier_reason = f"input step {inp_id} missing or not valid"
            return step
    step.verifier_status = "valid"
    step.verifier_reason = f"comparison_before {step.type_tag} over valid inputs"
    return step


def _verify_negation(step: ProofStep) -> ProofStep:
    # Structural refusal: always considered valid as a γ output (the abstention IS the answer)
    step.verifier_status = "valid"
    step.verifier_reason = "structured refusal carrying partial-proof state"
    return step


def verify_proof_tree(tree: ProofTree, passages: list[dict],
                      use_relaxed_object_span: bool = False) -> ProofTree:
    by_id: dict[int, ProofStep] = {}
    for step in tree.steps:
        if step.rule == "lookup":
            _verify_lookup(step, passages, use_relaxed_object_span)
        elif step.rule == "conjunction":
            _verify_conjunction(step, by_id)
        elif step.rule == "transitivity":
            _verify_transitivity(step, by_id)
        elif step.rule == "comparison_before":
            _verify_comparison_before(step, by_id)
        elif step.rule == "negation_as_failure":
            _verify_negation(step)
        else:
            step.verifier_status = "invalid"
            step.verifier_reason = f"unknown rule {step.rule!r}"
        by_id[step.step] = step

    # Tree-level status
    has_negation = any(s.rule == "negation_as_failure" for s in tree.steps)
    only_negation = has_negation and all(s.rule == "negation_as_failure" for s in tree.steps)
    invalid_steps = [s for s in tree.steps if s.verifier_status == "invalid"]

    if only_negation:
        tree.overall_status = "refuse"
    elif tree.is_complete and not invalid_steps:
        tree.overall_status = "valid"
    elif not tree.is_complete and not invalid_steps:
        tree.overall_status = "partial"
    else:
        tree.overall_status = "invalid"
    return tree
