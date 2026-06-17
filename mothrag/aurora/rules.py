"""γ pivot — 5-rule catalogue + claim-leaf schema (R3.1).

R3.1 updates:
- Rule 4 generalized to `comparison-before` with type_tag (temporal | numeric).
  One rule, not two twins. "X before Y" temporal and "A < B" numeric are the same binary
  predicate over an ordered domain.
- Verifier remains deterministic (no NLI). Architectural commitment.
- Rules 2 + 5 schemas added below; full proof tree (multi-step) is the unit of emit.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, Any


# -------- Source provenance --------

@dataclass
class Source:
    doc_id: str
    span_text: str
    char_offset: Optional[tuple[int, int]] = None  # filled by verifier


# -------- Step (one node in the proof tree) --------

RuleName = Literal["lookup", "conjunction", "transitivity", "comparison_before", "negation_as_failure"]


@dataclass
class ProofStep:
    """One node in a γ proof tree.

    For lookup / transitivity / comparison_before: subject + object + claim_text + 1 source.
    For conjunction: inputs (list of step ids) and a derived claim_text; sources inherited.
    For negation_as_failure: refusal carries the partial-proof state.
    """
    step: int                          # 1-based step id (for cross-step references)
    rule: RuleName
    predicate: Optional[str] = None    # None for conjunction (inherits from inputs) and refuse
    subject: Optional[str] = None
    object: Optional[str] = None
    claim_text: str = ""
    sources: list[Source] = field(default_factory=list)
    inputs: list[int] = field(default_factory=list)   # step ids combined (rule 2/3)
    type_tag: Optional[Literal["temporal", "numeric"]] = None  # rule 4 only
    verifier_status: str = "unchecked"  # valid | invalid | partial | unchecked
    verifier_reason: Optional[str] = None
    # How the object was grounded on a passing lookup:
    # "exact" (legacy: object token in the cited span) or "relaxed"
    # (--use-relaxed-object-span: object tokens covered by the passage). None
    # unless the step is a passing lookup.
    object_match_mode: Optional[str] = None


# -------- ProofTree (the unit emitted per question) --------

@dataclass
class ProofTree:
    """A γ proof tree: ordered sequence of proof steps + naturalized leaf answer."""
    qid: str
    steps: list[ProofStep] = field(default_factory=list)
    naturalized_answer: str = ""        # short answer for legacy Hard-EM
    is_complete: bool = False           # all required predicates discharged?
    refusal_reason: Optional[str] = None  # populated when is_complete=False
    overall_status: str = "unchecked"   # valid | partial | invalid | refuse

    def to_dict(self) -> dict[str, Any]:
        d = {
            "qid": self.qid,
            "naturalized_answer": self.naturalized_answer,
            "is_complete": self.is_complete,
            "refusal_reason": self.refusal_reason,
            "overall_status": self.overall_status,
            "steps": [],
        }
        for s in self.steps:
            sd = asdict(s)
            d["steps"].append(sd)
        return d


# -------- Prompt: emit full proof tree (one LLM call per question) --------

PROOF_TREE_SYSTEM_PROMPT = """You are a proof-step prior for a γ-pivot reasoning-quality QA system.

Your job is NOT to write a free-text answer. Your job is to emit a structured PROOF TREE that derives the answer from atomic claims grounded in passages.

You emit JSON only. The proof tree has ordered steps. Each step is one of 5 rules:

1. **lookup** — atomic fact extraction from a single passage.
   { "step": <int>, "rule": "lookup", "predicate": <str>, "subject": <str>,
     "object": <str>, "claim_text": <one-sentence statement>,
     "source": { "doc_id": <exact doc_id>, "span_text": <verbatim substring of passage> } }

2. **conjunction** — combines two earlier steps that share a bridge entity.
   { "step": <int>, "rule": "conjunction", "inputs": [<step_id_1>, <step_id_2>],
     "claim_text": <derived statement> }

3. **transitivity** — typed chain (located_in, subset_of, instance_of only).
   { "step": <int>, "rule": "transitivity", "inputs": [<step_id_1>, <step_id_2>],
     "predicate": <typed-transitive predicate>, "subject": <A>, "object": <C>,
     "claim_text": <derived statement> }

4. **comparison_before** — ordering on a temporal or numeric domain. Generalizes "X before Y" (temporal) and "A < B" (numeric) under a single rule with type_tag.
   { "step": <int>, "rule": "comparison_before", "type_tag": "temporal" | "numeric",
     "inputs": [<step_id_1>, <step_id_2>], "subject": <smaller/earlier>,
     "object": <larger/later>, "claim_text": <derived comparison statement> }

5. **negation_as_failure** — structured refusal carrying partial-proof state. Use this when no valid proof exists within the available passages.
   { "step": <int>, "rule": "negation_as_failure", "claim_text": <which predicate is unprovable>,
     "missing": <description of what evidence would discharge it> }

After emitting steps, also emit:
- "naturalized_answer": short answer (1-5 words) extracted from the proof's terminal step (legacy Hard-EM compatibility)
- "is_complete": true iff the proof discharges the question's predicate without rule-5 refuse
- "refusal_reason": null if is_complete else the unproven predicate

STRICT RULES:
1. Output ONLY valid JSON. No commentary, no preamble, no markdown fences. Use double quotes only. Escape any internal double quotes inside string values with backslash.
2. Every "span_text" MUST be copied VERBATIM from one of the provided passages — exact substring match, including punctuation and case (we will normalize whitespace only). If you cannot find a verbatim span that supports the predicate, you MUST emit negation_as_failure for that step.
3. Every "doc_id" MUST be one of the doc_ids listed in the passages.
4. Step ids start at 1 and are dense and ordered. Inputs reference earlier step ids only.
5. If the question is single-hop, emit ONE lookup (the lookup IS the proof). Don't emit conjunction over 1 input.
6. If the question is multi-hop bridge and the bridge entity is NOT discharged by any passage, emit a partial proof: one lookup over what you have + one negation_as_failure for the missing predicate + is_complete=false + naturalized_answer="REFUSE_NO_PROOF". Do NOT fabricate the missing fact.
7. **Relevance check**: a lookup is valid only if the (subject, predicate, object) triple it asserts directly answers the question. If the only verbatim spans you can find produce facts orthogonal to what was asked (e.g., the passage is about a different entity with the same name), DO NOT emit them as lookup — emit negation_as_failure with "missing": "<entity-disambiguated>". Grounded-but-irrelevant is worse than a refusal.
8. If you cannot ground anything, emit a single negation_as_failure step, is_complete=false, naturalized_answer="REFUSE_NO_PROOF".
9. naturalized_answer is the SHORT form (1-5 words) extracted from the proof's terminal step. Match the gold's expected granularity if the question implies one (e.g., "when?" → date range; "where?" → city or city+country; "who?" → full name).

Output schema (strict):
{
  "steps": [<step objects>],
  "naturalized_answer": <str>,
  "is_complete": <bool>,
  "refusal_reason": <str | null>
}
"""


# -------- Llama-friendly variant: simplified 2-rule prompt + inline example --------

PROOF_TREE_SYSTEM_PROMPT_LLAMA = """You answer questions by emitting a small JSON proof tree.

You output JSON only — no preamble, no markdown fences, no explanations. Use double quotes.

Each step is one of TWO rules:

1. **lookup** — a fact you found in a passage. Format:
   { "step": <int>, "rule": "lookup", "subject": "<entity>", "predicate": "<relation>",
     "object": "<value>", "claim_text": "<one-sentence statement>",
     "sources": [ { "doc_id": "<exact doc_id from passage>",
                    "span_text": "<verbatim substring of the passage>" } ] }

2. **negation_as_failure** — use this when the passages do not support an answer.
   { "step": <int>, "rule": "negation_as_failure",
     "claim_text": "no passage supports the requested fact" }

After the steps, emit:
- "naturalized_answer": the short answer (1-5 words). If you used negation_as_failure, set this to "REFUSE_NO_PROOF".
- "is_complete": true if you have a real answer, false if you used negation_as_failure.

CRITICAL RULES:
- "span_text" MUST be a substring copied VERBATIM from the passage text. Same words, same order. We will only normalize whitespace.
- "doc_id" MUST be one of the doc_ids printed in the passages list.
- For a single-hop question, ONE lookup step is enough.
- If you cannot find a verbatim span that supports an answer, output negation_as_failure (do NOT invent the fact).

EXAMPLE (single-hop):
QUESTION: Who directed Inception?
PASSAGES:
[1] doc_id: doc_inception
Inception is a 2010 science fiction film written and directed by Christopher Nolan.

OUTPUT:
{"steps": [{"step": 1, "rule": "lookup", "subject": "Inception", "predicate": "directed_by", "object": "Christopher Nolan", "claim_text": "Inception was directed by Christopher Nolan", "sources": [{"doc_id": "doc_inception", "span_text": "Inception is a 2010 science fiction film written and directed by Christopher Nolan"}]}], "naturalized_answer": "Christopher Nolan", "is_complete": true}

Now answer the user's question with the same JSON format. JSON only.
"""


def proof_tree_user_prompt(question: str, passages: list[dict]) -> str:
    blocks = []
    for i, p in enumerate(passages):
        blocks.append(f"[{i+1}] doc_id: {p['doc_id']}\n{p['text']}")
    return (
        f"QUESTION: {question}\n\n"
        f"PASSAGES:\n" + "\n\n".join(blocks) + "\n\n"
        f"Emit the proof tree as JSON. Spans must be VERBATIM substrings of the passages."
    )
