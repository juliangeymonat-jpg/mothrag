# Retry-on-abstain escalation cascade

When MothRag's arbitration produces an abstention signal (γ-invalid, H4 / H12 fired, iter abstained, or arm disagreement that cannot be arbitrated), the production pipeline walks an ordered list of recovery strategies. The *terminal* behaviour is mode-dependent:

- **`mode="loop"`** (default, production / customer-facing): a non-recovering cascade falls through to SoftFallback, which guarantees a non-empty answer.
- **`mode="abstention"`** (KB-audit / gap-discovery): a non-recovering cascade returns `qr.answer = ""` with `qr.metadata["terminal_abstain"] = True`, surfacing the gap signal to downstream auditors.

The cascade itself (same 7 strategies, same priorities) is shared across both modes — only the terminal diverges. The framework lives at [`mothrag/core/retry/`](../../mothrag/core/retry/).

## Quickstart

```python
from mothrag import MothRAG

# Default: all 7 strategies active, mode='loop' (non-empty guarantee).
rag = MothRAG.from_documents("docs/", production=True)
qr = rag.query("Multi-hop question that would normally abstain?")
print(qr.answer)
print(qr.metadata["mode"])                       # "loop" | "abstention"
print(qr.metadata["escalation_recovered_by"])    # which strategy fired, or None
print(qr.metadata["escalation_applied"])         # list of strategies tried
print(qr.metadata["final_answer_confidence"])    # high | medium_recovered | low_soft_fallback | terminal_abstain
print(qr.metadata["terminal_abstain"])           # True only in abstention mode + exhausted cascade
```

Disable for backward-compat:

```python
rag = MothRAG.from_documents("docs/", production=True, retry_strategies="off")
```

Zero-LLM "sweet spot" bundle:

```python
rag = MothRAG.from_documents("docs/", production=True, retry_strategies="sweet_spot")
# == ["iter_extension", "arm_fallback", "cross_arm_consensus", "soft_fallback"]
```

Custom explicit cascade (in loop mode, `soft_fallback` is auto-appended as the terminus; in abstention mode the list is honoured verbatim):

```python
rag = MothRAG.from_documents(
    "docs/", production=True,
    retry_strategies=["cross_arm_consensus", "bottom_up_boost"],
)
```

## Dual-mode deployment

The cascade serves two distinct deployment profiles from the same code path:

### `mode="loop"` — production / customer-facing (default)

The terminal SoftFallback always fires when every non-terminal strategy declines. The user-visible answer is **guaranteed non-empty** — recovered when possible, soft-fallen-back otherwise. This is the right mode when:

- The system fronts a customer-facing product where empty answers are a worse UX than a low-confidence answer.
- Downstream consumers cannot handle a structured "I don't know" outcome.
- The application has its own confidence-gated UI layer that uses `qr.metadata["final_answer_confidence"]` to decide whether to show the answer prominently.

```python
rag = MothRAG.from_documents("docs/", production=True, mode="loop")
qr = rag.query("...")
assert qr.answer          # non-empty by construction
assert qr.metadata["mode"] == "loop"
assert qr.metadata["terminal_abstain"] is False
```

### `mode="abstention"` — KB-audit / gap-discovery

The terminal SoftFallback is skipped. When every strategy declines, the orchestrator surfaces `qr.answer = ""` with `qr.metadata["terminal_abstain"] = True`. This is the right mode when:

- The deployment is an **internal KB audit pipeline**: empty-answer events are the *signal* you're looking for (they identify which customer documents need expansion / clarification).
- The downstream consumer is a **self-improving-KB primitive** — terminal abstains feed gap-detection → fetch → re-validate.
- You want a clean separation between cascade-recovered answers and terminal-abstain events in your own analytics.

```python
rag = MothRAG.from_documents("docs/", production=True, mode="abstention")
qr = rag.query("...")
if qr.metadata["terminal_abstain"]:
    log_kb_gap(qr.metadata["original_abstention_signal"], qr.metadata["escalation_applied"])
else:
    use(qr.answer)
```

In abstention mode the strategy list is honoured verbatim — SoftFallback is **not** auto-appended. You can still include it explicitly if you want a graceful "give me something" fallback for *some* queries while letting others terminate-abstain cleanly. The orchestrator's terminal-must-be-SoftFallback invariant only applies in loop mode.

### Same cascade, different terminus

Both modes use the same priority list (`DEFAULT_PRIORITY`, `SWEET_SPOT_PRIORITY`, or explicit). The non-terminal strategies (#1 `iter_extension` through #3 `query_reformulation`) fire identically; only #7 `soft_fallback` is mode-conditional.

## The seven strategies

The default priority order is **cheap-zero-LLM-first, expensive-LLM-last, terminal-fallback-always-last**:

| # | Name | Cost | Fires on signal | What it does |
|---|---|---:|---|---|
| 1 | `iter_extension` | 0 LLM | gamma_refuse, iter_abstain, empty_answer | Re-runs the iter arm with doubled `max_iter_steps` + wider `top_k`. Best when iter exhausted its budget without converging. |
| 2 | `arm_fallback` | 0 LLM | iter_abstain, h4_refuse, h12_refuse, empty_answer, cross_arm_disagree | Picks the highest-quality non-empty sibling arm output (preference: iter > dec > v3bu). |
| 4 | `cross_arm_consensus` | 0 LLM | any signal, ≥2 non-empty arms | Embeds the arm answers, cosine-clusters them at `threshold=0.7`. If ≥2 arms cluster, returns the cluster majority. Otherwise returns None. |
| 5 | `bottom_up_boost` | 0 LLM | gamma_refuse, iter_abstain, h4_refuse, empty_answer | Extracts entities from question + passages (NER if `[retrieval]` installed, naive Capitalised-NP regex otherwise). Re-retrieves with entity-augmented query. Re-runs iter arm. |
| 6 | `l4b_anchor_retry` | 0 LLM | iter_abstain, gamma_refuse | Swaps the L4b temporal anchor from rank-1 to rank-2 and re-runs iter. *Wired but inactive on the v0.5.0 alpha pipeline; becomes live when the full iter pipeline lands in v0.5.1.* |
| 3 | `query_reformulation` | 1 LLM call | any signal | LAST RESORT before SoftFallback. Asks the reader to rewrite the question more precisely given passage excerpts. Re-runs v3bu on the rewrite. Guarded against infinite recursion via `RetryContext.escalation_depth`. |
| 7 | `soft_fallback` | 0 LLM | always-final (loop mode) | Terminal guarantee. Picks the best non-empty arm output (iter > dec > v3bu), or echoes the original chosen, or returns a `[no answer recovered]` placeholder. |

## Abstention signal taxonomy

Strategies dispatch on a string drawn from `mothrag.core.retry.protocol.ABSTENTION_SIGNALS`:

| Signal | Source | Heuristic at v0.5.0 alpha |
|---|---|---|
| `gamma_refuse` | γ verifier flagged answer invalid | substring `γ`/`gamma`/`refuse` in `arbitrate_reason` |
| `h4_refuse` | pre-registered H4 selective rule | substring `h4`/`refuse` |
| `h12_refuse` | pre-registered H12 `how_many` rule | substring `h12`/`how_many` |
| `iter_abstain` | iter arm exhausted budget | `c7_info["l4b"]["cancelled"]` truthy |
| `cross_arm_disagree` | arbiter could not pick a winner | `c7_info["disagreement"] is True` |
| `empty_answer` | chosen is empty or uncertainty template | fall-through fallback |

The full γ / H4 / H12 / L4b signal surface from `mothrag.core.selective_ensemble` and `mothrag.eval.iterative_pipeline` will be wired cleanly when the production iter pipeline ports to the public API in v0.5.1; the strategies are already coded against the post-v0.5.1 trigger names so no further refactor is needed at strategy level.

## Telemetry

`QueryResult.metadata` gains seven keys when production mode is on:

```python
{
    "mode":                        "loop",             # "loop" | "abstention"
    "original_abstention_signal":  "iter_abstain",      # str | None
    "escalation_applied":          ["iter_extension", "cross_arm_consensus"],
    "escalation_recovered_by":     "cross_arm_consensus", # str | None
    "final_answer_confidence":     "medium_recovered",   # high | medium_recovered | low_soft_fallback | terminal_abstain
    "escalation_budget_used":      0,                    # LLM-call count spent in cascade
    "terminal_abstain":            False,                # True iff mode='abstention' AND cascade exhausted
}
```

## Configuration matrix

Available `retry_strategies` presets × `mode`:

| `retry_strategies` value | mode | What runs |
|---|---|---|
| `"off"` / `None` / `[]` | either | nothing — original arbitration preserved |
| `"soft_fallback_only"` | `loop` | #7 only |
| `"sweet_spot"` | `loop` | #1 + #2 + #4 + #7 |
| `"all"` (default) | `loop` | full cascade #1, #2, #4, #5, #6, #3, #7 |
| `"all"` | `abstention` | #1, #2, #4, #5, #6, #3 (no SoftFallback) |
| explicit `[name…]` | `loop` | named strategies + auto-appended #7 |
| explicit `[name…]` | `abstention` | named strategies verbatim |

## Budget guard

`RetryContext` carries a `budget_used` / `budget_limit` pair (default 8 LLM calls). Strategies that issue LLM calls (currently only `query_reformulation` at cost 2) check `ctx.spend(...)` before firing and refuse to recurse when the budget is exhausted. This caps worst-case cascade cost regardless of how many strategies the user enables.

## Extending the cascade

Adding a new strategy is a single file under [`mothrag/core/retry/strategies/`](../../mothrag/core/retry/strategies/):

```python
from mothrag.core.retry.protocol import RetryContext

class MyRecoveryStrategy:
    name = "my_recovery"
    cost_estimate = 0   # or 1+ if it makes additional LLM calls

    def applicable(self, ctx: RetryContext) -> bool:
        return ctx.abstention_signal == "iter_abstain" and bool(ctx.iter_pred)

    def try_recover(self, ctx: RetryContext) -> str | None:
        # Read ctx; mutate only ctx.budget_used via ctx.spend(...).
        # Return the recovered answer or None to defer to the cascade.
        return None
```

Register it in `mothrag.core.retry.orchestrator._instantiate` and add the name to `DEFAULT_PRIORITY` in the position you want it to fire. Tests live next to the existing ones in [`tests/test_retry_strategies.py`](../../tests/test_retry_strategies.py).

## See also

- [`mothrag/core/retry/protocol.py`](../../mothrag/core/retry/protocol.py) — `RetryContext`, `RetryStrategy`, `RetryOutcome`.
- [`mothrag/core/retry/orchestrator.py`](../../mothrag/core/retry/orchestrator.py) — `EscalationOrchestrator`, `build_default_orchestrator`, priority constants.
- [API reference](api.md#retry-on-abstain-cascade) — full constructor signature.
