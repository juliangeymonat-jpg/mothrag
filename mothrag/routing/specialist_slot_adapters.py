# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Adapters: M8 retrieval-shaping specialists -> decompose-slot runners.

Both M8 specialists are RETRIEVAL shapers: ``CompareArm.retrieve`` /
``DecomposeArmV2.retrieve`` return ``ranked_passage_ids`` (+ a ``fired`` /
``fallback`` flag) — neither emits a final answer. The pool-safe slot router
(``SpecialistSlotRouter``), however, fills the polymorphic ``decompose`` slot
with a *candidate dict* of the shape ``route_prospective`` arbitration consumes:

    {pred, retrieved_chunk_ids, n_llm_calls, prompt_tokens,
     completion_tokens, latency_s, metadata?}

So the adapter turns ``specialist.retrieve(q) -> ranked_passage_ids`` into that
dict by reading the specialist's passages with the **same reader** the generic
decompose slot uses. The substitution therefore swaps only the RETRIEVAL feeding
the slot, never the answer machinery — and the pool stays exactly 4 arms.

Decline semantics (pool-safety): the adapter returns ``None`` whenever the
specialist did not genuinely fire (no passages / ``fired is False`` /
``fallback is True``) or the reader produced nothing. ``None`` makes
``SpecialistSlotRouter.run_decompose_slot`` fall back to the generic decompose
arm — the slot is never dropped, so the pool never shrinks below 4.

Everything is injected (the specialist, the reader, the id->text fetch), so the
adapters are fully offline-testable; the live ``route_prospective`` wiring just
supplies the real pipeline reader + passage store.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger(__name__)

# read_slot(question, passage_ids) -> slot_dict
ReadSlot = Callable[[str, Sequence[str]], Optional[dict]]
# reader with a .read(question, passages_text) -> str surface (e.g. _PipelineReaderShim)
Reader = Any
# fetch_texts(passage_ids) -> list[str]
FetchTexts = Callable[[Sequence[str]], Sequence[str]]


def _did_fire(result: Any) -> bool:
    """True iff a retrieval specialist genuinely produced a usable passage set."""
    if getattr(result, "fallback", False):          # DecomposeArmV2 chain broke
        return False
    if getattr(result, "fired", None) is False:     # CompareArm split didn't run
        return False
    return bool(getattr(result, "ranked_passage_ids", None))


def make_specialist_slot_runner(
    *,
    specialist: Any,
    read_slot: ReadSlot,
    name: str,
) -> Callable[..., Optional[dict]]:
    """Wrap a retrieval-shaping specialist as a decompose-slot runner.

    ``specialist`` exposes ``.retrieve(question) -> result`` where ``result`` has
    ``ranked_passage_ids`` and a ``fired`` / ``fallback`` flag. ``read_slot``
    turns ``(question, passage_ids)`` into the candidate slot dict (the reader
    pass). Returns ``runner(question, **kw) -> slot_dict | None`` (``None`` ⇒ the
    router falls back to the generic decompose slot; pool stays 4).
    """

    def runner(question: str, **_kw: Any) -> Optional[dict]:
        try:
            result = specialist.retrieve(question)
        except Exception:  # noqa: BLE001 — a specialist crash ⇒ generic fallback
            logger.warning("specialist %s.retrieve failed; slot falls back to generic",
                           name, exc_info=True)
            return None
        if not _did_fire(result):
            return None
        ids = [str(p) for p in result.ranked_passage_ids]
        slot = read_slot(question, ids)
        if not slot or not slot.get("pred"):
            return None
        out = dict(slot)
        out.setdefault("retrieved_chunk_ids", ids)
        meta = dict(out.get("metadata") or {})
        meta["specialist_slot"] = name
        meta["specialist_passage_ids"] = ids
        out["metadata"] = meta
        return out

    runner.__name__ = f"{name}_slot_runner"
    return runner


def make_reader_slot_reader(
    *,
    reader: Reader,
    fetch_texts: FetchTexts,
) -> ReadSlot:
    """Build a ``read_slot(question, ids)`` from a ``.read`` reader + id->text fetch.

    ``reader.read(question, passages_text) -> str`` (e.g. ``_PipelineReaderShim``);
    ``fetch_texts(ids) -> [text, ...]`` resolves passage ids to texts via the live
    pipeline store. Produces the slot dict with cost/latency fields stubbed at the
    reader granularity (one reader call). Returns ``None`` on reader failure so the
    adapter degrades to the generic slot.
    """

    def read_slot(question: str, ids: Sequence[str]) -> Optional[dict]:
        t0 = time.time()
        try:
            texts = list(fetch_texts(ids))
        except Exception:  # noqa: BLE001
            return None
        if not texts:
            return None
        try:
            pred = reader.read(question, texts)
        except Exception:  # noqa: BLE001
            return None
        return {
            "pred": (pred or "").strip(),
            "retrieved_chunk_ids": list(ids),
            "n_llm_calls": 1,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_s": float(time.time() - t0),
        }

    return read_slot


__all__ = [
    "make_specialist_slot_runner",
    "make_reader_slot_reader",
    "ReadSlot",
]
