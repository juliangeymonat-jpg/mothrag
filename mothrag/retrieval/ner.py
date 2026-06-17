# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""LLM-NER cache for query entity linking.

Workflow:
  1. Build a cache once per question set: ``build_ner_cache(client, model, questions, ...)``
  2. At eval time, ``link_query_entities_with_cache`` resolves cached spans
     against the entity registry, falling back to a domain plugin when a
     question has no cache hit.
"""

import json
import re
from pathlib import Path


NER_SYSTEM = """Extract all NAMED ENTITIES from the question. Output a JSON list of strings.

Named entities include:
- People (full names): "Karen Joy Fowler", "Bruce Chatwin"
- Places: "Boston", "Finnmark", "Singapore"
- Works (books, films, shows): "The Newcomers", "The Adventure of the Seven Clocks"
- Organizations: "BBC Formula One", "Six Flags Great America"
- Specific objects/concepts: "Pratt & Whitney F100", "ICC KnockOut Trophy"

DO NOT include:
- Generic words ("city", "person", "team")
- Question words ("what", "who", "where")
- Common adjectives unless part of proper name

Output STRICTLY JSON list, no other text:
["entity1", "entity2", ...]"""


def call_ner(client, model: str, question: str) -> list[str]:
    msgs = [
        {"role": "system", "content": NER_SYSTEM},
        {"role": "user", "content": question},
    ]
    resp = client.chat.completions.create(model=model, messages=msgs,
                                          max_tokens=200, temperature=0)
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    lb, rb = raw.find("["), raw.rfind("]")
    if lb < 0 or rb <= lb:
        return []
    try:
        parsed = json.loads(raw[lb : rb + 1])
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass
    return []


def slugify(name: str) -> str:
    s = re.sub(r"[^\w\s-]", "", name).strip().lower()
    return re.sub(r"[-\s]+", "-", s) or "unknown"


def normalize_for_lookup(text: str) -> str:
    s = re.sub(r"[^\w\s]", " ", text).lower()
    return " ".join(s.split())


def build_entity_index(entities: list[dict]) -> dict[str, str]:
    """Map normalized entity-name -> entity_id."""
    idx: dict[str, str] = {}
    for e in sorted(entities, key=lambda x: -len(x.get("name") or x.get("id", ""))):
        if e.get("type") not in ("document", "entity"):
            continue
        name = e.get("name") or e.get("id", "").replace("doc_", "").replace("ent_", "")
        n = normalize_for_lookup(name.replace("-", " "))
        if n and n not in idx:
            idx[n] = e["id"]
        for tok in n.split():
            if len(tok) > 4 and tok not in idx:
                idx[tok] = e["id"]
    return idx


def load_cache(path: str | Path) -> dict[str, list[str]]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_cache(cache: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def link_query_entities_with_cache(question: str, entities_by_id: dict,
                                   cache: dict[str, list[str]],
                                   fallback_plugin=None) -> list[str]:
    """Resolve query entities using the LLM-NER cache + KB index.

    Falls back to ``fallback_plugin.link_query_entities`` on cache miss or when
    the cached spans cannot be resolved against the entity registry.
    """
    spans = cache.get(question, None)
    if spans is None and fallback_plugin is not None:
        return fallback_plugin.link_query_entities(question, entities_by_id)

    idx = build_entity_index(list(entities_by_id.values()))
    resolved = []
    seen = set()
    for span in spans or []:
        n = normalize_for_lookup(span)
        eid = idx.get(n)
        if eid is None:
            first = n.split()[0] if n else ""
            if first and first in idx:
                eid = idx[first]
        if eid and eid not in seen:
            seen.add(eid)
            resolved.append(eid)

    if not resolved and fallback_plugin is not None:
        return fallback_plugin.link_query_entities(question, entities_by_id)
    return resolved


def build_ner_cache(client, model: str, questions: list[dict],
                    out_path: str | Path,
                    save_every: int = 10) -> dict[str, list[str]]:
    """Build NER cache for a list of HotpotQA-style questions.

    Each question is a dict with at least a ``"question"`` key. Cache is
    persisted every ``save_every`` questions and at the end.
    """
    cache = load_cache(out_path)
    n_new = 0
    for i, q in enumerate(questions):
        qtext = q["question"] if isinstance(q, dict) else str(q)
        if qtext in cache:
            continue
        try:
            cache[qtext] = call_ner(client, model, qtext)
        except Exception as e:  # noqa: BLE001
            print(f"  ! NER failed q[{i}]: {e}")
            cache[qtext] = []
        n_new += 1
        if (i + 1) % save_every == 0:
            save_cache(cache, out_path)
    save_cache(cache, out_path)
    return cache
