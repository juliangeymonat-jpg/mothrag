# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Pre-process a Wikipedia corpus -> MothRAG-compatible entities/edges/chunks.

Input format (HippoRAG corpus.json)::

    [{"title": "...", "text": "...", "idx": 0}, ...]

Output (under ``output_dir``):
    entities.json   list of {id, name, type='document'|'entity', ...}
    edges.json      list of {src, dst, type='triple', predicate, ...}
    chunks.jsonl    one JSON per line: {id, text, entity_id (=doc), mentions: [...]}
    openie_stats.json   token counts + cost estimate

CLI::

    python -m mothrag.data.preprocess_wikipedia \
        --corpus path/to/corpus.json --output_dir path/to/data_wiki/ --max_docs 100
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


def slugify(name: str) -> str:
    """Canonical entity ID from name."""
    s = re.sub(r"[^\w\s-]", "", name).strip().lower()
    s = re.sub(r"[-\s]+", "-", s)
    return s or "unknown"


def chunk_paragraphs(text: str, min_chars: int = 100, max_chars: int = 1500) -> list[str]:
    """Split text into chunks at paragraph boundaries; merge tiny ones."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else []
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) < max_chars:
            current = (current + "\n\n" + p) if current else p
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    merged = []
    for c in chunks:
        if merged and len(c) < min_chars:
            merged[-1] = merged[-1] + "\n\n" + c
        else:
            merged.append(c)
    return merged


def preprocess(corpus_path: str | Path, output_dir: str | Path, *,
               api_key: str | None = None,
               base_url: str = "https://api.together.xyz/v1",
               model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
               max_docs: int = 0) -> dict:
    """Run preprocessing end-to-end. Resumes from prior partial output."""
    from tqdm import tqdm

    from mothrag.retrieval.openie import OpenIEClient

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    if max_docs > 0:
        corpus = corpus[:max_docs]
    print(f"Loaded {len(corpus)} documents from {corpus_path}")

    client = OpenIEClient(api_key=api_key, base_url=base_url, model=model)

    entities_by_id: dict[str, dict] = {}
    chunks_jsonl: list[dict] = []
    edges_set: set[tuple] = set()
    edges_list: list[dict] = []

    total_tokens_in = total_tokens_out = 0
    total_latency = 0.0
    failed_chunks = 0

    chunks_path = out_dir / "chunks.jsonl"
    done_doc_ids: set[str] = set()
    if chunks_path.exists():
        try:
            for line in chunks_path.open("r", encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "entity_id" in rec:
                    done_doc_ids.add(rec["entity_id"])
                    chunks_jsonl.append(rec)
            print(f"[resume] {len(done_doc_ids)} already-processed doc IDs in {chunks_path}",
                  file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[resume] Failed reading existing chunks ({e}); starting fresh", file=sys.stderr)
            done_doc_ids = set()
            chunks_jsonl = []

        for label, path in (("entities", out_dir / "entities.json"),
                             ("edges", out_dir / "edges.json")):
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if label == "entities":
                        for e in data:
                            entities_by_id[e["id"]] = e
                    else:
                        for ed in data:
                            key = (ed["src"], ed["predicate"], ed["dst"])
                            if key not in edges_set:
                                edges_set.add(key)
                                edges_list.append(ed)
                except Exception as e:  # noqa: BLE001
                    print(f"[resume] Failed to load {label}: {e}", file=sys.stderr)

    def _checkpoint():
        (out_dir / "entities.json").write_text(
            json.dumps(list(entities_by_id.values()), indent=2, ensure_ascii=False),
            encoding="utf-8")
        (out_dir / "edges.json").write_text(
            json.dumps(edges_list, indent=2, ensure_ascii=False),
            encoding="utf-8")

    chunks_fh = chunks_path.open("a", encoding="utf-8")

    pbar = tqdm(corpus, desc="Pre-processing")
    for doc_idx, doc in enumerate(pbar):
        doc_title = doc.get("title", f"doc_{doc_idx}")
        doc_text = doc.get("text", "")
        doc_id = f"doc_{slugify(doc_title)}"
        if doc_id in done_doc_ids:
            continue
        entities_by_id[doc_id] = {
            "id": doc_id, "name": doc_title, "type": "document",
            "summary": doc_text[:300],
        }

        chunks = chunk_paragraphs(doc_text)
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{chunk_idx}"
            try:
                result = client.process(chunk_text, chunk_id=chunk_id)
                total_tokens_in += result.tokens_in
                total_tokens_out += result.tokens_out
                total_latency += result.latency_s

                mentions = []
                for ent_name in result.entities:
                    eid = f"ent_{slugify(ent_name)}"
                    if eid not in entities_by_id:
                        entities_by_id[eid] = {"id": eid, "name": ent_name, "type": "entity"}
                    mentions.append(eid)

                for s, pred, o in result.triples:
                    sid = f"ent_{slugify(s)}"
                    oid = f"ent_{slugify(o)}"
                    if sid not in entities_by_id:
                        entities_by_id[sid] = {"id": sid, "name": s, "type": "entity"}
                    if oid not in entities_by_id:
                        entities_by_id[oid] = {"id": oid, "name": o, "type": "entity"}
                    key = (sid, pred, oid)
                    if key not in edges_set:
                        edges_set.add(key)
                        edges_list.append({"src": sid, "dst": oid, "type": "triple",
                                            "predicate": pred})

                chunk_record = {"id": chunk_id, "text": chunk_text,
                                "entity_id": doc_id, "mentions": mentions}
            except Exception as e:  # noqa: BLE001
                failed_chunks += 1
                print(f"\n  ! chunk {chunk_id} failed: {e}", file=sys.stderr)
                chunk_record = {"id": chunk_id, "text": chunk_text,
                                "entity_id": doc_id, "mentions": []}
            chunks_jsonl.append(chunk_record)
            chunks_fh.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
            chunks_fh.flush()

        pbar.set_postfix({"ents": len(entities_by_id), "edges": len(edges_set),
                          "tok_out": total_tokens_out})

        if (doc_idx + 1) % 20 == 0:
            _checkpoint()

    chunks_fh.close()

    entities_list = list(entities_by_id.values())
    (out_dir / "entities.json").write_text(
        json.dumps(entities_list, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "edges.json").write_text(
        json.dumps(edges_list, indent=2, ensure_ascii=False), encoding="utf-8")

    cost_usd = (total_tokens_in + total_tokens_out) / 1_000_000 * 0.88
    stats = {
        "n_docs": len(corpus),
        "n_chunks": len(chunks_jsonl),
        "n_entities_total": len(entities_list),
        "n_entities_documents": sum(1 for e in entities_list if e["type"] == "document"),
        "n_entities_ner": sum(1 for e in entities_list if e["type"] == "entity"),
        "n_edges": len(edges_list),
        "tokens_in": total_tokens_in,
        "tokens_out": total_tokens_out,
        "total_latency_s": total_latency,
        "failed_chunks": failed_chunks,
        "estimated_cost_usd": round(cost_usd, 4),
        "model": model,
    }
    (out_dir / "openie_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Path to corpus.json (HippoRAG format)")
    ap.add_argument("--output_dir", required=True, help="Output dir for entities/edges/chunks")
    ap.add_argument("--max_docs", type=int, default=0, help="Limit number of docs (0 = no limit)")
    ap.add_argument("--api_key", default=None, help="LLM API key (defaults to env)")
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    ap.add_argument("--base_url", default="https://api.together.xyz/v1")
    args = ap.parse_args()

    stats = preprocess(args.corpus, args.output_dir,
                        api_key=args.api_key, base_url=args.base_url,
                        model=args.model, max_docs=args.max_docs)
    print("\n=== Pre-processing complete ===")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
