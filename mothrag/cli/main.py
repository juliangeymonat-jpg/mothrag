# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Primary ``mothrag`` command-line entry point.

Subcommands:

- ``mothrag query "<question>" --text "..." [--docs PATH] [--json]`` --
  build an in-memory index over the supplied documents and answer a
  question with the high-level :class:`mothrag.MothRAG` API.
- ``mothrag smoke ...`` -- forward to the HotpotQA-style smoke eval
  (identical to the ``mothrag-smoke`` console script).
- ``mothrag version`` / ``mothrag --version`` -- print the installed version.

This module ships the ``mothrag = mothrag.cli.main:main`` console script
declared in ``pyproject.toml``. It intentionally does the loud thing that
the library layer cannot: when no real LLM reader is configured, it warns
on stderr that answers are echo fallbacks (the top retrieved chunk), so a
first-time user is never misled into reading a chunk-echo as a generated
answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from mothrag import __version__


def _reader_diagnostics() -> tuple[str, str | None]:
    """Classify the default-reader state the same way the library resolves it.

    Returns ``(mode, warning)`` where ``mode`` is one of ``"llm"``,
    ``"echo"`` (no key) or ``"key-but-no-sdk"`` (a key IS set but the
    OpenAI-compatible reader SDK is missing, so the library silently falls
    back to echo). ``warning`` is a human actionable string, or ``None``
    when a real reader will run.
    """
    key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("GROQ_API_KEY")
    try:
        from mothrag.core._api_adapters import OpenAICompatibleReader  # noqa: F401
        adapter_ok = True
    except Exception:  # noqa: BLE001 — any import failure means no real reader
        adapter_ok = False

    if key and adapter_ok:
        return "llm", None
    if key and not adapter_ok:
        return (
            "key-but-no-sdk",
            "An API key is set but the reader SDK is missing, so MothRAG is "
            "silently using the echo fallback (NO LLM call). Install it with: "
            "pip install 'mothrag[openai]'",
        )
    return (
        "echo",
        "No reader API key set (TOGETHER_API_KEY / GROQ_API_KEY). Answers are "
        "the top retrieved chunk echoed back, NOT LLM-generated. For real "
        "answers set GROQ_API_KEY (free tier) and: pip install 'mothrag[openai]'",
    )


def _cmd_query(args: argparse.Namespace) -> int:
    from mothrag import MothRAG

    if not args.docs and not args.text:
        print(
            "mothrag query: no documents supplied. Add at least one "
            "--text \"...\" or --docs PATH (a file or directory).",
            file=sys.stderr,
        )
        return 2

    mode, warning = _reader_diagnostics()
    if warning:
        print(f"NOTE: {warning}", file=sys.stderr)

    rag = MothRAG(
        embedder=args.embedder,          # None → auto-default chain
        production=args.production,
    )
    for path in args.docs:
        rag.ingest(path)
    if args.text:
        rag.ingest(list(args.text))

    query_kwargs: dict[str, Any] = {}
    if args.top_k is not None:
        query_kwargs["top_k"] = args.top_k
    result = rag.query(args.question, **query_kwargs)

    if args.json:
        print(json.dumps({
            "question": args.question,
            "answer": result.answer,
            "arm_used": result.arm_used,
            "arm_subset": result.arm_subset,
            "confidence": result.confidence,
            "n_retrieved": len(result.retrieved_chunks),
            "reader_mode": mode,
        }, ensure_ascii=False, indent=2))
    else:
        print(result.answer)
        print(
            f"[arm={result.arm_used or 'default'} "
            f"confidence={result.confidence} "
            f"retrieved={len(result.retrieved_chunks)} reader={mode}]",
            file=sys.stderr,
        )
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    """Forward to the HotpotQA-style smoke eval (the ``mothrag-smoke`` script)."""
    from mothrag.cli.smoke import main as smoke_main

    forwarded = list(args.smoke_args or [])
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    saved = sys.argv
    sys.argv = ["mothrag-smoke", *forwarded]
    try:
        return int(smoke_main() or 0)
    finally:
        sys.argv = saved


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"mothrag {__version__}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mothrag",
        description="MothRAG: adaptive multi-arm subset routing for multi-hop RAG.",
    )
    parser.add_argument(
        "--version", action="version", version=f"mothrag {__version__}",
    )
    sub = parser.add_subparsers(dest="command", metavar="{query,smoke,version}")

    q = sub.add_parser("query", help="Answer a question over supplied documents.")
    q.add_argument("question", help="The question to answer.")
    q.add_argument(
        "--text", action="append", metavar="TEXT",
        help="Inline document text (repeatable).",
    )
    q.add_argument(
        "--docs", action="append", default=[], metavar="PATH",
        help="Path to a document file or directory (repeatable).",
    )
    q.add_argument("--top-k", type=int, default=None, help="Chunks to retrieve.")
    q.add_argument(
        "--embedder", default=None, metavar="SPEC",
        help="Embedder spec, e.g. 'st' (offline), 'gemini', 'openai'. "
             "Default: auto (gemini → sentence-transformers → hash).",
    )
    q.add_argument(
        "--production", action="store_true",
        help="Run the full multi-arm orchestration (needs a real reader).",
    )
    q.add_argument("--json", action="store_true", help="Emit a JSON result.")
    q.set_defaults(func=_cmd_query)

    s = sub.add_parser(
        "smoke", help="Run the HotpotQA-style smoke eval (see also mothrag-smoke).",
    )
    s.add_argument("smoke_args", nargs=argparse.REMAINDER,
                   help="Arguments forwarded to mothrag-smoke.")
    s.set_defaults(func=_cmd_smoke)

    v = sub.add_parser("version", help="Print the installed MothRAG version.")
    v.set_defaults(func=_cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
