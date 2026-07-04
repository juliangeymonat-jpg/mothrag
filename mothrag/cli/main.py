# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Primary ``mothrag`` command-line entry point.

Subcommands:

- ``mothrag query "<question>" --text "..." [--docs PATH] [--json]`` --
  build an in-memory index over the supplied documents and answer a
  question with the high-level :class:`mothrag.MothRAG` API.
- ``mothrag smoke ...`` -- forward everything after ``smoke`` to the
  HotpotQA-style smoke eval (identical to the ``mothrag-smoke`` console
  script, including ``-h``).
- ``mothrag version`` / ``mothrag --version`` -- print the installed version.

This module ships the ``mothrag = mothrag.cli.main:main`` console script
declared in ``pyproject.toml``. Reader diagnostics are derived from the
RESOLVED reader instance (ground truth), not from a parallel re-implementation
of the library's resolution logic, so the CLI cannot report ``llm`` while the
library actually runs the echo fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from mothrag import __version__

_VERSION_STRING = f"mothrag {__version__}"


def _reader_note(reader: Any) -> tuple[str, str | None]:
    """Classify the reader instance the pipeline will actually use.

    Returns ``(mode, warning)``: ``mode`` is ``"llm"`` or ``"echo"``;
    ``warning`` is an actionable message for the echo cases (no key at all,
    or key set but the reader SDK missing so the library fell back).
    """
    if type(reader).__name__ != "_EchoReader":
        return "llm", None
    if os.environ.get("TOGETHER_API_KEY") or os.environ.get("GROQ_API_KEY"):
        return (
            "echo",
            "An API key is set but the reader SDK is missing, so answers are the "
            "echoed top chunk (NO LLM call). Fix: pip install 'mothrag[openai]'",
        )
    return (
        "echo",
        "No reader API key set (TOGETHER_API_KEY / GROQ_API_KEY). Answers are "
        "the top retrieved chunk echoed back, NOT LLM-generated. For real "
        "answers set GROQ_API_KEY (free tier available) and: "
        "pip install 'mothrag[openai]'",
    )


def _cmd_query(args: argparse.Namespace) -> int:
    from mothrag import MothRAG

    if not args.docs and not args.text:
        print(
            "mothrag query: no documents supplied. Add at least one "
            '--text "..." or --docs PATH (a file or directory).',
            file=sys.stderr,
        )
        return 2
    if args.top_k is not None and args.top_k < 1:
        print("mothrag query: --top-k must be a positive integer.", file=sys.stderr)
        return 2
    missing = [p for p in args.docs if not Path(p).exists()]
    if missing:
        print(
            "mothrag query: --docs path not found: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 2

    rag = MothRAG(
        embedder=args.embedder,  # None → auto-default chain
        production=args.production,
    )
    mode, warning = _reader_note(rag.reader)
    if warning:
        print(f"NOTE: {warning}", file=sys.stderr)

    # --require-llm lets automation opt into strictness: fail loudly rather than
    # silently answer with the echo fallback (which returns exit 0 like a real
    # LLM success, so a scripted caller cannot otherwise tell them apart).
    if getattr(args, "require_llm", False) and mode != "llm":
        print(
            "mothrag query: --require-llm is set but no LLM reader is active "
            "(the echo fallback would run). Set a reader key (GROQ_API_KEY) and "
            "install the SDK: pip install 'mothrag[openai]'.",
            file=sys.stderr,
        )
        return 4

    # The reader NOTE covers answer quality; retrieval quality has its own trap:
    # with no embedding backend the library silently drops to the offline hash
    # embedder, so a user could get real LLM answers over near-random retrieval.
    from mothrag.core.api import _HashEmbedder

    if isinstance(rag.embedder, _HashEmbedder):
        print(
            "NOTE: using the offline hash embedder (low retrieval quality, for "
            "smoke tests only). For production retrieval set GEMINI_API_KEY and "
            "pip install 'mothrag[prod]', or pass --embedder st for a local model.",
            file=sys.stderr,
        )

    # Loader failures (missing optional extra, unsupported extension, a BOM/bad
    # file) carry helpful messages but would otherwise escape as a raw traceback;
    # surface them as a single clean CLI error, like the other guards above.
    try:
        for path in args.docs:
            rag.ingest(path)
        if args.text:
            rag.ingest(list(args.text))
    except (
        ImportError,
        ValueError,
        NotImplementedError,
        FileNotFoundError,
        UnicodeError,
        OSError,
    ) as exc:
        print(f"mothrag query: could not load documents: {exc}", file=sys.stderr)
        return 2

    # A directory of unsupported types (or only-empty files) is silently skipped
    # during ingest, leaving an empty index that would answer "[no passages
    # retrieved]" with exit 0, a misleading success. Turn it into a clear error.
    if len(rag.retriever) == 0:
        from mothrag.loaders import _EXT_DISPATCH

        exts = ", ".join(sorted(_EXT_DISPATCH))
        print(
            "mothrag query: no searchable text was loaded (the files were empty "
            f"or an unsupported type; nothing to search). Supported: {exts}.",
            file=sys.stderr,
        )
        return 2

    query_kwargs: dict[str, Any] = {}
    if args.top_k is not None:
        query_kwargs["top_k"] = args.top_k
    result = rag.query(args.question, **query_kwargs)

    # A real reader that produced an empty answer almost always means the API
    # call failed (revoked key, network, rate limit): the adapters swallow the
    # exception and return "". Do not let that look like a clean success.
    empty_llm_answer = mode == "llm" and not (result.answer or "").strip()
    if empty_llm_answer:
        print(
            "WARNING: the reader returned an empty answer. The LLM call likely "
            "failed (invalid/expired API key, network, or rate limit); see the "
            "log lines above.",
            file=sys.stderr,
        )

    if args.json:
        print(
            json.dumps(
                {
                    "question": args.question,
                    "answer": result.answer,
                    "arm_used": result.arm_used,
                    "arm_subset": result.arm_subset,
                    "confidence": result.confidence,
                    "n_retrieved": len(result.retrieved_chunks),
                    "reader_mode": mode,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(result.answer)
        print(
            f"[arm={result.arm_used or 'default'} "
            f"confidence={result.confidence} "
            f"retrieved={len(result.retrieved_chunks)} reader={mode}]",
            file=sys.stderr,
        )
    return 3 if empty_llm_answer else 0


def _sample_corpus_dir() -> Path:
    """Absolute path to the demo corpus bundled inside the installed package."""
    return Path(__file__).resolve().parent.parent / "data" / "sample_corpus"


_DEMO_QUESTION = "Who was the Eiffel Tower named after, and in which country does it stand?"


def _cmd_demo(args: argparse.Namespace) -> int:
    corpus = _sample_corpus_dir()
    if not corpus.is_dir():
        print(
            f"mothrag demo: bundled sample corpus not found at {corpus} (reinstall mothrag).",
            file=sys.stderr,
        )
        return 2
    question = args.question or _DEMO_QUESTION
    print(f"mothrag demo: corpus = {corpus}", file=sys.stderr)
    print(f'mothrag demo: question = "{question}"', file=sys.stderr)
    print(
        f'(try your own: mothrag query "<question>" --docs {corpus})',
        file=sys.stderr,
    )
    # Reuse the full query path (reader/embedder NOTEs, error handling, exit codes).
    delegated = argparse.Namespace(
        question=question,
        text=None,
        docs=[str(corpus)],
        top_k=None,
        embedder=args.embedder,
        production=False,
        require_llm=False,
        json=args.json,
    )
    return _cmd_query(delegated)


def _cmd_version(_args: argparse.Namespace) -> int:
    print(_VERSION_STRING)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mothrag",
        description="MothRAG: adaptive multi-arm subset routing for multi-hop RAG.",
        epilog=(
            "smoke: everything after `mothrag smoke` is forwarded verbatim to the "
            "mothrag-smoke eval (try `mothrag smoke -h`)."
        ),
    )
    parser.add_argument("--version", action="version", version=_VERSION_STRING)
    sub = parser.add_subparsers(dest="command", metavar="{query,demo,smoke,version}")

    q = sub.add_parser("query", help="Answer a question over supplied documents.")
    q.add_argument("question", help="The question to answer.")
    q.add_argument(
        "--text",
        action="append",
        metavar="TEXT",
        help="Inline document text (repeatable).",
    )
    q.add_argument(
        "--docs",
        action="append",
        default=[],
        metavar="PATH",
        help="Path to a document file or directory (repeatable).",
    )
    q.add_argument("--top-k", type=int, default=None, help="Chunks to retrieve.")
    q.add_argument(
        "--embedder",
        default=None,
        metavar="SPEC",
        help="Embedder spec, e.g. 'st' (offline), 'gemini', 'openai'. "
        "Default: auto (gemini > sentence-transformers > hash).",
    )
    q.add_argument(
        "--production",
        action="store_true",
        help="Run the full multi-arm orchestration (needs a real reader).",
    )
    q.add_argument(
        "--require-llm",
        action="store_true",
        dest="require_llm",
        help="Exit non-zero (4) instead of answering with the echo fallback "
        "when no LLM reader is active (for scripts/CI).",
    )
    q.add_argument("--json", action="store_true", help="Emit a JSON result.")
    q.set_defaults(func=_cmd_query)

    d = sub.add_parser(
        "demo",
        help="Answer a sample multi-hop question over the bundled demo corpus.",
    )
    d.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Optional question (default: a canned Eiffel Tower multi-hop query).",
    )
    d.add_argument(
        "--embedder",
        default=None,
        metavar="SPEC",
        help="Embedder spec (default: auto). Use 'st' for a local offline model.",
    )
    d.add_argument("--json", action="store_true", help="Emit a JSON result.")
    d.set_defaults(func=_cmd_demo)

    v = sub.add_parser("version", help="Print the installed MothRAG version.")
    v.set_defaults(func=_cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Never crash on console encodings that cannot represent the answer
    # (classic Windows cp1252-when-piped trap): degrade to replacement chars.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:  # noqa: BLE001 — best-effort, never fatal
                pass

    args_list = list(sys.argv[1:] if argv is None else argv)

    # `smoke` is dispatched manually: argparse.REMAINDER cannot capture a
    # LEADING option token (so `mothrag smoke --corpus X` would die with
    # "unrecognized arguments" at the top-level parser). Forward verbatim,
    # including -h, to the real mothrag-smoke parser.
    if args_list and args_list[0] == "smoke":
        forwarded = args_list[1:]
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        from mothrag.cli.smoke import main as smoke_main

        return int(smoke_main(forwarded) or 0)

    parser = _build_parser()
    args = parser.parse_args(args_list)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
