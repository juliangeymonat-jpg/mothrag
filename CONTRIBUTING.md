# Contributing to MothRag

Thank you for considering a contribution. MothRag is a personal research and engineering project by Julian Geymonat, released under Apache 2.0 and developed in the open. The framework is **domain-agnostic by design**: contributed code, documentation, examples, and benchmarks should remain so.

This guide covers the contribution surface that is most useful at the alpha stage: opening issues, adding adapters, and submitting pull requests.

## Project status

- **Current release**: v0.5.0 alpha (PyPI: `pip install mothrag`).
- **Stability**: alpha — the public API surface (`mothrag.MothRAG`, the reader/embedder/vector-store adapters, and `mothrag.loaders`) may change between minor releases until v1.0.
- **Reproducibility**: the paper headline stack is preserved and tested under `tests/`; see `paper/REPRODUCE.md` for the end-to-end reproduction command and inputs.

## Before you start

1. **Open an issue first** for any change larger than a typo or a docs nit. Linking a PR to a discussion that has already aligned on scope and approach is far more likely to land than a surprise PR.
2. **Check existing issues** to avoid duplicate work.
3. **Respect the project's hard constraints**: keep all contributed content **domain-agnostic** (no company / sport / proprietary references), add the Apache 2.0 header to every new source file, and use conventional commits.

## Development setup

```bash
git clone https://github.com/juliangeymonat-jpg/mothrag.git
cd mothrag
pip install -e ".[dev]"
pytest
```

The `dev` extra includes `pytest`, `pytest-cov`, `ruff`, `build`, `mkdocs`, and `mkdocs-material`. The `prod` extra adds the production reader and embedder dependencies (Gemini, OpenAI, sentence-transformers, scikit-learn, networkx, rank-bm25).

## Code style

- **Python**: ≥ 3.10 supported (target 3.10–3.13).
- **Linter / formatter**: `ruff` configured in `pyproject.toml` with `line-length = 100` and `target-version = "py310"`.
- **Type hints**: use them on public function signatures; `typing-extensions` is a runtime dependency for forward-compatible types.
- **License header**: every new Python source file must start with the Apache 2.0 header used elsewhere in the package:
  ```python
  # Copyright 2026 Julian Geymonat
  # Licensed under the Apache License, Version 2.0
  ```

Before opening a PR:

```bash
ruff check .
ruff format --check .
pytest
```

## Adding a new adapter

Adapters are the integration points between MothRAG's pipeline and external LLM / embedding / vector-store providers. The expected pattern is:

1. Create a new module under the relevant package — `mothrag/readers/` for LLM readers (e.g. `mothrag/readers/myprovider.py`), `mothrag/embedders/` for embedders, `mothrag/vector_dbs/` for vector stores.
2. Subclass the protocol base for that package — e.g. `ReaderAdapter` from `mothrag/readers/base.py`, implementing `complete(messages, ...) -> ReaderResponse` (embedders and stores follow the same pattern in their own `base.py`).
3. Register it in that package's `__init__.py` (and, for top-level access like `mothrag.MyProviderReader`, in the lazy map in `mothrag/__init__.py`) so `MothRAG(..., reader="myprovider:model-id")` resolves correctly.
4. Add a smoke test under `tests/`. It should run against a mocked or recorded response — do **not** hit the live provider in CI.
5. Update `docs/guide/api.md` and the README installation table with the new optional-dependency group if your adapter has additional Python requirements.

A minimal reader-adapter skeleton:

```python
# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0

"""Adapter for MyProvider LLM API."""
from __future__ import annotations

from mothrag.readers.base import ReaderAdapter, ReaderResponse


class MyProviderReader(ReaderAdapter):
    name = "myprovider"

    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        # Resolve the key from an env var here, and lazy-import the provider SDK.

    def complete(self, messages: list[dict], *, max_tokens: int = 1024,
                 temperature: float = 0.0) -> ReaderResponse:
        # Provider-specific HTTP call; respect the base_url allowlist used by existing adapters.
        ...
        return ReaderResponse(text=..., n_input_tokens=..., n_output_tokens=..., raw=...)
```

**Security**: any new adapter that performs HTTP calls to an LLM provider must respect the `base_url` allowlist used by the existing adapters. If your provider host is not on the default allowlist, add it explicitly in your PR and motivate the addition; do **not** broaden the allowlist or bypass it implicitly.

## Adding a new document loader

Loaders convert external corpus formats into the internal `mothrag.loaders.Document` representation. The expected pattern mirrors adapters:

1. Module under `mothrag/loaders/`, e.g. `mothrag/loaders/my_format.py`.
2. Implement the `BaseLoader` protocol from `mothrag/loaders/base.py` (single `load(path) -> Iterable[Document]` method).
3. Register in `mothrag/loaders/__init__.py`.
4. Smoke test under `tests/test_loader_my_format.py` against a tiny sample fixture committed under `tests/fixtures/`.

## Pull request process

1. Branch from `main`. Solo-project velocity is low, so feature branches typically last days, not weeks.
2. Use **conventional commits** for commit messages: `feat:`, `fix:`, `docs:`, `research:`, `chore:`. The CHANGELOG is curated by hand at release time, but conventional commit prefixes make the diff review faster.
3. Reference the issue your PR resolves in the description (`Fixes #N` or `Refs #N`).
4. Run `ruff check .`, `ruff format --check .`, and `pytest` locally. CI runs the same commands.
5. PR review is solo (single maintainer, single approver). Reviews focus on: domain-agnostic content (no company / sport / proprietary references), Apache 2.0 header on new sources, test coverage on the new code path, and consistency with the project's hard rules (above).

## Citations and external claims

Every external citation (paper, repository, benchmark, claim about another system's performance) must be **verifiable from a primary source** at the time the PR is opened. Mark unverified citations with `[unverified]` rather than invent links or numbers. Performance comparisons in the README, `docs/`, or paper artifacts must trace to a peer-reviewed paper or arXiv pre-print.

## Reporting bugs

Please include:

- **Environment**: Python version, OS, `pip freeze | grep mothrag`.
- **Reproduction**: minimum command or code snippet that triggers the bug; corpus fixture if possible.
- **Expected vs observed** behavior, including any error trace.
- **MothRag version**: `python -c "import mothrag; print(mothrag.__version__)"`.

Performance regressions are bugs: include the F1 / EM / latency you observed and the headline number from the README or paper table you were comparing against.

## Questions and discussion

For research-flavoured questions (architectural choices, hypothesis registration, paper-reproducibility), please open a GitHub Discussion rather than an Issue. Issues are for bugs and concrete change proposals; Discussions are for open-ended exchange.

## License

By contributing to MothRag you agree that your contributions are licensed under Apache 2.0, the same license as the project. New files must include the Apache 2.0 header (see Code style above).
