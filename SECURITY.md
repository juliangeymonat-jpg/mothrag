# Security Policy

## Supported versions

Only the **latest release on PyPI** receives security fixes. Older releases are not patched; upgrade with `pip install mothrag` (add `-U` to force the upgrade).

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Report it privately through GitHub Security Advisories:
**[Report a vulnerability](https://github.com/juliangeymonat-jpg/mothrag/security/advisories/new)** (repository → Security → Report a vulnerability).

You will normally get a first response within **7 days**. Once a fix ships, the advisory is published and you are credited (unless you prefer otherwise).

## Scope notes

What is definitely in scope for this project:

- **Unexpected outbound calls.** MothRag must only call the LLM / embedding provider endpoints the user configured; the high-level API and the eval pipeline additionally validate provider base URLs against an allowlist (`mothrag/utils/url_safety.py`). Any way a crafted corpus or query (data, as opposed to the user's own configuration) can redirect a call to another host, or bypass the allowlist where it is enforced, is a vulnerability.
- **Secret handling.** Provider API keys are read from environment variables and must never be written to disk or logs. Any code path that persists, logs, or transmits a key anywhere other than the intended provider endpoint is a vulnerability.
- **Corpus-driven code execution.** Documents are data. Any way a document's content can lead to code execution, file writes outside expected output paths, or reads of files the user did not pass in is a vulnerability.

Prompt-injection *content* risks (a malicious document steering the model's answer text) are a known, documented property of every RAG system and are out of scope unless they cross into one of the categories above.
