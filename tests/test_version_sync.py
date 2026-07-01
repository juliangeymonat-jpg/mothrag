# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Version single-source-of-truth guard.

Fails when ``mothrag.__version__``, the installed distribution metadata, and
``CITATION.cff`` disagree -- the drift that let CITATION.cff sit at an old
version while PyPI shipped a newer one. Runnable standalone::

    python tests/test_version_sync.py
"""
from __future__ import annotations

import re
from importlib.metadata import version as dist_version
from pathlib import Path

import mothrag

_ROOT = Path(__file__).resolve().parent.parent


def _citation_version() -> str | None:
    cff = _ROOT / "CITATION.cff"
    if not cff.exists():
        return None
    m = re.search(r'^version:\s*["\']?([^"\'\n]+)', cff.read_text(encoding="utf-8"), re.M)
    return m.group(1).strip() if m else None


def test_dunder_matches_installed_metadata():
    assert mothrag.__version__ == dist_version("mothrag"), (
        f"mothrag.__version__ {mothrag.__version__!r} != installed metadata "
        f"{dist_version('mothrag')!r}"
    )


def test_citation_cff_matches_version():
    cffv = _citation_version()
    if cffv is None:
        return  # CITATION.cff not on this path (e.g. installed wheel) -> skip
    assert cffv == mothrag.__version__, (
        f"CITATION.cff version {cffv!r} != mothrag.__version__ "
        f"{mothrag.__version__!r}"
    )


if __name__ == "__main__":
    import sys

    problems: list[str] = []
    try:
        meta = dist_version("mothrag")
    except Exception as exc:  # noqa: BLE001
        meta = f"<error: {exc}>"
    if mothrag.__version__ != meta:
        problems.append(f"__version__ {mothrag.__version__!r} != installed metadata {meta!r}")
    cffv = _citation_version()
    if cffv is not None and cffv != mothrag.__version__:
        problems.append(f"CITATION.cff {cffv!r} != __version__ {mothrag.__version__!r}")
    if problems:
        print("VERSION DRIFT:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"OK: version {mothrag.__version__} consistent "
          f"(metadata + {'CITATION.cff' if cffv else 'no CITATION.cff'})")
    sys.exit(0)
