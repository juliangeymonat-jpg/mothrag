# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Version single-source-of-truth guard.

Fails when ``mothrag.__version__``, the installed distribution metadata, and
``CITATION.cff`` disagree -- the drift that let CITATION.cff sit at an old
version while PyPI shipped a newer one. Runnable standalone (the release
acceptance gate runs this file against the built wheel)::

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
        f"CITATION.cff version {cffv!r} != mothrag.__version__ {mothrag.__version__!r}"
    )


if __name__ == "__main__":
    # Standalone runner = the SAME assertions as pytest, not a parallel copy.
    import sys

    problems: list[str] = []
    for check in (test_dunder_matches_installed_metadata, test_citation_cff_matches_version):
        try:
            check()
        except AssertionError as exc:
            problems.append(str(exc).splitlines()[0])
    if problems:
        print("VERSION DRIFT:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"OK: version {mothrag.__version__} consistent")
    sys.exit(0)
