# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Entry-point completeness guard.

Every console_script the INSTALLED mothrag distribution declares must
resolve: importing its target module (and looking up the attribute) must
succeed. This reds instantly on the exact defect that shipped in 0.6.0 --
``mothrag = mothrag.cli.main:main`` was declared in pyproject while
``mothrag/cli/main.py`` did not ship, so ``mothrag --help`` raised
ModuleNotFoundError -- even though no other test invoked the entry point.

Runnable standalone against ANY environment's installed mothrag (this is
what the release acceptance gate runs against the built wheel)::

    python tests/test_entry_points.py
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points

_DIST = "mothrag"


def _console_scripts():
    # requires-python >= 3.10, where entry_points() supports keyword selection.
    return [
        ep
        for ep in entry_points(group="console_scripts")
        if ep.value.split(":")[0].split(".")[0] == _DIST
    ]


def check_console_scripts() -> list[str]:
    """Return failure messages; empty list means every script resolves."""
    failures: list[str] = []
    scripts = _console_scripts()
    if not scripts:
        return [f"no {_DIST} console_scripts found in the installed distribution"]
    for ep in scripts:
        module_path, _, attr = ep.value.partition(":")
        try:
            mod = importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001 — any import failure is a real defect
            failures.append(f"{ep.name} = {ep.value}: import failed: {type(exc).__name__}: {exc}")
            continue
        if attr and not hasattr(mod, attr.split(".")[0]):
            failures.append(f"{ep.name} = {ep.value}: {module_path} has no attribute {attr!r}")
    return failures


def test_all_console_scripts_resolve():
    failures = check_console_scripts()
    assert not failures, "Broken console scripts:\n  " + "\n  ".join(failures)


if __name__ == "__main__":
    import sys

    fails = check_console_scripts()
    if fails:
        print("BROKEN console scripts:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print(f"OK: {len(_console_scripts())} {_DIST} console script(s) resolve")
    sys.exit(0)
