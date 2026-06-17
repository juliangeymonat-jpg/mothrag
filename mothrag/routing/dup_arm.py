# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Duplicate-arm pool registration.

Enables a mechanism-attribution test: distinguish whether
the +14pp F1=0 lift on 5-arm pool extension comes from MothGraph
specifically or simply from pool-size redistribution. Test design:
register V3+bu TWICE under different arm IDs (``v3bu_dup_a`` or
``v3bu_dup_b``); if a 4-arm pool ``[v3bu, v3bu_dup_a, decompose, iter]``
shows the same lift as the 5-arm pool with MothGraph, the mechanism
is pure dispatch diversification, not arm utility.

Naming contract:
  * Pattern: ``<base>_dup_<suffix>`` where ``<suffix>`` matches
    ``[a-z0-9]+`` (alphanumeric, single-token).
  * ``<base>`` must be one of the registered legacy arms
    (``v3bu`` / ``decompose`` / ``iter``).
  * Each duplicate is treated by dispatch / arbitration as a
    SEPARATE candidate slot but executes the BASE arm's code path
    (identical inputs -> identical predictions, by design -- the test
    isolates pool-size effects from arm-specific contributions).

Pool-safety axiom interaction:
the dup arm's prediction IS the base arm's prediction. Pairwise
agreement WILL count it as agreeing with the base. THIS IS
INTENTIONAL -- the experiment exists to measure exactly this
double-counting effect against the +14pp pool-extension lift.
"""

from __future__ import annotations

import re


__all__ = [
    "DUP_ARM_RE",
    "LEGACY_BASE_ARMS",
    "is_dup_arm",
    "base_arm_of",
    "validate_dup_arm_name",
]


# Legacy base arms eligible for duplication. Opt-in arms (infobox_arm,
# mothgraph_arm) intentionally NOT included -- the test
# scope is dispatch-diversification vs MothGraph-utility, and the
# legacy bases are the cleanest control variables.
LEGACY_BASE_ARMS = ("v3bu", "decompose", "iter")


# Match e.g. ``v3bu_dup_a`` / ``decompose_dup_b1`` / ``iter_dup_xyz``.
# Suffix restricted to alphanumeric / single-token to keep CLI parsing
# and JSON keying trivial (no special chars).
DUP_ARM_RE = re.compile(
    r"^(?P<base>v3bu|decompose|iter)_dup_(?P<suffix>[a-z0-9]+)$",
)


def is_dup_arm(name: str) -> bool:
    """Return True iff ``name`` matches the dup-arm pattern.

    Example: ``is_dup_arm("v3bu_dup_a") == True``;
    ``is_dup_arm("v3bu") == False``;
    ``is_dup_arm("infobox_arm_dup_a") == False`` (base must be legacy).
    """
    if not name:
        return False
    return bool(DUP_ARM_RE.match(name))


def base_arm_of(name: str) -> str:
    """Return the base arm name for a dup arm, or the input unchanged.

    Raises ``ValueError`` only when ``name`` looks like a dup pattern
    but doesn't match the expected ``<base>_dup_<suffix>`` form with a
    legacy base. Non-dup names pass through unchanged (so callers can
    use this as a normalization helper without branching).

    Examples:
        base_arm_of("v3bu_dup_a") -> "v3bu"
        base_arm_of("v3bu")       -> "v3bu"
        base_arm_of("decompose")  -> "decompose"
    """
    if not name:
        return name
    m = DUP_ARM_RE.match(name)
    if m:
        return m.group("base")
    # Not a dup pattern at all -> pass through.
    if "_dup_" not in name:
        return name
    # Has "_dup_" but doesn't match our legacy-base contract.
    raise ValueError(
        f"Invalid dup-arm name {name!r}: must match "
        f"<base>_dup_<suffix> where base in {LEGACY_BASE_ARMS} and "
        f"suffix is [a-z0-9]+",
    )


def validate_dup_arm_name(name: str) -> None:
    """Strict validator: raise ``ValueError`` if ``name`` is intended to
    be a dup but malformed. Callers (CLI parsers, pool builders) use
    this to surface user errors early.
    """
    if "_dup_" not in name:
        raise ValueError(f"{name!r} is not a dup-arm name")
    if not is_dup_arm(name):
        raise ValueError(
            f"Invalid dup-arm name {name!r}: must match "
            f"<base>_dup_<suffix> where base in {LEGACY_BASE_ARMS} "
            f"and suffix is [a-z0-9]+",
        )
