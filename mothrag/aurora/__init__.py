# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Aurora primitives integrated into MothRAG (γ verifier + C7 cancellation).

Attribution: the deterministic proof-tree verifier (γ) and phase-cancellation
chain filter (C7) primitives originate in the Aurora project, an independent
research program by the same author. Aurora is not open-source; γ and C7 are
released here as part of the MothRAG framework under Apache 2.0 license.

Module mapping (Aurora codenames ↔ paper-friendly names):
- ``rules``             — ProofTree dataclass + PROOF_TREE_SYSTEM_PROMPT (γ)
- ``verifier``          — deterministic proof-tree verifier (γ-strict)
- ``verifier_liberal``  — liberal faithfulness variant (γ-liberal)
- ``c7``                — phase-cancellation chain filter (C7, Method-D auto-phase)

First public disclosure of γ + C7. See `mothrag/aurora/README.md` for spec.
"""
from mothrag.aurora.rules import (
    ProofTree, ProofStep, Source,
    PROOF_TREE_SYSTEM_PROMPT, PROOF_TREE_SYSTEM_PROMPT_LLAMA,
    proof_tree_user_prompt,
)
from mothrag.aurora.verifier import verify_proof_tree
from mothrag.aurora.verifier_liberal import (
    liberal_overall_status, liberal_status_distribution,
)
from mothrag.aurora.c7 import c7_aurora_rejected_chains

__all__ = [
    "ProofTree", "ProofStep", "Source",
    "PROOF_TREE_SYSTEM_PROMPT", "PROOF_TREE_SYSTEM_PROMPT_LLAMA",
    "proof_tree_user_prompt",
    "verify_proof_tree",
    "liberal_overall_status", "liberal_status_distribution",
    "c7_aurora_rejected_chains",
]
