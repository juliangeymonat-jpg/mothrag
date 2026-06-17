# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Anchor data structure — domain-agnostic primitive.

An Anchor is a stable named subset of the knowledge graph at a specific scope/layer.
Layer numbering is plugin-defined (e.g. WikipediaDomainPlugin uses layers 1-5
spanning topic clusters down to entity ego-graphs).
"""

import numpy as np


class Anchor:
    __slots__ = ("anchor_id", "layer", "parent", "members", "scope_text", "scope_vec", "tenant_id")

    def __init__(self, anchor_id, layer, members, scope_text, parent=None, tenant_id=None):
        self.anchor_id = anchor_id
        self.layer = layer
        self.parent = parent
        self.members = members  # set[str] entity_ids
        self.scope_text = scope_text
        self.scope_vec: np.ndarray | None = None
        self.tenant_id = tenant_id

    def __repr__(self):
        return f"Anchor({self.anchor_id}, L{self.layer}, |members|={len(self.members)})"
