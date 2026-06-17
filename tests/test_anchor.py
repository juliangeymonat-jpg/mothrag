# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Unit tests for :class:`mothrag.core.Anchor`."""

import numpy as np

from mothrag import Anchor


def test_anchor_basic():
    a = Anchor(anchor_id="L2_doc_x", layer=2, members={"doc_x", "ent_y"},
               scope_text="Document X")
    assert a.anchor_id == "L2_doc_x"
    assert a.layer == 2
    assert "doc_x" in a.members
    assert a.scope_vec is None
    assert "L2" in repr(a)


def test_anchor_with_parent_and_tenant():
    a = Anchor(anchor_id="L3_doc_x_para_0", layer=3, members={"doc_x"},
               scope_text="Para 0", parent="L2_doc_x", tenant_id="acme")
    assert a.parent == "L2_doc_x"
    assert a.tenant_id == "acme"


def test_anchor_scope_vec_assignment():
    a = Anchor(anchor_id="L1_root", layer=1, members={"x"}, scope_text="Root")
    a.scope_vec = np.zeros(8, dtype=np.float32)
    assert a.scope_vec.shape == (8,)
    assert a.scope_vec.dtype == np.float32
