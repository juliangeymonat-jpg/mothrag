# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Tests for :mod:`mothrag.core.decompose` (parser + offline behaviour)."""

from mothrag.core.decompose import _parse_sub_qs


def test_parse_plain_json_list():
    assert _parse_sub_qs('["a?", "b?"]') == ["a?", "b?"]


def test_parse_strips_markdown_fence():
    assert _parse_sub_qs('```json\n["a?"]\n```') == ["a?"]


def test_parse_handles_trailing_text():
    assert _parse_sub_qs('Sure, here you go: ["x?", "y?"] (done)') == ["x?", "y?"]


def test_parse_returns_empty_on_invalid_json():
    assert _parse_sub_qs("not a list at all") == []


def test_parse_rejects_non_string_items():
    assert _parse_sub_qs("[1, 2, 3]") == []
