# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Corpus preprocessing utilities (Wikipedia / HotpotQA / 2Wiki / MuSiQue)."""

from mothrag.data.preprocess_wikipedia import (
    chunk_paragraphs,
    slugify,
)

__all__ = ["chunk_paragraphs", "slugify"]
