# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Domain plugins for MothRAG.

Reference plugin:
  - :class:`mothrag.plugins.wikipedia.WikipediaDomainPlugin` for HotpotQA /
    2WikiMultiHopQA / MuSiQue benchmarks.

Custom plugins must satisfy the :class:`mothrag.core.DomainPlugin` protocol.
"""

from mothrag.plugins.wikipedia import WikipediaDomainPlugin

__all__ = ["WikipediaDomainPlugin"]
