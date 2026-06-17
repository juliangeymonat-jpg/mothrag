# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Smoke test: top-level imports succeed and the public API is reachable."""


def test_top_level_imports():
    import mothrag
    assert mothrag.__version__
    assert hasattr(mothrag, "Anchor")
    assert hasattr(mothrag, "DomainPlugin")
    assert hasattr(mothrag, "EntryPointClassifier")
    assert hasattr(mothrag, "build_anchor_registry")
    assert hasattr(mothrag, "simple")


def test_simple_run_signature():
    from mothrag import simple
    assert callable(simple.run)
    assert "default" in simple.CONFIG_PRESETS
    assert "fast" in simple.CONFIG_PRESETS


def test_subpackage_imports():
    from mothrag.core import Anchor, DomainPlugin
    from mothrag.plugins import WikipediaDomainPlugin
    from mothrag.eval import metrics, soft_em, faithfulness
    from mothrag.retrieval import embeddings, openie, ner

    assert Anchor is not None
    assert DomainPlugin is not None
    assert WikipediaDomainPlugin().name == "wikipedia"
    assert callable(metrics.em_score)
    assert callable(soft_em.soft_em_score)
    assert callable(faithfulness.faithfulness_score)
    assert hasattr(embeddings, "SentenceTransformerEmbedder")
    assert hasattr(openie, "OpenIEClient")
    assert callable(ner.build_ner_cache)
