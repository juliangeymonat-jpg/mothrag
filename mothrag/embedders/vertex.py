# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Vertex AI embedder — Google Cloud GDPR / SOC2 enterprise backend.

Targets the same Gemini-family embedding models as :class:`GeminiEmbedder`
but routes through Google Cloud Vertex AI instead of Generative Language
Studio. Use this adapter when:

- The data must stay within a specific GCP region (default ``europe-west4``)
  for GDPR / data-residency compliance.
- The deployment uses GCP service-account auth (``GOOGLE_APPLICATION_CREDENTIALS``)
  rather than a Studio API key.
- The customer already has a Vertex AI quota / billing setup and wants a
  single GCP invoice rather than a separate Studio key.

Auth resolution order:

1. Explicit ``credentials_path`` kwarg (path to service-account JSON).
2. ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable.
3. Application Default Credentials (workload identity / ``gcloud auth``).

Project resolution order:

1. Explicit ``project`` kwarg.
2. ``VERTEX_AI_PROJECT`` environment variable.
3. ``GOOGLE_CLOUD_PROJECT`` environment variable.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from mothrag.embedders.base import EmbedderAdapter

# Output dimensionality for the canonical Vertex text-embedding models.
# Note: ``gemini-embedding-2`` (Google's current production model used by
# :class:`GeminiEmbedder`) is NOT exposed on Vertex AI yet (404). Vertex
# users must pick a different model — default below = text-embedding-005.
_DIMS = {
    "text-embedding-005":         768,
    "text-embedding-004":         768,
    "textembedding-gecko@003":    768,
    "textembedding-gecko-multilingual@001": 768,
}

_DEFAULT_MODEL = "text-embedding-005"  # gemini-embedding-2 not on Vertex yet
_DEFAULT_LOCATION = "europe-west4"  # GDPR-friendly default.
_DEFAULT_TASK_TYPE = "SEMANTIC_SIMILARITY"


class VertexEmbedder(EmbedderAdapter):
    """Google Cloud Vertex AI text-embedding adapter.

    Defaults to ``text-embedding-005`` (768-d, GA) because Google has not yet
    released ``gemini-embedding-2`` on Vertex AI (only Studio API). If you need
    the higher-quality -2 model, use :class:`GeminiEmbedder` via Studio API.

    Parameters
    ----------
    model
        Vertex text-embedding model ID. Defaults to ``text-embedding-005``
        (768-d, GA). Other options: ``textembedding-gecko@003`` (768-d, legacy GA).
    project
        GCP project ID. If omitted, resolves from ``VERTEX_AI_PROJECT`` then
        ``GOOGLE_CLOUD_PROJECT`` env vars.
    location
        Vertex region. Defaults to ``europe-west4`` for GDPR alignment;
        override to ``us-central1`` etc. if the model is not enabled in
        the EU region.
    credentials_path
        Optional path to a service-account JSON file. If omitted, the SDK
        uses Application Default Credentials (``GOOGLE_APPLICATION_CREDENTIALS``
        env var or gcloud / workload-identity discovery).
    task_type
        Embedding task type. Defaults to ``SEMANTIC_SIMILARITY`` to match
        :class:`GeminiEmbedder` behaviour; use ``RETRIEVAL_DOCUMENT`` for
        corpus ingestion and ``RETRIEVAL_QUERY`` for queries when running
        the asymmetric retrieval protocol.
    output_dimensionality
        Optional embedding truncation (model-dependent support).
        ``None`` → full model dimension.
    """

    name = "vertex"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        project: str | None = None,
        location: str = _DEFAULT_LOCATION,
        credentials_path: str | None = None,
        task_type: str = _DEFAULT_TASK_TYPE,
        output_dimensionality: int | None = None,
    ) -> None:
        try:
            import vertexai  # noqa: F401
            from vertexai.language_models import TextEmbeddingModel  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "VertexEmbedder requires `google-cloud-aiplatform`. "
                "Install via `pip install mothrag[vertex]` or "
                "`pip install google-cloud-aiplatform`."
            ) from e

        import vertexai
        from vertexai.language_models import TextEmbeddingModel

        resolved_project = project or os.environ.get("VERTEX_AI_PROJECT") \
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not resolved_project:
            raise RuntimeError(
                "VertexEmbedder needs a GCP project. Pass `project=...` or set "
                "the VERTEX_AI_PROJECT (or GOOGLE_CLOUD_PROJECT) env var."
            )

        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        vertexai.init(project=resolved_project, location=location)
        self.model = model
        self.project = resolved_project
        self.location = location
        self.task_type = task_type
        self.output_dimensionality = output_dimensionality
        self._client = TextEmbeddingModel.from_pretrained(model)

    @property
    def dim(self) -> int:
        if self.output_dimensionality is not None:
            return self.output_dimensionality
        return _DIMS.get(self.model, 768)

    def embed(self, texts: Sequence[str], *, batch_size: int = 32):
        from vertexai.language_models import TextEmbeddingInput

        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        out: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            window = texts[start:start + batch_size]
            inputs = [TextEmbeddingInput(t, self.task_type) for t in window]
            kwargs: dict = {}
            if self.output_dimensionality is not None:
                kwargs["output_dimensionality"] = self.output_dimensionality
            embeddings = self._client.get_embeddings(inputs, **kwargs)
            for emb in embeddings:
                v = np.asarray(emb.values, dtype=np.float32)
                norm = np.linalg.norm(v) or 1.0
                out.append(v / norm)
        return np.stack(out, axis=0)


__all__ = ["VertexEmbedder"]
