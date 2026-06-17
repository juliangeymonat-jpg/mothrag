# Vertex AI setup (enterprise / GDPR)

MothRag ships two Google embedding backends:

| Backend | Adapter | Auth | Region control | Use when |
|---|---|---|---|---|
| Gemini Studio | `GeminiEmbedder` | `GEMINI_API_KEY` | none (Google-managed) | personal / research / fastest setup |
| Vertex AI | `VertexEmbedder` | GCP service account / ADC | yes (per-project, per-region) | enterprise / GDPR / SOC2 / single-invoice |

⚠️ **Embedder asymmetry**: Studio exposes `gemini-embedding-2` (3072-d, MOTHRAG production default); Vertex AI does **not** ship that model yet (as of 2026-05). VertexEmbedder therefore defaults to `text-embedding-005` (768-d, GA). For headline F1 numbers (3072-d gemini-embedding-2) use Studio; for GDPR / data-residency / SOC2 use Vertex with text-embedding-005. The two paths produce different embedding spaces and are not interchangeable on a single corpus.

## 1. Provision a GCP project

```bash
gcloud projects create my-mothrag-project
gcloud config set project my-mothrag-project
gcloud services enable aiplatform.googleapis.com
```

Confirm Vertex AI is enabled in your target region. The default region in `VertexEmbedder` is **`europe-west4`** (GDPR-aligned, low-latency for EU customers); override via the `location` kwarg or by passing `VertexEmbedder(location="us-central1")`.

## 2. Create a service account

```bash
gcloud iam service-accounts create mothrag-embedder \
    --display-name "MothRag embedding service account"

gcloud projects add-iam-policy-binding my-mothrag-project \
    --member "serviceAccount:mothrag-embedder@my-mothrag-project.iam.gserviceaccount.com" \
    --role "roles/aiplatform.user"

gcloud iam service-accounts keys create ./mothrag-sa.json \
    --iam-account mothrag-embedder@my-mothrag-project.iam.gserviceaccount.com
```

`roles/aiplatform.user` is the minimum needed to call `TextEmbeddingModel.get_embeddings(...)`. For production, prefer **Workload Identity** (no JSON keys) on GKE / Cloud Run; the adapter picks up Application Default Credentials automatically.

## 3. Install and configure

```bash
pip install mothrag[enterprise]    # Vertex + production retrieval stack
# or
pip install mothrag[vertex]        # bare adapter only
```

Configure via env vars (recommended) or constructor kwargs:

```bash
export VERTEX_AI_PROJECT=my-mothrag-project
export GOOGLE_APPLICATION_CREDENTIALS=$PWD/mothrag-sa.json
```

```python
from mothrag import MothRAG

# Env-var auto-resolve (preferred for 12-factor apps)
rag = MothRAG.from_documents("docs/")

# Explicit constructor (preferred when the same process serves multiple tenants)
from mothrag.embedders import VertexEmbedder
rag = MothRAG.from_documents(
    "docs/",
    embedder=VertexEmbedder(
        model="text-embedding-005",
        project="my-mothrag-project",
        location="europe-west4",
        credentials_path="/secrets/mothrag-sa.json",
    ),
)

# String spec (one-liner)
rag = MothRAG.from_documents("docs/", embedder="vertex:text-embedding-005")
```

## 4. Model selection

| Model | Dim | Notes |
|---|---:|---|
| `text-embedding-005` | 768 | **VertexEmbedder default**; GA in all regions; cheaper, faster |
| `text-embedding-004` | 768 | Legacy GA |
| `textembedding-gecko@003` | 768 | Legacy English-only; available in regions where 004 / 005 are not yet rolled out |
| `textembedding-gecko-multilingual@001` | 768 | Multilingual legacy model |

Note: `gemini-embedding-2` (3072-d, MOTHRAG production headline) is currently Studio-only. Watch [Vertex AI release notes](https://cloud.google.com/vertex-ai/docs/release-notes) for availability.

Match the embedding dimensionality to your existing vector store schema. All current Vertex-available models are 768-d.

## 5. Cost comparison (rough order-of-magnitude)

Both Studio and Vertex bill per million tokens; pricing has been close historically but the SKUs are billed independently and update on different schedules. As of 2026-Q2:

- Studio `gemini-embedding-2`: charged on the Generative Language API line item.
- Vertex `text-embedding-005`: charged on the Vertex AI prediction line item.

Single-invoice consolidation is the practical reason to choose Vertex even when per-token pricing matches — finance teams want one SKU per cloud vendor, not two. Always re-check current pricing at <https://cloud.google.com/vertex-ai/pricing> before signing enterprise contracts.

## 6. Region selection

| Region | Latency from | Compliance | Recommended |
|---|---|---|---|
| `europe-west4` (Netherlands) | EU | GDPR | EU customers, default |
| `europe-west1` (Belgium) | EU | GDPR | EU fallback |
| `us-central1` (Iowa) | NA | — | US customers, widest model availability |
| `asia-southeast1` (Singapore) | APAC | — | APAC customers |

If a model is not enabled in your preferred region, the adapter raises a clear error from the Vertex SDK; fall back to `us-central1` for the broadest model availability while preparing the multi-region rollout.

## 7. Troubleshooting

- **`google.auth.exceptions.DefaultCredentialsError`** — set `GOOGLE_APPLICATION_CREDENTIALS` to the JSON path or run `gcloud auth application-default login`.
- **`PermissionDenied`** — service account is missing `roles/aiplatform.user`. Re-bind and retry.
- **`Model not found`** — model is not enabled in the chosen region. Switch to `us-central1` or pick a regionally-available model from the table in §4.
- **Quota exceeded** — request a Vertex AI quota increase via the GCP console; default quotas are conservative for new projects.

## See also

- [High-level API reference](api.md) for the full constructor signature.
- [Production setup](production.md) for the full Llama / Together / multi-arm orchestration stack.
