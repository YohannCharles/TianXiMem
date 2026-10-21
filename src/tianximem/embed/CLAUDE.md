# embed

`embed/` converts rendered memory blocks and queries into vectors and persists reusable vectors.

## Files

```text
base.py               Embedder protocol, OpenAI-compatible client, and disk cache
text_embedding_v4.py  Submission embedding adapter
qwen3_embedding.py    Development embedding adapter
query_instruction.py  Optional query-side instruction wrapper
truncate.py           Query-side embedding-window truncation
```

## Interface

```python
class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...
    def encode_query(self, texts: list[str]) -> list[list[float]]: ...
```

`dim` comes from the embedding API response. The configured expected dimension is an assertion.

The architecture must not depend on sparse, ColBERT, or other model-specific multi-vector output.

## Inputs

| Side | Input |
| --- | --- |
| Document | Rendered QA block from `common/render.py`, one vector per block |
| Query | Original query, optionally with a configured instruction prefix |

Indexed text and returned evidence share the same base rendering. Response-only annotations do not enter embeddings.

The service does not run an independent pre-retrieval query rewriter. Multi-hop executors may issue new retrieval queries inside their own loop.

## Query Adaptation

The optional instruction prefix is part of the embedding input identity. Apply it before the cache lookup.

Long queries are truncated before sending to the embedding endpoint:

- truncate only the query side;
- apply the instruction prefix before truncation;
- keep a query-specific cache key;
- use tokenizer-based prefix and suffix budgets;
- never truncate indexed documents to hide source-data errors.

## Cache

The cache must be persistent and rebuildable.

- Cache key: hash of the exact rendered document or adapted query text.
- Do not use memory IDs as cache keys.
- Cache coordinate: embedding model ID, expected dimension, and rendering version.
- One coordinate uses one cache file.
- A model, dimension, rendering, or query-adaptation change must prevent old vectors from being reused silently.

The cache may use its own SQLite file. It is not the source of truth and must not mix with the business database.
