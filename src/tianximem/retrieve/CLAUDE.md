# retrieve

`retrieve/` owns retrieval policy and executes grounded-evidence plans. It stops before reranking and neighbor expansion.

## Files

```text
bm25.py       BM25 query construction
dense.py      dense query execution
fusion.py     hybrid parameters, stable deduplication, and final candidate ordering
checker.py    rank-consistency evidence checker
evidence.py   grounded QueryPlan execution and source review
router.py     MULTI_HOP / DEFAULT classification
```

## Hybrid Retrieval

```text
BM25 -----+
           +-> weighted RRF -> stable ordering -> candidate list
Dense ----+
```

- BM25 uses Qdrant `qdrant/bm25` with IDF.
- Dense uses the configured embedding backend.
- Every prefetch names a vector with `using`.
- Weighted RRF uses `k=61`.
- The root fusion limit covers the prefetch pool; final `top_k` truncation happens after deterministic ordering.
- Candidate IDs are deduplicated deterministically.
- `user_id` is the only isolation filter.
- `session_id` is never a Search filter.

`retrieve/` owns retrieval parameter selection and validation. `store/qdrant_store.py` owns the Qdrant request syntax and execution.

## Grounded Evidence

When enabled, the service compiles a query plan through `facts/query.py` and executes it with `evidence.py`.

| Plan | Execution |
| --- | --- |
| `filter` | Match source-backed subject, relation, object, and qualifiers. |
| `walk` | Follow bounded source-backed relation edges. |
| `interval` | Compare only source-stated time ranges. |
| `current` | Return a legal empty selection when full input is present. |

The executor also reviews source coverage and unexplained related statements. It must not treat physical coverage as semantic completeness.

Return control to hybrid retrieval when:

- no supported plan exists;
- source coverage is incomplete;
- a path has unresolved branches;
- relevant source text contains conflicts or unexplained statements;
- a source, hop, or evidence limit is exceeded.

Never return a partial fact set as if it were complete.

## Query Router

The router maps a query to exactly `MULTI_HOP` or `DEFAULT`.

- Invalid model output falls back to `DEFAULT`.
- The router does not rewrite the query or retrieve evidence.
- Blank-query validation remains at the HTTP boundary.
- Router failures never turn Search into a 500.

## Evidence Checker

The checker uses agreement between BM25-only and dense-only rankings. It must not use fused RRF scores as calibrated thresholds.

The current service path keeps the checker result observational; routing is owned by the query router.

## Boundaries

- Do not perform reranking, expansion, or packaging.
- Do not return generated answers.
- Do not bind fact fields in the router.
- Do not duplicate question syntax from `facts/`.
- Do not read configuration directly.
