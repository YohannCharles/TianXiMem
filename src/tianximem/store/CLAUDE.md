# store

`store/` owns access to the source database, grounded-fact tables, and Qdrant index.

```text
schema.sql       current SQLite schema
sqlite_store.py  source transactions, batch guards, fact indexing, and adjacency queries
qdrant_store.py  collection management, point writes, and hybrid query execution
```

## Storage Roles

| Store | Contents | Recovery |
| --- | --- | --- |
| SQLite `qa_pairs` | Source blocks and explicit adjacency | Source of truth |
| SQLite `applied_batches` | Batch idempotency and payload fingerprints | Source of truth |
| SQLite `memory_facts` | Source-backed derived facts | Rebuild from `qa_pairs` |
| SQLite `evidence_coverage` | Extractor-version coverage | Rebuild from `qa_pairs` |
| Qdrant | Dense and BM25 vectors plus filter metadata | Rebuild from SQLite |

Qdrant does not store source text. Search resolves selected IDs back to SQLite.

## Source Records

```text
qa_pairs
  id, user_id, session_id, request_id, local_index
  prev_memory_id, next_memory_id
  question, answer, status, event_time
```

- `id = hash(user_id, session_id, request_id, local_index)`.
- `request_id` is opaque.
- `local_index` is the block index within one Add.
- `prev_memory_id` and `next_memory_id` form an Add-scoped chain.
- New source rows always use `status='complete'`.
- `user_id` is stored on every source row and every derived fact.

## Idempotency

`applied_batches` is append-only and contains the canonical payload hash.

- Same `request_id` and hash: replay.
- Same `request_id` and different hash: conflict.
- The source insert and guard-row insert share one transaction.
- Never infer idempotency from `qa_pairs.request_id`.

## Grounded Facts

- `memory_facts` is derived data.
- A fact quote must exist verbatim in the declared source side.
- A fact user must equal its parent memory user.
- Fact IDs are deterministic and insert-once.
- `evidence_coverage` records extractor versions, including empty extraction.
- Only current-version coverage counts as scanned.
- Rebuilding derived facts must not modify `qa_pairs` or `applied_batches`.

## Connection Model

SQLite connections are short-lived.

- Each logical read uses its own connection.
- Each write uses `BEGIN IMMEDIATE` and one transaction.
- Helpers inside a transaction reuse the caller's connection.
- Helpers do not commit or roll back independently.
- `journal_mode=WAL` is configured at database open.
- Connection-scoped pragmas are applied to every new connection.
- `read_snapshot()` uses one read transaction for a consistent multi-query view.
- SQLite serializes writers through the write lock and busy timeout. Do not add an application-level write lock.

## Qdrant

- Use server mode with persistent storage.
- Use one shard and named `dense` and `bm25` vectors.
- Create payload indexes before writing data.
- Index `user_id` as keyword with tenant optimization.
- Index `session_id` as keyword and `event_time` as integer.
- Filter Search on `user_id` only.
- Use `wait=true` for writes.
- Keep parser and server versions compatible and pinned.

`retrieve/` owns retrieval parameters. `qdrant_store.py` translates provided parameters into Qdrant syntax and returns domain objects.

## Invariants

1. Memory IDs are position-derived.
2. Adjacency comes from stored links, not array arithmetic.
3. Batch idempotency lives in `applied_batches`.
4. `UNIQUE(user_id, session_id, request_id, local_index)` preserves one write position and supplies Add order.
5. Embedding cache identity comes from rendered text, not memory ID.
6. A Qdrant gap is recoverable; a source gap is not.
7. Unknown or malformed schema state fails loudly.
