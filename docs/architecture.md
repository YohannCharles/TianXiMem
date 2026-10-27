# Architecture

## Runtime Summary

```text
Add
  -> fingerprint request
  -> check applied_batches
  -> compose blocks within this Add
  -> write source rows, derived facts, and the batch guard in one SQLite transaction
  -> render and embed changed blocks
  -> upsert Qdrant points with wait=true
  -> return

Search
  -> grounded evidence
       complete: package evidence and return
       incomplete: continue
  -> query router
       DEFAULT: hybrid retrieval
       MULTI_HOP: react or plan executor
  -> BM25 + dense -> weighted RRF
  -> optional rerank
  -> optional neighbor expansion
  -> segment merge and budgeted packaging
  -> return at most top_k
```

The service stores memory and returns evidence. It does not generate or judge
final answers.

## Add Flow

1. Compute a canonical payload fingerprint.
2. Check `applied_batches`.
   - Matching payload: replay without rewriting source rows.
   - Different payload: return a conflict.
3. Compose memory blocks from this request only.
4. Link adjacent complete blocks inside this request only.
5. Atomically write source rows, original messages, derived facts, and the
   idempotency guard.
6. Synchronously update Qdrant.
7. Return the echoed request identifiers.

SQLite is the source of truth. A Qdrant failure leaves a recoverable
derived-index gap; replaying the same Add repairs it.

## Block Composition

Contiguous same-role runs are merged before a user run is paired with the
immediately following non-user runs. Leading non-user runs and trailing user
runs remain standalone blocks. Pairing and adjacency never cross an Add.

Every original message is retained with its role, source index, and timestamp.
Fact extraction runs on those original records and attaches derived facts to the
composed parent block.

## Search Flow

### Grounded Evidence

A deterministic grounded-evidence path may return complete source-backed
evidence before normal retrieval. Unsupported plans, incomplete source
coverage, conflicts, and exceeded limits fall back to the broader retrieval
path.

### Routing and Multi-Hop

After grounded evidence is incomplete, an optional router chooses `DEFAULT` or
`MULTI_HOP`. Multi-hop execution augments the candidate set but shares the same
isolation, final reranking, expansion, merge, and packaging path.

### Hybrid Retrieval

```text
BM25 query -----+
                 +-> weighted RRF -> stable candidate IDs
Dense query ----+
```

- BM25 uses Qdrant sparse vectors with IDF.
- Dense retrieval uses the configured embedding backend.
- RRF uses `k=61`.
- `user_id` is the only retrieval isolation filter.

## Module Ownership

| Module | Responsibility |
| --- | --- |
| `service/` | HTTP schemas, orchestration, degradation boundaries, and assembly |
| `pairing/` | Block composition, idempotency, and source writes |
| `facts/` | Source-backed fact extraction and query plans |
| `store/` | SQLite and Qdrant access |
| `embed/` | Embedding adapters, query adaptation, and vector cache |
| `retrieve/` | Retrieval, evidence execution, and routing |
| `rank/` | Reranking, expansion, merge, and packaging |
| `agent/` | Multi-hop executors |
| `llm/` | OpenAI-compatible chat client |
| `common/` | Rendering, token counting, and configuration |
| `observability/` | Metrics snapshots |

## Core Invariants

1. Memory IDs derive from position, not content.
2. Batch idempotency is stored in `applied_batches`.
3. Composition and adjacency are scoped to one Add.
4. Source text is stored only in SQLite.
5. Grounded facts require verbatim source evidence.
6. Search isolation is enforced on every retrieval path.
7. Final item and token limits are enforced after packaging.
8. Optional services degrade observably.
9. Invalid configuration and invalid model output fail loudly.

## Storage

| Store | Role | Loss behavior |
| --- | --- | --- |
| SQLite | Source of truth | Source memory is lost; required backup |
| Qdrant | Derived vector and metadata index | Rebuild from SQLite |
| Embedding cache | Rebuildable cost cache | Recompute vectors |

Qdrant runs in server mode. Its client and server versions must remain
compatible.
