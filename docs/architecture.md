# Architecture

## Runtime Summary

```text
Add
  -> fingerprint request
  -> check applied_batches
  -> compose blocks within this Add
  -> write blocks, original messages and derived facts in one SQLite transaction
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
  -> hybrid retrieval: BM25 + dense -> weighted RRF
  -> rerank
  -> neighbor expansion
  -> segment merge
  -> slot/token budgeted packaging
  -> return at most top_k
```

The service owns memory storage and evidence retrieval only. It does not generate final answers or judge them.

## Add Flow

1. Compute a canonical payload fingerprint.
2. Check `applied_batches`.
   - Matching payload: treat as a replay and repair missing derived indexes if needed.
   - Different payload: return a conflict.
3. Compose memory blocks from this request only.
4. Link adjacent complete blocks inside this request only.
5. Atomically write:
   - `qa_pairs` source rows.
   - `qa_pair_messages` original roles, text, source indices and timestamps.
   - `memory_facts` and `evidence_coverage` when grounded extraction is enabled.
   - The `applied_batches` guard row.
6. Synchronously update Qdrant.
7. Return the echoed request identifiers.

SQLite is the source of truth. A Qdrant failure leaves a recoverable derived-index gap; a retry can rebuild it without changing source rows.

## Block Composition

A block is composed by merging contiguous runs of the same role, then pairing a
user run with the following non-user runs. Leading non-user runs and trailing
user runs remain standalone blocks, with their same-role messages still merged.
For example, `U U U` becomes `[UUU]`, `U U A A` becomes `[UU+AA]`, and
`U A U A` becomes `[U+A] [U+A]`.

The original messages remain separate source records in SQLite. Fact extraction
processes each original message with its own role and timestamp and associates
the derived facts with the composed parent block. Older databases without those
records retain the parent-block extraction fallback; their original boundaries
cannot be inferred from joined text.

Only one role predicate is used: `role == "user"`. Non-user roles are retained and receive role labels during rendering. No role allowlist is used.

Pairing and adjacency never cross an Add boundary.

## Search Flow

### Grounded Evidence

The deterministic grounded-evidence path runs first when enabled.

```text
question -> common query plan -> source-backed fact execution
         -> source coverage and conflict review
         -> complete evidence: render and package
         -> incomplete: continue to routing and retrieval
```

Supported plan types are `filter`, `walk`, `interval`, and `current`. Lists and counts share the same evidence plan. Final list construction and counting remain AML responsibilities.

The path falls back as a whole when a plan is unsupported, source coverage is incomplete, evidence limits are exceeded, unresolved branches remain, or source review finds conflicts.

For explicit source-backed relation walks, a selected quotation can be projected
to an original user message when its full content and timestamp match exactly
one retained source record. These records are returned in selected hop order,
with their own dates and identities. Omitted portions of the parent still
participate in the related-source audit. Missing or ambiguous original records
retain whole-parent projection. Source-record limits count original messages;
an incomplete result after slot or token packaging falls back as a whole.
This presentation path does not alter indexed blocks or their adjacency links.

### Routing

After grounded evidence is incomplete, the router may choose:

- `DEFAULT`: normal hybrid retrieval.
- `MULTI_HOP`: the configured agent executor.

The router only classifies. It does not retrieve, bind fact fields, generate answers, or rewrite stored memory.

### Multi-Hop Executors

`react` repeatedly retrieves, reranks a round-local candidate set, asks the model for sufficiency and the next query, and merges evidence by stable ID.

`plan` first decomposes the original query into one to four ordered steps. Each round answers only the step whose evidence was just retrieved. Program code binds placeholders and decides when to stop. Program code adds one recovery retrieval for `unresolved` or `conflict` outcomes and reranks that recovery pool against the original query.

Both modes share normal retrieval, isolation, and final packaging. Final reranking occurs once in the Search pipeline after candidates from all rounds are merged.

### Hybrid Retrieval

```text
BM25 query -----+
                 +-> weighted RRF -> stable candidate IDs
Dense query ----+
```

- BM25 uses Qdrant sparse vectors with IDF.
- Dense retrieval uses the configured embedding backend.
- RRF uses `k=61`.
- Final truncation happens after deterministic ordering.
- `user_id` is the only retrieval filter.

## Module Ownership

| Module | Owns |
| --- | --- |
| `service/` | HTTP schemas, request orchestration, degradation boundaries, and service assembly |
| `pairing/` | Block composition, Add-scoped adjacency, idempotency, and the source write transaction |
| `facts/` | Deterministic source-backed fact extraction and query-plan compilation |
| `store/` | SQLite source access, grounded-fact persistence, Qdrant index access, and adjacency queries |
| `embed/` | Embedding protocol, model adapters, query adaptation, and persistent vector cache |
| `retrieve/` | Retrieval policy, candidate fusion, grounded-evidence execution, and query routing |
| `rank/` | Reranking, neighbor expansion, segment merge, and final packaging |
| `agent/` | `react` and `plan` multi-hop executors |
| `llm/` | OpenAI-compatible chat client and LLM availability errors |
| `common/` | Rendering, token counting, and configuration loading |
| `observability/` | Per-request metrics and aggregated metric snapshots |

## Dependency Direction

```text
service
  -> pairing -> facts, store
  -> retrieve -> facts, store, embed, llm
  -> rank -> store, common
  -> agent -> retrieve, rank, llm
  -> observability

common depends on no business module.
```

`store/` is the only module that accesses the business SQLite database and Qdrant. The embedding cache is the declared exception: it is a rebuildable cache with its own database file and connection policy.

`common/render.py` is the only implementation for QA and fact text rendering. Indexing, reranking, and response packaging use the same rendered representation, with one declared exception: response-only relative-time annotation may add text to `content` as long as removing the annotation restores the indexed text exactly.

## Core Invariants

1. Memory IDs are derived from position, not content.
2. Batch idempotency is stored in `applied_batches`, not inferred from `request_id` shape.
3. Block composition and adjacency are scoped to one Add.
4. Adjacency uses stored `prev_memory_id` and `next_memory_id` links.
5. Source text is stored only in SQLite.
6. Grounded facts require verbatim source evidence.
7. Search isolation is enforced on every retrieval path.
8. Final item count and token budgets are enforced after packaging.
9. Optional services degrade observably.
10. Invalid configuration, invalid model output, and source corruption fail loudly.

## Storage Boundaries

| Store | Role | Loss behavior |
| --- | --- | --- |
| SQLite | Source of truth for blocks, idempotency, facts, and coverage | Source memory is lost; this is the required backup |
| Qdrant | Derived vector and metadata index | Rebuild from SQLite |
| Embedding cache | Rebuildable cost cache | Recompute vectors |

Qdrant must run in server mode with persistent storage. Embedded local mode does not provide the required payload-index behavior.
