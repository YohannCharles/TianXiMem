# TianXiMem System Specification

## 1. Purpose

TianXiMem is an AML Agentic Memory service. It stores source conversations as searchable memory blocks and returns ranked evidence. Answer generation, answer judging, and metric aggregation remain outside this service.

The system must optimize end-to-end evidence usefulness. Retrieval recall alone is not the objective: the important decisions are which evidence to keep, how to order it, and how to fit it into the answer model's input window.

## 2. HTTP Contract

### 2.1 Add

`POST /add`

Request fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `request_id` | string | Opaque request identifier. It is echoed unchanged and never parsed for business meaning. |
| `user_id` | string | Search isolation identity. |
| `session_id` | string | Grouping metadata. It is not a Search filter. |
| `messages` | array | Source-order messages. Each item has `role`, non-empty `content`, and optional `timestamp` in Unix milliseconds. |

Successful response:

```json
{
  "success": true,
  "request_id": "same value",
  "user_id": "same value",
  "session_id": "same value"
}
```

The request is successful only after SQLite and Qdrant are ready for immediate Search. Qdrant writes use `wait=true`.

AML may retry Add up to 32 times with the same `request_id` and payload. Add must therefore be idempotent. A replay does not write source rows again. The same `request_id` with a different canonical payload is a conflict.

### 2.2 Search

`POST /search`

Request fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `query` | string | Non-blank query. The retrieval service does not generate an answer. |
| `user_id` | string | Required isolation identity. |
| `top_k` | positive integer | Maximum number of returned evidence items. AML normally sends `100`. |

Successful response:

```json
{
  "data": [
    {
      "id": "stable-id",
      "content": "rendered evidence",
      "created_at": "YYYY-MM-DD",
      "score": 1.0
    }
  ]
}
```

Rules:

- `data` is always an array. Empty results use `[]`.
- The final item count is at most `top_k`; count packaged segments, not raw candidates.
- `id` is stable for the same stored segment.
- `content` is evidence, never an answer.
- `created_at` is present and uses UTC day granularity, or `""` when the source has no timestamp.
- `score` is a monotonic rank placeholder such as `1 / (rank + 1)`, not a calibrated similarity or RRF score.
- The answer model reads the returned order as a token prefix. Items that do not fit can be discarded entirely.

### 2.3 Health

`GET /health` returns `{"status":"ok"}` without probing Qdrant or model endpoints. It is a liveness check, not a readiness check.

## 3. Hard Constraints

- `user_id` is the only Search isolation boundary.
- `session_id` is not a Search filter.
- Add requests and successful responses preserve `request_id`, `user_id`, and `session_id` verbatim.
- The service runs with one worker.
- Internal failures leave the Add batch retryable. Partial success is forbidden.
- Non-200 behavior is not a contract guarantee. Clients may retry, so all write paths must remain idempotent.
- The service must not train or fine-tune on evaluation data.
- The service must not return generated answers as memory.

## 4. Models and External Services

| Component | Runtime form |
| --- | --- |
| Embedding | OpenAI-compatible HTTP API. Submission uses `text-embedding-v4`. Development may use a configured Qwen embedding model. |
| LLM | OpenAI-compatible chat completions API. Submission uses `gpt-4o-mini`. Development may use a configured Qwen chat model. |
| Reranker | Optional OpenAI-compatible rerank API. The application supports both `query` and `queries` request envelopes. |

Embedding dimensions come from the embedding API response. The configured expected dimension is an assertion, not a source of truth.

Models, endpoints, and secrets belong in environment variables. Retrieval and agent thresholds belong in YAML.

## 5. Storage Model

### 5.1 SQLite

SQLite is the source of truth. It stores source blocks, batch idempotency state, and rebuildable grounded-fact tables.

Core tables:

```text
qa_pairs
  id, user_id, session_id, request_id, local_index,
  prev_memory_id, next_memory_id,
  question, answer, status, event_time

applied_batches
  request_id, user_id, session_id, payload_hash, applied_at

memory_facts
  source-backed atomic facts derived from qa_pairs

evidence_coverage
  extractor-version coverage per parent memory
```

`id` is derived from `(user_id, session_id, request_id, local_index)`. `request_id` participates as an opaque whole string.

`local_index` is the zero-based block index within one Add. Adjacency is stored explicitly through `prev_memory_id` and `next_memory_id`. Only complete question-answer blocks are linked, and links never cross an Add boundary.

`applied_batches` is append-only. It stores a canonical payload hash. A matching hash is a replay; a different hash is a conflict.

`status` is `complete` for all new rows. Historical `pending` rows may remain readable, but the current write path never creates or completes pending rows.

### 5.2 Qdrant

Qdrant is a rebuildable derived index. It stores named `dense` and `bm25` vectors plus filter metadata. It does not store source text.

Required properties:

- Server mode, not embedded local mode.
- Pinned compatible server and client versions.
- One shard.
- Payload indexes created before data writes.
- `user_id` keyword index with tenant optimization.
- Writes use `wait=true`.
- Search filters on `user_id`; `session_id` is not a retrieval filter.

If Qdrant is lost, rebuild it from SQLite. If SQLite is lost, source memory is lost.

## 6. Add Pipeline

```text
1. Canonicalize the request payload and compute its SHA-256 hash.
2. Check applied_batches.
   - same request_id and same hash: repair derived indexes if needed, then return
   - same request_id and different hash: return a conflict
3. Compose memory blocks using only this request's messages.
4. Build explicit adjacency links between complete blocks in this request.
5. In one SQLite transaction:
   - insert qa_pairs rows
   - derive and insert grounded facts and coverage rows
   - insert the applied_batches guard row
6. Render changed blocks, obtain embeddings, and upsert Qdrant points.
7. Return the successful Add response.
```

### 6.1 Block Composition

A block is formed from the last user message in a contiguous user segment plus the following contiguous non-user segment. Earlier consecutive user messages remain independent blocks. Messages that do not form such a pair are stored independently.

The pairing decision only asks whether a message has role `user`. It does not use a role allowlist. All non-user roles are retained with role labels.

Examples:

| Input roles | Output blocks |
| --- | --- |
| `U A U A` | `[U+A] [U+A]` |
| `U U A A U A` | `[U] [U+A+A] [U+A]` |
| `A U A` | `[A] [U+A]` |
| `U U U A` | `[U] [U] [U+A]` |
| `U U U U` | `[U] [U] [U] [U]` |

One Add is the only pairing and adjacency boundary. There is no cross-Add completion, repair, or merging.

### 6.2 Idempotency and Failure Recovery

A committed SQLite batch with a failed Qdrant write is recoverable. A retry with the same request payload finds the guard, verifies the payload hash, and rebuilds the missing derived index without changing source rows.

Grounded-fact extraction shares the SQLite write transaction with source rows. A fact extraction failure rolls back the whole Add. A later replay may backfill missing facts and coverage while leaving source rows unchanged.

## 7. Grounded Evidence

Grounded evidence is a deterministic short path that runs before query routing and hybrid retrieval.

A source-backed fact contains:

```text
subject / relation / object
qualifiers stated by the source
source side and verbatim source quote
source record date
```

The extractor uses explicit source syntax and does not call an LLM. The query compiler maps supported question forms into a common plan:

| Plan | Behavior |
| --- | --- |
| `filter` | Match subject, relation, object, and source-stated qualifiers. |
| `walk` | Follow a bounded sequence of source-backed relation edges. |
| `interval` | Compare only time ranges explicitly present in source text. |
| `current` | Return a legal empty result when the question's full input is present and no history is required. |

Lists and counts use the same evidence plan. The service returns the supporting facts; AML generates the final list or count.

Grounded evidence falls back to the normal retrieval chain when any required condition fails:

- No supported query plan.
- The current user's source coverage is incomplete.
- A bounded source or hop limit is exceeded.
- The plan has unresolved branches.
- Relevant source text contains unexplained related statements or conflicts.
- Packaging cannot return complete evidence.

A partial fact index is never treated as a complete answer set.

## 8. Retrieval

The normal retrieval chain is:

```text
BM25 + dense
  -> weighted RRF
  -> stable deduplication
  -> optional routing and multi-hop execution
  -> optional rerank
  -> optional neighbor expansion
  -> segment merge
  -> slot and token budgeted packaging
```

### 8.1 BM25 and Dense

- BM25 uses Qdrant's `qdrant/bm25` sparse representation with IDF.
- Dense retrieval uses the configured embedding backend.
- The query is sent unchanged to embedding. The service does not use a pre-retrieval query rewriter.
- Indexed text is the rendered QA block. One block produces one vector.

### 8.2 Weighted RRF

Weighted RRF combines the BM25 and dense ranked lists.

- `k` is explicitly `61` to match the literature's 1-based constant `60` under Qdrant's zero-based ranks.
- Both prefetches declare their named vector with `using`.
- The root fusion limit covers the full prefetch pool; final truncation happens after deterministic ordering.
- The configured prefetch limit is not the seed count or `top_k`.

### 8.3 Query Routing

After grounded evidence fails to complete, the optional router classifies the query as one of:

- `MULTI_HOP`: invoke the configured multi-hop executor.
- `DEFAULT`: use the normal hybrid retrieval chain.

The router must return an exact enum value. Invalid output falls back to `DEFAULT`. It does not generate or rewrite memory content.

### 8.4 Multi-Hop Executor

The executor supports two modes.

`react` mode:

```text
retrieve -> rerank round candidates -> ask LLM for sufficiency and next query
         -> stop or repeat with the next query
         -> merge evidence by stable ID
```

`plan` mode:

```text
decompose the original query into 1-4 ordered steps
  -> retrieve and answer one completed step per round
  -> bind the next query through program-controlled placeholders
  -> stop on completion, conflict, unresolved evidence, budget, or failure
```

Both modes reuse the normal retrieval, reranking, isolation, and packaging layers. They do not generate the final answer. Intermediate plan answers are diagnostics only and never enter Search responses.

The final rerank happens once in the Search pipeline. It reranks the merged candidate list using the original query plus the queries actually executed.

## 9. Ranking and Packaging

### 9.1 Rerank

Reranking is optional. The client sends candidates through the configured rerank endpoint and maps scores back to input positions.

Reranker failure is an observable fallback to RRF order. Disabled and degraded states are counted separately.

Reranking may change order but must not change the candidate ID set.

### 9.2 Neighbor Expansion

Neighbor expansion runs after reranking. It follows explicit SQLite adjacency links for the configured radius. It never scans a session and infers adjacency from array position.

Expansion is disabled when `neighbor.radius` is zero. The implementation remains available for configured use.

### 9.3 Segment Merge

Selected blocks are merged only when stored adjacency links connect them. Groups are isolated by `user_id`, `session_id`, and `request_id`.

A segment keeps source order. A segment anchor and priority come from the highest-ranked real candidate in the segment, never from an unranked expanded neighbor.

### 9.4 Budgeted Packaging

Packaging is the only place that enforces final output limits.

- Apply `top_k` and token budgets to final segments.
- Treat a segment as atomic. Do not return half a segment.
- Stop when the next segment does not fit; do not skip it to fit a lower-ranked segment.
- Count the actual joined output for reporting and use a safe upper bound for admission.
- Use `o200k_base` for the submission token budget.
- Return at most `top_k` segments, in final rank order.

## 10. Rendering and Time

`common/render.py` is the single renderer for QA blocks and grounded facts.

The QA template is:

```text
Q: {question}
A: {answer}
```

Rules:

- Omit a line when its side is empty.
- Add a role label to every non-user answer segment.
- Keep the rendered text self-delimiting because AML joins items with newlines.
- Do not include leading or trailing whitespace.
- The same rendered text is used for indexing and reranking.

With absolute-date injection enabled, each block is prefixed with a day-granularity source date:

```text
[YYYY-MM-DD] Q: ...
A: ...
```

Changing indexed rendering requires rebuilding the vector collection and invalidating the embedding cache.

Relative-time annotation is a presentation-only exception. It may add an absolute date in parentheses after a relative expression in returned `content`. It preserves the original text and does not change embedding input or Qdrant.

Never emit second-level timestamps. Do not substitute Add arrival time for source event time.

## 11. Configuration

Configuration has two non-overlapping layers.

Environment variables own:

- model endpoints and credentials
- model IDs
- file and directory paths
- worker count and bind/port settings

YAML profiles own:

- retrieval thresholds and weights
- feature switches
- context budgets
- collection names
- storage timeouts

Loading order is built-in defaults, `default.yaml`, selected profile, environment overrides, and validation. Unknown keys are errors.

`configs/default.yaml` is the product profile. `configs/local.yaml` is a development override. The active key list and defaults are documented in `docs/config-reference.md`.

## 12. Evaluation

Evaluation drives the running service over HTTP. Harness code must not import service internals.

The local evaluation path covers:

- Add ingestion in AML-shaped batches.
- Search calls with isolated users.
- Answer and judge adapters for supported datasets.
- Run records containing configuration, model, dataset, response, and score metadata.

Different datasets have different injection fields, answer prompts, time rules, and judges. Local scores are valid for relative comparisons inside the same dataset and contract. They are not a universal leaderboard prediction.

## 13. Operations

- Use Qdrant server mode with persistent storage.
- Use a persistent volume for SQLite and the embedding cache.
- Run the application as a single process.
- Keep `/health` independent of downstream availability.
- Pin dependency and container versions used for final evaluation.
- Back up SQLite. Rebuild Qdrant from SQLite when necessary.
- Treat request capture files as sensitive evaluation data. Do not commit or share them.

## 14. Documentation Policy

Documentation describes the current design only. Do not add chronological change logs, superseded designs, decision-number narratives, dated experiment summaries, or migration histories. Put current behavior in the owning document and cite its source with a relative link. Generated measurements belong under `eval/reports/` and must not become a second specification.
