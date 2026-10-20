# service

`service/` owns HTTP shapes, orchestration, service assembly, and degradation boundaries.

## Files

```text
app.py                FastAPI application and lifespan
wiring.py             dependency graph assembly
routes.py             /add, /search, /health
schemas.py            request and response models
add_pipeline.py       Add orchestration
search_pipeline.py    Search orchestration and final rerank
search_stages.py      grounded-evidence snapshot, backfill, and selection stages
grounded_planner.py   bounded LLM plan generation
errors.py             HTTP error mapping
capture.py            raw request capture
pipeline.py           compatibility facade for Add and Search pipelines
```

## HTTP Rules

- Translate request and response models literally; do not leak internal fields.
- Echo `request_id`, `user_id`, and `session_id` unchanged on Add.
- Return `data=[]`, not `null`, for no matches.
- Enforce `len(data) <= top_k` after final packaging.
- Keep `created_at` present on every Search item.
- Keep `/health` independent of downstream services.
- Treat non-200 responses as retryable unless the payload is definitively invalid or conflicting.
- Never return partial Add success.

## Add Orchestration

1. Call the pairing transaction.
2. Render committed source blocks.
3. Resolve vectors through the embedding cache.
4. Upsert Qdrant points with `wait=true`.
5. On derived-index failure, leave the batch recoverable through a same-payload replay.
6. Return only after the source and derived index are ready.

The service does not choose block boundaries or write source SQL itself.

## Search Orchestration

```text
grounded evidence
  -> complete: package and return
  -> incomplete: router
       -> DEFAULT: hybrid retrieval
       -> MULTI_HOP: executor
  -> merge candidates
  -> rerank once
  -> expansion
  -> segment merge
  -> package
  -> enforce top_k and token budget
```

The grounded-evidence path runs first when enabled. If it cannot return complete evidence, it falls back as a whole rather than mixing partial evidence with hybrid candidates.

The final rerank runs exactly once on the merged candidate list. Round-local reranking inside the multi-hop executor is separate and never enters the response.

Optional failures must produce contract-valid responses with a named diagnostic reason or fallback state. They must not silently claim successful execution.

## Configuration and Assembly

`wiring.py` builds services from `AppConfig`. `service/` does not read environment variables or YAML.

The service runs with one worker. Configuration and process checks must both reject multi-worker startup.

The downstream stack is blocking. Keep blocking storage and HTTP calls in synchronous route handlers or an explicit thread pool; do not block the event loop from `async def` request handlers.

## Request Capture

Capture records raw Add and Search requests before normal processing. It must not:

- mutate request text;
- alter downstream behavior;
- suppress normal request errors;
- turn capture failure into a response failure.

Capture files may contain sensitive evaluation data. Keep them out of Git and do not share them.

## Tests

Use focused tests for schemas, HTTP behavior, orchestration order, failure recovery, isolation, optional-service degradation, and response shape. Use the real preflight suite for the complete contract.
