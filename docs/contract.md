# HTTP Contract

This document defines the public HTTP surface. The service specification is [`../AML Agentic Memory 增强框架 PRD.md`](../AML%20Agentic%20Memory%20增强框架%20PRD.md).

## Add

`POST /add`

```json
{
  "request_id": "opaque-string",
  "user_id": "user",
  "session_id": "session",
  "messages": [
    {
      "role": "user",
      "content": "message text",
      "timestamp": 1697900000000
    }
  ]
}
```

Success:

```json
{
  "success": true,
  "request_id": "opaque-string",
  "user_id": "user",
  "session_id": "session"
}
```

Rules:

- `messages` must contain at least one item.
- `timestamp` is optional and uses Unix milliseconds.
- The three identifiers are echoed unchanged.
- A successful response means the source transaction is committed and the Qdrant index is ready for immediate Search.
- A replay with the same `request_id` and canonical payload returns success without writing source rows again.
- A different payload under the same `request_id` is a conflict.

## Search

`POST /search`

```json
{
  "query": "question text",
  "user_id": "user",
  "top_k": 100
}
```

Success:

```json
{
  "data": [
    {
      "id": "stable-id",
      "content": "rendered evidence",
      "created_at": "2026-07-26",
      "score": 1.0
    }
  ]
}
```

Rules:

- `query` must not be blank.
- `top_k` must be at least 1.
- `data` is always an array; no match returns `[]`.
- The final array length is at most `top_k`.
- The returned items are evidence, not answers.
- `id` is stable for the same stored segment.
- `content` is self-delimiting, has no leading or trailing whitespace, and contains no second-level timestamp.
- `created_at` is always present as `YYYY-MM-DD` or `""`.
- `score` is a monotonic rank placeholder such as `1 / (rank + 1)`.

## Health

`GET /health`

Response:

```json
{"status":"ok"}
```

This endpoint does not probe downstream services.

## Isolation

- Search filters only by `user_id`.
- `session_id` is grouping metadata.
- No retrieval path may reveal another user's blocks or facts.

## Error Handling

AML defines retries and timeouts. The service must assume a failed request can be retried.

| Status | Meaning | Client action |
| --- | --- | --- |
| `200` | Successful contract response | Validate response shape and echoed fields. |
| `409` | Same `request_id` with a different payload | Use a new `request_id`; retrying the same conflict cannot succeed. |
| `422` | Invalid request shape | Correct the request. |
| `5xx` | Internal or dependency failure | Retry when the operation remains retryable. |

Add must not return partial success. Search may fall back from optional reranking, routing, or multi-hop execution, but the final response must remain contract-valid and expose degradation through metrics.

## Compliance Checklist

### Shape

- [ ] Add returns `success: true` and echoes all identifiers byte-for-byte.
- [ ] Search returns a `data` array.
- [ ] Every item has string `id`, string `content`, and `created_at`.
- [ ] `score`, when present, decreases monotonically with output rank.

### Counting and Budgets

- [ ] `len(data) <= top_k`.
- [ ] Count packaged segments, not raw memories.
- [ ] The final joined context stays within the answer model input budget.
- [ ] Budget decisions do not split a segment.

### Isolation

- [ ] No path crosses `user_id`.
- [ ] `session_id` is never used as a Search filter.

### Idempotency

- [ ] Replaying the same request does not create new source rows.
- [ ] Replaying with a different payload returns a conflict.
- [ ] The guard is read from `applied_batches`.

### Content

- [ ] Relative time expressions remain in the source text.
- [ ] Absolute dates use day granularity only.
- [ ] Returned content is independently readable after newline joins.
- [ ] Search never returns generated answers.

## Preflight

Run the local compliance checks before spending AML Smoke quota:

```bash
uv run python -m tianximem.dev contract-check
```

To target a running instance:

```bash
uv run python eval/smoke/preflight.py --base-url http://127.0.0.1:8000
```
