# HTTP Contract

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

- `request_id`, `user_id`, and `session_id` are non-empty strings.
- `request_id` is opaque and is echoed unchanged.
- `messages` contains at least one item.
- Every message has a non-empty `role` and non-blank `content`.
- `timestamp` is optional Unix milliseconds.
- A successful response means SQLite is committed and Qdrant is ready.
- A same-payload replay is idempotent.
- Reusing `request_id` with a different payload returns `409`.

## Search

`POST /search`

```json
{
  "user_id": "user",
  "query": "question text",
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

- `user_id` and `query` are non-empty; `query` must not be blank.
- `top_k` is at least 1.
- `data` is always an array; no match returns `[]`.
- The final array length is at most `top_k`.
- Items are evidence, not answers.
- `id` is stable for the same stored segment.
- `content` has no leading or trailing whitespace.
- `created_at` is always `YYYY-MM-DD` or `""`.
- `score` is a monotonic rank placeholder.

## Health

`GET /health`

```json
{"status":"ok"}
```

Health checks process liveness only and does not probe downstream services.

## Isolation

- Search filters only by `user_id`.
- `session_id` is grouping metadata, not a Search filter.
- No retrieval path may reveal another user's records.

## Errors

| Status | Meaning | Client action |
| --- | --- | --- |
| `200` | Successful response | Validate response shape. |
| `409` | Same `request_id` with a different payload | Use a new `request_id`. |
| `422` | Invalid request shape | Correct the request. |
| `5xx` | Internal or dependency failure | Retry when the operation remains retryable. |

Add never returns partial success. Optional Search components degrade to a
contract-valid response and record the fallback in metrics.
