# eval/smoke

`eval/smoke/` contains AML-facing contract checks.

```text
preflight.py  HTTP contract preflight
```

## Preflight

Preflight validates the public contract without importing service internals.

It covers:

- Add success shape and identifier echo;
- Search array shape and required fields;
- `created_at` presence and day granularity;
- final `top_k` count;
- monotonic rank scores;
- user isolation;
- replay idempotency and payload conflict behavior;
- response readability after newline joins;
- absence of second-level timestamps.

Run an isolated instance:

```bash
uv run python -m tianximem.dev contract-check
```

Run against an existing service:

```bash
uv run python eval/smoke/preflight.py --base-url http://127.0.0.1:8000
```

The existing-service mode writes a small isolated test user. Do not target a service whose storage must remain untouched.

## Boundary

Smoke checks handle platform-facing contract behavior. Dataset quality and most retrieval comparisons belong in local evaluation.

Do not use AML Smoke quota for a question local evaluation can answer.
