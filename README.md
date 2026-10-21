# TianXiMem

TianXiMem is an AML Agentic Memory service. It implements `Add` and `Search`; AML remains responsible for answer generation, judging, and aggregation.

## What It Does

```text
Add request
  -> idempotency guard
  -> block composition
  -> SQLite source of truth
  -> Qdrant derived index

Search request
  -> grounded evidence short path
  -> optional multi-hop route
  -> hybrid BM25 + dense retrieval
  -> weighted RRF
  -> optional rerank
  -> optional neighbor expansion
  -> segment merge and budgeted packaging
  -> ranked evidence
```

The service returns source text or source-backed fact projections. It does not answer the user directly.

## Requirements

- Python 3.12
- `uv`
- Docker with Compose for Qdrant and containerized deployment
- OpenAI-compatible embedding and LLM endpoints; a reranker endpoint is optional

## Start

```bash
cp .env.example .env
uv sync --all-extras
uv run python -m tianximem.dev qdrant-up
uv run python -m tianximem.dev check
uv run python -m tianximem.dev serve
```

The service reads `configs/default.yaml` unless `TIANXIMEM_PROFILE` selects another profile. `configs/local.yaml` is for development and disables reranking while using a separate collection.

## API

```bash
curl -s http://127.0.0.1:8000/health

curl -s -X POST http://127.0.0.1:8000/add \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id": "example-1",
    "user_id": "user-1",
    "session_id": "session-1",
    "messages": [
      {"role": "user", "content": "I moved to Hangzhou.", "timestamp": 1697900000000},
      {"role": "assistant", "content": "Noted."}
    ]
  }'

curl -s -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "user-1", "query": "Where did the user move?", "top_k": 10}'
```

See [`docs/api-usage.md`](docs/api-usage.md) for request examples and error handling.

## Evaluate

```bash
uv run python -m tianximem.dev data-check --dataset locomo-refined
uv run python -m tianximem.dev baseline
uv run python -m tianximem.dev contract-check
```

Evaluation drives the running service over HTTP. Evaluation code must not import `tianximem` internals.

## Deploy

```bash
cp deploy/.env.example deploy/.env
uv run python -m tianximem.dev up
uv run python -m tianximem.dev deploy-check
```

The deployment is a single-process application plus a pinned Qdrant server. SQLite and Qdrant storage must be persistent; SQLite is the only required backup.

## Documentation

- [System specification](AML%20Agentic%20Memory%20增强框架%20PRD.md)
- [Architecture](docs/architecture.md)
- [HTTP contract](docs/contract.md)
- [Configuration reference](docs/config-reference.md)
- [Development commands](docs/active-spec/development.md)
- [Benchmark data](docs/benchmark-data.md)
