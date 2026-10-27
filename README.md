# TianXiMem

TianXiMem is an AML Agentic Memory service. It implements only the public `Add`
and `Search` endpoints; answer generation, judging, and aggregation remain AML
responsibilities.

## Runtime

```text
Add
  -> idempotency guard
  -> compose blocks within this request
  -> SQLite source of truth
  -> synchronous Qdrant index update

Search
  -> grounded evidence
  -> optional query routing and multi-hop retrieval
  -> hybrid BM25 + dense retrieval
  -> weighted RRF
  -> optional rerank, expansion, and merge
  -> token-budgeted evidence packaging
```

SQLite is the source of truth. Qdrant and the embedding cache are rebuildable
derived state. The service runs as one process.

## Requirements

- Python 3.12 for the container image; Python 3.11-3.13 for local execution
- `uv`
- Docker with Compose
- An OpenAI-compatible embedding endpoint
- An OpenAI-compatible chat endpoint when router, planner, or multi-agent
  execution is enabled
- An optional OpenAI-compatible reranker endpoint

## Local Start

```bash
cp .env.example .env
uv sync
uv run python -m tianximem.dev qdrant-up
uv run python -m tianximem.dev config-check
uv run python -m tianximem.dev serve
```

The service defaults to `configs/default.yaml`. Set `TIANXIMEM_PROFILE=local`
for the development collection.

## API

Health:

```bash
curl -s http://127.0.0.1:8000/health
```

Add:

```bash
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
```

Search:

```bash
curl -s -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "user-1", "query": "Where did the user move?", "top_k": 10}'
```

Valid Add requests have at least one message with non-empty `content`. Search
results contain at most `top_k` evidence items and never generated answers.

## Container Deployment

```bash
cp deploy/.env.example deploy/.env
uv run python -m tianximem.dev up
```

Qdrant is exposed only on host loopback by default. The API has no built-in
authentication; add TLS and authentication at the ingress before exposing it.

## Configuration

Configuration is split into two layers:

- environment variables for secrets, endpoints, model IDs, paths, and process
  shape;
- YAML profiles for thresholds, weights, collection names, and feature switches.

`configs/default.yaml` is the product profile. `configs/local.yaml` is the
development override. Unknown YAML keys fail validation.

## Operational Notes

- Back up SQLite; it is the only non-rebuildable store.
- Qdrant writes use `wait=true`, so a successful Add is immediately searchable.
- A matching Add replay repairs a missing derived index without writing source
  rows again.
- The service must run with one worker.

## Documentation

- [Architecture](docs/architecture.md)
- [HTTP contract](docs/contract.md)
