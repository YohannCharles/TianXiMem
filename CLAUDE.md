# TianXiMem

TianXiMem is an AML Agentic Memory service. It exposes two business endpoints:

- `POST /add`: persist one request's memory blocks and make them searchable before returning.
- `POST /search`: return at most `top_k` evidence items in relevance order.

`GET /health` is a side-effect-free liveness endpoint. The service never generates a final answer, judges an answer, or writes generated text back as memory.

## Source of Truth

- Product and system requirements: [`AML Agentic Memory 增强框架 PRD.md`](AML%20Agentic%20Memory%20增强框架%20PRD.md)
- Architecture and dependencies: [`docs/architecture.md`](docs/architecture.md)
- HTTP contract: [`docs/contract.md`](docs/contract.md)
- Configuration keys: [`docs/config-reference.md`](docs/config-reference.md)
- Active defaults: [`configs/default.yaml`](configs/default.yaml) and [`docs/active-spec/config-defaults.md`](docs/active-spec/config-defaults.md)
- Cross-platform commands: [`docs/active-spec/development.md`](docs/active-spec/development.md)

If documents disagree, the PRD defines behavior; the code and tests define the implemented state.

## Architecture

```text
Add
  -> canonical payload fingerprint and applied_batches guard
  -> compose memory blocks inside this Add only
  -> write qa_pairs and grounded facts in one SQLite transaction
  -> upsert Qdrant points with wait=true
  -> return

Search
  -> deterministic grounded-evidence plan
       -> complete evidence: render and package evidence
       -> incomplete: continue
  -> optional query router
       -> MULTI_HOP: react or plan executor
       -> DEFAULT: hybrid retrieval
  -> hybrid retrieval: BM25 + dense -> weighted RRF
  -> optional rerank
  -> optional neighbor expansion and segment merge
  -> package with slot and token budgets
  -> return at most top_k segments
```

- SQLite is the source of truth. Qdrant is a rebuildable derived index.
- `pairing/` owns the Add write boundary and block composition.
- `facts/` derives source-backed facts without calling an LLM.
- `retrieve/` owns retrieval policy and grounded-evidence execution.
- `rank/` owns rerank, expansion, segment merge, and packaging.
- `service/` owns HTTP models, orchestration, and degradation boundaries.
- `common/` owns rendering, token counting, and configuration loading.

## Hard Invariants

- `user_id` is the only Search isolation boundary. `session_id` is grouping metadata, not a filter.
- `request_id` is opaque. Never parse business meaning from it.
- A memory ID is derived from `(user_id, session_id, request_id, local_index)`. An embedding cache key is derived from rendered text, never from the memory ID.
- One Add is the only block-composition and adjacency boundary. Messages from different Adds are never paired or linked.
- `applied_batches` owns batch idempotency. The same `request_id` with the same canonical payload is a replay; the same `request_id` with a different payload is a conflict.
- Add must be fully persisted and searchable before a successful response. Qdrant writes use `wait=true`.
- Search returns evidence only. It must not generate or disguise an answer.
- The response contains at most the requested `top_k` items. Count the final packaged segments, not raw candidates.
- The final answer context is consumed as a prefix. Order is part of correctness.
- `created_at` is always present as `YYYY-MM-DD` or `""`; never use second-level timestamps.
- `content` is independently readable and has no leading or trailing whitespace.
- The service runs with one worker. Multi-worker operation has not been validated.
- Embedding and LLM endpoints use OpenAI-compatible HTTP clients. Add never calls an LLM.
- Training or fine-tuning on AML data is prohibited.

## Configuration Rules

- Environment variables own secrets, endpoints, model IDs, paths, and process shape.
- YAML profiles own thresholds, weights, collection names, and feature switches.
- `common/config/` is the only package allowed to read environment variables or YAML.
- Unknown keys are errors.
- A key is configuration only when a runtime consumer exists.
- `rrf.k = 61`, the `117,760` token ceiling, and `top_k` from the request are correctness constraints, not tuning knobs.
- `configs/default.yaml` is the product profile. `configs/local.yaml` contains only development overrides.

## Working Rules

1. Read the nearest module `CLAUDE.md` before editing that module.
2. Keep one owner for each rule, key, and record shape. Link to the owner instead of copying prose.
3. Preserve original memory text. Derived facts must retain a verbatim source quote.
4. Fail loudly on corrupt inputs, unknown configuration, invalid model output, and contract violations.
5. Treat degraded optional services as observable fallbacks, not silent success.
6. Add or update focused tests for behavior changes. Run the narrowest relevant test first, then broader checks when shared behavior changes.

## Common Commands

```bash
cp .env.example .env
uv sync --all-extras
uv run python -m tianximem.dev help
uv run python -m tianximem.dev test
uv run python -m tianximem.dev serve
uv run python -m tianximem.dev check
```

`Makefile` targets are compatibility aliases for `tianximem.dev`.
