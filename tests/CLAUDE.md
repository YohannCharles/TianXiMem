# tests

The test suite verifies the current design without calling real model or storage services unless a dedicated integration fixture is marked.

## Coverage Areas

| Area | Tests |
| --- | --- |
| HTTP contract | `test_contract.py`, `test_contract_preflight.py` |
| Add and idempotency | `test_pairing.py`, `test_apply.py`, `test_idempotency.py`, `test_service_add.py` |
| Isolation and storage | `test_isolation.py`, `test_store.py`, `test_qdrant_store.py` |
| Rendering and time | `test_render.py`, `test_annotate.py`, `test_packaging.py` |
| Embedding | `test_embed_cache.py`, `test_embed_truncation.py`, `test_query_instruction.py` |
| Retrieval and routing | `test_retrieve.py`, `test_router.py`, `test_reranker.py`, `test_neighbor.py` |
| Grounded evidence | `test_grounded_evidence.py`, `test_grounded_planner.py`, `test_evidence_operators.py`, fact and language tests |
| Multi-hop execution | `test_agent.py`, `test_plan.py` |
| Evaluation harness | `test_harness.py`, dataset and experiment tests |
| Configuration | `test_config.py`, `test_public_api.py`, `test_packaging.py` |
| Observability | `test_observability.py`, `test_capture.py` |

## Core Invariants to Test

- Add replays do not create source rows.
- Same `request_id` with a different payload conflicts.
- `request_id` accepts any opaque string.
- Composition and adjacency never cross an Add.
- `user_id` isolation holds on every retrieval path.
- Returned item count is measured after packaging.
- `created_at` is day-granularity or empty.
- Rendering used for indexing equals rendering used for reranking.
- Response-only annotation does not alter indexed text or candidate identity.
- Disabling a switch changes only the behavior named by that switch.
- Optional service failures degrade with a visible reason.
- Invalid configuration and invalid model output fail loudly.

## Test Style

- Prefer pure functions and protocol-level fakes.
- Do not use real sleeps to simulate HTTP timeout behavior; raise transport errors directly.
- Build minimal state explicitly so a test cannot pass through an unrelated fallback.
- Assert both positive and negative cases when a check can silently pass on an empty result.
- Keep integration skips visible; do not convert an unavailable dependency into a passing test.
- Patch the module that owns an implementation, not a compatibility re-export.

## Commands

```bash
uv run python -m tianximem.dev test
uv run python -m tianximem.dev test tests/test_router.py
uv run python -m tianximem.dev contract-check
```

`contract-check` validates a real HTTP boundary. Unit tests alone cannot prove the public contract.

## Fixtures

Use temporary directories and isolated users. Do not point tests at a production database, Qdrant collection, model endpoint, or capture file.

Tests that require a real Qdrant instance must skip only when the dependency is unavailable, with an explicit skip reason.
