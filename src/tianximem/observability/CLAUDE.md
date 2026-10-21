# observability

`observability/` aggregates per-request metrics and writes an optional process-level JSON snapshot.

## Metrics

| Area | Fields |
| --- | --- |
| Request | Search latency |
| Retrieval | Grounded-evidence outcome, operators, projections, selected facts, and fallback reason |
| Packaging | Returned segments, slot truncation, token truncation, and missing-source count |
| Rerank | Calls, disabled count, degraded count, and model identity |
| Router | Route distribution, calls, disabled/degraded state, reason, and latency |
| Multi-hop | Status, rounds, LLM calls, and latency |

Metrics are per-request increments. Do not infer request-local values by subtracting process totals.

The metrics file is the interface between the service and the HTTP-only evaluation harness because the response shape is fixed and harness code cannot import service internals. One service process writes one snapshot path.

## Snapshot Behavior

- Aggregate once in this package.
- Update the in-process snapshot under a lock.
- Write atomically using a temporary file and replacement.
- Treat metric-write failure as diagnostic, not as request failure.
- Separate `disabled` from `degraded`; they have different operational meanings.

## Boundaries

- Business modules emit events; this package aggregates them.
- This package must not import business modules.
- Metrics never appear in the public Search response.
- Do not add metrics that do not change an engineering action.
