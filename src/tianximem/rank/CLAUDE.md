# rank

`rank/` owns the post-retrieval evidence pipeline:

```text
rerank -> neighbor expansion -> segment merge -> budgeted packaging
```

The order is part of the design.

## Files

```text
reranker.py   remote cross-encoder client
neighbor.py   expansion and segment construction
packaging.py  final slot and token budgeting
```

## Rerank

The configured reranker receives candidate documents and returns one score per input position.

- The client maps scores back to the input order.
- The caller applies ordering to the same candidate list.
- Reranking must not add or remove candidate IDs.
- Endpoint errors degrade to RRF order.
- `disabled` and `degraded` are separate states.

The configured request envelope is an endpoint identity. Send one configured envelope; do not probe multiple shapes per request.

## Neighbor Expansion

Expansion follows stored adjacency links in SQLite.

- Expansion runs after reranking.
- Each hop moves to a complete QA block.
- A missing link ends the walk.
- Expansion never crosses an Add, session, or user.
- Expanded neighbors have no fabricated rerank score.
- Duplicate candidates keep the best existing rank.

When `neighbor.radius` is zero, no expansion occurs.

## Segment Merge

A segment is a maximal run of selected blocks connected by stored adjacency links. Group keys are `user_id`, `session_id`, and `request_id`.

- Preserve source order inside a segment.
- Segment priority and anchor come from the best real candidate in the segment.
- Expanded neighbors cannot increase segment priority.
- A gap ends the segment; do not bridge it by array position.

## Rendering

Use `common/render.py` for QA and factual content. Do not create a second template.

Response-only relative-time annotation may modify `content` after the indexed rendering. The base text must remain recoverable exactly.

The rendered QA template is:

```text
Q: {question}
A: {answer}
```

Omit an empty side. Label each non-user answer segment with its role. Keep content self-delimiting and free of leading or trailing whitespace.

## Packaging

Final `top_k` and token limits apply to packaged segments, not raw candidates.

- Treat a segment as atomic.
- Add segments in priority order.
- Stop at the first segment that does not fit.
- Do not split a segment or skip it to fit a lower-ranked segment.
- Use the joined output length when reporting.
- Use a safe upper bound when admitting a segment.
- Use the configured answer-model tokenizer.

Response fields:

| Field | Source |
| --- | --- |
| `id` | Segment anchor memory ID |
| `content` | Complete segment rendering |
| `created_at` | Anchor event time in UTC day granularity, or `""` |
| `score` | `1 / (output position + 1)` |

Do not return raw RRF or reranker scores.
