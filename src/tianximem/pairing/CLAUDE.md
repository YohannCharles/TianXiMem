# pairing

`pairing/` owns memory-block composition and the source write boundary.

```text
pairing.py  messages -> memory blocks; complete blocks -> Add-scoped adjacency links
apply.py    idempotency check, composition, source insert, fact indexing, and transaction
```

## Block Rules

A block is formed from the last user message in a contiguous user segment plus the immediately following contiguous non-user segment. Earlier consecutive user messages remain independent. Any remaining message is stored independently.

The role test is exactly `role == "user"`. Do not use a role allowlist. Non-user roles are preserved and labeled during rendering.

Examples:

| Messages | Blocks |
| --- | --- |
| `U A U A` | `[U+A] [U+A]` |
| `U U A A U A` | `[U] [U+A+A] [U+A]` |
| `A U A` | `[A] [U+A]` |
| `U U U A` | `[U] [U] [U+A]` |

Composition and adjacency are scoped to one Add. Different Adds are never paired, completed, merged, or repaired across the boundary.

## Position and ID

```text
request_id   opaque string from the request
local_index  zero-based block index in this Add
id           hash(user_id, session_id, request_id, local_index)
```

The ID does not depend on content or arrival order. Never parse `request_id`.

## Adjacency

`link_blocks()` writes `prev_memory_id` and `next_memory_id` when the block is inserted.

- Only complete question-answer blocks participate in the chain.
- Incomplete blocks are still stored, embedded, and searchable.
- Links are never backfilled and never cross an Add.
- A missing block breaks the chain; do not infer a replacement from index order.

## Add Transaction

`apply.py` performs:

1. Check `applied_batches` and payload fingerprint.
2. Compose blocks and adjacency links.
3. Insert all source rows with `status='complete'`.
4. Derive and index grounded facts when enabled.
5. Insert the batch guard row.
6. Commit.
7. Let the service perform the synchronous Qdrant upsert and recovery path.

A matching replay changes no source rows. It may backfill missing derived facts. A different payload under the same `request_id` raises a conflict.

`BEGIN IMMEDIATE` serializes SQLite writers. There is no application-level position allocator or session lock.

## Failure Semantics

- Position is a pure function of the request, so duplicate application would collide rather than create a second location.
- The batch guard still handles normal client retries without relying on a uniqueness failure.
- Fact extraction is part of the source transaction.
- Replay and bounded backfill may write derived facts but never alter source rows or the batch guard.

## Tests

Tests must cover composition shapes, unknown roles, source index preservation, Add-scoped adjacency, opaque request IDs, replay from a committed-but-unacknowledged request, payload mismatch, and fact backfill.
