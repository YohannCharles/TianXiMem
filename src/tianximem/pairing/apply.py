from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from tianximem.store.sqlite_store import STATUS_COMPLETE, SqliteStore, make_pair_id

from .pairing import MemoryBlock, Message, compose_memory_blocks, link_blocks

__all__ = [
    "AddBatch",
    "ApplyBatchResult",
    "PayloadMismatchError",
    "apply_batch",
    "payload_fingerprint",
]


@dataclass(frozen=True, slots=True)
class AddBatch:
    request_id: str
    user_id: str
    session_id: str
    messages: tuple[Message, ...]


class PayloadMismatchError(RuntimeError):
    pass


def payload_fingerprint(batch: AddBatch) -> str:
    canonical = json.dumps(
        {
            "user_id": batch.user_id,
            "session_id": batch.session_id,
            "messages": [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                for m in batch.messages
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ApplyBatchResult:
    applied: bool

    blocks: tuple[MemoryBlock, ...]

    new_pair_ids: tuple[str, ...]

    @property
    def new_pair_count(self) -> int:
        return len(self.new_pair_ids)


def _neighbour_id(batch: AddBatch, local_index: int | None) -> str | None:
    if local_index is None:
        return None
    return make_pair_id(batch.user_id, batch.session_id, batch.request_id, local_index)


def apply_batch(
    store: SqliteStore,
    batch: AddBatch,
) -> ApplyBatchResult:
    if not batch.messages:
        raise ValueError("AddBatch.messages 不得为空（messages 是批次的全部内容）")

    fingerprint = payload_fingerprint(batch)

    with store.transaction() as conn:
        if store.is_batch_applied(conn, batch.request_id):
            stored = store.applied_batch_payload_hash(conn, batch.request_id)
            if stored is not None and stored != fingerprint:
                raise PayloadMismatchError(
                    f"同一个 request_id 收到了**不同的 payload**：{batch.request_id!r}\n"
                    f"  已记录：{stored[:16]}…  本次：{fingerprint[:16]}…\n"
                    "  ⚠ 这不是重放，**不能静默挑一份落库**（另一份记忆会凭空消失，"
                    "而检索侧看不出来）。要么换一个新的 request_id，要么把 payload 改回去。"
                )
            return ApplyBatchResult(applied=False, blocks=(), new_pair_ids=())

        blocks = compose_memory_blocks(batch.messages)

        links = link_blocks(blocks)

        written: list[str] = []
        for local_index, (block, (prev_index, next_index)) in enumerate(
            zip(blocks, links, strict=True)
        ):
            pair = store.insert_pair(
                conn,
                user_id=batch.user_id,
                session_id=batch.session_id,
                request_id=batch.request_id,
                local_index=local_index,
                prev_memory_id=_neighbour_id(batch, prev_index),
                next_memory_id=_neighbour_id(batch, next_index),
                question=block.question,
                answer=block.answer,
                status=STATUS_COMPLETE,
                event_time=block.event_time,
            )
            written.append(pair.id)
            store.insert_pair_messages(
                conn,
                pair.id,
                tuple(
                    (source_index, message.role, message.content, message.timestamp)
                    for source_index, message in zip(block.source_idxs, block.messages, strict=True)
                ),
            )

        store.record_batch(
            conn,
            batch.request_id,
            batch.user_id,
            batch.session_id,
            payload_hash=fingerprint,
        )

    return ApplyBatchResult(
        applied=True,
        blocks=blocks,
        new_pair_ids=tuple(written),
    )
