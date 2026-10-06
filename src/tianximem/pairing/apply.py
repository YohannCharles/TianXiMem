"""一次 `Add` 的落库：**幂等守卫（含 payload 校验）→ 组合 → 按 Add 内位置写入**（一个事务）。

```text
0. payload 指纹                 canonical(user_id, session_id, messages) → sha256（D28）
1. 幂等守卫（必须最先做）        SELECT ... FROM applied_batches WHERE request_id = ?
                               命中且指纹相同 → 本批已应用过（重试）→ **真源一行不动**；
                                                启用共同取证时经 index_pair_facts() 补事实索引
                               命中但指纹不同 → **响亮冲突**（非 5xx），不许静默当重放
2. 组合                        compose_memory_blocks(messages)   ← 只看本批，不看库
3. Add 内邻接                   link_blocks(blocks) → 每个块的 (前, 后) 的 local_index
4. 写入 + 记下本批              位置 = (request_id, local_index)；同一个事务里
                               INSERT qa_pairs + INSERT applied_batches
                               （启用共同取证时再加 index_pair_facts() 写的派生表）
```

> **⚠ 第 1 步不能省。** 它是**唯一**能防"同一批被应用两次"的东西：服务在事务提交之后、
> 响应发出之前崩溃（或响应丢失），AML 会重试同一批（`request_id` 与 payload 不变）。

## `request_id` 是 **opaque string**（D28）

它只做三件事：**幂等键**、**溯源**、**成功响应原样回显**。位置的另一半也是它
（`(request_id, local_index)`），但那只是"拿它当个字符串去 hash"——**一个字节都不解析**。

> ⚠ 这推翻 D25。D25 从 `request_id` 里正则取 chunk 序号（`chunk-<n>`）当位置，
> 而**平台实发的是 `r_3115…` 这种不透明 id**（2026-09-29 用请求采集抓到的真实请求）
> ⇒ 那条路在真实流量上 **100% 失败**（`ValueError` → 500 → Add 全挂）。
> 详见 `docs/decisions.md` 的 **D28**。

## 位置与邻接都来自**这一次** Add

`local_index` 是块在本批里的 0-based 序号；`prev` / `next` 由 `link_blocks` 在同一批内
算出（只连完整 QA）。**跨 Add 既没有顺序、也没有邻接**——那道边界是结构性的：
本函数从头到尾只看得见 `batch` 与它自己写下的那几行。

## D25 起**没有"分配位置"这一步**（D28 保留）

位置是 `(request_id, local_index)`，**两者都是请求的纯函数**：
没有共享计数器、没有读-改-写 ⇒ **不需要按 session 串行化**，不同 Add 也可以并发到达
（它们在 `store.transaction()` 的 `BEGIN IMMEDIATE` 处排队，那是数据库级写者串行）。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass

from tianximem.facts.evidence import EVIDENCE_VERSION
from tianximem.facts.grammar import extract_evidence
from tianximem.store.sqlite_store import STATUS_COMPLETE, QaPair, SqliteStore, make_pair_id

from .pairing import MemoryBlock, Message, compose_memory_blocks, link_blocks

__all__ = [
    "AddBatch",
    "ApplyBatchResult",
    "PayloadMismatchError",
    "apply_batch",
    "index_pair_facts",
    "payload_fingerprint",
]


@dataclass(frozen=True, slots=True)
class AddBatch:
    """一次 `Add` 的 **canonical 契约输入**（§2.1）。

    ⚠ 只有这四样东西。**不得**出现任何数据集专属字段——加载层负责归一化（D16）。
    ⚠ **`request_id` 是不透明字符串**（D28）：它同时是**幂等键**与位置的**一半**
    （另一半是本批内的 `local_index`），但**绝不被解析**——任何形状都等价可用。
    """

    request_id: str
    user_id: str
    session_id: str
    messages: tuple[Message, ...]


class PayloadMismatchError(RuntimeError):
    """**同一个 `request_id`、不同的 payload**（D28）。

    ⚠ 这不是"重放"，也**不是 500**：它是调用方的 bug（id 复用了，内容却换了），
    必须**响亮**地冲突，而不是静默挑一份落库——静默挑一份意味着**另一份记忆凭空消失**，
    而检索侧完全看不出来。映射到 409 的那一处见 `service/errors.py`。
    """


def payload_fingerprint(batch: AddBatch) -> str:
    """规范化 payload 的 sha256（D28）。

    **只算"这一批要写什么"**：`user_id` / `session_id` / `messages`（含 `role` /
    `content` / `timestamp`）。`request_id` **不在里面**——它是键，不是内容。

    ⚠ 规范化在 **canonical 形状**上做（`Message` 已经把首尾空白 strip 过）⇒
    两次只差尾随空白的投递**不会**被误判成冲突；而真正换了内容（哪怕一个字）必然不同。

    ⚠ 键排序 + 紧凑分隔符：让"同一份内容"只对应**一个**指纹，与字典的插入顺序无关。
    """
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
    """False = 幂等守卫命中（本批已应用过），**没有写任何东西**。"""

    blocks: tuple[MemoryBlock, ...]
    """本批组合出的记忆块。守卫命中时是**空元组**（本批什么都没写）。"""

    new_pair_ids: tuple[str, ...]
    """本批写入的行的 `id`，与 `blocks` 一一对应且同序。

    ⚠ 回的是 **`id` 而不是位置**：位置是两个值，
    而消费方（`AddPipeline`）要的是"去真源取这几行"——`id` 正是那个东西。
    """

    @property
    def new_pair_count(self) -> int:
        return len(self.new_pair_ids)


def _neighbour_id(batch: AddBatch, local_index: int | None) -> str | None:
    """把"同批第几个块"翻成 `memory_id`（`None` 原样返回 = 链端）。"""
    if local_index is None:
        return None
    return make_pair_id(batch.user_id, batch.session_id, batch.request_id, local_index)


def index_pair_facts(store: SqliteStore, conn: sqlite3.Connection, pair: QaPair) -> None:
    """Add/重放/旧来源补索引共用一个入口，并在同一事务记版本覆盖。"""
    store.insert_evidence(
        conn,
        extract_evidence(
            parent_memory_id=pair.id,
            user_id=pair.user_id,
            question=pair.question,
            answer=pair.answer,
            event_time=pair.event_time,
        ),
    )
    store.mark_fact_index_coverage(conn, pair, version=EVIDENCE_VERSION)


def apply_batch(
    store: SqliteStore,
    batch: AddBatch,
    *,
    grounded_evidence: bool = False,
) -> ApplyBatchResult:
    """应用一批 `Add` 消息。**幂等**：同一 `request_id` 至多被应用一次。

    整个写入在**一个事务**里：崩溃或异常 ⇒ 整批回滚 ⇒ 保持"可重试"，
    不会留下半个批次（contract.md §6）。

    ⚠ **空批次直接抛 `ValueError`。** §2.1 的 `messages` 就是批次的全部内容，
    空批次是**调用方的 bug**（harness 或加载层喂错了），不是 AML 的行为。
    静默当成 no-op 会让"整个 session 一条都没写进去"伪装成成功。

    ⚠ **指纹在开事务之前算**：它是纯粹的请求解析，失败时**不该**先拿一次写锁。
    """
    if not batch.messages:
        raise ValueError("AddBatch.messages 不得为空（§2.1：messages 是批次的全部内容）")

    fingerprint = payload_fingerprint(batch)

    with store.transaction() as conn:
        # ── 第 1 步：幂等守卫（必须最先做，且必须是查 applied_batches）──
        if store.is_batch_applied(conn, batch.request_id):
            stored = store.applied_batch_payload_hash(conn, batch.request_id)
            # ⚠ `stored is None` = **核不了**（D28 之前写下的行没有指纹）⇒ 放行。
            #    判据是 is_batch_applied 那个布尔，不是这个值——把它当成"不同"会把
            #    老库里每一个 request_id 的重放都变成冲突。
            if stored is not None and stored != fingerprint:
                raise PayloadMismatchError(
                    f"同一个 request_id 收到了**不同的 payload**：{batch.request_id!r}\n"
                    f"  已记录：{stored[:16]}…  本次：{fingerprint[:16]}…\n"
                    "  ⚠ 这不是重放，**不能静默挑一份落库**（另一份记忆会凭空消失，"
                    "而检索侧看不出来）。要么换一个新的 request_id，要么把 payload 改回去。"
                )
            if grounded_evidence:
                for pair in store.fetch_by_request(
                    conn, batch.user_id, batch.session_id, batch.request_id
                ):
                    index_pair_facts(store, conn, pair)
            return ApplyBatchResult(applied=False, blocks=(), new_pair_ids=())

        # ── 第 2 步：组合（纯函数，只看本批）──
        blocks = compose_memory_blocks(batch.messages)

        # ── 第 3 步：Add 内邻接（只连完整 QA；跨 Add 一律不连）──
        links = link_blocks(blocks)

        # ── 第 4 步：写入（位置 = request_id + 本块在批内的序号）──
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
                # D24：块在写下的那一刻就是最终形状 ⇒ 状态恒为 complete。
                # `status` 列保留，但**不再有第二个取值**。
                status=STATUS_COMPLETE,
                event_time=block.event_time,
            )
            written.append(pair.id)
            if grounded_evidence:
                index_pair_facts(store, conn, pair)

        # 应用成功时，在【同一个事务】里记下本批（含 payload 指纹）
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
