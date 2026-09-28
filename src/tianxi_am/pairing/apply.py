"""一次 `Add` 的落库：**幂等守卫 → 组合 → 按 chunk 派生位置写入**（一个事务）。

```text
0. 解析 chunk 序号              parse_chunk_ordinal(request_id) ← 取不到就响亮失败（D25）
1. 幂等守卫（必须最先做）      SELECT 1 FROM applied_batches WHERE request_id = ?
                              命中 → 本批已应用过（重试）→ 直接返回，不写任何东西
2. 组合                        compose_memory_blocks(messages)   ← 只看本批，不看库、不看别的 Add
3. 写入 + 记下本批             位置 = (chunk_ordinal, local_index)
                              同一个事务里 INSERT qa_pairs + INSERT applied_batches
```

> **⚠ 第 1 步不能省。** 它是**唯一**能防"同一批被应用两次"的东西：服务在事务提交之后、
> 响应发出之前崩溃（或响应丢失），AML 会重试同一批（`request_id` 与 payload 不变）。

## D25 起**没有"分配位置"这一步**

位置 = `(chunk_ordinal, local_index)`，**两者都是请求的纯函数**：

| 位置分量 | 从哪来 | 为什么不是到达顺序的产物 |
| --- | --- | --- |
| `chunk_ordinal` | `request_id` 里的 `chunk-N` | 平台给定的批次序号 |
| `local_index` | `compose_memory_blocks` 输出里的块序号 | 组合是纯函数（D24）⇒ 同批必然得同号 |

⇒ 没有共享计数器、没有读-改-写 ⇒ **不需要按 session 串行化**，
且**乱序到达不会翻转会话顺序**——若位置改由到达顺序决定，`chunk-1` 先提交就会拿到更小的位置，
于是 §10 扩窗与 §11.2 段合并把对话顺序读反，**且不报错**。

> ⚠ **幂等的两层仍然是两层**：这里删掉的是"位置分配"那一半的读-改-写，
> 而"同一批被应用两次"由 `applied_batches` 守卫挡住（D4）——
> 取消串行化**不会**让守卫变得不必要，两件事正交。

## D24 起不再有"跨批续接"

旧的三步挂接（`3a′ / 3a / 3b / 3d`）与 `pending` 状态、`open_pair` / `append_*` /
`mark_complete` **已整体删除**：组合的边界就是这一次 Add，库里既有多少块与本批无关。

> ⚠ 由此推出一条**新的**读库纪律：`qa_pairs` 里不再有"半成品"行。
> 每一行在写下的那一刻就是它最终的样子，**没有任何后台任务会回头改它**（没有 pending、
> 没有 repair、没有重新 embedding）。
"""

from __future__ import annotations

from dataclasses import dataclass

from tianxi_am.common.config import DEFAULT_CHUNK_ORDINAL_PATTERN
from tianxi_am.pairing.pairing import (
    MemoryBlock,
    Message,
    compose_memory_blocks,
    parse_chunk_ordinal,
)
from tianxi_am.store.sqlite_store import STATUS_COMPLETE, SqliteStore

__all__ = ["AddBatch", "ApplyBatchResult", "apply_batch"]


@dataclass(frozen=True, slots=True)
class AddBatch:
    """一次 `Add` 的 **canonical 契约输入**（§2.1）。

    ⚠ 只有这四样东西。**不得**出现任何数据集专属字段——加载层负责归一化（D16）。
    ⚠ **`request_id` 不是"只是幂等键"**（D25 起）：位置模型的 `chunk_ordinal`
    就从它里面解析。它同时是**幂等键**与**顺序来源**，两个角色都要留。
    """

    request_id: str
    user_id: str
    session_id: str
    messages: tuple[Message, ...]


@dataclass(frozen=True, slots=True)
class ApplyBatchResult:
    applied: bool
    """False = 幂等守卫命中（本批已应用过），**没有写任何东西**。"""

    blocks: tuple[MemoryBlock, ...]
    """本批组合出的记忆块。守卫命中时是**空元组**（本批什么都没写）。"""

    chunk_ordinal: int
    """本批的 chunk 序号（从 `request_id` 解析）。守卫命中时是 `-1`（没解析过）。"""

    new_pair_ids: tuple[str, ...]
    """本批写入的行的 `id`，与 `blocks` 一一对应且同序。

    ⚠ 回的是 **`id` 而不是位置**：位置是两个数，
    而消费方（`AddPipeline`）要的是"去真源取这几行"——`id` 正是那个东西，
    少一层"位置 → id"的往返换算。
    """

    @property
    def new_pair_count(self) -> int:
        return len(self.new_pair_ids)


def apply_batch(
    store: SqliteStore,
    batch: AddBatch,
    *,
    chunk_ordinal_pattern: str = DEFAULT_CHUNK_ORDINAL_PATTERN,
) -> ApplyBatchResult:
    """应用一批 `Add` 消息。**幂等**：同一 `request_id` 至多被应用一次。

    整个写入在**一个事务**里：崩溃或异常 ⇒ 整批回滚 ⇒ 保持"可重试"，
    不会留下半个批次（contract.md §6）。

    ⚠ **空批次直接抛 `ValueError`。** §2.1 的 `messages` 就是批次的全部内容，
    空批次是**调用方的 bug**（harness 或加载层喂错了），不是 AML 的行为。
    静默当成 no-op 会让"整个 session 一条都没写进去"伪装成成功——
    正是本层最怕的失败模式。代价是它会被重试 32 次，但触发它的是我们自己的代码。

    ⚠ **chunk 序号在开事务之前解析**：它是一个纯粹的请求解析，失败时**不该**先拿一次写锁。
    """
    if not batch.messages:
        raise ValueError("AddBatch.messages 不得为空（§2.1：messages 是批次的全部内容）")

    # ── 第 0 步：位置的一半（取不到 ⇒ 响亮失败，见 parse_chunk_ordinal 的 docstring）──
    chunk_ordinal = parse_chunk_ordinal(batch.request_id, chunk_ordinal_pattern)

    with store.transaction() as conn:
        # ── 第 1 步：幂等守卫（必须最先做，且必须是查 applied_batches）──
        if store.is_batch_applied(conn, batch.request_id):
            return ApplyBatchResult(
                applied=False, blocks=(), chunk_ordinal=-1, new_pair_ids=()
            )

        # ── 第 2 步：组合（纯函数，只看本批）──
        blocks = compose_memory_blocks(batch.messages)

        # ── 第 3 步：写入（位置 = chunk_ordinal + 本块在批内的序号）──
        written: list[str] = []
        for local_index, block in enumerate(blocks):
            pair = store.insert_pair(
                conn,
                user_id=batch.user_id,
                session_id=batch.session_id,
                chunk_ordinal=chunk_ordinal,
                local_index=local_index,
                question=block.question,
                answer=block.answer,
                # D24：块在写下的那一刻就是最终形状 ⇒ 状态恒为 complete。
                # `status` 列保留，但**不再有第二个取值**。
                status=STATUS_COMPLETE,
                event_time=block.event_time,
                request_id=batch.request_id,
            )
            written.append(pair.id)

        # 应用成功时，在【同一个事务】里记下本批
        store.record_batch(conn, batch.request_id, batch.user_id, batch.session_id)

    return ApplyBatchResult(
        applied=True,
        blocks=blocks,
        chunk_ordinal=chunk_ordinal,
        new_pair_ids=tuple(written),
    )
