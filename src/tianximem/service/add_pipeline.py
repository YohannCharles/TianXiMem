"""`Add` 链的**编排**（`AddOutcome` / `AddPipeline`）。

```text
canonical AddRequest
  → apply_batch：payload 指纹 → 幂等守卫 → compose_memory_blocks → link_blocks
                 → SQLite 事务提交（真源，位置 = (request_id, local_index)）
  → 取出本批写下的行（按 id）→ render → dense embedding → Qdrant upsert(wait=True)
  → 响应
```

**没有 session 锁**：位置由请求派生 ⇒ 同 `(user_id, session_id)` 的 Add 可以并发，
它们在 `store.transaction()` 的 `BEGIN IMMEDIATE` 处排队（数据库级写者串行）。

## 那个**必须专门处理**的失败窗口

SQLite 事务**已提交** → Qdrant 的 embedding/upsert **失败** → 客户端重试**同一个
`request_id`**。此时 `applied_batches` 已有这一批 ⇒ 幂等守卫命中 ⇒ `apply_batch`
直接返回、**不写任何东西**。若这里跟着返回 200，就会留下 **SQLite 有真源、Qdrant
永久缺索引** ⇒ 那些记忆**永远检索不到，且没有任何报错**。

### 修复路径：按 **request_id** 幂等重建

`ApplyBatchResult.applied` 在守卫命中时是 `False`，但修复不需要那个列表：
`fetch_by_request(...)` 能拿到**那一次 Add** 写下的全部块。于是：

* `point_id` 是位置派生的纯函数 ⇒ 同一对永远是同一个 point
* `upsert` 幂等 ⇒ 已经写对的会被原样覆盖
* embedding 缓存按内容哈希 ⇒ 之前成功索引过的对**必然已在缓存里**

⇒ 不变量"该 Add 的每个 SQLite 行都在 Qdrant 里"被恢复，且**真源不变**。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from tianximem.common.render import render_pair
from tianximem.pairing import AddBatch, ApplyBatchResult, apply_batch
from tianximem.store.qdrant_store import QdrantStore
from tianximem.store.sqlite_store import QaPair, SqliteStore

__all__ = ["AddOutcome", "AddPipeline"]


@dataclass(frozen=True, slots=True)
class AddOutcome:
    """一次 Add 的结果，仅供内部编排和诊断使用，不进入响应。"""

    applied: bool
    """False = 幂等守卫命中（该批次已经应用）。"""

    new_pair_count: int

    repaired: bool
    """是否走了"按 `request_id` 幂等重建"的修复路径（即这是一次重试；作用域见 `_batch_pairs`）。"""


class AddPipeline:
    """`Add` 的完整时序。**它自己不做配对、不碰 SQL、不算向量**——全部往下调。"""

    def __init__(
        self,
        *,
        store: SqliteStore,
        qdrant: QdrantStore,
        embedder: object,
        inject_abs_time: bool = False,
        grounded_evidence: bool = False,
    ) -> None:
        self._store = store
        self._qdrant = qdrant
        self._embedder = embedder
        #: `packaging.inject_abs_time` 同时作用于索引渲染；正文变化会改变 embedding 输入。
        self._inject_abs_time = inject_abs_time
        self._grounded_evidence = grounded_evidence

    def apply(self, batch: AddBatch) -> AddOutcome:
        """应用一批。**不持 session 锁**。

        ⚠ **为什么可以并发**：位置 = `(request_id, local_index)`，**两者都是请求的
        纯函数** ⇒ 不同 Add 触碰**互不相交**的位置（`request_id` 不同 ⇒ `id` 不同），
        不会撞 `UNIQUE`；而"已提交、Qdrant 还没写完"那个窗口对并发方也安全
        （各自的 `id` 不同，重试走 per-id 幂等 upsert）。

        ⚠ **任何一步失败都直接抛**：SQLite 未提交 ⇒ 整批可重试；
        SQLite 已提交而 Qdrant 失败 ⇒ 抛出去（客户端重试时会走修复路径）。
        **不要在这里 catch 后返回成功**——那会让那批记忆永远检索不到。
        """
        result = apply_batch(
            self._store,
            batch,
            grounded_evidence=self._grounded_evidence,
        )
        pairs, repaired = self._pairs_to_index(batch, result)
        # ⚠ **这一步有副作用**（真的写 Qdrant），不是"算一个计数"——别因为它没有返回值就删掉。
        if pairs:
            self._qdrant.index_pairs(
                pairs,
                self._embedder,
                renderer=partial(render_pair, inject_abs_time=self._inject_abs_time),
            )
        return AddOutcome(
            applied=result.applied,
            new_pair_count=result.new_pair_count,
            repaired=repaired,
        )

    # ── 决定"该索引哪些对" ─────────────────────────────────────────────

    def _pairs_to_index(
        self, batch: AddBatch, result: ApplyBatchResult
    ) -> tuple[list[QaPair], bool]:
        """返回 `(要索引的对, 是否走了修复路径)`。"""
        if not result.applied:
            # 守卫命中（重试）⇒ **按这一次 Add 幂等重建**
            return self._batch_pairs(batch), True

        # 正常路径：本批写下的**全部**行（之后没有"被续接/被关闭的既有对"——
        # 一次 Add 触碰的恰好是它自己新写的那几行，不多不少）
        ids = list(result.new_pair_ids)
        if not ids:
            return [], False
        # ⚠ 正文**从 SQLite 读**，不用 `result.blocks` 里的文本——索引的内容必须与真源逐字一致。
        # 这是一次**独立的只读操作**（SQLite 事务已经提交），所以自己开一个短生命周期连接。
        with self._store.read() as conn:
            return self._store.fetch_pairs_by_ids(conn, ids), False

    def _batch_pairs(self, batch: AddBatch) -> list[QaPair]:
        """**这一次 Add**写下的全部块（真源）——修复路径的输入。

        ⚠ 该修复路径的作用域是 **`request_id`**：

        * **更准**：那次 Add 提交了哪些行、Qdrant 就可能缺哪些行——一行不多、一行不少
        * **更便宜**：长 session 上"整段重建"的代价随 session 增长，而它其实只需要一批
        * **不会碰到别人**：修复路径再也不需要回答"我会不会覆盖另一个 Add 的 point"

        ⚠ 返回空列表是**合法**的（那一批没写出任何行、或行已被清掉），调用方照常走完。
        """
        with self._store.read() as conn:
            return self._store.fetch_by_request(
                conn, batch.user_id, batch.session_id, batch.request_id
            )
