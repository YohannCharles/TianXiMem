"""`Add` 与 `Search` 的**编排**——把 ① ② ③-a ③-b 串成真能调用的两条路径。

> **为什么编排在本目录**：`service/CLAUDE.md` 说这一层"不做检索、不做配对、不碰存储"——
> 那是指它**不重新实现**那些逻辑（全部往下调用）。而"**按什么顺序调**"必须有人拥有：
> Search 的链条横跨 `retrieve/` 与 `rank/`，Add 的链条横跨 `pairing/`、`store/`、`embed/`，
> **没有任何单个下层模块能拥有整条链**。所以顺序在这里，逻辑在下面。

## 当前 Step 1 的 Search 链（**不是最终形态**）

```text
User Query → DenseArm（每 query 恰好 1 次）
           → HybridRetriever（BM25 + Dense + RRF）
           → Initial Candidates
           → EvidenceChecker（v1 passthrough）
           → Minimal Packaging（rank/）
           → ≤ top_k
```

最终 v1 还要在 Checker 与 Packaging 之间插入 **Remote Rerank** 与
**Neighbor Expansion**——两个阶段都写在 [`../rank/CLAUDE.md`](../rank/CLAUDE.md)。

## `Add` 的链，以及那个**必须专门处理**的失败窗口

```text
canonical AddRequest
  → session 锁（按 (user_id, session_id)）
  → pairing / continuation  ──── SQLite 事务提交（真源）
  → 取出"本批应索引的 QaPair"
  → render → dense embedding → Qdrant upsert(wait=True)
  → 响应
```

**失败窗口**：SQLite 事务**已提交** → Qdrant 的 embedding/upsert **失败** → 客户端重试
**同一个 `request_id`**。

此时 `applied_batches` 里**已经有这一批**，幂等守卫会命中 ⇒ `apply_batch` 直接返回、
**不写任何东西**。若这里跟着返回 200，就会留下：**SQLite 有真源、Qdrant 永久缺索引**
⇒ 那些记忆**永远检索不到，且没有任何报错**。

### 现有 API 的一个缺口，以及不碰 ① 的修法

`ApplyBatchResult.plan` 在**守卫命中时是 `None`**（① 的 `apply_batch` 里那行早返回）——
所以重放时**拿不到"本批触碰了哪些 pair"**。

**但修复不需要那个列表**：`user_id` / `session_id` 就在请求里，而
`SqliteStore.fetch_pairs_by_idx_range(conn, user_id, session_id, 0, MAX)` 能拿到**该 session
的全部对**。于是修复路径是**按 session 做幂等重建**：

* `point_id` 是位置派生的纯函数 ⇒ 同一对永远是同一个 point
* `upsert` 幂等 ⇒ 已经写对的会被原样覆盖
* embedding 缓存按内容哈希 ⇒ 之前成功索引过的对**必然已在缓存里**，
  重放几乎不产生远程调用

⇒ **净效果**：不变量"该 session 的每个 SQLite 行都在 Qdrant 里"被恢复，
且**没有让 SQLite 去迁就 Qdrant**（真源不变，派生索引被修复）。

> **代价**：重放要 upsert 该 session 的全部对，而不是只有本批那几条。这是**刻意的**——
> 换取了"不改 ① 的 DDL"。若日后要精确到批，需要给 `applied_batches` 加一列记录
> 触碰的 `pair_id`s（那是 ① 的改动，**本切片没有做**）。
"""

from __future__ import annotations

from dataclasses import dataclass

from tianxi_am.pairing import AddBatch, ApplyBatchResult, apply_batch
from tianxi_am.pairing.pairing import BatchLimits
from tianxi_am.rank import PackagedResponse, package
from tianxi_am.retrieve import EvidenceChecker, HybridRetriever
from tianxi_am.service.locks import SessionLocks
from tianxi_am.store.qdrant_store import QdrantStore
from tianxi_am.store.sqlite_store import QaPair, SqliteStore, make_pair_id

__all__ = ["AddOutcome", "AddPipeline", "SearchPipeline"]

#: `fetch_pairs_by_idx_range` 的上界。取 int64 的极大值 ⇒ "该 session 的全部对"。
_MAX_PAIR_IDX: int = 2**63 - 1


@dataclass(frozen=True, slots=True)
class AddOutcome:
    """一次 Add 的结果（**不进响应**——响应只有 §2.1 那三个回显字段）。"""

    applied: bool
    """False = 幂等守卫命中（本批此前已应用过）。"""

    new_pair_count: int
    indexed: int
    """本次写进 Qdrant 的 point 数。"""

    repaired: bool
    """是否走了"按 session 幂等重建"的修复路径（即这是一次重试）。"""


class AddPipeline:
    """`Add` 的完整时序。**它自己不做配对、不碰 SQL、不算向量**——全部往下调。"""

    def __init__(
        self,
        *,
        store: SqliteStore,
        qdrant: QdrantStore,
        embedder: object,
        locks: SessionLocks,
        limits: BatchLimits | None = None,
    ) -> None:
        self._store = store
        self._qdrant = qdrant
        self._embedder = embedder
        self._locks = locks
        self._limits = limits

    def apply(self, batch: AddBatch) -> AddOutcome:
        """应用一批；**持锁直到 Qdrant 写完**。

        ⚠ 锁覆盖到 upsert 结束（不只是 SQLite 事务）——否则同一 session 的下一个批次
        可能在"真源已提交、索引还没写完"的窗口里插进来，让重放的范围与顺序变得难判。
        代价是同 session 的 Add 完全串行（那是 §15 本来就要的）。

        ⚠ **任何一步失败都直接抛**：SQLite 未提交 ⇒ 整批可重试；
        SQLite 已提交而 Qdrant 失败 ⇒ 抛出去（客户端重试时会走修复路径）。
        **不要在这里 catch 后返回成功**——那会让那批记忆永远检索不到。
        """
        with self._locks.hold(batch.user_id, batch.session_id):
            result = apply_batch(self._store, batch, limits=self._limits)
            pairs, repaired = self._pairs_to_index(batch, result)
            indexed = self._qdrant.index_pairs(pairs, self._embedder) if pairs else 0
        return AddOutcome(
            applied=result.applied,
            new_pair_count=result.new_pair_count,
            indexed=indexed,
            repaired=repaired,
        )

    # ── 决定"该索引哪些对" ─────────────────────────────────────────────

    def _pairs_to_index(
        self, batch: AddBatch, result: ApplyBatchResult
    ) -> tuple[list[QaPair], bool]:
        """返回 `(要索引的对, 是否走了修复路径)`。"""
        plan = result.plan
        if not result.applied or plan is None:
            # 守卫命中（重试）或状态不明 ⇒ **按 session 幂等重建**
            return self._session_pairs(batch.user_id, batch.session_id), True

        # 正常路径：精确到本批【触碰过】的对——新建的 + 被续接/关闭的那个 pending
        ids = [make_pair_id(batch.user_id, batch.session_id, d.pair_idx) for d in plan.new_pairs]
        if plan.resume is not None:
            ids.append(plan.resume.pending_id)
        # 去重但保序（`dict.fromkeys`）：本批可能既新建又关闭，id 不会重合，但别依赖这一点
        ids = list(dict.fromkeys(ids))
        if not ids:
            return [], False
        # ⚠ 正文**从 SQLite 读**，不用 plan 里的草稿——索引的内容必须与真源逐字一致。
        # 这是一次**独立的只读操作**（SQLite 事务已经提交），所以自己开一个短生命周期连接。
        with self._store.read() as conn:
            return self._store.fetch_pairs_by_ids(conn, ids), False

    def _session_pairs(self, user_id: str, session_id: str) -> list[QaPair]:
        """该 session 的全部对（真源）——修复路径的输入。"""
        with self._store.read() as conn:
            return self._store.fetch_pairs_by_idx_range(conn, user_id, session_id, 0, _MAX_PAIR_IDX)


class SearchPipeline:
    """`Search` 的完整时序。**不重排、不扩窗、不生成答案。**"""

    def __init__(
        self,
        *,
        store: SqliteStore,
        qdrant: QdrantStore,
        retriever: HybridRetriever,
        checker: EvidenceChecker,
    ) -> None:
        self._store = store
        self._qdrant = qdrant
        self._retriever = retriever
        self._checker = checker

    def run(self, *, user_id: str, query: str, top_k: int) -> PackagedResponse:
        """跑完整条链，返回打包好的响应。

        ⚠ **空库直接返回空结果**（`data: []` 是合法的，§2.1）：集合还不存在时
        Qdrant 会抛"collection not found"，而"没有数据"不是错误。
        顺带也**不为一次必然空的检索付远程 embedding 调用**。
        """
        if not self._qdrant.exists():
            return PackagedResponse(items=(), dropped_missing=0)

        # ① 混合检索（DenseArm 内部保证每 query 恰好 1 次 embedding）
        candidates = self._retriever.search(user_id=user_id, query=query, top_k=top_k)

        # ② Evidence Checker：v1 恒"充足"，但每轮判定都记账（D13）
        self._checker.decide(query=query)

        # ③ 最小打包：render + created_at + score(1/(rank+1)) + 精确 ≤ top_k
        return package(candidates, store=self._store, top_k=top_k)
