"""`Add` 与 `Search` 的**编排**——把各层的调用串成真能跑的两条路径。

> **为什么编排在本目录**：`service/CLAUDE.md` 说这一层"不做检索、不做配对、不碰存储"——
> 那是指它**不重新实现**那些逻辑（全部往下调用）。而"**按什么顺序调**"必须有人拥有：
> Search 的链条横跨 `retrieve/` 与 `rank/`，Add 的链条横跨 `pairing/`、`store/`、`embed/`，
> **没有任何单个下层模块能拥有整条链**。所以顺序在这里，逻辑在下面。

## Search 链（2026-09-24：接上扩窗、合并与 token 预算）

```text
User Query → DenseArm（每 query 恰好 1 次）
           → HybridRetriever（BM25 + Dense + RRF）
           → memory_id 稳定去重
           → EvidenceChecker（v1 passthrough，每轮记账）
           → rerank（恰好一次；不可用则降级回 RRF 顺序）
           → Neighbor Expansion（全部候选保留，只对前 N 条扩 ±radius）
           → Context Segment Merge（段内会话序，段间 best_rank 序）
           → Token Budget + Final Packaging
           → ≤ top_k 个**段**
```

**唯一还没实现的一环曾经是 rerank 的远端调用**——**2026-09-24 已接**（`RemoteReranker`
打主网关的 `/rerank`，线格式实测过）。端点不可用时按 D12 降级回 RRF 顺序并记
`rerank_degraded`；没配/关掉时记 `rerank_disabled`。理由与降级契约写在
[`../rank/reranker.py`](../rank/reranker.py)。

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

from tianxi_am.common.config import DEFAULT_EXPANSION_SEED_LIMIT, DEFAULT_RADIUS
from tianxi_am.common.render import render
from tianxi_am.common.tokens import TokenCounter
from tianxi_am.pairing import AddBatch, ApplyBatchResult, apply_batch
from tianxi_am.pairing.pairing import BatchLimits
from tianxi_am.rank import (
    PackagedResponse,
    Reranker,
    RerankUnavailable,
    expand_neighbors,
    merge_segments,
    package,
)
from tianxi_am.retrieve import EvidenceChecker, HybridRetriever, dedup_candidates
from tianxi_am.retrieve.fusion import Candidate
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
    """`Search` 的完整时序。**不生成答案**（§2.1 的红线）。

    ## 顺序（每一步都有理由，不要调换）

    ```text
    ① 混合检索（BM25 + Dense → RRF）   每路 1 次查询；DenseArm 保证每 query 恰好 1 次 embedding
    ② memory_id 稳定去重 + 重编号      必须在 rerank **之前**：重复项会被排两次、占两个名额
    ③ Evidence Checker（v1 恒"充足"）   每轮判定都记账（D13）
    ④ rerank（**恰好一次**）            `RemoteReranker`；不可用 ⇒ 降级回 RRF 顺序（D12）
    ⑤ Neighbor Expansion               全部候选保留；只对前 N 条扩 ±radius
    ⑥ Context Segment Merge            连续 pair_idx 合成段（段内会话序，段间 best_rank 序）
    ⑦ Token Budget + Final Packaging   段是原子单位；`<= top_k` 的计数在这里收口
    ```

    ## 两个"顺序错了也不会报错"的地方

    * **去重必须在 rerank 之前**：不去重的话，同一份证据会被 rerank 两次、
      在预算里占两个名额——而**名次看起来完全正常**。
    * **`top_k` 必须在扩窗/合并之后生效**：它约束的是**段数**。在扩窗之前按
      raw memory 数截断，会把本该形成段的邻居砍掉，最后返回的段数**少于该有的**。
    """

    def __init__(
        self,
        *,
        store: SqliteStore,
        qdrant: QdrantStore,
        retriever: HybridRetriever,
        checker: EvidenceChecker,
        counter: TokenCounter,
        budget_tokens: int,
        seed_limit: int = DEFAULT_EXPANSION_SEED_LIMIT,
        radius: int = DEFAULT_RADIUS,
        reranker: Reranker | None = None,
    ) -> None:
        self._store = store
        self._qdrant = qdrant
        self._retriever = retriever
        self._checker = checker
        self._counter = counter
        self._budget_tokens = budget_tokens
        self._seed_limit = seed_limit
        self._radius = radius
        self._reranker = reranker
        #: 诊断计数（§14）。**不进响应**——响应的形状是契约，一个字段都不能多。
        self.rerank_calls = 0
        self.rerank_degraded = 0
        self.rerank_disabled = 0

    @property
    def reranker(self) -> Reranker | None:
        """当前接的 reranker（`None` = 没配 / 显式关掉）。

        只读暴露是给两处用的：**测试**（断言装配真的接上了，不是又传了 `None`）
        与**诊断**（§14 的 run record 要记"这次用的哪个 reranker"，D12）。
        """
        return self._reranker

    def run(self, *, user_id: str, query: str, top_k: int) -> PackagedResponse:
        """跑完整条链，返回打包好的响应。

        ⚠ **空库直接返回空结果**（`data: []` 是合法的，§2.1）：集合还不存在时
        Qdrant 会抛"collection not found"，而"没有数据"不是错误。
        顺带也**不为一次必然空的检索付远程 embedding 调用**。
        ⚠ 注意这里的短路**早于** token 计数——空库时不该去加载分词器。
        """
        if top_k <= 0 or not self._qdrant.exists():
            return PackagedResponse(items=())

        # ① 混合检索（DenseArm 内部保证每 query 恰好 1 次 embedding）
        candidates = self._retriever.search(user_id=user_id, query=query, top_k=top_k)

        # ② 稳定去重 + 重编号（**在 rerank 之前**，见类 docstring）
        ranked = dedup_candidates(candidates)

        # ③ Evidence Checker：v1 恒"充足"，但每轮判定都记账（D13）
        self._checker.decide(query=query)

        # ④ rerank —— **恰好一次**
        ranked = self._maybe_rerank(query=query, ranked=ranked)

        # ⑤⑥ 扩窗 + 合并成段（全部候选保留，只对前 N 条扩窗）
        expansion = expand_neighbors(
            ranked, store=self._store, seed_limit=self._seed_limit, radius=self._radius
        )
        segments = merge_segments(expansion.selected, counter=self._counter)

        # ⑦ 预算 + 打包（段是原子单位；`top_k` 约束的是**段数**）
        return package(
            segments,
            top_k=top_k,
            counter=self._counter,
            max_tokens=self._budget_tokens,
            dropped_missing=expansion.missing_rows,
        )

    def _maybe_rerank(self, *, query: str, ranked: list[Candidate]) -> list[Candidate]:
        """对候选做**一次** rerank；不可用就退回原顺序。

        ⚠ **"降级"与"没开"是两件事，分开计数**（见 [`../rank/reranker.py`](../rank/reranker.py)）：
        两者产出的名次一样，但前者说明端点坏了、后者是 v1 的既定状态。
        混为一个计数会让"reranker 一直失败"看起来像"我们没打算用它"。

        **判据是"我们有没有打算调用它"**，不是"有没有调用成功"：

        | 情况 | 计数 | 为什么 |
        | --- | --- | --- |
        | `reranker is None`（没配 / 显式关掉） | `disabled` | 本来就没打算调 |
        | 没有候选 | `disabled` | 没有可排序的东西，**调用是没有意义的** |
        | 真源缺行 ⇒ 拼不出文档 | `degraded` | **打算调了**，是这一步没能兑现 |
        | 端点不可用 / 返回形状不符 | `degraded` | 同上 |

        ⚠ **降级不是吞异常**（D12 明确要求）：reranker 是唯一位于关键路径上、
        又不被规则保证可用的组件。它挂了必须**照样产出合法响应**，而不是 5xx——
        但**降级这件事必须留下痕迹**，不能像什么都没发生。
        """
        if self._reranker is None or not ranked:
            self.rerank_disabled += 1
            return ranked

        # 取每条候选的**渲染文本**当 rerank 的输入——与索引侧、与最终 content 是同一份渲染
        # （§7.2 / 不变式 I1）。**一次批量读**：短生命周期连接模型下，逐条查询要付 N 次 connect。
        pairs = self._fetch_for_rerank([c.memory_id for c in ranked])
        documents = [render(p.question, p.answer) for p in pairs if p is not None]
        if len(documents) != len(ranked):
            # 真源缺行 ⇒ 不 rerank（少了几条就没法一一对应）。扩窗那一步会把缺行的记下来。
            # ⚠ 计 `degraded` 而不是 `disabled`：**我们本来是要调的**。
            self.rerank_degraded += 1
            return ranked

        self.rerank_calls += 1
        try:
            scores = self._reranker.score(query=query, documents=documents)
        except RerankUnavailable:
            self.rerank_degraded += 1
            return ranked  # ★ 退回未重排的 RRF 顺序（D12）

        if len(scores) != len(ranked):
            # 形状不符与端点挂了一样不可用——**静默按前缀对齐会让后半段名次错位**。
            # ⚠ `RemoteReranker.score()` 自己保证长度一致（它校验 index 集合），
            #   所以走到这里说明**换了一个不守规矩的实现**——那正是这条判断存在的理由。
            self.rerank_degraded += 1
            return ranked

        # 分数降序；**同分按原名次**（稳定）——否则同分项的先后会随排序实现漂移，
        # 而那是"消融不可复现"的经典来源。
        #
        # ★ 这里只对**同一个列表**重排 ⇒ 候选集合不可能改变（§13 的开关纯度）。
        #   把"重排"留在调用方而不是 reranker 内部，就是为了让这件事是结构性的。
        order = sorted(range(len(ranked)), key=lambda i: (-float(scores[i]), i))
        return [
            Candidate(memory_id=ranked[i].memory_id, rank=new_rank)
            for new_rank, i in enumerate(order)
        ]

    def _fetch_for_rerank(self, ids: list[str]) -> list[QaPair | None]:
        with self._store.read() as conn:
            pairs = self._store.fetch_pairs_by_ids(conn, ids)
        by_id = {p.id: p for p in pairs}
        return [by_id.get(memory_id) for memory_id in ids]
