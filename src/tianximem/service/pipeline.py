"""`Add` 与 `Search` 的**编排**——把各层的调用串成真能跑的两条路径。

> **为什么编排在本目录**：本层"不做检索、不做配对、不碰存储"指的是**不重新实现**那些
> 逻辑（全部往下调用）；而"**按什么顺序调**"必须有人拥有——Search 的链条横跨 `retrieve/`
> 与 `rank/`，Add 的链条横跨 `pairing/`、`store/`、`embed/`，**没有任何单个下层模块能拥有
> 整条链**。所以顺序在这里，逻辑在下面。

## Search 链

启用共同取证时先将问题编译为字段条件/有限连接/显式区间计划，并统一返回原文或
有来源的独立事实。索引未齐、冲突、超限或句法不适用时沿用下列混合检索链。

```text
User Query → DenseArm（每 query 恰好 1 次）
           → HybridRetriever（BM25 + Dense + RRF）
           → memory_id 稳定去重
           → EvidenceChecker（v1 passthrough，每轮记账）
           → rerank（恰好一次；不可用则降级回 RRF 顺序）
           → Neighbor Expansion（全部候选保留，只对前 N 条扩 ±radius）
           → Context Segment Merge（段内 local_index 序，段间 best_rank 序）
           → Token Budget + Final Packaging
           → ≤ top_k 个**段**
```

端点不可用时按 D12 降级回 RRF 顺序并记 `rerank_degraded`；没配/关掉时记
`rerank_disabled`。理由与降级契约写在 [`../rank/reranker.py`](../rank/reranker.py)。

## `Add` 的链，以及那个**必须专门处理**的失败窗口

```text
canonical AddRequest
  → apply_batch：payload 指纹 → 幂等守卫 → compose_memory_blocks → link_blocks（Add 内邻接）
                 → SQLite 事务提交（真源，位置 = (request_id, local_index)）
  → 取出本批写下的行（按 id）
  → render → dense embedding → Qdrant upsert(wait=True)
  → 响应
```

**D24 起组合只看本批**（`compose_memory_blocks` 是纯函数、无跨 Add 状态），
**D28 起邻接也只看本批**（`link_blocks`），所以这一段没有"跨批续接"带来的额外分支：
本批触碰的行**恰好**是它自己新写的那几行。

**没有 session 锁**：位置由请求派生（不是 `MAX+1`）⇒ **同 `(user_id, session_id)`
的 Add 可以并发**，它们在 `store.transaction()` 的 `BEGIN IMMEDIATE` 处排队（数据库级
写者串行），而不是在应用层互斥。**D28 之后连"顺序"这个概念都不再存在**——
不同 Add 之间既不比较先后，也不建立邻接。

**失败窗口**：SQLite 事务**已提交** → Qdrant 的 embedding/upsert **失败** → 客户端重试
**同一个 `request_id`**。此时 `applied_batches` 里**已经有这一批**，幂等守卫会命中 ⇒
`apply_batch` 直接返回、**不写任何东西**。若这里跟着返回 200，就会留下
**SQLite 有真源、Qdrant 永久缺索引** ⇒ 那些记忆**永远检索不到，且没有任何报错**。

### 修复路径：按 **request_id** 幂等重建（不动 `pairing/`）

`ApplyBatchResult.applied` 在**守卫命中时是 `False`**（`apply_batch` 里那行早返回），
所以重放时拿不到"本批写了哪些行"。**但修复不需要那个列表**：`user_id` / `session_id` /
`request_id` 都在请求里，`fetch_by_request(...)` 能拿到**那一次 Add**写下的全部块
（**D28 起作用域就是这一次 Add**），于是：

* `point_id` 是位置派生的纯函数 ⇒ 同一对永远是同一个 point
* `upsert` 幂等 ⇒ 已经写对的会被原样覆盖
* embedding 缓存按内容哈希 ⇒ 之前成功索引过的对**必然已在缓存里**，重放几乎不产生远程调用

⇒ **净效果**：不变量"该 Add 的每个 SQLite 行都在 Qdrant 里"被恢复，
且**没有让 SQLite 去迁就 Qdrant**（真源不变，派生索引被修复）。

> **代价（D28 之前）**：旧口径下重放要 upsert 该 session 的**全部**对，因为
> `applied_batches` 没记"本批碰了哪些行"。现在位置里带着 `request_id`，
> 按它一查就是本批——**代价没有了**，而且顺带把"修复会不会碰到别的 Add 的行"
> 这个问题取消掉了（它碰不到）。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from time import monotonic

from tianximem.common.config import (
    DEFAULT_EXPANSION_SEED_LIMIT,
    DEFAULT_RADIUS,
)
from tianximem.common.render import render_pair
from tianximem.common.tokens import TokenCounter
from tianximem.facts.evidence import EVIDENCE_VERSION
from tianximem.facts.query import FactPattern, compile_query
from tianximem.observability import MetricsSink, NullMetricsSink, SearchObservation
from tianximem.pairing import AddBatch, ApplyBatchResult, apply_batch
from tianximem.pairing.apply import index_pair_facts
from tianximem.rank import (
    PackagedResponse,
    Reranker,
    RerankUnavailable,
    expand_neighbors,
    merge_segments,
    package,
)
from tianximem.rank.neighbor import ContextSegment
from tianximem.retrieve import EvidenceChecker, HybridRetriever, dedup_candidates
from tianximem.retrieve.evidence import EvidenceSelection, select_evidence
from tianximem.retrieve.fusion import Candidate
from tianximem.store.qdrant_store import QdrantStore
from tianximem.store.sqlite_store import QaPair, SqliteStore

__all__ = ["AddOutcome", "AddPipeline", "SearchPipeline"]


@dataclass(frozen=True, slots=True)
class AddOutcome:
    """一次 Add 的结果（**不进响应**——响应只有 §2.1 那三个回显字段）。"""

    applied: bool
    """False = 幂等守卫命中（本批此前已应用过）。"""

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
        #: T1 的渲染变体（`packaging.inject_abs_time`）——**索引侧也要用它**：
        #: 正文一变，embedding 输入就变（§7.2 的"同一份渲染"）。
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
            # 守卫命中（重试）⇒ **按这一次 Add 幂等重建**（D28：作用域就是 request_id）
            return self._batch_pairs(batch), True

        # 正常路径：本批写下的**全部**行（D24 之后没有"被续接/被关闭的既有对"——
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

        ⚠ D28 起作用域是 **`request_id`**：

        * **更准**：那次 Add 提交了哪些行、Qdrant 就可能缺哪些行——一行不多、一行不少
        * **更便宜**：长 session 上"整段重建"的代价随 session 增长，而它其实只需要一批
        * **不会碰到别人**：修复路径再也不需要回答"我会不会覆盖另一个 Add 的 point"

        ⚠ 返回空列表是**合法**的（那一批没写出任何行、或行已被清掉），调用方照常走完。
        """
        with self._store.read() as conn:
            return self._store.fetch_by_request(
                conn, batch.user_id, batch.session_id, batch.request_id
            )


class SearchPipeline:
    """`Search` 的完整时序。**不生成答案**（§2.1 的红线）。

    ## 顺序（每一步都有理由，不要调换）

    ```text
    ① 混合检索（BM25 + Dense → RRF）   每路 1 次查询；DenseArm 保证每 query 恰好 1 次 embedding
    ② memory_id 稳定去重 + 重编号      必须在 rerank **之前**：重复项会被排两次、占两个名额
    ③ Evidence Checker（v1 恒"充足"）   每轮判定都记账（D13）
    ④ rerank（**恰好一次**）            `RemoteReranker`；不可用 ⇒ 降级回 RRF 顺序（D12）
    ⑤ Neighbor Expansion               全部候选保留；只对前 N 条扩 ±radius
    ⑥ Context Segment Merge            沿显式链合成段（段内 local_index 序）
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
        inject_abs_time: bool = False,
        seed_placement: str = "keep",
        annotate_relatives: bool = False,
        grounded_evidence: bool = False,
        evidence_limit: int = 12,
        evidence_hop_limit: int = 4,
        fact_backfill_limit: int = 1024,
        metrics: MetricsSink | None = None,
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
        #: T1 的渲染变体（`packaging.inject_abs_time`）。**搜索链上有两处要用它**：
        #: 精排的输入文本（`_maybe_rerank`）与最终的 `content`（`merge_segments` 逐对渲染）。
        #: 漏掉任何一处 ⇒ 与索引侧的渲染分叉，**而分叉不报错**（不变式 I1）。
        self._inject_abs_time = inject_abs_time
        #: 段内顺序（§11.2 的"组内顺序"，明文列为可消融项）。
        #: **只影响正文怎么排**——`id` / 段数 / 锚点 / `best_rank` 一个都不动。
        self._seed_placement = seed_placement
        #: 正文里的相对时间就地注解成绝对日期（`packaging.annotate_relatives`）。
        #: ⚠ **只在 `content` 这一条路上生效**——精排输入与索引侧仍是 `render_pair`，
        #: 所以开它**不改 embedding 输入**（不用重建索引）。理由与实测见
        #: [`neighbor.py`](../rank/neighbor.py) 的 `_build_segment`。
        self._annotate_relatives = annotate_relatives
        self._grounded_evidence = grounded_evidence
        self._evidence_limit = evidence_limit
        self._evidence_hop_limit = evidence_hop_limit
        if fact_backfill_limit <= 0:
            raise ValueError("fact_backfill_limit 必须为正")
        self._fact_backfill_limit = fact_backfill_limit
        #: 观测值的出口（§14）。**没配就是 `Null`**——与 reranker 同一套口径：缺省不是错误。
        self._metrics = metrics if metrics is not None else NullMetricsSink()
        #: 诊断计数（§14）。**不进响应**——响应的形状是契约，一个字段都不能多。
        #:
        #: ⚠ **它们是累计值**（`run()` 每次读前后差值发给 `_metrics`，见那里的注释），
        #: 所以读的人拿到的总是"这个进程开到现在"。出口是
        #: [`../observability/`](../observability/)——**别在响应里给它们找位置**。
        self.rerank_calls = 0
        self.rerank_degraded = 0
        self.rerank_disabled = 0

    @contextmanager
    def _fact_snapshot(self, user_id: str) -> Iterator[sqlite3.Connection | None]:
        """同一版本只补扫描一次；覆盖未齐全时继续普通检索。"""
        with self._store.read_snapshot() as conn:
            if self._store.has_fact_index_coverage(conn, user_id, version=EVIDENCE_VERSION):
                yield conn
                return
        with self._store.transaction() as conn:
            for pair in self._store.fetch_unindexed_pairs(
                conn, user_id, version=EVIDENCE_VERSION, limit=self._fact_backfill_limit
            ):
                index_pair_facts(self._store, conn, pair)
        with self._store.read_snapshot() as conn:
            yield (
                conn
                if self._store.has_fact_index_coverage(conn, user_id, version=EVIDENCE_VERSION)
                else None
            )

    @property
    def reranker(self) -> Reranker | None:
        """当前接的 reranker（`None` = 没配 / 显式关掉）。

        只读暴露给**测试**（断言装配真的接上了，不是又传了 `None`）与**诊断**
        （§14 的 run record 要记"这次用的哪个 reranker"，D12）。
        """
        return self._reranker

    @property
    def metrics(self) -> MetricsSink:
        """当前的指标出口（`NullMetricsSink` = 没配路径）。

        与 `reranker` 同一个理由：**装配漏传是静默的**——出口没接上时 Search 照常工作、
        计数照常累计，只是**永远不写文件**（又一条 V12）。
        """
        return self._metrics

    def run(self, *, user_id: str, query: str, top_k: int) -> PackagedResponse:
        """跑完整条链，返回打包好的响应，并**把这一次的观测值发出去**（§14）。

        ⚠ 三个 rerank 计数在实例上是**累计值**，而 `SearchObservation` 要的是**增量**
        （[`../observability/metrics.py`](../observability/metrics.py) 的模块 docstring）
        ⇒ 这里前后各取一次、相减。**直接把累计值发出去会让聚合端把它们加成
        `1+2+3+…`——而那个数字只是"有点大"，不报错。**

        ⚠ **失败的那一次不发**：耗时与计数只统计"真的返回了响应"的请求
        （非 200 由 AML 重试，§15；把失败混进 latency 会让均值没有解释）。
        """
        started = monotonic()
        calls, disabled, degraded = self.rerank_calls, self.rerank_disabled, self.rerank_degraded
        response = self._search(user_id=user_id, query=query, top_k=top_k)
        self._metrics.record(
            SearchObservation(
                latency_ms=round((monotonic() - started) * 1000.0, 3),
                rerank_calls=self.rerank_calls - calls,
                rerank_disabled=self.rerank_disabled - disabled,
                rerank_degraded=self.rerank_degraded - degraded,
                rerank_name=self._reranker.name if self._reranker is not None else None,
            )
        )
        return response

    def _search(self, *, user_id: str, query: str, top_k: int) -> PackagedResponse:
        """链本身（顺序见类 docstring）。

        ⚠ **空库直接返回空结果**（`data: []` 是合法的，§2.1）：集合还不存在时
        Qdrant 会抛"collection not found"，而"没有数据"不是错误。
        顺带也**不为一次必然空的检索付远程 embedding 调用**。
        ⚠ 注意这里的短路**早于** token 计数——空库时不该去加载分词器。
        """
        if top_k <= 0:
            return PackagedResponse(items=())
        if self._grounded_evidence:
            plan = compile_query(query)
            if plan is not None:
                if plan.operator == "current":
                    return PackagedResponse(items=())
                fetch_limit = self._evidence_limit * 8
                patterns = plan.patterns
                if plan.resolve_references:
                    patterns = tuple(FactPattern(subject=p.subject) for p in patterns)
                if plan.operator == "walk":
                    patterns = (*patterns, FactPattern(("explicit replacement",)))
                with self._fact_snapshot(user_id) as conn:
                    evidence = (
                        []
                        if conn is None
                        else self._store.fetch_evidence(
                            conn, user_id, patterns, limit=fetch_limit + 1
                        )
                    )
                    sources = []
                    audit = []
                    if conn is not None and plan.guard_literal:
                        sources = self._store.fetch_evidence_sources(
                            conn, user_id, plan.guard_literal, limit=self._evidence_limit + 1
                        )
                        if sources:
                            audit = self._store.fetch_evidence(
                                conn,
                                user_id,
                                (FactPattern(source_ids=tuple(p.id for p in sources)),),
                                limit=fetch_limit + 1,
                                deduplicate=False,
                            )
                if (
                    len(evidence) <= fetch_limit
                    and len(audit) <= fetch_limit
                    and len(sources) <= self._evidence_limit
                ):
                    selected = select_evidence(
                        evidence,
                        plan,
                        source_limit=self._evidence_limit,
                        hop_limit=self._evidence_hop_limit,
                        sources=sources,
                        audit_facts=audit,
                    )
                    if selected is not None:
                        return self._package_evidence(selected, top_k=top_k)
        if not self._qdrant.exists():
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
        segments = merge_segments(
            expansion.selected,
            counter=self._counter,
            inject_abs_time=self._inject_abs_time,
            seed_placement=self._seed_placement,
            annotate_relatives=self._annotate_relatives,
        )

        # ⑦ 预算 + 打包（段是原子单位；`top_k` 约束的是**段数**）
        return package(
            segments,
            top_k=top_k,
            counter=self._counter,
            max_tokens=self._budget_tokens,
            dropped_missing=expansion.missing_rows,
        )

    def _package_evidence(self, selection: EvidenceSelection, *, top_k: int) -> PackagedResponse:
        """原文/原子事实共用出处验证、段结构与双预算；此路径半径始终为零。"""
        facts = selection.facts
        with self._store.read() as conn:
            parents = {
                p.id: p
                for p in self._store.fetch_pairs_by_ids(conn, [f.parent_memory_id for f in facts])
            }
        valid = [
            f
            for f in facts
            if f.parent_memory_id in parents and parents[f.parent_memory_id].user_id == f.user_id
        ]
        if selection.projection == "source":
            expansion = expand_neighbors(
                [Candidate(memory_id=f.parent_memory_id, rank=i) for i, f in enumerate(valid)],
                store=self._store,
                seed_limit=0,
                radius=0,
            )
            segments = merge_segments(
                expansion.selected,
                counter=self._counter,
                inject_abs_time=self._inject_abs_time,
                annotate_relatives=self._annotate_relatives,
            )
        else:
            segments = [
                ContextSegment(
                    anchor_memory_id=f.id,
                    anchor_event_time=f.event_time,
                    content=f.content,
                    token_count=self._counter.count(f.content),
                    source_memory_ids=(f.parent_memory_id,),
                    user_id=f.user_id,
                    session_id=parents[f.parent_memory_id].session_id,
                    request_id=parents[f.parent_memory_id].request_id,
                    start_local_index=parents[f.parent_memory_id].local_index,
                    end_local_index=parents[f.parent_memory_id].local_index,
                    best_rank=i,
                    rerank_member_count=1,
                )
                for i, f in enumerate(valid)
            ]
        return package(segments, top_k=top_k, counter=self._counter, max_tokens=self._budget_tokens)

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

        ⚠ **降级不是吞异常**（D12 明确要求）：reranker 是唯一位于关键路径上、又不被规则
        保证可用的组件。它挂了必须**照样产出合法响应**，而不是 5xx——但**降级这件事
        必须留下痕迹**，不能像什么都没发生。
        """
        if self._reranker is None or not ranked:
            self.rerank_disabled += 1
            return ranked

        # 取每条候选的**渲染文本**当 rerank 的输入——与索引侧、与最终 content 是同一份渲染
        # （§7.2 / 不变式 I1）。**一次批量读**：短生命周期连接模型下，逐条查询要付 N 次 connect。
        pairs = self._fetch_for_rerank([c.memory_id for c in ranked])
        documents = [
            render_pair(p, inject_abs_time=self._inject_abs_time) for p in pairs if p is not None
        ]
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
