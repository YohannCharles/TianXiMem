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
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from time import monotonic
from typing import Protocol, runtime_checkable

from tianximem.agent import AgenticResult, PlanResult
from tianximem.common.amounts import annotate_unit_total
from tianximem.common.config import (
    DEFAULT_EXPANSION_SEED_LIMIT,
    DEFAULT_RADIUS,
)
from tianximem.common.render import render_pair
from tianximem.common.tokens import TokenCounter
from tianximem.facts.evidence import EVIDENCE_VERSION
from tianximem.facts.query import FactPattern, QueryPlan, compile_query
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
from tianximem.retrieve.evidence import (
    EvidenceDecision,
    EvidenceSelection,
    evaluate_evidence,
    walk_source_literals,
    walk_sources_supported,
    walk_subjects,
)
from tianximem.retrieve.fusion import Candidate
from tianximem.retrieve.router import QueryRouter, Route
from tianximem.store.qdrant_store import QdrantStore
from tianximem.store.sqlite_store import QaPair, SqliteStore

__all__ = ["AddOutcome", "AddPipeline", "AgenticSearch", "SearchPipeline"]


@runtime_checkable
class AgenticSearch(Protocol):
    """多跳执行器的**接口约定**（§3.3）——两个实现都在 [`../agent/`](../agent/)。

    | `agentic.mode` | 实现 | 怎么推进 |
    | --- | --- | --- |
    | `react`（默认） | `CoqAgent` | 每轮重新决定"充分了吗 / 下一跳问什么" |
    | `plan` | `PlanAgent` | 先拆一份固定计划，再逐题落地，**不重规划** |

    ⚠ **本层不认识模式**：装配期（`service/app.py` 的 `build_agentic`）已经选定一个
    执行器注入进来，这里只调它——于是"选路"与"执行器是谁"是两件事，
    ⛔ 不让编排层再判一次模式（那会让"接错了"表现为"跑的是另一套机制"，
    而**两边都正常返回**）。

    它是 `MULTI_HOP` 那条路的**唯一入口**，形状刻意定成"**再跑一遍底层检索**"：

    ```text
    agentic.search(user_id=..., query=..., top_k=...) → AgenticResult | PlanResult
              ↓  与混合检索那条路**汇入同一个尾巴**
        去重 → 精排（恰好一次，用它给的 rerank_query）→ 扩窗 → 段合并 → 双预算打包
    ```

    ## 四条约定（写在这里，因为它是"接口"而不是"实现细节"）

    1. **它返回候选，不返回答案**（§2.1 的红线）。多跳只是"多取几轮证据"，
       产出物仍然是**按名次排列的证据**——`Candidate` 这个类型本身就说明了这一点。
       ⚠ Plan 模式**确实会生成中间跳的答案**（用来填占位），但那**不是**这条约定的
       反例：它不进响应、不进索引、不落库，只以**执行记录**的身份被交出去
       （`PlanResult.plan_history`）。
    2. **它自己去重之前可以有重复**：合并多轮结果会产生同一个 `memory_id` 多次命中，
       而"去重恰好发生一次、且发生在精排之前"是**调用方（本文件）**的责任
       （顺序错了不报错：重复项会被排两次、占两个名额）。
    3. **最终精排不归它做**：那一步**恰好一次**，由本文件对**合并后的那一份列表**做
       （§7.2 的"每查询恰好一次"同款纪律）。⚠ 但它**可以说该用什么 Query 去排**——
       那就是 `rerank_query`（多跳的联合精排，§9.1）。
       ⛔ 让执行器自己排最后那一次会造出**两次最终精排**，而 §9.4 专门点了这个坑。
       ⚠ 它**内部**每轮还会精排一次（为了把交给 LLM 的证据筛干净）——
       那是另一件事，产出物不进响应。
    4. **它不抛异常**：失败一律变成 `fallback_reason`。
       `fallback_reason is not None` ⇒ **回到混合检索**（§10.1 第 6 条）。

    ⚠ **两种结果的公共面**就是上面这四个字段（`candidates` / `rerank_query` /
    `stats` / `fallback_reason`），`_record_agentic` 只读它们 ⇒ **"两条路怎么记账"
    只有一处实现**。`PlanResult` 另外带 `plan_history`（诊断用，不进响应）。

    ⚠ **`top_k` 是"每轮取多少"的口径提示，不是最终名额**：最终 `len(data) <= top_k`
    由打包那一步收口（段级计数，§2.2）。执行器不必也不该自己截断到 `top_k`。
    ⚠ **两个实现在这个参数上的行为相同**（2026-10-13 起）：每轮按请求的 `top_k` 召回，
    精排之后再取 `per_round_top_k` 条。⇒ 两条路的差别**只剩推进机制**——
    这正是"多跳模式值不值"这个对照要的单变量（第一版 `plan` 只召回 20 条，
    两个自变量缠在一起，见 `PlanAgent.search` 的 docstring）。
    """

    def search(self, *, user_id: str, query: str, top_k: int) -> AgenticResult | PlanResult: ...


@dataclass(slots=True)
class _SearchTrace:
    """请求局部诊断；并发搜索不共享状态，也不改变 HTTP 响应。"""

    grounded_status: str = "disabled"
    grounded_operator: str | None = None
    grounded_projection: str | None = None
    selected_facts: int = 0
    selected_sources: int = 0
    rerank_calls: int = 0
    rerank_disabled: int = 0
    rerank_degraded: int = 0
    #: 路由（§4.3 的三分法，与 rerank 同形）。
    #: `disabled` = **路由器没开**；`success` = 模型正常分类；`degraded` = 想用但降级了。
    #: ⚠ 后两者**必须分得开**——混为一谈会把"端点一直挂"看成"我们本来就没开路由"（D12）。
    route: str = "disabled"
    route_status: str = "disabled"
    #: 分类**本身**降级的原因（`llm_unavailable` / `llm_timeout` / `invalid_output` …）。
    route_reason: str | None = None
    #: 分类成功、但**选中的那条路没跑成**的原因（`agentic_unavailable` / `retrieval_error` …）。
    #: ⚠ 与上面那个字段是两件事：一个是"没分成类"，一个是"分对了但执行器没跑成"。
    route_fallback: str | None = None
    route_latency_ms: float | None = None
    #: 多跳执行器的汇总（§14）。⚠ `agentic_rounds = 0` 且 `agentic_status = "disabled"`
    #: 表示**这条路没被走到**，不是"走了 0 轮"——两者要分得开（同 rerank 的三分口径）。
    agentic_status: str = "disabled"
    agentic_rounds: int = 0
    #: 它自己停下来的原因（§11 的 `stop_reason`）。⚠ 与 `route_fallback` 是两件事：
    #: 一个是"这条路的结局"，一个是"它压根没跑成"。
    agentic_stop_reason: str | None = None
    agentic_llm_calls: int = 0
    agentic_rerank_calls: int = 0
    agentic_selected_evidence: int = 0
    agentic_candidates: int = 0
    agentic_joint_queries: int = 0
    agentic_latency_ms: float | None = None


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
    ⓿ 查询路由（**仅在 `router.enabled` 时**） 分类 → COUNT / MULTI_HOP / DEFAULT
    ⓿′ 共同取证分支（`grounded_evidence`）   关着：**无条件先试**；开着：**仅 COUNT 走**
    ② memory_id 稳定去重 + 重编号             必须在 rerank **之前**：重复项会被排两次、占两个名额
    ③ Evidence Checker（v1 恒"充足"）          每轮判定都记账（D13）
    ④ rerank（**恰好一次**）                   `RemoteReranker`；不可用 ⇒ 降级回 RRF 顺序（D12）
    ⑤ Neighbor Expansion                      全部候选保留；只对前 N 条扩 ±radius
    ⑥ Context Segment Merge                   沿显式链合成段（段内 local_index 序）
    ⑦ Token Budget + Final Packaging          段是原子单位；`<= top_k` 的计数在这里收口
    ```

    ⚠ **①–⑦ 是"两条候选来源共用的尾巴"**：`MULTI_HOP` 走的是多跳执行器给的候选
    （`AgenticSearch`，实现见 [`../agent/`](../agent/)），其余情况走混合检索——
    之后**逐字共用** `_rank_and_package()`。那一段被抽出来**不是为了好看**：
    ② 与 ⑦ 是两条"顺序错了不报错"的纪律，写两遍就一定会漂。

    ⚠ **多跳那条路也不会多精排一次**：执行器交出候选 + **该用哪句 Query 精排**
    （联合精排，§9.1），④ 仍然只做一次——这是结构性的，不是靠记性。

    ## 两个"顺序错了也不会报错"的地方

    * **去重必须在 rerank 之前**：不去重的话，同一份证据会被 rerank 两次、
      在预算里占两个名额——而**名次看起来完全正常**。
    * **`top_k` 必须在扩窗/合并之后生效**：它约束的是**段数**。在扩窗之前按
      raw memory 数截断，会把本该形成段的邻居砍掉，最后返回的段数**少于该有的**。

    ## 路由带来的第三处（**这一处会改行为——所以它默认关着**）

    * **路由开着时，`DEFAULT` 类问题不再先试共同取证。** 共同取证是一条
      **确定性**的短路（编译问题 → 取值 → 打包），而路由是**模型判断**
      ⇒ 这是一次"用确定性换分类能力"的取舍，不是白拿的。
    * ⚠ **它的代价必须被测出来**：`compile_query` 支持 `walk` / `interval` /
      `current` 这些**非计数**问法，而它们在三分类里大多会落到 `DEFAULT` 或
      `MULTI_HOP` ⇒ **那些问题会失去专门的取证能力**。回归数据见
      `eval/reports/` 的路由报告——**不许用"改旧断言"把它盖过去**。
    * ⇒ 这正是 `router.enabled` **默认 `false`** 的理由：一个未经代理评测标定的
      行为改动，不该默认生效在提交口径上。
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
        annotate_unit_totals: bool = False,
        grounded_evidence: bool = False,
        evidence_limit: int = 12,
        evidence_hop_limit: int = 4,
        fact_backfill_limit: int = 1024,
        router: QueryRouter | None = None,
        agentic: AgenticSearch | None = None,
        agentic_force: bool = False,
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
        self._annotate_unit_totals = annotate_unit_totals
        self._grounded_evidence = grounded_evidence
        self._evidence_limit = evidence_limit
        self._evidence_hop_limit = evidence_hop_limit
        if fact_backfill_limit <= 0:
            raise ValueError("fact_backfill_limit 必须为正")
        self._fact_backfill_limit = fact_backfill_limit
        #: 查询路由器（`router.enabled`）。**`None` = 没有路由器** ⇒ 这条链上
        #: 一次 LLM 调用都不发生，且共同取证分支照旧**无条件先试**（= 本接缝存在之前的行为）。
        #: ⚠ 它是 `None` 的两种来源**要分得开**（WARNING 由装配侧发，见 `build_router`）：
        #: "明确关掉"（默认）与"开着却没配齐端点"——后者由 `validate()` 在启动时拒绝，
        #: 所以走到这里时 `None` 只可能是前者。
        self._router = router
        #: 多跳执行器（§3.3 的接缝）。实现见 [`../agent/`](../agent/)（2026-10-10）。
        #: ⚠ **它不需要一个 `enabled` 布尔**：装上了就是开，没装上就是关——
        #: 再加一个布尔会造出"开关说开着、执行器是 None"这种自相矛盾的状态。
        #: `agentic.enabled = false`（默认）⇒ 装配侧不给它，于是每条 `MULTI_HOP`
        #: 都降级成 `DEFAULT` 并记 `agentic_unavailable`。
        self._agentic = agentic
        #: **实验开关**（`agentic.force`，§12.3 的 E1）：绕过路由器，一律按 MULTI_HOP 走。
        #: ⛔ 它不是给提交口径用的——打开它等于关掉另外两条路的选择能力。
        #: 关着（默认）时这条链与本开关存在之前**逐字相同**。
        self._agentic_force = agentic_force
        #: 观测值的出口（§14）。**没配就是 `Null`**——与 reranker 同一套口径：缺省不是错误。
        self._metrics = metrics if metrics is not None else NullMetricsSink()
        #: 诊断计数（§14）。**不进响应**——响应的形状是契约，一个字段都不能多。
        #:
        #: ⚠ **它们是累计值**（`run()` 单独用请求局部诊断发增量给 `_metrics`），
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

    @property
    def router(self) -> QueryRouter | None:
        """当前接的查询路由器（`None` = 没开）。

        与 `reranker` 同一个理由：**装配漏传是静默的**——路由没接上时 Search 照常工作、
        响应照旧合法，只是**每一题都走原来的路**（而"Router On"那一臂的结论就是错的）。
        """
        return self._router

    @property
    def agentic(self) -> AgenticSearch | None:
        """当前接的多跳执行器（**今天恒为 `None`**，D13）。"""
        return self._agentic

    def _grounded_decision(self, user_id: str, plan: QueryPlan) -> EvidenceDecision:
        """事实选择与相关原文审查共用一个读快照；选择器仅执行一次。"""
        fetch_limit = self._evidence_limit * 8
        patterns = plan.patterns
        if plan.resolve_references:
            patterns = tuple(FactPattern(subject=p.subject) for p in patterns)
        if plan.operator == "walk":
            patterns = (*patterns, FactPattern(("explicit replacement",)))
        with self._fact_snapshot(user_id) as conn:
            if conn is None:
                return EvidenceDecision(None, "index_incomplete")
            evidence = self._store.fetch_evidence(conn, user_id, patterns, limit=fetch_limit + 1)
            if len(evidence) > fetch_limit:
                return EvidenceDecision(None, "fetch_limit")
            sources, audit = [], []
            if plan.guard_literal:
                source_scan_limit = (
                    fetch_limit if plan.guard_assertion_pattern else self._evidence_limit
                )
                by_id: dict[str, QaPair] = {}
                for literal in (plan.guard_literal, *plan.guard_source_literals):
                    for source in self._store.fetch_evidence_sources(
                        conn, user_id, literal, limit=source_scan_limit + 1
                    ):
                        by_id.setdefault(source.id, source)
                    if len(by_id) > source_scan_limit:
                        return EvidenceDecision(None, "source_scan_limit")
                sources = list(by_id.values())
                if sources:
                    audit = self._store.fetch_evidence(
                        conn,
                        user_id,
                        (FactPattern(source_ids=tuple(p.id for p in sources)),),
                        limit=fetch_limit + 1,
                        deduplicate=False,
                    )
                    if len(audit) > fetch_limit:
                        return EvidenceDecision(None, "audit_limit")
            decision = evaluate_evidence(
                evidence,
                plan,
                source_limit=self._evidence_limit,
                hop_limit=self._evidence_hop_limit,
                sources=sources,
                audit_facts=audit,
            )
            if (
                plan.operator != "walk"
                or not plan.require_complete_walk
                or decision.selection is None
            ):
                return decision
            subjects = walk_subjects(evidence, decision.selection, plan)
            related: dict[str, QaPair] = {}
            for literal in walk_source_literals(subjects):
                for source in self._store.fetch_evidence_sources(
                    conn, user_id, literal, limit=fetch_limit + 1
                ):
                    related[source.id] = source
                if len(related) > fetch_limit:
                    return EvidenceDecision(None, "source_scan_limit")
            audit = (
                self._store.fetch_evidence(
                    conn,
                    user_id,
                    (FactPattern(source_ids=tuple(related)),),
                    limit=fetch_limit + 1,
                    deduplicate=False,
                )
                if related
                else []
            )
            if len(audit) > fetch_limit:
                return EvidenceDecision(None, "audit_limit")
            if not walk_sources_supported(
                tuple(related.values()), decision.selection, audit, subjects
            ):
                return EvidenceDecision(None, "unsupported_source")
            return decision

    def run(self, *, user_id: str, query: str, top_k: int) -> PackagedResponse:
        """跑完整条链，返回打包好的响应，并**把这一次的观测值发出去**（§14）。

        ⚠ 三个 rerank 计数在实例上是累计值；观测采用请求局部增量，
        避免并发搜索把其他请求的调用或取证原因算进来。

        ⚠ **失败的那一次不发**：耗时与计数只统计"真的返回了响应"的请求
        （非 200 由 AML 重试，§15；把失败混进 latency 会让均值没有解释）。
        """
        started = monotonic()
        trace = _SearchTrace()
        response = self._search(user_id=user_id, query=query, top_k=top_k, trace=trace)
        self._metrics.record(
            SearchObservation(
                latency_ms=round((monotonic() - started) * 1000.0, 3),
                rerank_calls=trace.rerank_calls,
                rerank_disabled=trace.rerank_disabled,
                rerank_degraded=trace.rerank_degraded,
                rerank_name=self._reranker.name if self._reranker is not None else None,
                grounded_status=trace.grounded_status,
                grounded_operator=trace.grounded_operator,
                grounded_projection=trace.grounded_projection,
                selected_facts=trace.selected_facts,
                selected_sources=trace.selected_sources,
                route=trace.route,
                route_status=trace.route_status,
                route_reason=trace.route_reason,
                route_fallback=trace.route_fallback,
                route_latency_ms=trace.route_latency_ms,
                agentic_status=trace.agentic_status,
                agentic_rounds=trace.agentic_rounds,
                agentic_stop_reason=trace.agentic_stop_reason,
                agentic_llm_calls=trace.agentic_llm_calls,
                agentic_rerank_calls=trace.agentic_rerank_calls,
                agentic_selected_evidence=trace.agentic_selected_evidence,
                agentic_candidates=trace.agentic_candidates,
                agentic_joint_queries=trace.agentic_joint_queries,
                agentic_latency_ms=trace.agentic_latency_ms,
                returned_segments=len(response.items),
                truncated_by_budget=response.truncated_by_budget,
                truncated_by_top_k=response.truncated_by_top_k,
                dropped_missing=response.dropped_missing,
            )
        )
        return response

    def _search(
        self, *, user_id: str, query: str, top_k: int, trace: _SearchTrace | None = None
    ) -> PackagedResponse:
        """链本身（顺序见类 docstring 与下面的注释）。

        ⚠ **空库直接返回空结果**（`data: []` 是合法的，§2.1）：集合还不存在时
        Qdrant 会抛"collection not found"，而"没有数据"不是错误。
        顺带也**不为一次必然空的检索付远程 embedding 调用**。
        ⚠ 注意这里的短路**早于** token 计数——空库时不该去加载分词器。
        """
        if trace is None:
            trace = _SearchTrace()
        if top_k <= 0:
            trace.grounded_status = "invalid_top_k"
            return PackagedResponse(items=())
        # ⚠ **空白 query 在这里短路，绝不喂给路由器**（2026-10-10）：
        #   ① 空查询不是"分类问题"，让模型去判它是把**一个输入校验**外包给 LLM
        #      ——既贵又不可靠（提示词一旦没有 NONE 这一档，它就只能瞎猜三类之一）；
        #   ② 契约要的是 `{"data": []}`（§2.1 明确空结果合法），而这条短路给出的**正是它**；
        #   ③ **放在路由之前**是关键：`_route()` 只要被调到就会花一次 LLM 调用。
        # 与 `top_k <= 0` 同一条出口、同一个记法（`grounded_status` 是既有先例）。
        if not query.strip():
            trace.grounded_status = "blank_query"
            return PackagedResponse(items=())

        # ⓿ 查询路由（**在一切检索之前**）。`None` = 没开路由器 ⇒ 下面的分支
        #    与"本接缝存在之前"逐字相同（§13 的开关纯度）。
        route = self._route(query, trace)

        # ⓿′ 共同取证分支。**它刻意早于 Qdrant 存在性检查**（事实住在 SQLite）。
        #    ⚠ 开关打开时**只有 `COUNT` 走这里**；没开时**一律走这里**（`route is None`）。
        if self._grounded_evidence and route in (None, Route.COUNT):
            trace.grounded_status = "no_plan"
            plan = compile_query(query)
            if plan is not None:
                trace.grounded_operator = plan.operator
                if plan.operator == "current":
                    trace.grounded_status = "current_input"
                    return PackagedResponse(items=())
                decision = self._grounded_decision(user_id, plan)
                trace.grounded_status = decision.reason
                selected = decision.selection
                if selected is not None:
                    trace.grounded_projection = selected.projection
                    trace.selected_facts = len(selected.facts)
                    trace.selected_sources = len({f.parent_memory_id for f in selected.facts})
                    return self._package_evidence(selected, top_k=top_k)
        if not self._qdrant.exists():
            return PackagedResponse(items=())

        # ① 取候选。**两条来源，同一个尾巴**：
        #    多跳执行器（`MULTI_HOP` 且真的接上了）或混合检索（其余全部情况）。
        rerank_query: str | None = None
        if route is Route.MULTI_HOP and self._agentic is not None:
            # ⚠ 执行器**自己去不去重、精不精排**是本文件的责任（见 `AgenticSearch` 的四条约定）：
            #   它只负责"多取几轮证据"，②–⑦ 与混合那条路**逐字共用**。
            outcome = self._agentic.search(user_id=user_id, query=query, top_k=top_k)
            self._record_agentic(trace, outcome)
            if outcome.fallback_reason is not None:
                # §10.1 第 6 条：执行器在拿到任何证据之前就失败了 ⇒ **改走混合检索**。
                # ⚠ 原因**换到 `route_fallback` 上**（不再是 `agentic_unavailable`）：
                #   "分对了但执行器没接上"与"执行器接上了却没跑成"是两件事，
                #   混成一个记法会让"多跳这条路的失败率"无法统计。
                trace.route_fallback = outcome.fallback_reason
                candidates: Sequence[Candidate] = self._retriever.search(
                    user_id=user_id, query=query, top_k=top_k
                )
            else:
                candidates = outcome.candidates
                # 多跳的联合精排（§9.1）：原问题 ＋ **实际执行过的**改写。
                # ⚠ 最终精排**仍然只有一次**，就在下面那一步——这不是"第二次精排"。
                rerank_query = outcome.rerank_query
        else:
            candidates = self._retriever.search(user_id=user_id, query=query, top_k=top_k)
        return self._rank_and_package(
            candidates, query=query, top_k=top_k, trace=trace, rerank_query=rerank_query
        )

    def _record_agentic(self, trace: _SearchTrace, outcome: AgenticResult | PlanResult) -> None:
        """把执行器的汇总记进 `trace`（§14）。

        ⚠ **执行器不写 `trace`**：它是被调用的那一层，"怎么记账"属于编排层，
        与 `rerank` 的分工一样（计数住在 `SearchPipeline`，见 `docs/architecture.md`）。

        ⚠ **两种模式共用这一处**：`PlanStats` 的前九个字段与 `AgenticStats` 同名同义
        （那是刻意的，不是巧合）⇒ 模式换了、记账的代码**一行都不用改**。
        模式专属的量（拆解调用数、计划步数、`depends_on` 被规范化的步数）**不进 `trace`**：
        它们要看得更细，走 `plan_history`（诊断记录），⛔ 不为它们扩 §14 的出口。
        """
        stats = outcome.stats
        trace.agentic_status = "success" if outcome.fallback_reason is None else "degraded"
        trace.agentic_rounds = len(stats.rounds)
        trace.agentic_stop_reason = stats.stop_reason
        trace.agentic_llm_calls = stats.llm_calls
        trace.agentic_rerank_calls = stats.rerank_calls
        trace.agentic_selected_evidence = stats.selected_evidence_count
        trace.agentic_candidates = stats.unique_candidate_count
        trace.agentic_joint_queries = stats.joint_query_count
        trace.agentic_latency_ms = stats.latency_ms

    def _route(self, query: str, trace: _SearchTrace) -> Route | None:
        """跑一次分类，把结果记进 `trace`；**没开路由器时返回 `None`**。

        ⚠ **绝不抛异常**：`QueryRouter.route()` 的所有失败形态都已经变成
        `Route.DEFAULT` + `status="degraded"`（§4.3：路由的失败不能让一次正常 Search 变 500）。
        本方法只做两件事：**把三分类映射成"走哪条分支"**，以及**记下为什么**。

        ⚠ `MULTI_HOP` 在**没有执行器**时**在这里就把降级原因记下来**——而不是等到分支里。
        理由：那条路会一直往下走到混合检索，如果不在这里留痕，
        "分类成功了但执行器不在"与"分类结果就是 DEFAULT"在 §14 上**长得一模一样**。
        """
        if self._agentic_force and self._agentic is not None:
            # ⚠ **实验开关**（`agentic.force`，§12.3 的 E1）：绕过路由器，一律按 MULTI_HOP 走。
            #   它必须**先于** `self._router is None` 的早返回被检查，否则"只开 agentic
            #   不开 router"这一臂会安静地什么都不做——正是 E1 要避免的归因错误。
            #   `trace.route` 记成 `MULTI_HOP` 且 `route_status = "forced"`：
            #   ⛔ **不要记成 `success`**——那会让"模型分了什么"与"我们强制了什么"分不开。
            trace.route = Route.MULTI_HOP.value
            trace.route_status = "forced"
            return Route.MULTI_HOP
        if self._router is None:
            return None
        decision = self._router.route(query)
        trace.route = decision.route.value
        trace.route_status = decision.status
        trace.route_reason = decision.reason
        trace.route_latency_ms = decision.latency_ms
        if decision.route is Route.MULTI_HOP and self._agentic is None:
            trace.route_fallback = "agentic_unavailable"
        return decision.route

    def _rank_and_package(
        self,
        candidates: Sequence[Candidate],
        *,
        query: str,
        top_k: int,
        trace: _SearchTrace,
        rerank_query: str | None = None,
    ) -> PackagedResponse:
        """②–⑦：**混合检索与多跳共用同一个尾巴**。

        ⚠ 它只认"一份候选"，**不关心它们从哪来**——这正是 §3.3 要的那个汇合点。
        把 ②–⑦ 抽出来（而不是在 `_search` 里写第二遍）是**结构性的**：
        "去重恰好一次、且在精排之前"与"`top_k` 在扩窗/合并之后生效"这两条
        **顺序错了不报错**的纪律，从此**只有一处实现**。

        ⚠ `rerank_query` **只换精排用的那一句**，不换别的：Checker 的判据、打包的
        `top_k`、扩窗的种子都仍然看**原问题**。它默认 `None` ⇒ 用 `query`，
        也就是混合链那条路的行为**逐字不变**。
        """
        # ② 稳定去重 + 重编号
        ranked = dedup_candidates(candidates)

        # ③ Evidence Checker（v1 恒"充足"）
        self._checker.decide(query=query)

        # ④ rerank —— 恰好一次
        ranked = self._maybe_rerank(
            query=query if rerank_query is None else rerank_query, ranked=ranked, trace=trace
        )

        # ⑤⑥ 扩窗 + 合并成段
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

        # ⑦ 预算 + 打包
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
            contents = [
                annotate_unit_total(f.content) if self._annotate_unit_totals else f.content
                for f in valid
            ]
            segments = [
                ContextSegment(
                    anchor_memory_id=f.id,
                    anchor_event_time=f.event_time,
                    content=content,
                    token_count=self._counter.count(content),
                    source_memory_ids=(f.parent_memory_id,),
                    user_id=f.user_id,
                    session_id=parents[f.parent_memory_id].session_id,
                    request_id=parents[f.parent_memory_id].request_id,
                    start_local_index=parents[f.parent_memory_id].local_index,
                    end_local_index=parents[f.parent_memory_id].local_index,
                    best_rank=i,
                    rerank_member_count=1,
                )
                for i, (f, content) in enumerate(zip(valid, contents, strict=True))
            ]
        return package(segments, top_k=top_k, counter=self._counter, max_tokens=self._budget_tokens)

    def _record_rerank(self, trace: _SearchTrace | None, field: str) -> None:
        setattr(self, field, getattr(self, field) + 1)
        if trace is not None:
            setattr(trace, field, getattr(trace, field) + 1)

    def _maybe_rerank(
        self, *, query: str, ranked: list[Candidate], trace: _SearchTrace | None = None
    ) -> list[Candidate]:
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
            self._record_rerank(trace, "rerank_disabled")
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
            self._record_rerank(trace, "rerank_degraded")
            return ranked

        self._record_rerank(trace, "rerank_calls")
        try:
            scores = self._reranker.score(query=query, documents=documents)
        except RerankUnavailable:
            self._record_rerank(trace, "rerank_degraded")
            return ranked  # ★ 退回未重排的 RRF 顺序（D12）

        if len(scores) != len(ranked):
            # 形状不符与端点挂了一样不可用——**静默按前缀对齐会让后半段名次错位**。
            # ⚠ `RemoteReranker.score()` 自己保证长度一致（它校验 index 集合），
            #   所以走到这里说明**换了一个不守规矩的实现**——那正是这条判断存在的理由。
            self._record_rerank(trace, "rerank_degraded")
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
