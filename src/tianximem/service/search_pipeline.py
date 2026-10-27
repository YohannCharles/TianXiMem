"""`Search` 链的**编排**（`AgenticSearch` / `_SearchTrace` / `SearchPipeline`）。

> **两条链各住一个文件**：本文件是 Search 链的家，Add 链在
> [`add_pipeline.py`](add_pipeline.py)；[`pipeline.py`](pipeline.py) 是薄 façade——只重新
> 导出两边的名字，“为什么编排在本目录”那条理由写在它的模块 docstring。

## Search 链

启用共同取证时先将问题编译为字段条件/有限连接/显式区间计划，并统一返回原文或
有来源的独立事实。索引未齐、冲突、超限或句法不适用时沿用下列混合检索链。

```text
User Query → DenseArm（每 query 恰好 1 次）
           → HybridRetriever（BM25 + Dense + RRF）
           → memory_id 稳定去重
           → EvidenceChecker（观测性判定，每轮记账）
           → rerank（恰好一次；不可用则降级回 RRF 顺序）
           → Neighbor Expansion（全部候选保留，只对前 N 条扩 ±radius）
           → Context Segment Merge（段内 local_index 序，段间 best_rank 序）
           → Token Budget + Final Packaging
           → ≤ top_k 个**段**
```

端点不可用时降级回 RRF 顺序并记 `rerank_degraded`；未配置或关闭时记
`rerank_disabled`。降级规则见 [`../rank/reranker.py`](../rank/reranker.py)。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
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
from tianximem.facts.query import QueryPlan, compile_query
from tianximem.observability import MetricsSink, NullMetricsSink, SearchObservation
from tianximem.rank import (
    PackagedResponse,
    Reranker,
    RerankUnavailable,
    expand_neighbors,
    merge_segments,
    package,
)
from tianximem.rank.neighbor import ContextSegment, SelectedMemory
from tianximem.retrieve import EvidenceChecker, HybridRetriever, dedup_candidates
from tianximem.retrieve.evidence import (
    EvidenceDecision,
    EvidenceSelection,
    evaluate_evidence,
)
from tianximem.retrieve.fusion import Candidate
from tianximem.retrieve.router import QueryRouter, Route
from tianximem.service.grounded_planner import GroundedPlanner
from tianximem.service.search_stages import GroundedEvidenceStage
from tianximem.store.qdrant_store import QdrantStore
from tianximem.store.sqlite_store import QaPair, SqliteStore, make_source_message_id

__all__ = ["AgenticSearch", "SearchPipeline"]


@runtime_checkable
class AgenticSearch(Protocol):
    """多跳执行器的接口约定；两个实现都在 [`../agent/`](../agent/)。

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

    1. **它返回候选，不返回答案**。多跳只是"多取几轮证据"，
       产出物仍然是**按名次排列的证据**——`Candidate` 这个类型本身就说明了这一点。
       ⚠ Plan 模式**确实会生成中间跳的答案**（用来填占位），但那**不是**这条约定的
       反例：它不进响应、不进索引、不落库，只以**执行记录**的身份被交出去
       （`PlanResult.plan_history`）。
    2. **它自己去重之前可以有重复**：合并多轮结果会产生同一个 `memory_id` 多次命中，
       而"去重恰好发生一次、且发生在精排之前"是**调用方（本文件）**的责任
       （顺序错了不报错：重复项会被排两次、占两个名额）。
    3. **最终精排不归它做**：本文件对合并后的候选列表精排恰好一次。执行器可以返回
       `rerank_query`，即多跳联合精排使用的查询。执行器内部每轮精排只用于筛选交给
       LLM 的证据，不进入响应。
    4. **它不抛异常**：失败一律变成 `fallback_reason`。该值非空表示调用方回到混合检索。

    ⚠ **两种结果的公共面**就是上面这四个字段（`candidates` / `rerank_query` /
    `stats` / `fallback_reason`），`_record_agentic` 只读它们 ⇒ **"两条路怎么记账"
    只有一处实现**。`PlanResult` 另外带 `plan_history`（诊断用，不进响应）。

    ⚠ **`top_k` 是"每轮取多少"的口径提示，不是最终名额**：最终 `len(data) <= top_k`
    由打包那一步按段数收口。执行器不自行截断最终结果。
     ⚠ **两个实现在这个参数上的行为相同**：每轮按请求的 `top_k` 召回，
     精排之后再取 `per_round_top_k` 条。⇒ 两条路只在推进机制上不同，
     这是比较多跳模式的单变量前提（见 `PlanAgent.search` 的 docstring）。
    """

    def search(self, *, user_id: str, query: str, top_k: int) -> AgenticResult | PlanResult: ...


@dataclass(slots=True)
class _SearchTrace:
    """请求局部诊断；并发搜索不共享状态，也不改变 HTTP 响应。"""

    grounded_status: str = "disabled"
    grounded_operator: str | None = None
    grounded_projection: str | None = None
    grounded_plan_source: str | None = None
    grounded_planner_status: str = "not_called"
    grounded_planner_calls: int = 0
    grounded_planner_latency_ms: float | None = None
    selected_facts: int = 0
    selected_sources: int = 0
    rerank_calls: int = 0
    rerank_disabled: int = 0
    rerank_degraded: int = 0
    #: 路由状态，与 rerank 的 disabled/degraded 区分方式一致。
    #: `disabled` = **路由器没开**；`success` = 模型正常分类；`degraded` = 想用但降级了。
    #: ⚠ 后两者**必须分得开**——混为一谈会把"端点一直挂"看成"我们本来就没开路由"。
    route: str = "disabled"
    route_status: str = "disabled"
    #: 分类**本身**降级的原因（`llm_unavailable` / `llm_timeout` / `invalid_output` …）。
    route_reason: str | None = None
    #: 分类成功、但**选中的那条路没跑成**的原因（`agentic_unavailable` / `retrieval_error` …）。
    #: ⚠ 与上面那个字段是两件事：一个是"没分成类"，一个是"分对了但执行器没跑成"。
    route_fallback: str | None = None
    route_latency_ms: float | None = None
    #: 多跳执行器的汇总。`agentic_rounds = 0` 且 `agentic_status = "disabled"`
    #: 表示**这条路没被走到**，不是"走了 0 轮"——两者要分得开（同 rerank 的三分口径）。
    agentic_status: str = "disabled"
    agentic_rounds: int = 0
    #: 多跳执行器的 `stop_reason`。它与 `route_fallback` 是两件事：
    #: 一个是"这条路的结局"，一个是"它压根没跑成"。
    agentic_stop_reason: str | None = None
    agentic_llm_calls: int = 0
    agentic_rerank_calls: int = 0
    agentic_selected_evidence: int = 0
    agentic_candidates: int = 0
    agentic_joint_queries: int = 0
    agentic_latency_ms: float | None = None


class SearchPipeline:
    """`Search` 的完整时序，**不生成答案**。

    ## 顺序（每一步都有理由，不要调换）

    ```text
    ⓿ 共同取证分支：完整证据直接返回；实验 force 模式除外
    ⓿′ 取证失败后查询路由（仅在 router.enabled 时）→ 多跳或混合检索
    ② memory_id 稳定去重 + 重编号             必须在 rerank **之前**：重复项会被排两次、占两个名额
    ③ Evidence Checker                        每轮判定都记账，结果保持观测性
    ④ rerank（**恰好一次**）                   `RemoteReranker`；不可用 ⇒ 降级回 RRF 顺序
    ⑤ Neighbor Expansion                      全部候选保留；只对前 N 条扩 ±radius
    ⑥ Context Segment Merge                   沿显式链合成段（段内 local_index 序）
    ⑦ Token Budget + Final Packaging          段是原子单位；最终数量在这里收口
    ```

    ⚠ **①–⑦ 是"两条候选来源共用的尾巴"**：`MULTI_HOP` 走的是多跳执行器给的候选
    （`AgenticSearch`，实现见 [`../agent/`](../agent/)），其余情况走混合检索——
    两条路径随后**逐字共用** `_rank_and_package()`。抽出该函数是为了固定顺序：
    ② 与 ⑦ 是两条"顺序错了不报错"的纪律，写两遍就一定会漂。

    多跳执行器只交出候选和用于最终精排的查询；④ 对合并后的列表仍然只执行一次。

    ## 两个"顺序错了也不会报错"的地方

    * **去重必须在 rerank 之前**：不去重的话，同一份证据会被 rerank 两次、
      在预算里占两个名额——而**名次看起来完全正常**。
    * **`top_k` 必须在扩窗/合并之后生效**：它约束的是**段数**。在扩窗之前按
      raw memory 数截断，会把本该形成段的邻居砍掉，最后返回的段数**少于该有的**。

    ## 取证与路由

    共同事实完整时直接返回，不调用路由器；未完成时才分类并交给多跳或混合链。
    路由器默认开启，agentic 仍默认关闭。
    agentic.force 保留显式绕过事实的实验语义。

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
        grounded_planner: GroundedPlanner | None = None,
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
        #: `packaging.inject_abs_time` 同时用于精排输入和最终 `content` 渲染，
        #: 必须与索引侧保持一致。
        self._inject_abs_time = inject_abs_time
        #: 段内顺序只影响正文排列，不改变 ID、段数、锚点或 `best_rank`。
        self._seed_placement = seed_placement
        #: 正文里的相对时间就地注解成绝对日期（`packaging.annotate_relatives`）。
        #: ⚠ **只在 `content` 这一条路上生效**——精排输入与索引侧仍是 `render_pair`，
        #: 所以开它**不改 embedding 输入**（不用重建索引）。理由与见
        #: [`neighbor.py`](../rank/neighbor.py) 的 `_build_segment`。
        self._annotate_relatives = annotate_relatives
        self._annotate_unit_totals = annotate_unit_totals
        self._grounded_evidence = grounded_evidence
        self._evidence_limit = evidence_limit
        self._evidence_hop_limit = evidence_hop_limit
        self._fact_backfill_limit = fact_backfill_limit
        #: Optional LLM interpretation for query surfaces the deterministic
        #: compiler does not recognize.  It only returns a bounded plan;
        #: execution and source auditing stay in GroundedEvidenceStage.
        self._grounded_planner = grounded_planner
        self._grounded_stage = GroundedEvidenceStage(
            store=store,
            evidence_limit=evidence_limit,
            evidence_hop_limit=evidence_hop_limit,
            fact_backfill_limit=fact_backfill_limit,
        )
        #: 查询路由器（`router.enabled`）。**`None` = 没有路由器** ⇒ 这条链上
        #: 一次 LLM 调用都不发生，且共同取证分支仍**无条件先试**。
        #: ⚠ 它是 `None` 的两种来源**要分得开**（WARNING 由装配侧发，见 `build_router`）：
        #: "明确关掉"（默认）与"开着却没配齐端点"——后者由 `validate()` 在启动时拒绝，
        #: 所以走到这里时 `None` 只可能是前者。
        self._router = router
        #: 多跳执行器。实现见 [`../agent/`](../agent/)。
        #: ⚠ **它不需要一个 `enabled` 布尔**：装上了就是开，没装上就是关——
        #: 再加一个布尔会造出"开关说开着、执行器是 None"这种自相矛盾的状态。
        #: `agentic.enabled = false`（默认）⇒ 装配侧不给它，于是每条 `MULTI_HOP`
        #: 都降级成 `DEFAULT` 并记 `agentic_unavailable`。
        self._agentic = agentic
        #: **实验开关** `agentic.force`：绕过路由器，把所有查询按 `MULTI_HOP` 处理。
        #: 它不是产品默认值；关闭时由正常证据路径和路由器选择路径。
        self._agentic_force = agentic_force
        #: 观测值出口；未配置时使用 `NullMetricsSink`，缺省不是错误。
        self._metrics = metrics if metrics is not None else NullMetricsSink()
        #: 进程级诊断计数，不进入 Search 响应。
        #:
        #: ⚠ **它们是累计值**（`run()` 单独用请求局部诊断发增量给 `_metrics`），
        #: 所以读的人拿到的总是"这个进程开到现在"。出口是
        #: [`../observability/`](../observability/)——**别在响应里给它们找位置**。
        self.rerank_calls = 0
        self.rerank_degraded = 0
        self.rerank_disabled = 0

    @contextmanager
    def _fact_snapshot(self, user_id: str) -> Iterator[sqlite3.Connection | None]:
        """向事实快照阶段转发请求。"""
        with self._grounded_stage.snapshot(user_id) as conn:
            yield conn

    @property
    def reranker(self) -> Reranker | None:
        """当前接的 reranker（`None` = 没配 / 显式关掉）。

                只读暴露给**测试**（断言装配真的接上了，不是又传了 `None`）与**诊断**
        。
        """
        return self._reranker

    @property
    def metrics(self) -> MetricsSink:
        """当前的指标出口（`NullMetricsSink` = 没配路径）。

        与 `reranker` 同一个理由：**装配漏传是静默的**——出口没接上时 Search 照常工作、
        计数照常累计，只是**永远不写文件**。
        """
        return self._metrics

    @property
    def router(self) -> QueryRouter | None:
        """当前接的查询路由器（`None` = 没开）。

        与 `reranker` 同一个理由：**装配漏传是静默的**——路由没接上时 Search 照常工作、
        响应照旧合法，只是每一题都走默认的混合检索路径。
        """
        return self._router

    @property
    def agentic(self) -> AgenticSearch | None:
        """当前接的多跳执行器。"""
        return self._agentic

    @property
    def grounded_planner(self) -> GroundedPlanner | None:
        """LLM query interpreter used only after deterministic planning fails."""
        return self._grounded_planner

    def _grounded_decision(self, user_id: str, plan: QueryPlan) -> EvidenceDecision:
        """事实选择与相关原文审查的兼容转发。"""
        return self._grounded_stage.decide(user_id, plan, evaluator=evaluate_evidence)

    def run(self, *, user_id: str, query: str, top_k: int) -> PackagedResponse:
        """跑完整条链，返回打包响应并发送本次请求的观测值。

        ⚠ 三个 rerank 计数在实例上是累计值；观测采用请求局部增量，
        避免并发搜索把其他请求的调用或取证原因算进来。

        失败的请求不会发送观测值；耗时和计数只统计成功返回响应的请求，避免失败污染
        延迟均值。
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
                grounded_plan_source=trace.grounded_plan_source,
                grounded_planner_status=trace.grounded_planner_status,
                grounded_planner_calls=trace.grounded_planner_calls,
                grounded_planner_latency_ms=trace.grounded_planner_latency_ms,
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

        空库直接返回空结果：集合不存在时的 “collection not found” 等价于没有数据，
        不是请求错误。
        顺带也**不为一次必然空的检索付远程 embedding 调用**。
        ⚠ 注意这里的短路**早于** token 计数——空库时不该去加载分词器。
        """
        if trace is None:
            trace = _SearchTrace()
        if top_k <= 0:
            trace.grounded_status = "invalid_top_k"
            return PackagedResponse(items=())
        # ⚠ **空白 query 在这里短路，绝不喂给路由器**：
        #   ① 空查询不是"分类问题"，让模型去判它是把**一个输入校验**外包给 LLM
        #      ——既贵又不可靠（提示词一旦没有 NONE 这一档，它就只能瞎猜三类之一）；
        #   ② 空查询的合法响应是 `{"data": []}`；
        #   ③ **放在路由之前**是关键：`_route()` 只要被调到就会花一次 LLM 调用。
        # 与 `top_k <= 0` 同一条出口、同一个记法（`grounded_status` 是既有先例）。
        if not query.strip():
            trace.grounded_status = "blank_query"
            return PackagedResponse(items=())

        # Deterministic evidence precedes routing. The explicit force-agent
        # experiment retains its deliberate bypass of all direct evidence.
        if self._grounded_evidence and not self._agentic_force:
            trace.grounded_status = "no_plan"
            plan = compile_query(query)
            if plan is not None:
                trace.grounded_plan_source = "rule"
            if plan is None and self._grounded_planner is not None:
                planning = self._grounded_planner.decide(query)
                trace.grounded_planner_status = planning.status
                trace.grounded_planner_calls = planning.llm_calls
                trace.grounded_planner_latency_ms = planning.latency_ms
                plan = planning.plan
                if plan is not None:
                    trace.grounded_plan_source = "llm"
            if plan is not None:
                trace.grounded_operator = plan.operator
                if plan.operator == "current":
                    trace.grounded_status = "current_input"
                    return PackagedResponse(items=())
                decision = self._grounded_decision(user_id, plan)
                trace.grounded_status = decision.reason
                selected = decision.selection
                if selected is not None:
                    packaged = self._package_evidence(selected, top_k=top_k)
                    trace.grounded_projection = selected.projection
                    trace.selected_facts = len(selected.facts)
                    trace.selected_sources = selected.source_count
                    if not (
                        packaged.truncated_by_top_k
                        or packaged.truncated_by_budget
                        or packaged.dropped_missing
                    ):
                        return packaged
                    trace.grounded_status = "incomplete_packaging"
        route = self._route(query, trace)
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
                # 执行器在拿到任何证据之前失败 ⇒ **改走混合检索**。
                # ⚠ 原因记录在 `route_fallback`，而不是 `agentic_unavailable`：
                #   "分对了但执行器没接上"与"执行器接上了却没跑成"是两件事，
                #   混成一个记法会让"多跳这条路的失败率"无法统计。
                trace.route_fallback = outcome.fallback_reason
                candidates: Sequence[Candidate] = self._retriever.search(
                    user_id=user_id, query=query, top_k=top_k
                )
            else:
                candidates = outcome.candidates
                # 多跳的联合精排查询：原问题加上实际执行过的改写。
                # ⚠ 最终精排**仍然只有一次**，就在下面那一步——这不是"第二次精排"。
                rerank_query = outcome.rerank_query
        else:
            candidates = self._retriever.search(user_id=user_id, query=query, top_k=top_k)
        return self._rank_and_package(
            candidates, query=query, top_k=top_k, trace=trace, rerank_query=rerank_query
        )

    def _record_agentic(self, trace: _SearchTrace, outcome: AgenticResult | PlanResult) -> None:
        """把执行器的汇总记进 `trace`。

        ⚠ **执行器不写 `trace`**：它是被调用的那一层，"怎么记账"属于编排层，
        与 `rerank` 的分工一样（计数住在 `SearchPipeline`，见 `docs/architecture.md`）。

        ⚠ **两种模式共用这一处**：`PlanStats` 的前九个字段与 `AgenticStats` 同名同义
        （那是刻意的，不是巧合）⇒ 模式换了、记账的代码**一行都不用改**。
        模式专属的量（拆解调用数、计划步数、`depends_on` 被规范化的步数）**不进 `trace`**：
        更细的数据记录在 `plan_history` 诊断信息中，不扩展指标快照字段。
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
        `Route.DEFAULT` + `status="degraded"`；路由失败不能让正常 Search 变成 500。
        本方法只做两件事：**把二分类映射成"走哪条分支"**，以及**记下为什么**。

        ⚠ `MULTI_HOP` 在**没有执行器**时**在这里就把降级原因记下来**——而不是等到分支里。
        理由：那条路会一直往下走到混合检索，如果不在这里留痕，
        "分类成功但执行器不存在"与"分类结果就是 DEFAULT"在指标中会无法区分。
        """
        if self._agentic_force and self._agentic is not None:
            # 实验开关 `agentic.force`：绕过路由器，一律按 MULTI_HOP 处理。
            #   它必须**先于** `self._router is None` 的早返回被检查，否则"只开 agentic
            #   不开 router 的运行会静默地不启用强制多跳，导致归因错误。
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

        它只处理一份候选，不关心候选来自混合检索还是多跳执行器。
        把 ②–⑦ 抽出来（而不是在 `_search` 里写第二遍）是**结构性的**：
        "去重恰好一次、且在精排之前"与"`top_k` 在扩窗/合并之后生效"这两条
        **顺序错了不报错**的纪律，从此**只有一处实现**。

        ⚠ `rerank_query` **只换精排用的那一句**，不换别的：Checker 的判据、打包的
        `top_k`、扩窗的种子都仍然看**原问题**。它默认 `None` ⇒ 用 `query`，
        也就是混合链那条路的行为**逐字不变**。
        """
        # ② 稳定去重 + 重编号
        ranked = dedup_candidates(candidates)

        # ③ Evidence Checker（观测性判定）
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
            messages = (
                {
                    parent: {
                        m.source_index: m for m in self._store.fetch_pair_messages(conn, parent)
                    }
                    for parent in parents
                }
                if selection.projection == "source_record"
                else {}
            )
        valid = [
            f
            for f in facts
            if f.parent_memory_id in parents and parents[f.parent_memory_id].user_id == f.user_id
        ]
        if selection.projection == "source_record":
            original_records: dict[tuple[str, int], SelectedMemory] = {}
            for fact, index in zip(facts, selection.source_indices, strict=True):
                parent = parents.get(fact.parent_memory_id)
                message = messages.get(fact.parent_memory_id, {}).get(index)
                if (
                    parent is None
                    or parent.user_id != fact.user_id
                    or message is None
                    or message.role != "user"
                    or message.content != fact.source_quote
                    or message.timestamp != fact.event_time
                ):
                    return PackagedResponse(items=(), dropped_missing=1)
                record_key = (parent.id, index)
                if record_key not in original_records:
                    original_records[record_key] = SelectedMemory(
                        memory_id=make_source_message_id(parent.id, index),
                        pair=replace(
                            parent,
                            id=make_source_message_id(parent.id, index),
                            local_index=index,
                            question=message.content,
                            answer=None,
                            event_time=message.timestamp,
                            prev_memory_id=None,
                            next_memory_id=None,
                        ),
                        rerank_rank=len(original_records),
                        is_neighbor=False,
                    )
            # Detached presentation records cannot join the parent's adjacency
            # chain. The existing renderer and annotations preserve each message's
            # date, while best_rank retains the selected walk order.
            segments = merge_segments(
                tuple(original_records.values()),
                counter=self._counter,
                inject_abs_time=self._inject_abs_time,
                annotate_relatives=self._annotate_relatives,
            )
        elif selection.projection == "source":
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
        packaged = package(
            segments, top_k=top_k, counter=self._counter, max_tokens=self._budget_tokens
        )
        return replace(packaged, dropped_missing=packaged.dropped_missing + len(facts) - len(valid))

    def _record_rerank(self, trace: _SearchTrace | None, field: str) -> None:
        setattr(self, field, getattr(self, field) + 1)
        if trace is not None:
            setattr(trace, field, getattr(trace, field) + 1)

    def _maybe_rerank(
        self, *, query: str, ranked: list[Candidate], trace: _SearchTrace | None = None
    ) -> list[Candidate]:
        """对候选做**一次** rerank；不可用就退回原顺序。

        ⚠ **"降级"与"没开"是两件事，分开计数**（见 [`../rank/reranker.py`](../rank/reranker.py)）：
        两者产出的名次相同，但前者表示端点失败，后者表示功能未启用。
        混为一个计数会让"reranker 一直失败"看起来像"我们没打算用它"。

        **判据是"我们有没有打算调用它"**，不是"有没有调用成功"：

        | 情况 | 计数 | 为什么 |
        | --- | --- | --- |
        | `reranker is None`（没配 / 显式关掉） | `disabled` | 本来就没打算调 |
        | 没有候选 | `disabled` | 没有可排序的东西，**调用是没有意义的** |
        | 真源缺行 ⇒ 拼不出文档 | `degraded` | **打算调了**，是这一步没能兑现 |
        | 端点不可用 / 返回形状不符 | `degraded` | 同上 |

        ⚠ **降级不是吞异常**：reranker 是唯一位于关键路径上、又不被规则
        保证可用的组件。它挂了必须**照样产出合法响应**，而不是 5xx——但**降级这件事
        必须留下痕迹**，不能像什么都没发生。
        """
        if self._reranker is None or not ranked:
            self._record_rerank(trace, "rerank_disabled")
            return ranked

        # 使用与索引侧和最终 content 相同的基础渲染作为 rerank 输入。
        # 一次批量读取，避免逐条查询产生 N 次连接。
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
            return ranked  # ★ 退回未重排的 RRF 顺序

        if len(scores) != len(ranked):
            # 形状不符与端点挂了一样不可用——**静默按前缀对齐会让后半段名次错位**。
            # ⚠ `RemoteReranker.score()` 自己保证长度一致（它校验 index 集合），
            #   所以走到这里说明**换了一个不守规矩的实现**——那正是这条判断存在的理由。
            self._record_rerank(trace, "rerank_degraded")
            return ranked

        # 分数降序；**同分按原名次**（稳定）——否则同分项的先后会随排序实现漂移，
        # 而那是"消融不可复现"的经典来源。
        #
        # 这里只对同一个列表重排，因此候选集合不会改变。
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
