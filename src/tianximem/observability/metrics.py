"""§14 指标的聚合与导出（**本目录唯一的实现文件**）。

分工与出口形态写在 [`__init__.py`](./__init__.py) 与本目录的 `CLAUDE.md`——
本文件只写"怎么发、怎么聚"。

## 一条必须记住的语义：观测值**是每次请求的增量**，不是累计值

`SearchObservation` 的三个 rerank 计数是**这一请求引起的**增量。
`SearchPipeline` 上的 `rerank_calls` / `rerank_degraded` / `rerank_disabled`
**仍然是累计的**（`tools/probe_reranker.py` 与测试按累计值读它们，那是不该动的接口）
⇒ 发射方用请求局部计数记录增量，不能用并发请求共享累计值的前后差。

⚠ **聚合成累计值时不要拿累计值再相加**：那会把 346 次请求的 `rerank_calls`
加成 `1+2+3+…+346`——而它看起来只是个"有点大"的数字，不会报错。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "MetricsSink",
    "NullMetricsSink",
    "SearchObservation",
    "SnapshotMetricsSink",
]


@dataclass(frozen=True, slots=True)
class SearchObservation:
    """一次 `Search` 的观测值。**发射方（`service/`）填好，本模块只聚合。**"""

    latency_ms: float
    """这一次请求的墙钟耗时（毫秒）——§14 的 latency/query。"""

    rerank_calls: int
    """本请求里真正调用了 reranker 的次数（**增量**：0 或 1）。"""

    rerank_disabled: int
    """本请求里"没有 reranker 可调"的次数（**增量**）：**没构造**（没配端点 / 显式关）。"""

    rerank_degraded: int
    """本请求里"端点不可用而降级"的次数（**增量**）。"""

    rerank_name: str | None = None
    """这次用的哪个 reranker（D12 要求 run record 记它）。`None` = 这条链上没接。"""

    grounded_status: str = "not_recorded"
    grounded_operator: str | None = None
    grounded_projection: str | None = None
    selected_facts: int = 0
    selected_sources: int = 0

    route: str = "not_recorded"
    """这一次**实际走的**策略：`disabled` / `MULTI_HOP` / `DEFAULT`。

    ⚠ `disabled` 说明**路由器没开**——它与 `DEFAULT` **不是一回事**：
    后者是"路由器开着、分类结果是默认路"，前者是"根本没有分类这一步"。
    两者在响应上完全一样（都是原混合检索链），**只有这里能分开**。
    ⚠ 取值域**只有这三个**：2026-10-14 起路由是二分类，`COUNT` 不再是合法输出
    （模型万一吐它 ⇒ 走 `degraded`，落 `DEFAULT`）。
    """

    route_status: str = "not_recorded"
    """路由的三种状态：`disabled`（没开）/ `success`（模型正常分类）/ `degraded`（降级）。

    ⚠ 与 `rerank` 的 `disabled` / `degraded` 三分法同形，**理由也同一条**（D12）：
    混为一谈会把"端点一直挂"看成"我们本来就没开路由"。
    """

    route_reason: str | None = None
    """**分类本身**降级的原因（`llm_unavailable: …` / `invalid_output` …）。成功时为 `None`。"""

    route_fallback: str | None = None
    """分类成功、但**选中的那条路没跑成**的原因（`agentic_unavailable` / `retrieval_error` …）。

    ⚠ 它与 `route_reason` 是两件事：一个是"没分成类"，一个是"分对了但执行器没跑成"。
    """

    route_latency_ms: float | None = None
    """这一次路由本身的墙钟耗时（**是 `latency_ms` 的一个可归因子集**，不是并列的量）。

    ⚠ 没有路由时为 `None`——**不要记 0**：0 会让"Router On"与"Router Off"两臂的
    平均路由耗时都算得出来，而其中一臂的 0 是假的（§6 的成本分析会读错）。
    """

    agentic_status: str = "not_recorded"
    """多跳执行器这一次的状态：`disabled`（没走到那条路）/ `success` / `degraded`。

    ⚠ 与 `rerank` 的三分法**同一条纪律**（D12）：`disabled` 指的是
    **"这条路没被走到"**（路由判了别的、或执行器没装），不是"走了 0 轮"。
    把两者混起来，"多跳值不值"这个问题的分母就没了。
    """

    agentic_rounds: int = 0
    """**实际执行的检索轮数**（含第一轮）。§14 的"Agent 平均轮数"就是它。"""

    agentic_stop_reason: str | None = None
    """它为什么停（§11 的 `stop_reason`）：`sufficient` / `max_rounds` / `duplicate_query` …

    ⚠ 与 `route_fallback` 是两件事：一个是"这条路的结局"，一个是"它压根没跑成"。
    """

    agentic_llm_calls: int = 0
    """多跳自己发起的 LLM 调用数（**每轮最多一次**）。"""

    agentic_rerank_calls: int = 0
    """多跳**每轮内部**的精排调用数。

    ⚠ 它**不包含**最终那一次精排——那一次算在 `rerank_calls` 里（§9.4 的"最多四次"=
    这里的 ≤3 ＋ 那里的一次）。⛔ 把两次混进同一个数，§9.4 那个上限就没法核对了。
    """

    agentic_selected_evidence: int = 0
    """跨轮累积的"模型选中"的证据条数（§8.1 的 `selected_evidence_ids`）。"""

    agentic_candidates: int = 0
    """去重后的跨轮候选总数（§9.2）。

    ⚠ 它是**去重后**的，与 `rounds × per_round_top_k` 不是一回事。
    """

    agentic_joint_queries: int = 0
    """最终联合精排 Query 里**实际执行过的**查询条数（含原始问题）。"""

    agentic_latency_ms: float | None = None
    """多跳执行器本身的墙钟耗时。

    ⚠ 没走到那条路时为 `None`——**不要记 0**（同 `route_latency_ms`）。
    """

    returned_segments: int = 0
    truncated_by_budget: bool = False
    truncated_by_top_k: bool = False
    dropped_missing: int = 0


@runtime_checkable
class MetricsSink(Protocol):
    """观测值的出口。**与 `retrieve/checker.py` 的 `CheckerInstrument` 同一套形态**：
    发射在各自层，聚合在这里。"""

    def record(self, observation: SearchObservation) -> None: ...


class NullMetricsSink:
    """什么都不记——**没配 `TIANXIMEM_METRICS_PATH` 时的默认**。

    缺省不是错误：与 reranker 同一套口径（D12）——**没配就是不用**，服务照常起。
    """

    __slots__ = ()

    def record(self, observation: SearchObservation) -> None:
        pass


@dataclass(slots=True)
class SnapshotMetricsSink:
    """把累计量写成一份 JSON 快照，**每来一条观测就重写一次**。

    **为什么是快照而不是逐行追加**：聚合只该做一次（见 `__init__.py`）——
    追加式 JSONL 会把"怎么加"推给读的人，而 harness **不许 import 本模块**。

    ⚠ **写失败不抛异常**：指标是诊断，不能因为它写不进去就让一次 Search 变成 500
    （Search 的失败必须只来自契约或下游，§2.1）。写失败会在 `stderr` 上留一条。
    """

    path: Path
    searches: int = 0
    latency_ms_sum: float = 0.0
    latency_ms_max: float = 0.0
    rerank_calls: int = 0
    rerank_disabled: int = 0
    rerank_degraded: int = 0
    rerank_name: str | None = None
    grounded_outcomes: dict[str, int] = field(default_factory=dict)
    grounded_operators: dict[str, int] = field(default_factory=dict)
    grounded_projections: dict[str, int] = field(default_factory=dict)
    #: 路由的三个分布 + 耗时。**它们是"路由到底有没有在跑"的唯一出口**
    #: （响应里装不下——形状是契约；§14 那条 V12 的同款处境）。
    router_routes: dict[str, int] = field(default_factory=dict)
    router_statuses: dict[str, int] = field(default_factory=dict)
    router_reasons: dict[str, int] = field(default_factory=dict)
    router_fallbacks: dict[str, int] = field(default_factory=dict)
    #: 路由调用的次数与耗时之和。⚠ **`router_calls` 数的是"真的分类过"的次数**
    #: （即 `route_latency_ms` 非 None 的那些），所以 `router_calls <= searches`——
    #: 两者之差就是"没开路由器"的那些请求。**均值必须按 `router_calls` 除**，
    #: 按 `searches` 除会把两臂混成一个没有解释的数。
    router_calls: int = 0
    router_latency_ms_sum: float = 0.0
    router_latency_ms_max: float = 0.0
    selected_facts: int = 0
    selected_sources: int = 0
    #: 多跳执行器的分布与总量（§14）。⛔ **`agentic_calls` 数的是"真的跑过多跳"的次数**
    #: （即 `agentic_latency_ms` 非 None 的那些）——均值必须按它除，不是按 `searches`。
    #: 与路由器那条「`router_calls <= searches`」同型。
    agentic_statuses: dict[str, int] = field(default_factory=dict)
    agentic_stops: dict[str, int] = field(default_factory=dict)
    agentic_calls: int = 0
    agentic_rounds_sum: int = 0
    agentic_llm_calls: int = 0
    agentic_rerank_calls: int = 0
    agentic_selected_evidence: int = 0
    agentic_candidates: int = 0
    agentic_joint_queries: int = 0
    agentic_latency_ms_sum: float = 0.0
    agentic_latency_ms_max: float = 0.0
    returned_segments: int = 0
    truncated_by_budget: int = 0
    truncated_by_top_k: int = 0
    dropped_missing: int = 0
    _lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    def record(self, observation: SearchObservation) -> None:
        with self._lock:
            self.searches += 1
            self.latency_ms_sum += observation.latency_ms
            self.latency_ms_max = max(self.latency_ms_max, observation.latency_ms)
            self.rerank_calls += observation.rerank_calls
            self.rerank_disabled += observation.rerank_disabled
            self.rerank_degraded += observation.rerank_degraded
            if observation.rerank_name:
                self.rerank_name = observation.rerank_name
            for counts, value in (
                (self.grounded_outcomes, observation.grounded_status),
                (self.grounded_operators, observation.grounded_operator),
                (self.grounded_projections, observation.grounded_projection),
                (self.router_routes, observation.route),
                (self.router_statuses, observation.route_status),
                (self.router_reasons, observation.route_reason),
                (self.router_fallbacks, observation.route_fallback),
            ):
                if value:
                    counts[value] = counts.get(value, 0) + 1
            if observation.route_latency_ms is not None:
                self.router_calls += 1
                self.router_latency_ms_sum += observation.route_latency_ms
                self.router_latency_ms_max = max(
                    self.router_latency_ms_max, observation.route_latency_ms
                )
            self.selected_facts += observation.selected_facts
            self.selected_sources += observation.selected_sources
            for counts, value in (
                (self.agentic_statuses, observation.agentic_status),
                (self.agentic_stops, observation.agentic_stop_reason),
            ):
                if value:
                    counts[value] = counts.get(value, 0) + 1
            if observation.agentic_latency_ms is not None:
                self.agentic_calls += 1
                self.agentic_rounds_sum += observation.agentic_rounds
                self.agentic_llm_calls += observation.agentic_llm_calls
                self.agentic_rerank_calls += observation.agentic_rerank_calls
                self.agentic_selected_evidence += observation.agentic_selected_evidence
                self.agentic_candidates += observation.agentic_candidates
                self.agentic_joint_queries += observation.agentic_joint_queries
                self.agentic_latency_ms_sum += observation.agentic_latency_ms
                self.agentic_latency_ms_max = max(
                    self.agentic_latency_ms_max, observation.agentic_latency_ms
                )
            self.returned_segments += observation.returned_segments
            self.truncated_by_budget += observation.truncated_by_budget
            self.truncated_by_top_k += observation.truncated_by_top_k
            self.dropped_missing += observation.dropped_missing
            self._write()

    def snapshot(self) -> dict[str, Any]:
        """当前的聚合值——写进文件的就是它。

        runner 原样塞进 run record 的 `metrics=`（不是 `counters=`）。
        """
        with self._lock:
            return self._snapshot()

    def _snapshot(self) -> dict[str, Any]:
        return {
            "searches": self.searches,
            "latency_ms": {
                "count": self.searches,
                "mean": (round(self.latency_ms_sum / self.searches, 3) if self.searches else None),
                "max": round(self.latency_ms_max, 3),
            },
            "rerank": {
                "calls": self.rerank_calls,
                "disabled": self.rerank_disabled,
                "degraded": self.rerank_degraded,
                "name": self.rerank_name,
            },
            "grounded_evidence": {
                "outcomes": dict(self.grounded_outcomes),
                "operators": dict(self.grounded_operators),
                "projections": dict(self.grounded_projections),
                "selected_facts": self.selected_facts,
                "selected_sources": self.selected_sources,
            },
            "router": {
                "routes": dict(self.router_routes),
                "statuses": dict(self.router_statuses),
                "reasons": dict(self.router_reasons),
                "fallbacks": dict(self.router_fallbacks),
                "latency_ms": {
                    # ⚠ 分母是 `router_calls`（真的分类过的次数）**不是** `searches`
                    #   ——见 `router_calls` 字段的说明。
                    "count": self.router_calls,
                    "mean": (
                        round(self.router_latency_ms_sum / self.router_calls, 3)
                        if self.router_calls
                        else None
                    ),
                    "max": round(self.router_latency_ms_max, 3),
                },
            },
            "agentic": {
                "statuses": dict(self.agentic_statuses),
                "stops": dict(self.agentic_stops),
                "rounds": self.agentic_rounds_sum,
                "llm_calls": self.agentic_llm_calls,
                "rerank_calls": self.agentic_rerank_calls,
                "selected_evidence": self.agentic_selected_evidence,
                "candidates": self.agentic_candidates,
                "joint_queries": self.agentic_joint_queries,
                "latency_ms": {
                    # ⚠ 分母是 `agentic_calls`（真的跑过多跳的次数）**不是** `searches`
                    #   ——见 `agentic_calls` 字段的说明。
                    "count": self.agentic_calls,
                    "mean": (
                        round(self.agentic_latency_ms_sum / self.agentic_calls, 3)
                        if self.agentic_calls
                        else None
                    ),
                    "max": round(self.agentic_latency_ms_max, 3),
                },
            },
            "packaging": {
                "returned_segments": self.returned_segments,
                "truncated_by_budget": self.truncated_by_budget,
                "truncated_by_top_k": self.truncated_by_top_k,
                "dropped_missing": self.dropped_missing,
            },
            "note": (
                "服务进程内累计到此刻的观测值（§14）。"
                "⚠ `rerank.disabled` 是**没构造 reranker**（没配端点或显式关）、"
                "`rerank.degraded` 是**端点不可用而退回 RRF 顺序**——"
                "混为一谈会把「端点一直挂」看成「我们本来就没打算用它」（D12）。"
                "共同取证成功与空库短路不调用精排，也不计入 disabled。"
                "⚠ 路由器同型：`router.statuses.disabled` 是**没开路由**、"
                "`router.routes.DEFAULT` 是**开着但这一题分到了默认路**——"
                "两者在响应上完全一样，只有这里能分开。"
                "⚠ `router.latency_ms` 的分母是 `router.calls`（真的分类过的次数），"
                "**不是** `searches`：两臂对照时按 `searches` 平均会把没开路由的那一臂算成 0。"
            ),
        }

    def _write(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            staged = self.path.with_name(self.path.name + ".tmp")
            staged.write_text(
                json.dumps(self.snapshot(), ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            staged.replace(self.path)
        except OSError as exc:  # 见类 docstring：诊断写不进去不该让 Search 失败
            print(f"⚠ 指标快照写不进 {self.path}（{exc}）——计数仍在进程内累计", file=sys.stderr)
