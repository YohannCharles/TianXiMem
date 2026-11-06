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
    latency_ms: float

    rerank_calls: int

    rerank_disabled: int

    rerank_degraded: int

    rerank_name: str | None = None

    fact_status: str = "disabled"

    fact_snippets: int = 0

    route: str = "not_recorded"

    route_status: str = "not_recorded"

    route_reason: str | None = None

    route_fallback: str | None = None

    route_latency_ms: float | None = None

    agentic_status: str = "not_recorded"

    agentic_rounds: int = 0

    agentic_stop_reason: str | None = None

    agentic_llm_calls: int = 0

    agentic_rerank_calls: int = 0

    agentic_selected_evidence: int = 0

    agentic_candidates: int = 0

    agentic_joint_queries: int = 0

    agentic_latency_ms: float | None = None

    returned_segments: int = 0
    truncated_by_budget: bool = False
    truncated_by_top_k: bool = False
    dropped_missing: int = 0


@runtime_checkable
class MetricsSink(Protocol):
    def record(self, observation: SearchObservation) -> None: ...


class NullMetricsSink:
    __slots__ = ()

    def record(self, observation: SearchObservation) -> None:
        pass


@dataclass(slots=True)
class SnapshotMetricsSink:
    path: Path
    searches: int = 0
    latency_ms_sum: float = 0.0
    latency_ms_max: float = 0.0
    rerank_calls: int = 0
    rerank_disabled: int = 0
    rerank_degraded: int = 0
    rerank_name: str | None = None
    fact_statuses: dict[str, int] = field(default_factory=dict)
    fact_snippets: int = 0
    router_routes: dict[str, int] = field(default_factory=dict)
    router_statuses: dict[str, int] = field(default_factory=dict)
    router_reasons: dict[str, int] = field(default_factory=dict)
    router_fallbacks: dict[str, int] = field(default_factory=dict)
    router_calls: int = 0
    router_latency_ms_sum: float = 0.0
    router_latency_ms_max: float = 0.0
    selected_facts: int = 0
    selected_sources: int = 0
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
                (self.fact_statuses, observation.fact_status),
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
            self.fact_snippets += observation.fact_snippets
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
            "fact_augmentation": {
                "statuses": dict(self.fact_statuses),
                "snippets": self.fact_snippets,
            },
            "router": {
                "routes": dict(self.router_routes),
                "statuses": dict(self.router_statuses),
                "reasons": dict(self.router_reasons),
                "fallbacks": dict(self.router_fallbacks),
                "latency_ms": {
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
                "服务进程内累计到此刻的观测值。"
                "⚠ `rerank.disabled` 是**没构造 reranker**（没配端点或显式关）、"
                "`rerank.degraded` 是**端点不可用而退回 RRF 顺序**——"
                "混为一谈会把“端点一直挂”看成“我们本来就没打算用它”。"
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
        except OSError as exc:
            print(f"⚠ 指标快照写不进 {self.path}（{exc}）——计数仍在进程内累计", file=sys.stderr)
