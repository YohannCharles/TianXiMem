"""§14 指标的聚合与导出（**本目录唯一的实现文件**）。

分工与出口形态写在 [`__init__.py`](./__init__.py) 与本目录的 `CLAUDE.md`——
本文件只写"怎么发、怎么聚"。

## 一条必须记住的语义：观测值**是每次请求的增量**，不是累计值

`SearchObservation` 的三个 rerank 计数是**这一请求引起的**增量。
`SearchPipeline` 上的 `rerank_calls` / `rerank_degraded` / `rerank_disabled`
**仍然是累计的**（`tools/probe_reranker.py` 与测试按累计值读它们，那是不该动的接口）
⇒ 发射方负责**在请求前后各取一次、相减**。

⚠ **聚合成累计值时不要拿累计值再相加**：那会把 346 次请求的 `rerank_calls`
加成 `1+2+3+…+346`——而它看起来只是个"有点大"的数字，不会报错。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
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

    def record(self, observation: SearchObservation) -> None:
        self.searches += 1
        self.latency_ms_sum += observation.latency_ms
        self.latency_ms_max = max(self.latency_ms_max, observation.latency_ms)
        self.rerank_calls += observation.rerank_calls
        self.rerank_disabled += observation.rerank_disabled
        self.rerank_degraded += observation.rerank_degraded
        if observation.rerank_name:
            self.rerank_name = observation.rerank_name
        self._write()

    def snapshot(self) -> dict[str, Any]:
        """当前的聚合值——写进文件的就是它。

        runner 原样塞进 run record 的 `metrics=`（不是 `counters=`）。
        """
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
            "note": (
                "服务进程内累计到此刻的观测值（§14）。"
                "⚠ `rerank.disabled` 是**没构造 reranker**（没配端点或显式关）、"
                "`rerank.degraded` 是**端点不可用而退回 RRF 顺序**——"
                "混为一谈会把「端点一直挂」看成「我们本来就没打算用它」（D12）。"
                "`disabled == searches` ⇒ 这个 run 全程没有精排。"
            ),
        }

    def _write(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self.snapshot(), ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:  # 见类 docstring：诊断写不进去不该让 Search 失败
            print(f"⚠ 指标快照写不进 {self.path}（{exc}）——计数仍在进程内累计", file=sys.stderr)
