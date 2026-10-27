"""逐请求指标的聚合与可选快照导出。

各业务层发射请求局部观测值，本包负责聚合，不能反向导入业务模块。
响应结构固定，HTTP-only harness 也不能导入服务内部模块，因此
`TIANXIMEM_METRICS_PATH` 是服务与评测 harness 之间的公共接口。

当前覆盖 Search 延迟、grounded evidence、打包、rerank、路由、多跳执行器及其聚合状态。
指标不会进入 Search 响应；快照读法见 `SnapshotMetricsSink.snapshot()` 的 `note`。
"""

from tianximem.observability.metrics import (
    MetricsSink,
    NullMetricsSink,
    SearchObservation,
    SnapshotMetricsSink,
)

__all__ = [
    "MetricsSink",
    "NullMetricsSink",
    "SearchObservation",
    "SnapshotMetricsSink",
]
