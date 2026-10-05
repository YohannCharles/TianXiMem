"""§14 指标的**聚合与导出**（[`CLAUDE.md`](./CLAUDE.md) 的落点）。

## 分工：各层发射、本目录聚合

[`../../../docs/architecture.md`](../../../docs/architecture.md) 的三条硬性边界之一：
**每个指标由产生它的那一层发射**，本目录只聚合 ——
所以 ⛔ **本模块不 import 任何业务层**（反向依赖就是分层坏了）。

## 出口为什么是文件

响应形状是契约（§2.1），**一个字段都不能多**；而 harness 打 HTTP、**禁 import `src/`**
（[`../../../eval/CLAUDE.md`](../../../eval/CLAUDE.md)）⇒ 两边唯一的公共面是**磁盘**。
⇒ 服务把**已聚合的快照**写到 `TIANXIMEM_METRICS_PATH`，runner 原样塞进 run record 的
`metrics=`（**不是 `counters=`**——那份是 §6.5 的 pending 计数器，键必须恒在场）。
**聚合只在本模块做一次**——eval 侧若自己再算一遍，那就是两处会漂的判据。

## 今天接了哪些、没接哪些（逐项，别读成"§14 已完成"）

| §14 的指标 | 状态 |
| --- | --- |
| latency/query | ✅ 本模块——发射方是 `service/pipeline.py` 的请求边界 |
| rerank 的 `calls` / `disabled` / `degraded` + 模型名 | ✅ 本模块——**V12 要的那几个量** |
| embedding API 调用数 / 缓存命中率 | ⬜ 发射方 `embed/` 还没发 |
| Agent Trigger Rate · 平均轮数 · Rewrite 次数 | ⬜ 发射方 `agent/` 属于 v2（D13） |

⚠ **为什么 rerank 那几个量非要有出口**：它们在响应里装不下（契约），所以**只看响应和
`contract-check` 的 14 条，端点长期挂掉不会让任何东西变红**
（[`../../../docs/open-questions.md`](../../../docs/open-questions.md) 的 **V12**）。
本模块就是那条出口；读法见 `SnapshotMetricsSink.snapshot()` 的 `note`。
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
