# observability/ — 指标与计数器

**PRD**：§14（监控指标）

## 有什么

```text
metrics.py     ✅ 已实现——§14 指标的**聚合与导出**（出口 = 一份 JSON 快照）
```

**本目录是"指标怎么看"的唯一声明处。** 各业务层不重复描述读法，只发射。

**今天接了哪些**（逐项，**别读成"§14 已完成"**）：

| §14 指标 | 发射方 | 状态 |
| --- | --- | --- |
| latency/query | [`../service/`](../service/) 的请求边界 | ✅ |
| rerank `calls` / `disabled` / `degraded` + 模型名 | [`../service/`](../service/)（计数住在 `SearchPipeline`） | ✅——**V12 要的那几个量** |
| embedding 调用数 / 缓存命中率 | [`../embed/`](../embed/) | ⬜ 还没发 |
| Agent Trigger Rate · 平均轮数 · Rewrite 次数 | [`../agent/`](../agent/) | ⬜ v2（D13） |

**出口为什么是文件**：响应形状是契约（**一个字段都不能多**）、而 harness **禁 import `src/`**
⇒ 磁盘是两边唯一的公共面。服务把**已聚合的快照**写到 `TIANXIMEM_METRICS_PATH`（空 ⇒ 不写），
runner 读进 run record 的 `metrics=`——**不是 `counters=`**，那份是 §6.5 的 pending 计数器，
`schema.validate()` 要求它的键**恒在场**。聚合**只在本目录做一次**（eval 侧自己再算一遍
就是两处会漂的判据）。

## 一条职责边界

**本目录聚合，不发射。** 每个指标由**产生它的那一层**发出来：

| 指标 | 发射方 | 为什么在那里 |
| --- | --- | --- |
| **embedding API 调用数 / 缓存命中率** | [`../embed/`](../embed/) | 它是**缓存键**的探针——**不下降说明缓存键写错了** |
| **Agent Trigger Rate** · 平均轮数 · Rewrite 次数 | [`../agent/`](../agent/) | 只有 agent 循环知道被触发了几次 |
| latency/query | [`../service/`](../service/) | 请求边界 |

**本目录把它们汇集、按 §13 的记录格式导出**，供 `eval/reports/` 使用（字段清单见 [`../../eval/reports/CLAUDE.md`](../../../eval/reports/CLAUDE.md)）。

**反向依赖是错的**——若 `observability/` import 了业务层，分层就坏了。

---

## §14：只留会触发动作的指标

> **其余不看**（§14）。这条纪律比指标清单本身重要——本项目**不写论文**，任何"看着有信息量但不改变下一步动作"的图都是净负债。

| 指标 | 为什么看它 | 异常时触发什么 |
| --- | --- | --- |
| 端到端总分 + 各维度子分 | **唯一的目标** | 决定下一步改哪 |
| **Agent Trigger Rate** | 核心 claim 的开关频率 | 太高 → Checker 太保守；**接近 0 → agent 没起作用** |
| **latency/query** | 契约允许 30 分钟，但 **Full run 要连续跑 0.5–2 天** | 接近上限就**削减 agent 轮数** |
| Agent 平均轮数 / Rewrite 次数 | **prompt 健康的探针** | 异常升高说明 **prompt 崩了** |
| embedding API 调用数 | 缓存命中率 | 不下降说明**缓存键写错了** |

---

## 切分是否切坏了 QA 对（D24 起）

**没有独立计数器**：一次 Add 的块在写下那一刻就是最终形状，**不存在"新建 / 补全 / 落单"三种归宿**
⇒ 这件事**只体现在块形状里**——每次 Add 产出几个块、其中几个 `is_paired`（见
[`../pairing/pairing.py`](../pairing/pairing.py) 的 `MemoryBlock.is_paired`），
**一次读库就能算出来**（`SELECT count(*) ... WHERE answer IS NULL`），不需要任何跨请求状态。

> ⚠ **别为它加回一个计数器**：跨请求状态正是 D24 想消掉的东西。


---

## ⚠ 本地数字与线上对不上（必读）

> **词数那一路只能近似**（§6.5 / §12.3 第 5 条）：AML 按"20 条消息**或** 2,000 个 **Adapter 计数的词**"切分，而**计数口径官方是给了的**（`2,000 **Adapter-counted** words`）——**本仓复现不了的是那个 Adapter 本身**（平台侧冻结组件），本地那一路只能用**空白分词近似**。

**这只影响 [`../../eval/harness/batching.py`](../../../eval/harness/batching.py)**（它已声明两条预算都做、词数那半是近似）——
本目录没有依赖切批口径的计数器。S2 本身仍是 [`../../../docs/open-questions.md`](../../../docs/open-questions.md) 里的一条未决事项。

---

## 明确不做的（§14）

| 不做 | 理由 |
| --- | --- |
| **不看 Recall@K** | 本任务上它**已接近天花板**，涨跌都不代表什么 |
| **不做按问题类型的收益归因** | **那是论文的事** |
| latency / 成本的细粒度追踪 | **只在明显变差时才追**（§13） |

> §12.2 的**重点指标是端到端，不是 Recall@K**——检索召回已接近天花板（~96–99% @5/@10），而端到端 QA 只有约 60–95%，**缺口在"留哪些、按什么顺序留"**。
