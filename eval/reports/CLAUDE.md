# eval/reports/ — 数字的唯一落点

**PRD**：§13（记录）、§14（监控指标）、§3.2（七个维度）、§15（SQLite 随 run 归档）

## 要写什么

```text
ledger.md             **结果台账**（人工维护，**提交进 git**）——结论在本目录的唯一权威
schema.py             run record 的 schema
<主题>-<YYYYMMDD>.md   **主题卷**：一个主题一份，一次调查是它的一节（**2026-10-10 起，见下**）
runs/                 新跑批产物落点（默认 gitignored；旧实验已清理，见 runs/README.md）
```

### ⚠ 报告按**主题卷**归档，不再一次调查一份（2026-10-10）

**为什么改**：文档数量本身就是成本——AI 每次扫本目录都吃上下文，40 份一次性报告
的检索价值远低于它们的噪音。**一次调查一个文件是错的。**

**规矩**：

1. **按主题建卷**，不是按日期或按调查。新调查先找有没有同主题的卷，**有就加一节**；
2. 卷内用 `## <原报告标题>` 分节，文件头写**合并说明**（来源文件名 + 合并日期），
   **内容逐字保留、只降标题级别、不改写不删减**——删减是另一次有意的编辑，不是合并的附带效果；
3. 卷名仍是 `<主题>-<YYYYMMDD>.md`，日期取该主题**最后一次调查**的日期；
4. **`ledger.md` 不受影响**：它是结论的家，卷是证据的家。

**当前卷**（`ledger.md` + 本文件之外的全部）：

| 卷 | 覆盖 |
| --- | --- |
| [`datasets-20261006.md`](./datasets-20261006.md) | 数据集接入、目录整理、AML 输入与 Full 优化 |
| [`retrieval-evidence-20261005.md`](./retrieval-evidence-20261005.md) | 检索与共同事实取证的四项改进与验证 |
| [`optimization-20261007.md`](./optimization-20261007.md) | 12h 优化长跑、子任务优先级、临床多跳 |
| [`scores-20261007.md`](./scores-20261007.md) | 十一数据集成绩 + **v1.0/v1.1/v1.2 版本对照表** |
| [`memory-governance-20261009.md`](./memory-governance-20261009.md) | 记忆治理（当前状态 / 遗忘 / 摘要）实验与实施 |
| [`corporatebench-20261004.md`](./corporatebench-20261004.md) | CorporateBench 审计、错例分析、oracle 诊断、修复 |
| [`personamem-20261009.md`](./personamem-20261009.md) | PersonaMem pipeline 修复与遗忘题池调查 |
| [`analysis-20261009.md`](./analysis-20261009.md) | 查询路由、问题分类学、RAG 差距分析 |
| [`git-history-cleanup-20261005.md`](./git-history-cleanup-20261005.md) | Git 历史整理与 v1.2 发布记录 |

⚠ **`memory-governance-rollback-20261010.md` 与 `explicit-forgetting-research-20261010.md`
是新建的单篇**（2026-10-10，尚未并入卷）。

---

## 为什么需要一个"唯一落点"

数字散在三处就会漂：`docs/experiments.md`（协议）、`eval/experiments/`（怎么跑）、`reports/`（数字）。**约定：只有本目录的数字算数**——其它地方引用数字一律指回这里。

**并且**：`docs/experiments.md` 的登记表有一列"结论"——**那一列是链接，不是数字的家。**

---

## run record 的字段（§13 + §14 + §6.5）

| 组 | 字段 | 出处 |
| --- | --- | --- |
| **结果** | 端到端**总分** | §13 |
| | **七个维度子分**（逐维） | §3.2 / §14 |
| | **`breakdown.partial_credit`（部分分，⚠ 不进总分）** | 见下 |
| **配置指纹** | profile（`local` / `submit`）、各开关状态、**配置快照 hash** | §13 |
| **数据指纹** | 数据集 + 版本 + **切批口径** + **`add_shape`（add 正文形态）** | §13 |
| **模型** | embedder / LLM / reranker 各自的标识 | §12.1 R1 |
| **阶段** | 属于哪个 Step | §16 |
| **§14 指标** | Agent Trigger Rate · 平均轮数 / Rewrite 次数 · latency/query · embedding API 调用数 | §14 |
| **归档** | 该 run 的 SQLite 文件副本 | §15 |

> **§14 那一行今天只填了一半**：latency/query 与 rerank 的计数已接线（服务写
> `TIANXIMEM_METRICS_PATH` 的快照，runner 读进 `metrics=`）；embedding 调用数与
> Agent Trigger Rate 的发射方还没接。**逐项状态见
> [`../../src/tianximem/observability/CLAUDE.md`](../../src/tianximem/observability/CLAUDE.md)**——本文件不另列一份。

### ⚠ `breakdown.partial_credit`：**不进总分**的那一列（2026-10-03）

只对**判分方本来就给出部分分**的数据集存在（**CL-Bench / CorporateBench / BEAM**；BEAM 的那份来自官方采集重放，发射方见 [`../harness/official_capture_pipeline.py`](../harness/official_capture_pipeline.py)）：
`{"mean": …, "n": …}`，逐类还有 `partial`。**给不出的数据集写 `mean=None, n=0`**
（同 `_accuracy` 的纪律：空集合不写 `0.0`，那会被读成"全错"）。

CorporateBench 每题传递标量 exact-match / 列表 set-F1，完整 QA 均分另写入
`scores.dataset_score`（带 `metric` / `mean` / `n` / 本地复现范围）。
`overall` 保留二值完全匹配准确率，两者不能混用；它也不代表 AML 榜分。
实现与重判记录见 [修复报告](corporatebench-20261004.md)。

HybridQA / FEVEROUS 的逐题指标也汇总到 `scores.dataset_score`，
分别记录上游 EM/F1、标签准确率与严格证据组分，以及证据宏平均 precision/recall
的调和 F1。其语料候选范围和指标定义见 [接入报告](datasets-20261006.md)。
`overall` 分别对应二值 EM 或严格证据组正确率；候选池范围必须随结果一起引用。

**为什么需要它**：CL-Bench 的官方分是**全有全无**的 LLM rubric 判分
（[`../../dataset/.upstream/aml/clb_pipeline.py`](../../dataset/.upstream/aml/clb_pipeline.py) 的判分 prompt 原文：
`strict, all-or-nothing … The final score is binary`）——一道题从 0 翻到 1 要**每一条**
rubric 都满足 ⇒ **中间的所有进展在 `overall` 里都看不见**。实测有题
`rubric_clbench_score=0` 而 `requirement_ratio=0.50`（14 条里满足 7 条）：
二值分下它和"一条都没满足"**完全一样**。

⇒ 归档裁判本来就写着 `requirement_ratio`，本仓把它接成部分分：

| 用途 | 看哪个 |
| --- | --- |
| **对齐榜分 / 报对外数字** | **`overall`**（二值那一列） |
| **判断一个改动有没有效果** | **`partial_credit`**（分辨率高得多） |

⛔ **它不是榜分**——别把它当"我们的分数变好了"。守它的用例在
[`../../tests/test_harness.py`](../../tests/test_harness.py)：
`test_partial_credit_is_aggregated_but_never_touches_overall` 钉住"它一个字都不许进 `overall`"。

### ⚠ 七个维度必须逐维记录

官方维度**逐字采用**（§3.2）：

1. Explicit fact recall
2. Relational and multi-hop reasoning
3. Temporal and event understanding
4. **Memory governance**
5. Personalization and care
6. Rules and process execution
7. **Epistemic safety and privacy**

> **第 4、7 两维要显式回应**（§3.2，映射见 [`../../docs/architecture.md`](../../docs/architecture.md)）——**报告里这两维为空的 run，等于没测。**

### 切分有没有切坏 QA 对（D24 起）

**口径见 [`../../src/tianximem/observability/CLAUDE.md`](../../src/tianximem/observability/CLAUDE.md) 的"切分是否切坏了 QA 对"一节**——**本文件不另列一份**。

---

## Git 归档策略

| 类型 | 是否入库 | 理由 |
| --- | --- | --- |
| `ledger.md` | ✅ **提交** | 它是结论本身 |
| `schema.py` | ✅ 提交 | 代码 |
| `runs/` 中的新产物与一次性探针 | ❌ **默认忽略** | 旧分支实验已清理；原始数据与运行时产物仍不入库（§12.5） |
| `runs/README.md` | ✅ **提交** | 清理记录、后续通用评测入口与历史恢复方法 |
| 正式版本对照的配置和清单 | 按需显式提交 | 经核对后保留共享复核需要的指纹；通用程序放 `eval/experiments/` 或 `tools/` |
| **T2 的人工标注** | ✅ **提交**（但它不在这里，在 [`../experiments/`](../experiments/)） | **133 行人工劳动，丢了要重做** |

2026-10-10 起按用户计划仅重跑 `v1.0`、`v1.2`，不再保留旧分支实验的脚本、补丁或源码副本。
历史报告、成绩与 Git 出处保留；本机原始产物已移到本地归档，详情见
[`runs/README.md`](runs/README.md)。新原始数据与运行时状态继续由 `.gitignore` 覆盖。

---

## 两条记录纪律

| 纪律 | 出处 |
| --- | --- |
| **latency / 成本只在明显变差时才追** | §13 |
| **不记 Recall@K，不做按问题类型的收益归因** | §14——前者已接近天花板，**后者是论文的事** |

**另外**：结论怎么写见 [`./ledger.md`](./ledger.md) 的"怎么记一条"——**本文件不另列一份**。
