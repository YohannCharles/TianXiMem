# eval/reports/ — 数字的唯一落点

**PRD**：§13（记录）、§14（监控指标）、§3.2（七个维度）、§15（SQLite 随 run 归档）

## 要写什么

```text
ledger.md          **结果台账**（人工维护，**提交进 git**）
schema.py          run record 的 schema
runs/              每次 run 的原始产出与复现脚本（**原始数据 gitignored、复现脚本提交**，见下）
```

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

只对**判分方本来就给出部分分**的数据集存在（**CL-Bench / CorporateBench**）：
`{"mean": …, "n": …}`，逐类还有 `partial`。**给不出的数据集写 `mean=None, n=0`**
（同 `_accuracy` 的纪律：空集合不写 `0.0`，那会被读成"全错"）。

CorporateBench 每题传递标量 exact-match / 列表 set-F1，完整 QA 均分另写入
`scores.dataset_score`（带 `metric` / `mean` / `n` / 本地复现范围）。
`overall` 保留二值完全匹配准确率，两者不能混用；它也不代表 AML 榜分。
实现与重判记录见 [修复报告](corporatebench-fix-20261004.md)。

**为什么需要它**：CL-Bench 的官方分是**全有全无**的 LLM rubric 判分
（[`../../benchmark_data/clb_pipeline.py`](../../benchmark_data/clb_pipeline.py) 的判分 prompt 原文：
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

## git 忽略策略（有意不对称）

| 类型 | 是否入库 | 理由 |
| --- | --- | --- |
| `ledger.md` | ✅ **提交** | 它是结论本身 |
| `schema.py` | ✅ 提交 | 代码 |
| `runs/**/*.json` `*.jsonl` `*.csv` | ❌ **忽略** | 体积大、可重跑；且逐题结果可能含受许可约束的数据内容（§12.5） |
| `runs/**/*.py` | ✅ **提交** | 复现用的探针 / 审计脚本（`probe.py` / `audit.py` …）——重跑与复核要它们 |
| **T2 的人工标注** | ✅ **提交**（但它不在这里，在 [`../experiments/`](../experiments/)） | **133 行人工劳动，丢了要重做** |

**规则的形状**：**派生产物入库，原始数据与运行时状态不入库。** 后者由 `.gitignore` 覆盖；若你新增一类产物，先问它属于哪一边。

---

## 两条记录纪律

| 纪律 | 出处 |
| --- | --- |
| **latency / 成本只在明显变差时才追** | §13 |
| **不记 Recall@K，不做按问题类型的收益归因** | §14——前者已接近天花板，**后者是论文的事** |

**另外**：结论怎么写见 [`./ledger.md`](./ledger.md) 的"怎么记一条"——**本文件不另列一份**。
