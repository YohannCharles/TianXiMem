# eval/reports/ — 数字的唯一落点

**PRD**：§13（记录）、§14（监控指标）、§3.2（七个维度）、§15（SQLite 随 run 归档）

## 要写什么

```text
ledger.md          **结果台账**（人工维护，**提交进 git**）
schema.py          run record 的 schema
runs/              每次 run 的原始产出（**gitignored**，见下）
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
| **配置指纹** | profile（`local` / `submit`）、各开关状态、**配置快照 hash** | §13 |
| **数据指纹** | 数据集 + 版本 + **切批口径** | §13 |
| **模型** | embedder / LLM / reranker 各自的标识 | §12.1 R1 |
| **阶段** | 属于哪个 Step | §16 |
| **§14 指标** | Agent Trigger Rate · 平均轮数 / Rewrite 次数 · latency/query · embedding API 调用数 | §14 |
| ~~**§6.5 计数器**~~ | ~~`pending_created` / `pending_completed` / `pending_orphaned`~~ ⛔ **D24 起这个量已不存在**（字段保留、恒为 `None`，只为旧记录读得回来） | §6.5 → **D24** |
| **归档** | 该 run 的 SQLite 文件副本 | §15 |

### ⚠ 七个维度必须逐维记录

官方维度**逐字采用**（§3.2）：

1. Explicit fact recall
2. Relational and multi-hop reasoning
3. Temporal and event understanding
4. **Memory governance**
5. Personalization and care
6. Rules and process execution
7. **Epistemic safety and privacy**

> **第 4、7 两维要显式回应**（§3.2，映射见 [`../../../docs/architecture.md`](../../docs/architecture.md)）——**报告里这两维为空的 run，等于没测。**

### ⛔ ~~`pending_orphaned` 要拆成两个字段~~（D24 已作废）

**2026-09-27（D24）取消了 `pending` 概念**：块在写下那一刻就是最终形状，**不存在
"被补全"这种归宿**，三个计数器**没有发射方**。⇒ 新 run record 的这四个键**恒为 `None`**，
键本身保留只为**旧记录还读得回来**。**别再为它们写聚合代码。**

> **D24 之后的替代信号**："切分有没有切坏 QA 对"**一次读库就能算**：
> `SELECT count(*) FROM qa_pairs WHERE answer IS NULL`（只剩 question 的半块数），
> 分母是总行数。不需要任何跨请求状态。

---

## git 忽略策略（有意不对称）

| 类型 | 是否入库 | 理由 |
| --- | --- | --- |
| `ledger.md` | ✅ **提交** | 它是结论本身 |
| `schema.py` | ✅ 提交 | 代码 |
| `runs/**/*.json` `*.jsonl` `*.csv` | ❌ **忽略** | 体积大、可重跑；且逐题结果可能含受许可约束的数据内容（§12.5） |
| **T2 的人工标注** | ✅ **提交**（但它不在这里，在 [`../experiments/`](../experiments/)） | **133 行人工劳动，丢了要重做** |

**规则的形状**：**派生产物入库，原始数据与运行时状态不入库。** 后者由 `.gitignore` 覆盖；若你新增一类产物，先问它属于哪一边。

---

## 两条记录纪律

| 纪律 | 出处 |
| --- | --- |
| **latency / 成本只在明显变差时才追** | §13 |
| **不记 Recall@K，不做按问题类型的收益归因** | §14——前者已接近天花板，**后者是论文的事** |

**另外**：`docs/experiments.md` 的结论列要求写清**怎么清掉的**（哪个实验、什么数字、哪次 Smoke）。**结论比状态有用**——状态是"✅"，结论是"因为 X 所以 Y"。
