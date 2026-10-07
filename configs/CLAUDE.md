# configs/ — 运行时配置

> **状态**：`default.yaml` 与 `local.yaml` **已建**，由
> [`../src/tianximem/common/config.py`](../src/tianximem/common/config.py) 加载。
> **`runs/` 已建**：**44 个目录**的对照臂配置快照（分类与重放注意见下节）。
> `submit.yaml` **已建**（2026-09-28）——**现在只有 `models.embedder` 一项**
> （`text-embedding-v4`）；**由模型派生的量（token 预算实测量、全部标定阈值）待 Step 5 重标定后往这里补**。
> **完整配置项清单见 [`../docs/config-reference.md`](../docs/config-reference.md)**（每个配置项、默认值、出处 §）。本文件只说**为什么这么组织**。

---

## 一条纪律：**每个键只有一个家**

| 层 | 拥有哪些键 | 例子 |
| --- | --- | --- |
| **`.env`**（环境变量） | 密钥、端点、**路径**、进程形态（worker 数） | `AML_EMB_BASE_URL`、`TIANXIMEM_SQLITE_PATH`、`TIANXIMEM_METRICS_PATH`、`TIANXIMEM_WORKERS` |
| **`configs/*.yaml`** | 阈值、权重、模型名、集合名 | `retrieval.rrf.k`、`models.embedder`、`storage.qdrant.collection` |

**两者不重叠，也不许重叠。** 两处都能设的值，最终会变成"跑出来的结果和 yaml 里写的不一样，
而没人知道为什么"。在 yaml 里写一个 env 拥有的键（比如 `storage.sqlite.path`）会**直接报错**，
反之亦然——**不认识的键一律报错**，因为拼错的键被静默忽略等于跑在没人声明过的值上。

**为什么阈值不能藏在环境变量里**（§12.1 R1 对冲 4）：那会让 Step 5 的模型切换变成"改 shell 变量"——
**改了什么无法 diff、无法评审**，而归因恰恰是那一步唯一的目的。

**加载顺序**：内置默认值 ← `default.yaml` ← `<profile>.yaml` ← 环境变量 ← 校验。
**`default.yaml` 缺失或不存在的 profile 都报错**，不静默退回。

---

## 三个 profile

**profile 清单（文件 / 用途 / 模型 / 状态）在 [`../docs/config-reference.md`](../docs/config-reference.md) §1**，本文件不复制。**`runs/<arm>/` 不是 profile**，是对照臂的冻结快照（见下节）。

> ⚠ **`local.yaml` 必须保持短。** 它存在的意义就是让"开发期与提交期差在哪"一眼可见；
> 重述默认值等于把那个信息淹掉。`local.yaml` / `submit.yaml` **只覆盖模型与由模型派生的量**
> （向量维度、实测 token 预算、全部标定阈值），其余继承 `default.yaml`——全部复制一遍，
> Step 5 切换时**没人知道该重标定哪些**（R1 对冲 3）。
> **`local.yaml` 覆盖两项**：`storage.qdrant.collection`（开发期用单独的集合，避免 §6.3 的 upsert
> 把上一套实验的 point **静默留给下一套**，V9）与 `rerank.enabled: false`（开发期不花那份墙钟——
> 基线里是 `true`，那是提交口径）。**`submit.yaml` 现在只覆盖一项**：`models.embedder`。

**两个 profile 之外的加载细节**（实现在 [`../src/tianximem/common/config.py`](../src/tianximem/common/config.py)）：

| 变量 | 作用 |
| --- | --- |
| `TIANXIMEM_PROFILE` | 选 `configs/<name>.yaml`（默认 `default`） |
| `TIANXIMEM_CONFIG_DIR` | 换一份配置**目录**（替代集合的 arm 快照就靠它，见下 `runs/`） |
| `TIANXIMEM_ENV_FILE` | 换 `.env` 的位置（默认 cwd 下的 `.env`） |

⚠ **`.env` 由 `common/config.py` 读取**。它**不是**一条独立来源，而是"这台机器的环境"的本地副本——**排在真实环境变量之下**，`export` 过的值压过它。

---

## 目录里最终会有什么

```text
configs/
├── default.yaml     # 基线：全部有消费方的阈值与模型名（含共同取证的四个检索键）
├── local.yaml       # 开发期覆盖（**只写与基线不同的键**）
├── submit.yaml      # 提交期覆盖（🟡 已建：模型名；阈值待 Step 5 重标定）
└── runs/            # 每次对照实验的配置快照（哪个实验、什么时候、哪套模型）✅ 44 个目录
```

`runs/` 的用途：`docs/experiments.md` 要求记录**配置指纹**。让每个实验留下**冻结的配置副本**，而不是"当时的 local.yaml 大概是这样"——**后者在 Step 5 之后就无法重建了**。
（跑某个 arm 用 `TIANXIMEM_CONFIG_DIR=configs/runs/<arm>` 指向那份快照。）

44 个目录大致分四类，重放前先确认手上这份落在哪一类：

- **2026-09 的对照臂**：`t1-*` / `a3-*` / `annotate` / `seed-*` / `cov-wide` / `clbench` / `lme`——
  T1/A3 四份由 [`../eval/experiments/t1_timestamp.py`](../eval/experiments/t1_timestamp.py) 与
  [`a3_rerank.py`](../eval/experiments/a3_rerank.py) 的 `--freeze`（脚手架 `arms.py`）生成；其余几份是一次性冻结
  （各份 `local.yaml` 头部写着当时动了哪几件事）；数字在 [`../eval/reports/ledger.md`](../eval/reports/ledger.md)。
- **两个外部基线**：`refind`（B1）与 `invmem-qwen`（候选仓库 + 我们的 embedding shim）——这两份记的是
  **另一个服务**的启动环境变量，不是我们的 yaml（loader 读不了它们）；口径见
  [`../eval/baselines/CLAUDE.md`](../eval/baselines/CLAUDE.md)。
- **共同取证的各阶段臂**（`*-20261005`）：
  - **加载得起来**：`chinese-evidence-20261005`、`unified-evidence-20261005/{candidate,generalization}`、
    `facts-benchmark-20261005/{on,off}`（本轮冻结配置；`grounded_evidence` 的开 / 关两臂），以及
    `goal-20261005-{baseline,employment,subject}`（这三份的 `retrieval` 段不含已删除的业务键；
    其中 `-baseline` 还缺 `grounded_evidence` 键 ⇒ 事实取证按**关闭**跑）。
  - **加载会报错、只供对应旧提交复现**：其余带业务专用键的 `goal-20261005-*`、
    `generalization-20261005`、`generic-evidence-20261005/*`、`unified-evidence-20261005/baseline`——
    那些业务专用键（`employment_facts` / `statement_source_limit` / …）**不在当前键集里**，
    **当前代码遇到它们拒绝启动**。
  - 报告见 [`../eval/reports/`](../eval/reports/) 的 `*-20261005.md`。
- **新数据集接入臂**（`new-datasets-benchmark-20261006`）：HaluMem / MuSiQue / HybridQA / FEVEROUS
  等的接入基准——`default.yaml` 与基线**逐字相同**，`local.yaml` 只覆盖集合名与
  `rerank.enabled: false`（当前键集，可加载）。报告见
  [`../eval/reports/new-datasets-benchmark-20261006.md`](../eval/reports/new-datasets-benchmark-20261006.md)。

> ⛔ **2026-09 那批快照的 `models.embedder` 是旧网关 id**（`Qwen/Qwen3-Embedding-8B`；现役 id
> `qwen3-embedding-8b`，对照表见 [`../docs/config-reference.md`](../docs/config-reference.md) §9）。
> **故意不改**：它们是**历史记录**，改了就等于伪造"当时跑的是什么"。
> ⇒ **重放那一批之前必须先把 `models.embedder` 覆盖成当前 id**，否则第一步 embedding 就 404。
> 另外**维度也从 1024 变成了 4096** ⇒ 那批 arm 产出的集合与缓存**作废**
> （动作见 [`../deploy/CLAUDE.md`](../deploy/CLAUDE.md) §4）。**20261005 起冻结的快照已是当前 id**——
> 重放它们不用覆盖。

> ⚠ **`default.yaml` 只收"有代码消费方"的键**（③-d）。`checker.*` / `agent.*` 的落点已写在
> [`../docs/config-reference.md`](../docs/config-reference.md)，但**没有搬进 yaml**——消费方还没接线
> （`checker` / `agent` 是 v2 的事，D26 / D13），收进来等于预留字段。
> ⇒ **`default.yaml` 不是"§15 七个消融项都在这里"。**
> `neighbor.*` / `budget.*` / `rerank.*` 与**共同取证的四个检索键**（`retrieval.grounded_evidence` /
> `evidence_limit` / `evidence_hop_limit` / `fact_backfill_limit`——语义与回退条件见
> [`../docs/config-reference.md`](../docs/config-reference.md) 顶部）都已在，各自有 consumer。
> ⚠ `rerank` 段**只有开关、超时与 `envelope`**：端点 / 密钥 / 模型名在 `.env`（那是端点身份，不是阈值）。
> ⚠ `grounded_evidence` 的**代码内置缺省是 `false`**（`default.yaml` 写的是 `true`）——缺这个键的快照
> 就按"事实取证关闭"跑。

---

## 建 YAML 时先看这三条

1. **`k=61` 不是调参项**，是正确性常量——写进配置但**不要放进"可调阈值"分组**（§7.3 / D5）。
   配置层会**拒绝启动**，不是警告后照用。
2. **先分清常量与阈值**：见 [`../docs/config-reference.md`](../docs/config-reference.md) §1.5 的 A / B / C 三分类。**"配置化"不等于"可调"**——`top_k = 100` 与 `k = 61` 都是写进配置但**不许动**的。
3. **每个开关在"关"分支下只影响它命名的那一件事**（§13）——**开关的依赖图、"关掉时不得改变什么"与"不要实现"的连带项一处声明在 [`../docs/config-reference.md`](../docs/config-reference.md) §2**（`dense` / `rrf` **没有下游依赖**，§8 的 Checker 退化路径不存在，D15），本文件不另列一份。对应测试见 [`../tests/CLAUDE.md`](../tests/CLAUDE.md)。

**⚠ 开关的接线状态**（`checker.*` / `agent.*` / `rrf` 等）**逐项见 [`../docs/config-reference.md`](../docs/config-reference.md) §2 的"接线"列**——`checker` / `agent` 是 v2 的事（D26 / D13）；`rrf` 与 `dense` 是**无下游依赖**（D15），**不是待办**；`neighbor` 只有 `radius` 可关，而它默认就是 `0`（D31）。本文件不另列。
