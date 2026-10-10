# configs/ — 运行时配置

> **状态**：`default.yaml` 与 `local.yaml` **已建**，由
> [`../src/tianximem/common/config.py`](../src/tianximem/common/config.py) 加载。
> **`runs/` 仅保留新实验入口说明**：旧分支快照已于 2026-10-10 清理；后续计划只重跑 `v1.0`、`v1.2`。
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

**profile 清单（文件 / 用途 / 模型 / 状态）在 [`../docs/config-reference.md`](../docs/config-reference.md) §1**，本文件不复制。**`runs/<arm>/` 不是 profile**，用于新对照臂的冻结快照（目录状态见下节）。

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

## 目录与后续实验

```text
configs/
├── default.yaml     # 活动基线配置
├── local.yaml       # 开发期覆盖
├── submit.yaml      # 提交期覆盖；模型切换仍待 Step 5 重标定
└── runs/README.md   # 新实验快照入口；旧分支实验已清理
```

`runs/` 保留为新跑批快照的落点，`TIANXIMEM_CONFIG_DIR` 的读取方式不变。
旧 T1/A3、共同取证、记忆治理、PersonaMem 和外部基线等实验快照
已从当前工作树移出；历史记录从清理前提交恢复，原快照字节不改写。
目录状态、历史恢复与版本对照要求统一见 [`runs/README.md`](runs/README.md)。

后续计划仅重跑 Git 标签 `v1.0`、`v1.2` 的服务，配置从各自标签取得。
新对照仍按 §13 冻结配置、数据及模型指纹；临时快照默认忽略，
正式比较需要共享的配置经核对后显式提交。

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
