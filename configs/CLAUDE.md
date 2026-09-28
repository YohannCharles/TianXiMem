# configs/ — 运行时配置

> **状态**：`default.yaml` 与 `local.yaml` **已建**，由
> [`../src/tianxi_am/common/config.py`](../src/tianxi_am/common/config.py) 加载。
> **`runs/` 已建**：`configs/runs/{t1-plain,t1-dated,a3-on,a3-off}/` 是各对照臂的冻结快照（由
> [`../eval/experiments/t1_timestamp.py`](../eval/experiments/t1_timestamp.py) 与
> [`a3_rerank.py`](../eval/experiments/a3_rerank.py) 的 `--freeze` 生成）。
> `submit.yaml` **待建**（Step 5 产出）。
> **完整配置项清单见 [`../docs/config-reference.md`](../docs/config-reference.md)**（每个配置项、默认值、出处 §）。本文件只说**为什么这么组织**。

---

## 一条纪律：**每个键只有一个家**

| 层 | 拥有哪些键 | 例子 |
| --- | --- | --- |
| **`.env`**（环境变量） | 密钥、端点、**路径**、进程形态（worker 数） | `AML_EMB_BASE_URL`、`TIANXI_SQLITE_PATH`、`TIANXI_METRICS_PATH`、`TIANXI_WORKERS` |
| **`configs/*.yaml`** | 阈值、权重、模型名、集合名 | `retrieval.rrf.k`、`models.embedder`、`pairing.batch_max_messages` |

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
> **`local.yaml` 只覆盖一项**：`storage.qdrant.collection`——开发期用单独的集合，避免 §6.3 的 upsert
> 把上一套实验的 point **静默留给下一套**（V9）。

**两个 profile 之外的加载细节**（实现在 [`../src/tianxi_am/common/config.py`](../src/tianxi_am/common/config.py)）：

| 变量 | 作用 |
| --- | --- |
| `TIANXI_PROFILE` | 选 `configs/<name>.yaml`（默认 `default`） |
| `TIANXI_CONFIG_DIR` | 换一份配置**目录**（替代集合的 arm 快照就靠它，见下 `runs/`） |
| `TIANXI_ENV_FILE` | 换 `.env` 的位置（默认 cwd 下的 `.env`） |

⚠ **`.env` 由 `common/config.py` 读取**。它**不是**一条独立来源，而是"这台机器的环境"的本地副本——**排在真实环境变量之下**，`export` 过的值压过它。

---

## 目录里最终会有什么

```text
configs/
├── default.yaml     # 基线：全部有消费方的阈值与模型名（§15 的七个消融项接完后才齐）
├── local.yaml       # 开发期覆盖（**只写与基线不同的键**）
├── submit.yaml      # 提交期覆盖（⬜ Step 5 产出）
└── runs/            # 每次对照实验的配置快照（哪个实验、什么时候、哪套模型）✅ T1 两臂 + A3 两臂已建
```

`runs/` 的用途：`docs/experiments.md` 要求记录**配置指纹**。让每个实验留下**冻结的配置副本**，而不是"当时的 local.yaml 大概是这样"——**后者在 Step 5 之后就无法重建了**。
（跑某个 arm 用 `TIANXI_CONFIG_DIR=configs/runs/<arm>` 指向那份快照。）

> ⚠ **`default.yaml` 只收"有代码消费方"的键**（③-d）。`checker.*` / `agent.*` 的落点已写在
> [`../docs/config-reference.md`](../docs/config-reference.md)，但**没有搬进 yaml**——消费方还没接线，
> 收进来等于预留字段。⇒ **`default.yaml` 不是"§15 七个消融项都在这里"。**
> `neighbor.*` / `budget.*` / `rerank.*` 已搬进来（扩窗 + 段合并 + token 预算 + 远端精排各自的
> consumer 都在）——⚠ `rerank` 段**只有开关与超时**：端点 / 密钥 / 模型名在 `.env`（那是端点身份，不是阈值）。

---

## 建 YAML 时先看这三条

1. **`k=61` 不是调参项**，是正确性常量——写进配置但**不要放进"可调阈值"分组**（§7.3 / D5）。
   配置层会**拒绝启动**，不是警告后照用。
2. **先分清常量与阈值**：见 [`../docs/config-reference.md`](../docs/config-reference.md) §1.5 的 A / B / C 三分类。**"配置化"不等于"可调"**——`top_k = 100` 与 `k = 61` 都是写进配置但**不许动**的。
3. **每个开关在"关"分支下只影响它命名的那一件事**——否则 §13 的对照不成立，**而结果看起来完全正常，只是结论错了**（§13）。

   **开关的依赖图与"关掉时不得改变什么"一处声明在 [`../docs/config-reference.md`](../docs/config-reference.md) §2**，本文件不另列一份。对应测试见 [`../tests/CLAUDE.md`](../tests/CLAUDE.md)。
   ⚠ **那两条 `dense → checker` / `dense → rrf` 的依赖边已随 D15 删除**，别再按旧图接。

**⚠ 一条待补的开关**：`checker.enabled` 必须是配置项（§13 的 A4 要关它做对照）——落点见 [`../docs/config-reference.md`](../docs/config-reference.md) §2。
