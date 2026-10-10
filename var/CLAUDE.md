# var/ — 运行时状态

**PRD**：§6.1（真源文件）、§6.3（Qdrant 卷）、§7.2（向量缓存）、§15（随 run 归档）

## 为什么叫 `var/` 而不叫 `data/`

`var/` 沿用 Unix 的 `/var` 语义：**只装运行时状态**。**装进这里的东西都应当是可重建或可弃的。**

⚠ **别与评测材料 `dataset/` 混为一谈**——混了会导致有人往里写、或反过来以为它可以删。

## 里面有什么

```text
var/
├── tianxi.db              SQLite 真源（§6.1）——**唯一的不可重建物**
├── tianxi.db.pre-d25      旧库留档（2026-09-30，schema 早于 D25）
├── tianxi.db.pre-official 旧库留档（2026-10-01，**add 形态改成 official 之前**那一代）
├── embed_cache/           embedding 缓存（§7.2）——**可重建，但重建要花钱**
├── capture/               请求原文采集（`capture.enabled`）——**诊断产物，可弃**
└── <实验与跑批的临时目录、日志>  快照库、逐题答案——**可弃**
```

> **2026-10-10：早期实验目录已清理。** 15 个跑批目录（`optimization-12h-20261007`、
> `new-datasets-benchmark-20261006`、`facts-benchmark-20261005`、`targeted-six-hours-20261005`、
> `unified-evidence-20261005`、`generic-evidence-20261005`、`chinese-evidence-20261005`、
> `generalization-20261005`、`ab-{new,old}`、`d24-locomo`、`order-{a,b}`、
> `order-probe-cache`、`logs/`）按"可弃"口径删除，**释放 ~6.9 GB**。
> ⇒ **各报告里指向这些路径的引用不再有效**（`eval/reports/*.md` 里有十余处叙述性指向，
> 少数复现命令随之作废）。删除前已把"不在任何 ref、且不可重建"的部分摘进
> 当时 `eval/reports/runs/` 的同名 `var-artifacts-20261010/` 目录
> （补丁、配置快照、修订版代码），并建 tag `archive/optimization-12h-baseline`
> 保活原 `baseline` worktree 的悬空提交链。清理方法（逐文件 `hash-object` 比对）
> 与逐目录清单保存在清理前提交 `05c0c35` 中。
> 同日随后按用户要求清理旧分支实验，以上摘出材料也已移入本地归档；
> 当前恢复入口见 [`评测产物说明`](../eval/reports/runs/README.md)。
> `capture/`（官方请求原文）与两个 `pre-*` 祖库**未动**；`embed_cache/` 因仍在服务使用中保留。

**整目录 gitignored**（`.gitignore` 的"运行时产物"段），**只保留本文件**，让它在 checkout 后依然存在且有说明。

---

## ⚠ 四样东西的可重建性完全不同

| 物 | 丢了会怎样 |
| --- | --- |
| **`tianxi.db`** | **没了就没了。**它是真源，**备份对象只有它** |
| **Qdrant 卷**（`deploy/compose.yaml` 的 Docker 命名卷，**不在本目录**） | 可从 `tianxi.db` 的正文全量重建（这正是"Qdrant 是派生读存储"的意思，§6.3） |
| `embed_cache/` | 可重建，**但要重付一遍 embedding 的钱**——本地模型是电费，线上是 API 账单 |
| `capture/` | **丢了就丢了，不用重建**——它是诊断产物（要再来一次就再打开开关跑一遍）；⚠ 但 **§4 的备份是整卷 `tar`**，它会顺带被拷进备份包——**恢复时可弃** |

**所以 Step 5 的重建 runbook 里"备份 SQLite"是第 2 步**（见 [`../deploy/CLAUDE.md`](../deploy/CLAUDE.md) §4）——**顺序不能反**。

**并且 SQLite 文件随 run 归档**（§15）——它是 [`../eval/reports/`](../eval/reports/) 里那条"归档"字段指向的东西。

---

## 两条约束

| 约束 | 说明 |
| --- | --- |
| **不要往这里提交任何东西** | 整目录被忽略，提交了也看不见——**若你写的东西需要入库，它就不属于 `var/`**（派生【产物】的归属见 `.gitignore` 里那条说明） |
| **embedding 缓存必须落盘**（§7.2） | 不是优化项，是**成本结构**：v1 的 Add 阶段不调用任何 LLM，embedding 是唯一的 Add 侧成本，**且只与内容有关——缓存后即成为一次性成本，与迭代次数无关**（实现要求见 [`../src/tianximem/embed/CLAUDE.md`](../src/tianximem/embed/CLAUDE.md)） |

> **缓存条目键与"一个坐标系一个缓存文件"的口径在根 `CLAUDE.md` 的「两个 ID / 缓存键」表与 [`../src/tianximem/embed/CLAUDE.md`](../src/tianximem/embed/CLAUDE.md)**——本目录不重复。Step 5 切模型时按 [`../deploy/CLAUDE.md`](../deploy/CLAUDE.md) §4 的 runbook 第 3 步**作废整个缓存目录**。

---

## ⚠ 换库要**连 Qdrant 集合一起换**（2026-09-30）

**库与集合是两个坐标系，只有一个方向会自动跟上**：`SqliteStore.open()` 只在"表里有
`chunk_ordinal`，**或**缺 `prev_memory_id`"时自动搬一次——那是给 **D25 那代库**写的
（它 SELECT 的列名是 `local_index`）。**D25 之前那代库（列名 `pair_idx`）没有迁移路径**：
服务一启动就 `no such column: local_index`，**起不来**。

**换库时只换 SQLite 是不够的**：旧集合里的 point 是旧库的派生索引，**新库不会覆盖它们**
（§15 写路径第 5 步：同 `id` upsert 覆盖）⇒ 检索会返回"新真源里根本不存在的记忆"，而**不报错**（这就是 **V9**）。

```bash
mv var/tianxi.db var/tianxi.db.pre-d25                          # 留档。**别删**——它是唯一不可重建的那份
curl -X DELETE http://127.0.0.1:6333/collections/memories_dev   # 派生索引；下次 Add 会自动重建
```

> 本机（WSL）2026-09-30 已这么做过一次：`tianxi.db.pre-d25` 是 9/27 那代（schema 早于 D25）的库，
> `memories_dev` 已删。⚠ **那份留档仅供取证，不能直接喂给服务或 `tools/reindex.py`**——D25 之前那代库
> 没有迁移路径（见上），要用它得先 checkout 当时的代码，或先写转换脚本。

**2026-10-01 又做了一次，理由不同但手法一样**：add 形态从 `native` 改成 `official`
（逐数据集加 `<标签>: ` 前缀、`system` 折成 `user`、单条 8,000 字符切分、每条 Add 2,000 词预算）——
**正文变了 ⇒ 渲染出来的记忆也变**，旧库是另一套坐标系。

```bash
mv var/tianxi.db var/tianxi.db.pre-official
curl -X DELETE http://127.0.0.1:6333/collections/memories_dev
```

⚠ **换形态时必须同时换库**，理由比换 schema 更硬：`request_id` 是**确定性**的
（`user|session|index`），而 payload 变了 ⇒ 同 `request_id` **不同 payload** ⇒
D28 的指纹守卫会**响亮 409**（那不是重试，是两套输入撞在一起）。**不是"会串数据"，是跑不起来。**

---

## 一条本地限制（别误读数字）

**本地复现的批次边界与线上不一定一致**（§12.3 第 5 条：官方给的计数口径是"由冻结的 Adapter 计"，**本仓复现不了那个 Adapter**）。D24 之后它不再影响任何埋点，但**它决定有多少 QA 对被批界切成两半**——原因与实测见 [`../docs/decisions.md`](../docs/decisions.md) 的 **D24**。**本目录不重复。**
