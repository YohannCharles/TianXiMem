# TianXiMem

**AML（Agent Memory Leaderboard，记忆之巅排行榜）参赛系统。**只提供 `Add` 与 `Search` 两个 HTTP 端点，答案生成、评判、聚合全部由 AML 完成。

> **完整实现规格见 [`AML Agentic Memory 增强框架 PRD.md`](./AML%20Agentic%20Memory%20增强框架%20PRD.md)。**
> **约束速查见 [`CLAUDE.md`](./CLAUDE.md)**——它是写给 Claude Code 的、每次会话自动加载的那一份，内容与 PRD 同步维护。
> **任何冲突以 PRD 为准。**

本仓库**实现已开工**——存储层、配对、检索、打包与服务都已落地，见下方「当前状态」。

---

## 目标

把 AML 榜单分数做到**优于 ReFind 的 44.97**。

- **不写论文**，所以不做"为了讲清楚贡献"的实验，**只做能改变下一步动作的对照**（PRD §13）。
- **核心判断**：检索不是瓶颈（LongMemEval 召回已接近天花板，数字见 PRD §4），**选择和排序才是**。因此 Rerank + Context Packaging 是主线（PRD §4 / §11）；另有已落地的**有来源的共同事实取证**（见下方「架构」）。
- **核心 claim**：不让所有 Query 都进昂贵的 Agentic Search，**只有证据不足时才触发多轮搜索**（PRD §3.1 / §9）。

**资源**：3 人 / 4–6 周。运行形态已定（D12）：**模型三段全部经自建网关远程访问；检索服务与 Qdrant 跑在本机**——本机不需要 GPU。

---

## 必须知道的六件事

详细版本（含后果、出处、静默失败机制）在 [`CLAUDE.md`](./CLAUDE.md)；这里只列**踩了会丢分**的：

| # | 约束 |
| --- | --- |
| 1 | **`top_k` 由 AML 固定为 100**，返回超过它是**契约错误**、**不会被静默截断**——必须精确计数 |
| 2 | **答案阶段按返回顺序取 117,760 token 前缀**——排在后面的证据整段作废 |
| 3 | **`user_id` 是唯一的检索隔离字段**；`session_id` 不是 Search 的过滤器 |
| 4 | **幂等**：块写下即最终形状（D24）、位置是**请求的纯函数**（D28）⇒ 重试必然算出同一位置；但**批次级仍必须查 `applied_batches`** 旁表（重试是正常行为，不能靠撞 `UNIQUE` 兜），且**同 `request_id` 不同 payload ⇒ 409** |
| 5 | **Qdrant RRF 的 `k` 默认是 2**（不是文献里的 60），必须显式设 **61** |
| 6 | **`content` 里只许带日粒度日期**（`[YYYY-MM-DD]`，D21）——**秒级绝不许出现**，也别加星期 |

**模型规定**（§2.3）：Embedding 只能用 `text-embedding-v4`，LLM 只能用 `gpt-4o-mini`，**Reranker 不作规定**。
**法律约束**（§12.5）：**数据不得用于训练，「微调模型」不是暂不实现、是不允许。**

---

## 架构

当前 Add/Search 内部采用共同事实取证：

```text
Add → 原文真源 + 共同事实（主体、关系、客体、限定条件、来源）
Search → 共同查询计划 → 条件筛选 / 有限连接 / 显式区间比较 → 原文或事实片段
                    └→ 无适用计划、证据冲突或索引未齐 → 下方混合检索链
```

来源语义绑定集中在句法层，执行器和存储不再按业务拆分。
名单与计数共用证据，不在 Search 生成答案。模块分工见 [当前架构](docs/architecture.md)。

```text
                              User Query
                                  │
        ┌─────────────────────────┴─────────────────────────┐
        │ 共同取证（retrieval.grounded_evidence，默认开）        │
        │ 问题 → 共同计划：条件筛选 / 有限连接 / 显式区间比较      │
        └─────────────────────────┬─────────────────────────┘
     命中：原文或独立事实片段           不适用 / 索引未扫完 / 冲突 / 超限
     （不 rerank、半径 0）                          │
                  │                                ▼
                  │              ┌──────── Hybrid Retrieval ────────┐
                  │              │          BM25   +   Dense        │
                  │              └──────── Weighted RRF ────────────┘
                  │                              │
                  │                       Initial Candidates
                  │                              │
                  │                Evidence Checker 【v1：空实现】
                  │                              │
                  │                     Rerank（远程 API）
                  │                              │
                  │         Neighbor Expansion（默认 radius=0 ⇒ 不扩窗）
                  │                              │
                  └──────────────┬───────────────┘
                                 ▼
                   Context Packaging（同一份槽位 + token 双预算）
                                 ▼
                        ≤ Top-K（精确计数）

        ═══ 以下整块为 v2，v1 不实现（见 docs/decisions.md D13）═══
          Evidence Weak → Agentic Search
            - 关键词重写（Agent 每轮自产）
            - Multi-round Search / Temporal Search / Evidence Note
```

> **v1 是直通的**：Evidence Checker 恒返回「证据充足」，因此**没有证据补充路径**——正确的 QA 对不在初始候选里就永久丢了。这把 v1 的全部重量压在**排序 + token 预算分配**上（§4 / §8）。共同取证链是一条**短路**：只在问题被共同计划覆盖、且当前用户来源扫描齐全时命中；不覆盖的问题仍走上面这条链。
>
> v1 的 Checker 是**恒返回「充足」的空实现**（D13），判定结果被丢弃；**A4 已移出 v1**（D26）⇒ 那份反事实分布现在不攒（要补它得先给 `store/` 加一个单路查询——已登记、未实现）。

> **基础检索采用混合检索**（D15）；共同事实计划不适用时沿用这条链。

**存储分工不可互换**（§6.3）：**SQLite 是真源**（QA 对正文、`status`、位置 `(request_id, local_index)`、`event_time`；另有**可由正文重建的派生事实表** `memory_facts` / `evidence_coverage`），**Qdrant 是派生索引**（向量 + 过滤键 payload，**不含正文**，坏了可从 SQLite 全文重建）。

**索引单元 = 一个 QA 对**（不是一个 message），一对一个向量。一个 QA 对 = **一段连续 user 消息的最后一条 + 紧随其后的非 user 消息段**（§6.2；D20 → **D32**：前面几条 user 各自独立成块，`U U U A` ⇒ `[U] [U] [U+A]`）。

---

## 目录导航

> **每个目录下的是 `CLAUDE.md` 而不是 `README.md`**——内容面向执行：该目录要写什么、受哪条约束、对应 PRD 哪一节，以及本层的静默陷阱。根 [`CLAUDE.md`](./CLAUDE.md) 里有"动 X 之前先读 Y"的路由表。

| 路径 | 内容 | PRD |
| --- | --- | --- |
| [`CLAUDE.md`](./CLAUDE.md) | **约束速查 + 文档路由**（写给 Claude Code，也适合人读） | 全部 |
| [`AML Agentic Memory 增强框架 PRD.md`](./AML%20Agentic%20Memory%20增强框架%20PRD.md) | **权威实现规格** | 全部 |
| [`docs/`](./docs/) | 架构与七维映射、契约、配置项、实验登记、悬而未决清单、提交台账、决策日志 | — |
| [`dataset/`](./dataset/CLAUDE.md) | 按数据集组织的本地材料；评测前按需下载，数据与上游源码 gitignored | §12 / D35 |
| [`configs/`](./configs/) | 运行时配置：三个 profile + 完整开关清单 + 开关依赖图 | §15 |
| [`deploy/`](./deploy/) | Qdrant server（**版本钉死**）、Step 5 重建 runbook | §6.3、§7.3 |
| [`src/tianximem/`](./src/tianximem/) | 检索服务本体 | §6–§11、§14、§15 |
| [`eval/`](./eval/) | 代理评测：`datasets/` `harness/` `experiments/` `baselines/` `smoke/` `reports/` | §12、§13 |
| [`tests/`](./tests/) | 单元测试（记忆块组合 / 幂等 / 契约 / 隔离 / 开关纯度） | — |
| [`var/`](./var/) | 运行时产物（**gitignored**，只保留说明） | §6.1 |

MemMachine 参考源码位于 [`eval/baselines/memmachine/`](eval/baselines/memmachine/)，
研读路径与 TianXiMem 改进方向见 [MemMachine 学习入口](docs/memmachine-reference.md)。当前仅供源码参考。

---

## 路线图

**步骤定义、交付与逐项状态见 [`docs/roadmap.md`](./docs/roadmap.md)**——**本文件不复制那张表**
（两处各存一份必然漂移，而状态恰好是本仓最易过期的东西）。

两条不可犯：

- **Step 0 不可跳过**——没有它，后面每一步都是盲调，**而 Full 只有 2 次**。
- **Step 5 不可与任何设计改动合并**——否则分数变化无法归因（§12.1 R1）。

---

## 评测机会成本

**AML 不提供本地评测**（不给 gold answer、不给评分标准、不提供批量数据下载）。真实信号只有两条路：
**代理评测**（本地，无限次——全部迭代、消融、调参都在这里）与 **Smoke / Full**（次数是硬上限，
配额表与版本冻结规则见 [`docs/submission.md`](./docs/submission.md)）。

> **本项目没有"跑一遍看看"的余地。** 任何能在代理评测上回答的问题，都不该花 Smoke 的额度——
> Smoke 只做两件事：验证契约合规、消除本地无从验证的未知。

**代理评测的边界**：只覆盖 LoCoMo-Refined + LongMemEval——而这两份恰好是全部数据集里**唯一共用同一套契约**的。其余各份的记忆注入字段与裁判规则各不相同（PersonaMem 的本地答案输入适配见 [`eval/harness/CLAUDE.md`](eval/harness/CLAUDE.md)），因此**代理分数不能线性外推到全赛道**（§12.4）。

---

## 当前状态

**实现已开工，不是空脚手架。** 模块级的**状态表唯一权威在根 [`CLAUDE.md`](./CLAUDE.md)**（本文件不再镜像一份——镜像必然掉队）；分步交付与剩余项在 [`docs/roadmap.md`](./docs/roadmap.md)。

### Step 0 阻塞项

**运行形态已定**（见 [`docs/decisions.md`](./docs/decisions.md) D12）：**模型（LLM / embedding / reranker）全部经自建网关远程访问；检索服务与 Qdrant 跑在本机**——本机不需要 GPU，docker 已就位。

| # | 阻塞项 | 状态 |
| --- | --- | --- |
| 1 | **`api_config.py`** | ✅ **不是环境依赖**：**AML 公开仓自己就发布了这个文件**（520 字节、无凭据、只读 `os.environ`），放**仓库内** + 由 harness 在 subprocess 里注入 `PYTHONPATH` 即可——**"clone 下来不能直接跑"不成立**。实现见 [`eval/harness/api_config.py`](./eval/harness/api_config.py) |
| 2 | **本地评测数据** | ✅ 已整理为 `dataset/`，来源与哈希在 [`eval/datasets/manifest.py`](./eval/datasets/manifest.py)。按需准备，验证见 [迁移报告](./eval/reports/dataset-layout-20261006.md)。 |
| 3 | **Windows 那台机器的 `tmp_path` 故障** | ⬜ **仍开着，且与代码无关**（是机器状态）。绕法见 [`docs/roadmap.md`](./docs/roadmap.md) 的 Step 0 |

**reranker 走自建网关**（`TIANXIMEM_RERANKER_BASE_URL` + `rerank.envelope` 两半必须配套，观测点与两种信封见 [`src/tianximem/rank/CLAUDE.md`](./src/tianximem/rank/CLAUDE.md)）。它是 Step 3 及之后的**基础设施前置项**，不是代码任务——而 v1 不做 agentic 之后，**Rerank + Context Packaging 就是主线上的新增价值**，所以这条前置项直接压在主线上。开发期由 `configs/local.yaml` 显式关掉（墙钟约 3×）；关掉时链路照常（记 `rerank_disabled`）。

本机默认 Python 3.14.4，已按 `>=3.11,<3.14` 保守钉在 3.12（torch / qdrant-client 的 wheel 覆盖通常滞后）。

完整清单见 [`docs/roadmap.md`](./docs/roadmap.md) 的 Step 0。

## 快速开始

**新机器上从 clone 到跑通一轮**——每一行都是必需的，顺序也是：

```bash
cp .env.example .env      # ← 密钥要向管理员申请（仓库里全空，见下）
make sync                 # uv sync --all-extras；含 [local] 本地模型栈，数 GB
make fetch-data DATASET=locomo-refined  # 按需准备；make eval 也会自动准备缺失材料
make data-check           # 逐文件 sha256 校验
make qdrant-up            # 起 Qdrant（server 模式）
make check                # 环境自检：Qdrant / 三段模型端点 / 密钥已填
make serve                # ⚠ 另开一个终端：起检索服务
make baseline DATASET=locomo-refined   # 跑一轮冻结口径的基线
make baseline DATASET=halumem          # HaluMem-Medium 的检查点 QA
make baseline DATASET=musique          # MuSiQue-Full dev 的答案与拒答评测
make baseline DATASET=hybridqa         # 整表及链接段落的问答
make baseline DATASET=feverous         # claim 候选整页上的标签及证据评测
```

HybridQA / FEVEROUS 的语料下载、输入范围与评分见
[接入报告](./eval/reports/hybridqa-feverous-pipelines-20261006.md)。FEVEROUS 首次准备会下载
完整 Wikipedia 数据库并生成候选检索索引，空间与耗时也记录在那里。

**四件事先知道，否则会卡在半路：**

| | |
| --- | --- |
| **`.env` 的密钥要向管理员申请** | 仓库里**全空**（只提交 `.env.example`）。这是**唯一一个"仓库里查不到答案"的步骤**，其余都能自己跑通 |
| **数据集不进版本库** | `dataset/CLAUDE.md` 入库，数据不入库；首次评测自动按固定版本下载所选数据集及评分依赖，`--offline` 禁止下载。需要预取时运行 `make fetch-data DATASET=<名称>` |
| **Qdrant 必须 server 模式** | **local 模式会静默丢弃 payload 索引**，而 `user_id` / `session_id` / `event_time` 三个筛选**全依赖**它（机制与出处见 [`deploy/CLAUDE.md`](./deploy/CLAUDE.md) §1） |
| **服务得先起着** | `make eval` / `make baseline` 打的是**真 HTTP**（§13 的边界：harness 不 import `src/`）⇒ 另开一个终端跑 `make serve` |

`make fetch-data` 默认只准备 LoCoMo-Refined；其他数据集用 `DATASET=<名称>` 指定。
需要全档时显式用 `DATASET=all`。旧的按档下载仍可通过
`uv run --env-file .env python -m eval.datasets.prepare --fetch --tier required` 使用。

**部署到服务器**（只跑检索服务 + Qdrant，不需要 GPU）：

```bash
cp deploy/.env.example deploy/.env   # 填 AML_EMB_*（reranker 可选）
make up                              # 起整栈（容器形态）
make deploy-check                    # 在容器里跑契约预检（14 条，真 HTTP）
```

细节（单进程约束、真源卷、服务器要满足什么）见 [`deploy/CLAUDE.md`](./deploy/CLAUDE.md) §0。

`Makefile` 里服务与评测的目标**指向尚未实现的模块时会明确失败**——这是有意的，**避免误以为某一步已经实现**。
