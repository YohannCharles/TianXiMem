# TianXi_AM

**AML（Agent Memory Leaderboard，记忆之巅排行榜）参赛系统。**只提供 `Add` 与 `Search` 两个 HTTP 端点，答案生成、评判、聚合全部由 AML 完成。

> **完整实现规格见 [`AML Agentic Memory 增强框架 PRD.md`](./AML%20Agentic%20Memory%20增强框架%20PRD.md)。**
> **约束速查见 [`CLAUDE.md`](./CLAUDE.md)**——它是写给 Claude Code 的、每次会话自动加载的那一份，内容与 PRD 同步维护。
> **任何冲突以 PRD 为准。**

本仓库**实现已开工**（2026-09-23 起）——存储层与配对已落地，见下方「当前状态」。

---

## 目标

把 AML 榜单分数做到**优于 ReFind 的 44.97**。

- **不写论文**，所以不做"为了讲清楚贡献"的实验，**只做能改变下一步动作的对照**（PRD §13）。
- **核心判断**：检索不是瓶颈（LongMemEval 召回已 96–99%），**选择和排序才是**。因此 Rerank + Context Packaging 是主线（PRD §4 / §11）。
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
| 4 | **幂等分两层**：内容层"填空 + 追加"，但**位置分配不幂等**——批次级必须查 `applied_batches` 旁表 |
| 5 | **Qdrant RRF 的 `k` 默认是 2**，必须显式设 **61**；**local 模式会静默丢弃 payload 索引** |
| 6 | **不要在 content 里注入绝对时间戳前缀**——裁判 TIME 块有两条独立规则都会因此判负 |

**模型规定**（§2.3）：Embedding 只能用 `text-embedding-v4`，LLM 只能用 `gpt-4o-mini`，**Reranker 不作规定**。
**法律约束**（§12.5）：**数据不得用于训练，「微调模型」不是暂不实现、是不允许。**

---

## 架构

```text
                    User Query
                        │
                        ↓
        ┌──────── Hybrid Retrieval ────────┐
        │                                  │
     BM25                                Dense
        │                                  │
        └──────── Weighted RRF ────────────┘
                        │
                        ↓
                 Initial Candidates
                        │
                        ↓
             Evidence Checker 【v1：空实现】
                        │
                        ↓
                    Rerank（远程 API）
                           ↓
                  Neighbor Expansion（按名次依次扩窗，直到 Top-K 用尽）
                           ↓
                   Context Packaging
                           ↓
                  ≤ Top-K（精确计数）

        ═══ 以下整块为 v2，v1 不实现（见 docs/decisions.md D13）═══
          Evidence Weak → Agentic Search
            - 关键词重写（Agent 每轮自产）
            - Multi-round Search / Temporal Search / Evidence Note
```

> **v1 是直通的**：Evidence Checker 恒返回「证据充足」，因此**没有证据补充路径**——正确的 QA 对不在初始候选里就永久丢了。这把 v1 的全部重量压在**排序 + token 预算分配**上（§4 / §8）。
>
> v1 的 Checker 是**带日志的空实现**：它不做门控，但**必须记录每轮的判定**——不是为了留接口，是为了给 Step 4 攒下反事实分布（`docs/experiments.md` 的 A4）。

> **检索只有混合一种形态**（D15）：参照点由**混合主路径自身**承担（§13）。

**存储分工不可互换**（§6.3）：**SQLite 是真源**（QA 对正文、`status`、`pair_idx`、`event_time`），**Qdrant 是派生索引**（向量 + 过滤键 payload，**不含正文**，坏了可从 SQLite 全文重建）。

**索引单元 = 一个 QA 对**（不是一个 message），一对一个向量。一个 QA 对 = **从一条 user 消息开始，到（不含）下一条 user 消息之前的全部消息**（§6.2）。

---

## 目录导航

> **每个目录下的是 `CLAUDE.md` 而不是 `README.md`**——内容面向执行：该目录要写什么、受哪条约束、对应 PRD 哪一节，以及本层的静默陷阱。根 [`CLAUDE.md`](./CLAUDE.md) 里有"动 X 之前先读 Y"的路由表。

| 路径 | 内容 | PRD |
| --- | --- | --- |
| [`CLAUDE.md`](./CLAUDE.md) | **约束速查 + 文档路由**（写给 Claude Code，也适合人读） | 全部 |
| [`AML Agentic Memory 增强框架 PRD.md`](./AML%20Agentic%20Memory%20增强框架%20PRD.md) | **权威实现规格** | 全部 |
| [`docs/`](./docs/) | 架构与七维映射、契约、配置项、实验登记、悬而未决清单、提交台账、决策日志 | — |
| [`benchmark_data/`](./docs/benchmark-data.md) | AML 官方 pipeline 源码与数据集的**只读归档**（**整目录 gitignored**，说明见链接的文档） | §12 |
| [`configs/`](./configs/) | 运行时配置：三个 profile + 完整开关清单 + 开关依赖图 | §15 |
| [`deploy/`](./deploy/) | Qdrant server（**版本钉死**）、Step 5 重建 runbook | §6.3、§7.3 |
| [`src/tianxi_am/`](./src/tianxi_am/) | 检索服务本体 | §6–§11、§14、§15 |
| [`eval/`](./eval/) | 代理评测：`datasets/` `harness/` `experiments/` `baselines/` `smoke/` `reports/` | §12、§13 |
| [`tests/`](./tests/) | 单元测试（配对 / 续接 / 幂等 / 契约 / 隔离 / 开关纯度） | — |
| [`var/`](./var/) | 运行时产物（**gitignored**，只保留说明） | §6.1 |

---

## 路线图

| 阶段 | 交付 | 状态 |
| ---- | ---- | ---- |
| **Step 0** | 代理评测 harness（LoCoMo-Refined + LongMemEval） | ⬜ |
| Step 1 | 存储层 + Add/Search 服务 + **混合检索**（BM25 + Dense + RRF） | ⬜ |
| Step 2 | Neighbor Expansion + 双预算截断 | ⬜ |
| Step 3 | Rerank + Context Packaging（含 T1 实验） | ⬜ |
| Step 4 | Conditional Agentic Search | ⬜ |
| **Step 5** | **切换到提交模型**，重标定全部阈值，重跑 T2 | ⬜ |
| Step 6 | 对照实验（§13）+ Smoke 验证 + Full 定稿 | ⬜ |

**Step 0 不可跳过**——没有它，后面每一步都是盲调，**而 Full 只有 2 次**。
**Step 5 不可与任何设计改动合并**——否则分数变化无法归因（§12.1 R1）。

完整清单见 [`docs/roadmap.md`](./docs/roadmap.md)。

---

## 评测机会成本

**AML 不提供本地评测**（不给 gold answer、不给评分标准、不提供批量数据下载）。真实信号只有两条路：

| 层 | 用途 | 次数限制 |
| -- | ---- | -------- |
| **代理评测**（本地） | 全部迭代、消融、调参 | 无限 |
| **Smoke** | 验证契约合规、端到端连通 | 每轨道 **≤30 次**，每小时 1 次，不进榜 |
| **Full** | 最终定稿 | 每 Key 每轨道 **2 次**，第二次隔 30 天；**一旦接受即版本冻结** |

> **本项目没有"跑一遍看看"的余地。** 所有迭代必须在自建代理评测上完成，Smoke 用于验证契约合规，Full 只用于最终定稿。

**代理评测的边界**：只覆盖 LoCoMo-Refined + LongMemEval——而这两份恰好是全部六份数据集里**唯一共用同一套契约**的。其余四份的记忆注入字段与裁判规则各不相同（PersonaMem 甚至**根本不读检索字段**），因此**代理分数不能线性外推到全赛道**（§12.4）。

---

## 当前状态

**实现已开工（2026-09-23），不是空脚手架。**

| 状态 | 模块 |
| --- | --- |
| ✅ **已实现** | `store/`（SQLite 真源 + Qdrant + schema）· `pairing/`（配对 / 续接三步 / 计数器）· `embed/`（`Embedder` 协议 + Qwen3-Embedding-8B + 落盘缓存）· `common/render.py`（渲染唯一实现）——另有 **8 个测试文件**在 `tests/` |
| ⬜ **未实现** | `rank/` 的 `reranker.py` / `neighbor.py` · `llm/` · `observability/` · `common/` 的 `tokens.py` / `config.py` · `embed/` 的 `text_embedding_v4.py` · `configs/*.yaml` · `eval/` 的 `datasets/contracts.py` / `smoke/{quota,s1_discriminator,s2_probe,s3_probe}.py` / `reports/ledger.md` / `experiments/` / `baselines/`<br>（**状态表的唯一权威在根 `CLAUDE.md`**，这里只是镜像） |
| ⛔ **v1 不做** | `agent/`（D13） |

已实现的部分对应 `docs/roadmap.md` 的 **Step 1**。**每个目录下的 `CLAUDE.md` 说明了该目录要写什么、受哪条约束、对应哪一节。**

> **⚠ 一处已知漂移**：`docs/` 里仍有若干处状态描述停留在"脚手架阶段"（例如 `docs/roadmap.md` 的勾选状态）。以本节与代码为准。

### Step 0 阻塞项

**运行形态已定**（2026-09-23，见 [`docs/decisions.md`](./docs/decisions.md) D12）：**模型（LLM / embedding / reranker）全部经自建网关远程访问；检索服务与 Qdrant 跑在本机**——本机不需要 GPU，docker 已就位。

| # | 阻塞项 | 状态 |
| --- | --- | --- |
| 1 | **`api_config.py`** | ✅ **不再是环境依赖**（2026-09-24）：**AML 公开仓自己就发布了这个文件**（520 字节、无凭据、只读 `os.environ`），放**仓库内** + 由 harness 在 subprocess 里注入 `PYTHONPATH` 即可——**"clone 下来不能直接跑"不再成立**。实现未写。详见 [`eval/harness/CLAUDE.md`](./eval/harness/CLAUDE.md) |
| 2 | **归档 `benchmark_data/`** | ✅ **已解决**（2026-09-24）：已在本机（635MB→**370MB**，删掉了明令不用的 `lme_test.json` 与失败下载残留）；**且不需要任何共享副本**——每一份都能按 commit / revision 从公开源取回，**出处 + sha256 进版本库**：`make fetch-data` 取回、`make data-check` 校验，清单在 [`tools/fetch_benchmark_data.py`](./tools/fetch_benchmark_data.py)。见 [`docs/decisions.md`](./docs/decisions.md) 待决 6 / 7 的清除记录 |

**reranker 点尚未部署**（部署不在本项目范围内，后续进行）。它是 Step 3 及之后的**基础设施前置项**，不是代码任务——而 v1 不做 agentic 之后，**唯一的新增价值就是 Rerank + Context Packaging**，所以这条前置项直接压在主线上。

本机默认 Python 3.14.4，已按 `>=3.11,<3.14` 保守钉在 3.12（torch / qdrant-client 的 wheel 覆盖通常滞后）。

完整清单见 [`docs/roadmap.md`](./docs/roadmap.md) 的 Step 0。

## 快速开始

```bash
cp .env.example .env      # 填密钥与路径
make sync                 # uv sync --all-extras
make help                 # 看全部目标
```

服务与评测的目标**目前都指向尚未存在的模块，会明确失败**——这是有意的，**避免误以为某一步已经实现**。
