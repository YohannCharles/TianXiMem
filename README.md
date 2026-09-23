# TianXi_AM

**AML（Agent Memory Leaderboard，记忆之巅排行榜）参赛系统。**只提供 `Add` 与 `Search` 两个 HTTP 端点，答案生成、评判、聚合全部由 AML 完成。

> **完整实现规格见 [`AML Agentic Memory 增强框架 PRD.md`](./AML%20Agentic%20Memory%20增强框架%20PRD.md)。**
> 本 README 只做导航与速查；**任何与本 README 冲突的地方，以 PRD 为准**。
> 本仓库当前处于**脚手架阶段**——只有目录与文档，**没有任何实现代码**（见下方「当前状态」）。

---

## 目标

把 AML 榜单分数做到**优于 ReFind 的 44.97**。

- **不写论文**，所以不做"为了讲清楚贡献"的实验，**只做能改变下一步动作的对照**（PRD §13）。
- **核心判断**：检索不是瓶颈（LongMemEval 召回已 96–99%），**选择和排序才是**。因此 Rerank + Context Packaging 是主线（PRD §4 / §11），Hybrid Retrieval 降级为待验证假设。
- **核心 claim**：不让所有 Query 都进昂贵的 Agentic Search，**只有证据不足时才触发多轮搜索**（PRD §3.1 / §9）。

**资源**：3 人 / 4–6 周 / **1×L20（在自建服务器上，不在开发机）**。qwen3.5-9b（128K）+ BGE-M3 + reranker **三者必须共存于同一张卡**（PRD §1 / §11.2）。

---

## ⚠ 硬约束速查

这一节是踩了会丢分、**且不会报错**的那些。写任何模块前先对照本表。

### 契约（PRD §2）

| 约束 | 后果 |
| --- | --- |
| **`top_k` 由 AML 固定为 100** | 返回超过 `top_k` 是**契约错误**，不会被静默截断——必须**精确计数** |
| **答案阶段按返回顺序取 117,760 token 前缀** | 排在后面的证据**整段作废**，白白占掉名额 |
| **`user_id` 是唯一的检索隔离字段** | 跨 user 检索被禁止；`session_id` 只是分组字段，**不是 Search 过滤器** |
| **Add 最多被重试 32 次**（`request_id` 与 payload 不变） | 必须幂等，且**幂等分两层**（见下） |
| **响应前必须持久化完成且立即可搜索** | 不允许异步建索引 |
| **`Search` 不得生成最终答案**，也不得把答案伪装成记忆记录 | reranker 只重排证据、不生成答案 |

### 三个"静默出错"的重灾区

| 陷阱 | 为什么静默 | 出处 |
| --- | --- | --- |
| **幂等分两层**：内容层靠"填空 + 追加、绝不覆盖"，但**位置分配不幂等**——`pair_idx` 是读-改-写，事务提交后崩溃再重试会落到**新的 `pair_idx`**，产生重复记录。**批次级必须查 `applied_batches` 旁表** | 写入是 upsert，重复落库不报错 | §6.5 |
| **Qdrant RRF 的 `k` 默认是 `2`**，不是文献里的 60。**必须显式设 `k=61`**（Qdrant 秩 0-based，`1/(0+61) = 1/(1+60)`） | 不设会得到一个与所有参考实现都不同的融合行为，极难排查 | §7.3 |
| **Qdrant local 模式会静默丢弃 payload 索引**（`create_payload_index` 只打一行警告就返回） | 而 `user_id`/`session_id`/`event_time` 三个筛选全依赖它 | §6.3 |

### 两个 ID / 缓存键，方向正好相反

| 用途 | 键 | 理由 |
| --- | --- | --- |
| 记录 `id` | **位置派生** `hash(user_id, session_id, pair_idx)` | 用内容哈希会在补全时变 `id`，留下**孤儿 point** |
| embedding 缓存键 | **渲染后文本的哈希**，**不能用 `id`** | 补全时内容变了而 `id` 不变，用 `id` 会拿到**陈旧向量** |

缓存**必须落盘**，且要能在 Step 5 切模型时整体失效（§7.2）。

### 时间处理——两条独立机制指向同一个动作

**不要在 content 里注入绝对时间戳前缀。** 裁判 prompt 的 TIME 块有**两条独立规则**都会因此判负：

1. **粒度变细**（DAY → Second）——与 gold 是否相对**无关**
2. **相对 ↔ 绝对互转**（gold 是 "yesterday"，模型算成日期）

而答案 prompt 第 7 条**却要求**转换相对时间——**所以加绝对时间戳可能反而有害**（§11.3）。

推论：`event_time` 列负责筛选；正文**只保留原始时间表述**。但 **`created_at` 必须带，且只给到日粒度**（如 `2026-07-26`）——CL-Bench 那条路径靠它渲染时间，而秒级粒度会诱发模型按秒级回答。

### 模型规定（PRD §2.3）

| 组件 | 规定 |
| --- | --- |
| **Embedding** | **只能用 `text-embedding-v4`** |
| **LLM 相关组件** | **只能用 `gpt-4o-mini`** |
| **Reranker** | **不作规定**——整份规则里唯一不限模型的组件 |

由此推出一条设计约束：**架构必须 embedder-agnostic**。`text-embedding-v4` 不提供 sparse 或 ColBERT 输出，因此**任何依赖 BGE-M3 多向量能力的代码在提交时都是死重**。集合的向量维度必须由接口提供、**不能写死**。

**开发期用自建网关上的 BGE-M3 + qwen3.5-9b 替代**（§12.1 风险 R1，团队已接受）——代价是**所有阈值、权重、排序策略在切换后都不保证成立**，四条对冲见 PRD §12.1。

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
> v1 的 Checker 是**带日志的空实现**：它不做门控，但**必须记录每轮的判定**。这不是留接口，是为了给 Step 4 攒下反事实分布——不记，就永远无法用数据回答"agent 到底值不值"（`docs/experiments.md` 的 A4）。

> 图里的「**关键词重写**」是 **Agent 每轮自己产出的检索关键词**，属于 agent 循环；**不是**被砍掉的 Query Analyzer。**两者仍是两回事**——但注意：随 v1 不做 agentic，它**也一并归 v2**（v1 两条都没有，见 D13）。v1 没有检索之前那一次独立的 query 改写（§5 / §7.2）。

**存储分工不可互换**（§6.3）：

| | SQLite（真源） | Qdrant（派生索引） |
| -- | -- | ---- |
| 存 | QA 对正文、`status`、`pair_idx`、`event_time`、`id` | 向量（`dense` + `bm25`）+ 过滤键 payload |
| 不存 | 向量 | **正文** |
| 能做的事 | 事务、关系查询 | 向量检索、payload 过滤、RRF 融合 |
| 不能做的事 | 向量检索 | 跨 point 事务、关系查询 |

**索引单元 = 一个 QA 对**（不是一个 message），一对一个向量。一个 QA 对 = **从一条 user 消息开始，到（不含）下一条 user 消息之前的全部消息**（§6.2）。

---

## 目录导航

| 路径 | 内容 | PRD |
| --- | --- | --- |
| [`AML Agentic Memory 增强框架 PRD.md`](./AML%20Agentic%20Memory%20增强框架%20PRD.md) | **权威实现规格** | 全部 |
| [`benchmark_data/`](./docs/benchmark-data.md) | AML 官方 pipeline 源码与数据集的**只读归档**（**整目录 gitignored**，说明见链接的文档） | §12 |
| [`docs/`](./docs/) | 架构与七维映射、契约、配置项、实验登记、悬而未决清单、提交台账、归档说明 | — |
| [`configs/`](./configs/) | 运行时配置：三个 profile + 完整开关清单 + 开关依赖图 | §15 |
| [`deploy/`](./deploy/) | Qdrant server（**版本钉死**）、显存预算、Step 5 重建 runbook | §6.3、§7.3、§11.2 |
| [`src/tianxi_am/`](./src/tianxi_am/) | 检索服务本体 | §6–§11、§14、§15 |
| [`eval/`](./eval/) | 代理评测：`datasets/` `harness/` `experiments/` `baselines/` `smoke/` `reports/` | §12、§13 |
| [`tests/`](./tests/) | 单元测试（配对 / 续接 / 幂等 / 契约 / 隔离 / 开关纯度） | — |
| [`var/`](./var/) | 运行时产物（**gitignored**，只保留 README） | §6.1 |

---

## 路线图

| 阶段 | 交付 | 状态 |
| ---- | ---- | ---- |
| **Step 0** | 代理评测 harness（LoCoMo-Refined + LongMemEval）+ **确认显存预算** | ⬜ |
| Step 1 | 存储层 + Add/Search 服务 + **混合检索**（BM25 + Dense + RRF，含 T2 实验） | ⬜ |
| Step 2 | Neighbor Expansion + 双预算截断 | ⬜ |
| Step 3 | Rerank + Context Packaging（含 T1 实验） | ⬜ |
| Step 4 | Conditional Agentic Search | ⬜ |
| **Step 5** | **切换到提交模型**，重标定全部阈值，重跑 T2 与 A1 | ⬜ |
| Step 6 | 对照实验（§13）+ Smoke 验证 + Full 定稿 | ⬜ |

**Step 0 不可跳过**——没有它，后面每一步都是盲调，**而 Full 只有 2 次**。
**Step 5 不可与任何设计改动合并**——否则分数变化无法归因（§12.1 R1）。

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

## 法律约束

**数据不得用于训练。**「微调模型」**不是暂不实现，是不允许**（§12.5）。

> ⚠ **这条尤其针对两个地方**：`gpt-4o-mini` 不能微调，但**本地 qwen3.5-9b 可以**；而 §11.2 的 **reranker 也是自托管开放权重的**——**那两处是"顺手微调一下"最容易发生的地方。不允许。**（§17.3 P2：**无对冲余地**）

| 数据集 | 许可 |
| ---- | ---- |
| LoCoMo-Refined | **CC BY-NC 4.0** |
| ScriptMem | **CC BY-NC 4.0** |
| PersonaMem v2 | CC BY 4.0 |
| LongMemEval | **待确认**（归档内无依据——`lme_readme.md` 无 License 章节） |
| BEAM / CL-Bench | 归档内无许可证文本 |

> **NC（非商业）这一列值得注意**：LoCoMo-Refined 与 ScriptMem 都是 CC BY-NC 4.0。不影响参赛，但**这两份数据不能进任何商业用途的产物**——若后续想把系统或组件开源/商用，它们的评测结果是引用不了的。

---

## 当前状态

**脚手架阶段（2026-09-23）。** 已完成：目录结构、文档、元文件、依赖声明。

**尚未开始**：任何实现。仓内**没有任何 `.py` 文件**——每个目录下的 `README.md` 说明了该目录要写什么、受哪条约束、对应哪一节，实现按 Step 0 起逐个填入。

### Step 0 阻塞项

**运行形态已定**（2026-09-23，见 [`docs/decisions.md`](./docs/decisions.md) D12）：**模型（LLM / embedding / reranker）全部经自建网关远程访问；检索服务与 Qdrant 跑在本机**——本机不需要 GPU，docker 已就位。

| # | 阻塞项 | 状态 |
| --- | --- | --- |
| 1 | **`api_config.py` 不存在** | ⬜ **仍在**，但已降级——它只需是一个读 `.env` 的 **7 行适配器**，且**不含 embedding 配置**（归档 pipeline 不向量化）。详见 [`eval/harness/README.md`](./eval/harness/README.md) |
| 2 | ~~本机无 `docker`~~ | ✅ **已解决**——Docker Desktop 29.8.0（WSL2 后端）已装，`localhost:6333` 直接可用；§6.3 的 Qdrant server 模式可以落地 |
| 3 | ~~本机无 `nvidia-smi`~~ | ✅ **消失**——本机不跑任何模型，三段模型都在自建网关上 |

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
