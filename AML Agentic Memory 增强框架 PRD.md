# TianXi_AM — AML Agentic Memory 系统 PRD

> **文档定位**：**可直接照着写代码的实现规格**。凡是"待验证"的判断都显式标注，不写成已知。
>
> **目标**：把榜单分数做上去。**不写论文**——所以不做"为了讲清楚贡献"的实验，只做"能改变下一步动作"的对照。
>
> **版本**：v1 规格。**Query Analyzer 已砍**——它"要不要进 agent"的那部分职责由 §8 的 Evidence Checker 承担，不另设组件。实体 / 别名归并层属 v2，收在 **附录 A**（待 §13 的 T2 判定是否投入）。

---

## 1. 项目背景

**AML（Agent Memory Leaderboard，记忆之巅排行榜）** 是 2026-07-29 由 20 余所高校与机构联合发起的公开评测。参赛方**只提供 `Add` 与 `Search` 两个 HTTP 端点**，答案生成、评判、聚合全部由 AML 完成。

赛道的开源方法组前三名分数极为接近：

| 系统 | 分数 | 关键做法 |
| ---- | ---- | -------- |
| InvMem | 45.06 | 细粒度切分 + dense/BM25 混合 + 加权 RRF + 同 session 邻域扩展 |
| ReFind | 44.97 | turn 粒度纯 BM25 + ReAct 检索 agent（最多 4 轮）+ 会话级 RRF + 邻域扩展 |
| ActiveMemoryIndex | 44.84 | 原文与原子事实共表存储 + HyDE 式召回改写 + **原文优先排序** |

**两处必须先澄清的事实**（详见附录 B）：

1. **InvMem 不是论文，也没有可 fork 的代码。** 它是榜单上的显示名，无第一方文档；其对应的 repo 全库无 LICENSE，且代码里从不出现 "InvMem" 一词——名称到代码的映射是第三方推断。**不可作为参考实现依赖。**
2. **ReFind 是三者中唯一有论文（[arXiv:2608.12888](https://arxiv.org/abs/2608.12888)）且 MIT 许可的**，本项目的设计参照与外部基线均以它为准。

本项目**从零搭建**，不复用任何既有代码。

**资源与工期（2026-09-22 敲定）**：3 人 / 4–6 周 / **1×L20**。本地模型是 qwen3.5-9b（128K）与 BGE-M3，且 **reranker 也要跑在这同一张卡上**——三者共存是 §11.2 与 §12.1 的共同前提，**显存预算必须在 Step 0 就确认**；若挤不下，R1 的对冲方案需要重写。

---

## 2. AML 参赛契约（硬约束）

这一节的每一条都**不可协商**，实现时必须逐条对照。

### 2.1 接口形状

**`Add`**（POST）
- 请求：`request_id`（唯一，须原样回显）、`messages`（源序，每项含 `role` 与非空 `content`，可选 `timestamp`（Unix 毫秒））、`user_id`、`session_id`
- 响应 200：`success: true` + 原样回显 `request_id` / `user_id` / `session_id`
- **每次 Add 只喂 ≤20 条消息或 2,000 词**，由 AML 确定性切分
- **必须"持久化完成且立即可搜索"后才能响应**——不允许异步建索引

**`Search`**（POST）
- 请求：`query`、`user_id`、`top_k`
- 响应：`data`——**按相关性降序的数组**，每项 `id`（稳定字符串）、`content`、`created_at`（**必须始终带上**，理由见 §11.3）、可选 `score`。空结果返回 `[]`
- **不得生成最终答案，也不得把答案伪装成记忆记录**

### 2.2 硬性规则

| 规则 | 后果 |
| ---- | ---- |
| **`top_k` 由 AML 固定为 100** | 返回超过 `top_k` 是**契约错误**，不会被静默截断——必须精确计数 |
| **`user_id` 是唯一的检索隔离字段** | 跨 user 检索被禁止 |
| `session_id` 只是分组字段 | **不是 Search 的过滤条件** |
| **Add 最多被重试 32 次**，`request_id` 与 payload 不变 | 必须幂等 |
| **答案窗口 128k token**，扣掉输出与安全余量后剩 **117,760 input** | **按你返回的顺序取前缀**——排在后面的证据会被整段截掉 |
| 单请求最长 30 分钟 | 预算充足 |

### 2.3 模型规定

**官方规定（本项目赛道为开源方法组，按此执行）**：

| 组件 | 规定 |
| ---- | ---- |
| **Embedding** | **只能用 `text-embedding-v4`** |
| **LLM 相关组件** | **只能用 `gpt-4o-mini`** |
| **Reranker** | **不作规定**——整份规则里唯一不限模型的组件（§11.1） |

**开发期先用本地模型替代，提交前换回官方指定并重新适配。** 本地 `BGE-M3`（embedding）与 `qwen3.5-9b`（LLM 组件）只用于本地实验与代理评测，**不进提交链路**；这套替代方案及代价记在 §12.1 的 R1。

> 治理文件 `/rules` 另写着"不限定你使用的数据库、索引、向量模型或内部架构"，与上表字面冲突；AML 的 issue #19 触及此事但零回复。**本项目不据此放松**——按上表执行在两种读法下都合规。

由此推出一条设计约束：

- **架构必须 embedder-agnostic。** `text-embedding-v4` 不提供 sparse 或 ColBERT 输出，因此**任何依赖 BGE-M3 多向量能力的代码在提交时都是死重**。embedding 后端做成可替换接口；**集合的向量维度必须由 §7.4 的接口提供、不能写死**——Step 5 换模型时按新维度重建集合（§16）。

### 2.4 评测机会成本——本项目最大的约束

**AML 不提供本地评测**：不给 gold answer、不给评分标准、不提供批量数据下载。真实信号只有两条路：

- **Smoke**：每轨道 ≤30 次，每小时 1 次，跑通 Add→Search→Answer→Eval 但不进榜
- **Full**：每 Key 每轨道**最多 2 次**，第二次需等第一次完成 30 天后；**一旦接受即版本冻结**，不能因成绩不佳撤换

**这意味着：本项目没有"跑一遍看看"的余地。所有迭代必须在自建代理评测上完成，Smoke 用于验证契约合规，Full 只用于最终定稿。**

---

## 3. 项目目标

### 3.1 核心目标

在**不明显增加系统复杂度和推理成本**的前提下，提高 AML Textual 赛道的综合分数。核心原则：

> 不让所有 Query 都进入昂贵的 Agentic Search，优先使用低成本、高稳定性的检索；只有证据不足时才触发 Agent 多轮搜索。

### 3.2 官方评分维度（七个，逐字采用）

1. Explicit fact recall
2. Relational and multi-hop reasoning
3. Temporal and event understanding
4. **Memory governance**
5. Personalization and care
6. Rules and process execution
7. **Epistemic safety and privacy**

> **注意**：早期文档中的"Context Learning"与"Execution"**不是 AML 的维度名**（前者疑为数据集 CLBench 之误）；且**第 4、7 两维此前完全未被覆盖**，需要在设计中显式回应。

### 3.3 成功判据

**只有一条：AML 榜单分数。** 具体目标是优于 ReFind 的 44.97。

> 不写论文，因此不追求"讲清楚每个模块的贡献"。所有实验只服务于一个目的：**决定下一步改什么**。

---

## 4. 关键判断：本项目的赌注押在哪里

**四条独立证据指向同一结论——检索不是瓶颈，选择和排序才是：**

1. **LongMemEval 的检索召回已接近天花板**（系统普遍报 ~96–99% R@5/R@10），但端到端 QA 只有约 60–95%。缺口在"留哪些、按什么顺序留"。
2. **ActiveMemoryIndex 的自测**：排序改动（原文优先 .6333 vs 按相关度 .5887）是其找到的**最大单一杠杆**，大于其余所有检索改动之和。（**限定**：这是它自己 **本地 harness + 本地 judge** 的 LoCoMo 结果，不是平台分数，重跑还有约 0.2pp 漂移。）
3. **ReFind 的消融**：dense 与 hybrid 都打不过纯 BM25（BM25 93.2/89.3 vs dense 91.3/82.2 vs hybrid 91.3/86.7）。**但注意该组是 GPT-5-mini backbone 下的 matched 子集消融**——它证明的是"检索后端之间 BM25 不输"，**不能直接推及 §9 的 `gpt-4o-mini` 场景**。引用时别把两者混在一起。
4. **AML 契约**：答案阶段按返回顺序取 117,760 token 前缀，后续截断。

**因此本方案的重心分布是：**

| 环节 | 地位 |
| ---- | ---- |
| Hybrid Retrieval（§7） | **主体的检索形态**（不使用裸 BM25） |
| Evidence Checker（§8） | 保留 |
| Conditional Agentic Search（§9） | 保留，是项目的核心 claim |
| **Rerank + Context Packaging（§11）** | **升格为主线** |

> **代价**：上面第 3 条证据（ReFind 消融"dense 与 hybrid 都打不过纯 BM25"）**不再被本地检验**——**若分数不及预期，第一个该复检的就是它**。

---

## 5. 整体架构

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
                Evidence Checker
                   /           \
          Evidence Enough     Evidence Weak
                │                  │
                │                  ↓
                │          Agentic Search
                │          - 关键词重写（Agent 每轮自产）
                │          - Multi-round Search
                │          - Temporal Search
                │          - Evidence Note
                │                  │
                └──────────┬───────┘
                           ↓
                    Rerank（本地模型）
                           ↓
                  Neighbor Expansion
                  （按名次依次扩窗，
                    直到 Top-K 用尽）
                           ↓
                   Context Packaging
                           ↓
                  ≤ Top-K（精确计数）
                      Search Result
```

> 图里的"**关键词重写**"是 **Agent 每轮自己产出的检索关键词**（§9 的 `search_chatrecord`）——它属于 agent 循环，**不是**被砍掉的 Query Analyzer。§7.2 说的"查询改写属 v2"指的是**检索之前那一次独立的 query 改写**。两者别混：前者 v1 有，后者 v1 没有。

---

## 6. Memory 设计

### 6.1 真源：SQLite（一张业务表 + 一张幂等旁表）

```sql
qa_pairs(
  id          TEXT PRIMARY KEY,        -- hash(user_id, session_id, pair_idx)，位置派生
  user_id     TEXT NOT NULL,           -- 隔离契约
  session_id  TEXT NOT NULL,
  pair_idx    INTEGER NOT NULL,        -- session 内连续，从 0 起
  question    TEXT,                    -- 可空
  answer      TEXT,                    -- 可空
  status      TEXT NOT NULL,           -- 'complete' | 'pending'
  event_time  INTEGER,                 -- Unix 毫秒，取该对首条消息的 timestamp，可空
  request_id  TEXT NOT NULL            -- 最后触碰该行的 Add；仅用于溯源（幂等不靠它，见下）
)
UNIQUE(user_id, session_id, pair_idx);

-- 唯一的旁表：批次级幂等守卫（§6.5）。每批一行，只增不改
applied_batches(
  request_id  TEXT PRIMARY KEY,        -- AML 的 Add request_id，天然唯一
  user_id     TEXT NOT NULL,
  session_id  TEXT NOT NULL,
  applied_at  INTEGER NOT NULL
)
```

**不预留 v2 字段。** 早期版本留过 `kind` / `parent_id` 给 Fact 抽取用，还有一个无定义键的杂项袋 `meta`——三者现已一并删除，理由相同：按下面第一条原则，预留字段既不入索引也不进 `ORDER BY`，正属于"不该是列"的一类。真要做时再加。

**设计原则：**

> 除正文（`question` / `answer`）外，一个字段要么进索引、要么进 `ORDER BY`、要么有明确的运维用途（幂等守卫 / 溯源），否则它不该是列。

> 能被 `question` / `answer` 重建的东西，一律不进真源。

**为什么业务数据只有一张表**：内容写入的幂等由"只填空、只追加、绝不覆盖"保证（§6.5）。但**批次级幂等必须另记一笔**，那需要一张极小的旁表 `applied_batches`——"只填空不覆盖"免不掉它，理由见 §6.5。旁表每批一行、只增不改，10 万条消息也只约 5,000 行。

> **`qa_pairs.request_id` 不承担幂等职责。** 它会被后一批覆盖（后一批触碰过的每一行都会改写它），因此可能丢掉早先批次的指纹。**幂等一律查 `applied_batches`。**

**为什么必须有 `pair_idx`**：`event_time` 保证不了 session 内顺序——同一秒的多条消息排序未定义，会让 `±1` 邻域扩展产生抖动，进而让消融实验不可复现。`pair_idx` 必须连续，有空洞则邻域静默消失。

**为什么必须有 `user_id`**：AML 的唯一隔离字段。

**时间语义**：`event_time` 列负责筛选；`question` / `answer` **只保留原始时间表述，不加绝对时间前缀**。理由见 §11.3——答案 prompt 与裁判在"相对↔绝对转换"上互相矛盾，加绝对时间戳可能反而有害。

### 6.2 一个 QA 对是什么

> **一个 QA 对 = 从一条 user 消息开始，到（不含）下一条 user 消息之前的全部消息。**

这样定义而不是"严格 user + 紧邻 assistant"，是为了覆盖三种真实情况：连续多条 user 消息、一条 user 后跟多条 assistant 消息（工具调用等）、session 以 assistant 开头（此时 `question` 为空，构成一个无问的对）。

**配对的作用域是整个 session，不是一个 Add 批次。** 一个 QA 对可以跨批次（§6.5 的续接），批次只是 AML 的传输单位。这一点直接决定了 `pair_idx` 必须 session 内连续、新批次要接着数——§6.5 的三步续接全部由此而来。

**配对只做一种判断：这条消息的 `role` 是不是 `user`。** 其他 role（`assistant` / `system` / 工具输出…）一律归入当前对。**实现里不要枚举 role 白名单**——AML 传入的取值域没有文档，白名单会在遇到没见过的 role 时**静默丢消息**。

定义与 ReFind 的 turn 粒度一致——**与外部基线同粒度，消融对比才干净**。

### 6.3 派生索引：Qdrant

**两边的分工（不可互换）**：

| | SQLite（真源） | Qdrant（派生索引） |
| -- | -- | ---- |
| 存 | QA 对正文（`question` / `answer`）、`status`、`pair_idx`、`event_time`、`id` | 向量（`dense` + `bm25`）+ 过滤键 payload |
| 不存 | 向量 | **正文** |
| 能做的事 | 事务、关系查询 | 向量检索、payload 过滤、RRF 融合 |
| 不能做的事 | 向量检索 | 跨 point 事务、关系查询 |

**为什么两边都不能省**：能力不重叠——Qdrant 不做事务和关系查询，SQLite 不做向量检索。补全一个 `pending` 对是"读-改-写"，需要一个事务；而检索需要向量。

**为什么正文不复制进 Qdrant payload**：那样会有两份正文，一旦不一致就无法判断该信哪一份。而按主键批量取正文是微秒级操作，没有性能理由去复制。

**Qdrant 是派生读存储，不是真源**——索引损坏时可从 SQLite 的正文全量重建。Qdrant 官方同步指南的定位也是"真源在关系库，Qdrant 是派生读存储"。

`content` 字段不在 Qdrant 里，`Search` 返回时按检索到的 `id` 批量回 SQLite 取正文。

**必须用 server 模式（Docker）。** local 模式会**静默丢弃 payload 索引**（`create_payload_index` 只打一行警告就返回），且其数据格式与 server 不兼容。而本项目的 `user_id` / `session_id` / 时间筛选全部依赖 payload 索引。

配置要点：

| 项 | 值 | 理由 |
| -- | -- | ---- |
| 集合分片数 | **1** | 根级融合跨分片合并，分片数变化会改变排名——为可复现必须单分片 |
| 命名向量 | `dense` + `bm25` | 稀疏向量的距离固定为 Dot |
| payload 索引 | `user_id`（**keyword** + `is_tenant`）/ `session_id`（keyword）/ `event_time`（integer） | **必须在写入数据前建**，否则 HNSW 需要重建才有过滤感知。**`is_tenant` 只支持 keyword / uuid 两种类型**，别把 `user_id` 建成 integer |
| payload 内容 | `user_id` / `session_id` / `pair_idx` / `event_time`，**不含正文** | 够过滤和溯源即可 |
| 融合 | `prefetch` + `rrf{weights, k}` | 见 §7.3 |
| 写入 | `wait=true` | 契约要求"响应前立即可搜"；默认异步不保证 |

> **为什么不用 Qdrant-only**：那会让向量库成为唯一真源，索引不可重建；且 AML 只给 2 次 Full、版本冻结，把提交押在一次不可逆的存储格式迁移上不值得。SQLite 的成本是一个文件。

### 6.4 检索单元

**索引单元 = 一个 QA 对，一对一个向量。** embedding 的输入是该对渲染后的完整文本（§11.3）。

**代价必须记账**：一个槽位现在装的是一条完整问答，文本量约为单条消息的两倍。Top-100 虽然装得下 100 个对，但**答案窗口只有 117,760 token**，这才是真正的上限。因此返回时**必须同时按槽位数和 token 数双预算截断**，否则排在后面的对会整段作废，白白占掉名额。实际能返回多少对，需要在代理评测上量出来。

**token 怎么数**：用答案模型自己的分词器（提交期即 `gpt-4o-mini` 的 `o200k_base`），**不要用字符数或空格切分近似**——单次近似偏差会在 100 个对上被放大到几千 token。

> **又一处 R1 类风险。** 本地 qwen3.5-9b 的分词器与 `gpt-4o-mini` 不同，**本地量出的"能装多少对"不能直接搬到线上**。§12.1 的四条对冲里没有这一条，应补上：**Step 5 切换后必须重新量一次单请求实际返回的对数**，并把 token 预算做成配置项（与 §15 的"不得硬编码"一致）。

### 6.5 残缺 QA 对与跨批次补全

AML 按 20 条消息**或 2,000 词**切批，切点由它决定。**一个 QA 对可能被切在批次边界上**——例如前一批以一条 user 消息结尾，其 assistant 回复落在下一批。

#### 判定 `pending`

**只要本批命中任一上限（20 条消息 或 2,000 词），本批的最后一对就是 `pending`；两限都未命中，则最后一对是 `complete`**——AML 手上已经没有这个 session 的消息了，边界即 session 末端。

> **本地复现不了词数那一路。** "Adapter 计数的词"官方从未定义（§12.3 第 5 条），本地只能按 20 条复现，因此**本地测出的 `pending` 埋点数与线上必然对不上**，解读那三个计数器时必须记住。

#### 批次续接——三步，顺序不能换

新批次到达时，服务端必须先从库里恢复上下文。**`pair_idx` 是 session 内的全局序号，不能从 0 重开**——`id = hash(user_id, session_id, pair_idx)` 是位置派生的，重开必然与既有行撞车，而写入是 upsert，**会静默覆盖掉上一批的数据**。

```text
1. 幂等守卫（必须最先做，理由见下）
   SELECT 1 FROM applied_batches WHERE request_id = ? LIMIT 1
   命中 → 本批已应用过（重试）→ 直接返回 200，不写任何东西
   ※ 应用成功时，在**同一个事务**里向 applied_batches 插入本批这一行

2. 恢复位置
   next_idx = COALESCE(MAX(pair_idx) + 1, 0)        -- 限于该 (user_id, session_id)
   pending  = 该 session 中 status = 'pending' 的那一对
              （至多一个，且必在末尾：ORDER BY pair_idx DESC LIMIT 1）

3. 挂接本批消息
   a. 本批开头、首个 user 消息之前的消息 → 追加到 pending 的 answer
      若此时不存在 pending → 按 §6.2 处理（批次以 assistant 开头，建一个 question 为空的对）
   b. 本批出现首个 user 消息 → 把前一个 pending 对标 complete
      （它是被两条连续 user 消息关掉的，没有内容可填；漏了这一步它会永久挂在 pending）
   c. 其余消息按 §6.2 配对，pair_idx 从 next_idx 起连续赋值
   d. 收尾：按上面的"判定 pending"给涉及到的最后一对标 status——
      本批两限都未命中 ⇒ session 已结束 ⇒ 标 complete
      ※ **纯接续批（零条 user 消息）也走这一步**：此时被追加的那个 pending 对
        就是最后一带，同样要标 complete，否则它会一直挂到 session 结束
```

> **为什么第 1 步不能省——这是本项目最容易踩的一个陷阱。**
>
> "只填空不覆盖"确实让**内容**写入幂等，但**位置分配不幂等**。若服务在事务提交之后、响应发出之前崩溃（或响应丢失），AML 会重试同一批（`request_id` 与 payload 不变），而此时 `MAX(pair_idx)` 已经前移——重试会把**同一批消息重新分配到新的 `pair_idx` 上**，落成一份重复记录，**且不会报错**。
>
> 所以"只填空不覆盖"**不足以**免掉判重：它管不住**位置**。批次级幂等必须另有一笔记录，且这笔记录**只增不改**——这就是 §6.1 那张唯一的旁表 `applied_batches`。
>
> **为什么不能复用 `qa_pairs.request_id` 做守卫。** 因为那一列会被**后一批覆盖**：若某批唯一触碰过的行随后被下一批改写，这批的指纹就丢了，守卫查不到、重试照旧重复应用。旁表让它永不丢失。
>
> 这条守卫也是"`answer` 允许追加"得以安全的前提（见下）。别把它和内容幂等混为一谈——它们解决的是两个不同的问题。

#### 写入规则——填空 + 追加，绝不覆盖

```text
question      只在原值为 NULL 时写入
answer        只在原值为 NULL 时写入；跨批续接时允许**追加**（append-only）
status        只允许 pending → complete，不允许反向
```

`answer` 必须允许追加，是因为 §6.2 承认"一条 user 后跟多条 assistant 消息（工具调用等）"——这类对若跨批次，续接批次带来的 assistant 消息必须并进去，而不是丢掉。**追加的安全性由上面的批次级守卫保证**（同一批至多被应用一次），不需要额外的判重逻辑。

**补全时必须做的两件事**：

1. **重算该对的 embedding 并 upsert 覆盖原 Qdrant point。** `id` 不变——这是 `id` 必须**位置派生**（`hash(user_id, session_id, pair_idx)`）而不是内容哈希的原因：内容哈希会在补全时变，留下一个孤儿 point。
2. **埋点**：`pending_created` / `pending_completed` / `pending_orphaned`。

`orphaned` = 到 session 结束仍是 `pending` 的对。**这三个计数器是判断 AML 切分是否频繁打断 QA 对的唯一手段**，持续偏高说明配对规则需要调整。

> **但要区分 `pending_orphaned` 的两种来源**：**(i) 真·残缺**——session 就此结束，对里确实少了一半，这是非零的正常来源；**(ii) 误判残留**——本批恰好命中上限、最后一对其实已完整，而下一批以 user 消息开头，按 3b 它本该被关掉。**(ii) 非零就说明续接逻辑漏了 3b，是 bug，不是数据问题。**

---

## 7. 索引与检索（待验证环节）

### 7.1 BM25 一路

使用 **Qdrant 原生 `qdrant/bm25`**（服务端推理，真 BM25，sparse + `modifier: idf`）。

**不使用 BGE-M3 的 sparse 输出**，理由有二：它的 sparse 是学出来的词法权重而非 BM25；且提交时的 `text-embedding-v4` 不提供该能力。

> **"BM25 反超 BGE-M3 sparse"这条必须带前提说。** 在 MLDR 上：**用 Lucene Analyzer 分词的** BM25 得 **64.1**，反超 BGE-M3 sparse 的 **62.2**；但**改用 BGE-M3 自己的 XLM-R 分词器**时，BM25 只有 **53.6**，明显落败。**结论对分词器高度敏感。**
>
> 所以"真 BM25"这个选择成立，但理由不是"BM25 天然更强"，而是"**配正经分词器的 BM25** 更强"。由此推出一条实现要求：**§7.1 选 `qdrant/bm25` 时，它的分词行为要实测一次**（拿一两个多语言长文档的 case 对照），别默认它等价于 Lucene Analyzer。
>
> （数据来源：BGE-M3 论文 Table 11。本环境 `arxiv.org` 被拦截，该数值经三个独立二手来源交叉一致。）

### 7.2 Dense 一路

`text-embedding-v4`，通过可替换的 embedding 后端接口调用。

- **索引侧**：输入是一个 QA 对的**渲染文本**（§11.3），一对一个向量
- **查询侧**：query 原样送进去，**不做任何改写**——v1 没有 Query Analyzer（§5），查询改写属 v2。每查询恰好 1 次调用

**必须缓存向量，缓存键 = 渲染后 QA 文本的哈希。不能用 `id` 当键**——补全时内容变了而 `id` 不变，用 `id` 会拿到陈旧向量。**缓存必须落盘**，否则每次重启重付一遍；同时要能在 Step 5 切模型时整体失效（§12.1 R1 对冲 1）。

v1 的 Add 阶段不调用任何 LLM，embedding 是唯一的 Add 侧成本，且只与内容有关——缓存后即成为一次性成本，与迭代次数无关。

### 7.3 融合：Weighted RRF

```json
{"prefetch": [{"query": <bm25>,  "using": "bm25",  "limit": N},
              {"query": <dense>, "using": "dense", "limit": N}],
 "query": {"rrf": {"weights": [w_bm25, w_dense], "k": 61}},
 "limit": <top_k>}
```

**`k` 必须显式设为 `61`——不是默认值，也不是文献里的 60。** 两个坑叠在一起：

- **Qdrant 的默认值是 `k = 2`**（官方文档逐字："k is a constant (set to 2 by default)"）。不设就会得到一个与所有参考实现都不同的融合行为，且极难排查
- **`k = 61` 才等价于 RRF 文献里的 60**：Qdrant 的秩是 **0-based**（"the top result has r_d = 0"），公式为 `1/(rank + k)`；文献是 1-based 的 `1/(60 + rank)`，首位即 `1/(0+61) = 1/(1+60)`。官方调参文的原话："Qdrant defaults to k=2. The original RRF paper uses 60, **which maps to k=61 in Qdrant's formula**."

**版本门槛**：`k` 参数自 **v1.16.0** 起可用，`weights` 自 **v1.17.0** 起可用，上面的写法要求 **≥ v1.17.0**。**部署时把 Qdrant 版本钉死**——它与 §2.2 的可复现性同源。

**每个 `prefetch` 都要带 `using`。** 命名向量场景下不写 `using`，Qdrant 无法确定用哪一路；根级还要给 `limit`（以及需要时的 `filter`）。

**根级 `limit` 用请求里的 `top_k`，不要写死 100。** AML 说它固定是 100（§2.2），但**契约字段就是 `top_k`**——写死会在它传更小值时变成"返回超限"，那是契约错误而不是截断（§2.2 第一行）。取小值即可，不必精确等于。

- **每路 `prefetch` 的 `limit`（上式里的 `N`）必须显式给定并作为配置项。** 它是"进入 RRF 融合的候选池"大小，与 §10 的种子数（20）、Top-K（100）是**三个不同的量**，不要混用。初值取 `N = 200`（融合池至多 400 条，要给 100 个名额留出邻域折叠与去重的余量），在代理评测上标定
- 权重初值取 `[0.5, 0.5]`；Qdrant 官方明确警告"无评测集时手调权重不太可能稳定优于默认值"，因此**权重的任何调整都必须有 ablation 数据支撑**
- RRF 只用排名不用分数，因此 **BM25 分数与余弦相似度无需归一化对齐**——这绕开了 hybrid 检索最麻烦的一步，但也意味着权重代表的是"名次话语权"而非"分数重要性"

### 7.4 Embedding 后端

```python
class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...
```

v1 提供两个实现：`TextEmbeddingV4`（提交与本地评测用）与 `BGE_M3`（对照实验用）。**代码中不得出现任何依赖具体模型输出结构的逻辑。**

---

## 8. Evidence Checker

判断当前候选是否已足够回答 Query；足够则直接返回，不足才进入 Agentic Search。

**关键实现约束**：**RRF 融合后的分数不是校准量**，Qdrant 官方明确警告不要把单路阈值用到根级 `score_threshold`——"照搬 dense-only 的阈值会静默截断结果"。因此判据**不能用融合分数**。

**改用名次一致性判据**：分别跑一次 `bm25-only` 与一次 `dense-only` 的检索，用两者返回的排名做判断：

```text
足够（Direct Return）：
  两路 top1 相同
  或 两路 top-5 重叠 ≥ 3
  或 bm25 top1 的名次在 dense 结果中位于前 3

否则：
  → Agentic Search
```

**两次分离查询仍要跑**（它们是判据本身），但**不必为实验留档**。

阈值需在代理评测上标定，初值如上。

**Checker 的开关必须是配置项**（§15 的开关清单里原本漏了它）。

---

## 9. Conditional Agentic Search

只在 Evidence Checker 判定不足时启动。**这是本项目的核心 claim，不可砍。**

**参数沿用 ReFind 的形状**（这套参数是在真实评测上跑出来的，改它需要理由）：

```text
最多 4 轮
每轮内部检索 top_k = 5
工具集：
  search_chatrecord(query, date_from?, date_to?)  -- 复用 §7 的 hybrid 检索
  take_note()                                     -- 保存上一轮结果原文，存后即不可再访问
  finish_search()
```

- Agent 只能给出关键词与日期范围，其余控制由服务端自动施加
- 复用 ReFind 的四个 chat-native 控制：会话级 rank fusion、邻域扩展、时间收窄、已见 session 去重
- **`4` 是"检索轮数"的上限，不是"LLM 调用次数"的上限。** ReFind 实测每 query **平均约 5.0 次 LLM 调用**（论文 Table 11：4.99）——一轮里可能既有决策调用也有收尾调用。**§14 要监控的是这个均值，不要拿 4 去卡它。**

> 工具名沿用了 ReFind 论文的 `search_chatrecord`。我们的实现是自有的，改名无妨，**但保持与论文同形有利于对照实验时的归因**。
- **Agent 的产出与初始候选是"合并"关系，不是替换**：两者一起进 §11.2 的 Rerank，按 `id` 去重。理由——初始那一路里往往已经有正确答案，Agent 的价值是补上它找不到的那部分，而不是推翻它。（这也是 §4 "选择与排序是瓶颈"的直接应用：Agent 只负责把候选弄全，排序仍归 Rerank。）

重点处理：Multi-hop、Temporal、Knowledge Update、Ambiguous、信息分散。

---

## 10. Neighbor Expansion

**顺序：本步骤在 Rerank 之后执行**，理由见 §11.2。它是"排序 → 扩窗 → 打包"三连里的第二环，不是"扩窗 → 排序"。

对最终命中的 Candidate 补充同 session 邻近上下文。

```sql
SELECT * FROM qa_pairs
WHERE user_id = ? AND session_id = ? AND pair_idx BETWEEN ? - 1 AND ? + 1
ORDER BY pair_idx;
```

`UNIQUE(user_id, session_id, pair_idx)` 建出的索引**正好就是这个查询的键**。窗口从 ±1 改 ±2 只需改 `BETWEEN` 的界。

**注意粒度**：单位是 QA 对，所以 `±1` 拿回的是前后各一整对（最多 4 条消息），不是前后各一条消息。这条直接改变 §10 的预算——**扩窗比 message 粒度时更贵**。

**必须计入 Top-K 预算**：邻域槽位**占 top_k 名额**（InvMem 与 ActiveMemoryIndex 均如此）。完整流程：

```text
1. 融合后的候选 → Rerank（§11.2）
2. 取名次前 20 作为种子
3. 按名次依次扩窗（±1）
4. 去重
5. 回填，直到槽位用尽
```

**预算怎么算**：20 个种子 × 每颗最多 3 个槽位 = **最坏 60 个槽位**。实际通常远小于此——种子密集时相邻窗口大量重叠，去重会折叠掉大部分；种子分散时才逼近 60。所以：

- 最坏情况给回填留 40 个余量
- **但真正先撞上的限制是 token 而不是槽位**（§6.4）：一对的文本量约为单条消息的两倍，117,760 token 的答案窗口可能比 100 个槽位更早用尽

**种子数、窗口大小、Top-K、token 预算，是同一道题，任何一项调整都要重算其余三项。**

目的：避免检索到正确的局部信息，却丢失事件背景、原因或状态变化。

---

## 11. Rerank 与 Context Packaging（主线）

### 11.1 为什么这里是主线

见 §4 的四条证据。此外有一条规则上的理由：§2.3 把 embedding 与 LLM 两处都钉死了，**只有 Reranker 不作规定**——所以**这是唯一能自己投入算力的环节**，也是 §4 认定"排序是最大杠杆"之后唯一还能加码的地方。

### 11.2 Rerank

自托管开放权重的 cross-encoder reranker，跑在本地 L20 上（本地推理，无 API 成本，规则依据见 §11.1）。

**具体模型未定**（2026-09-23 决定：Step 3 前再选）。选型受两条约束：

- **显存是硬约束**——它要与 qwen3.5-9b（128K 上下文，KV cache 很占）和 BGE-M3 **共存于一张 L20**（48 GB）。**Step 0 就要把这条跑通**，否则 §1 的"显存预算"前提悬空、R1 的对冲方案要重写
- 数据集以英文为主（§12.2），**多语言能力非必需但便宜**。顺带记住：reranker 是整份规则里**唯一不限模型**的组件，选型时不必迁就 `text-embedding-v4`

对融合后的候选重排。**注意 reranker 只重排证据、不生成答案**，不触碰"Search 不得生成最终答案"的红线。

#### 顺序与预算的配合

rerank 给出种子名次后，**按名次依次扩窗，直到 Top-K 槽位用尽**。这样预算不足时被牺牲的是名次最低的种子，而不是随机某一个。

#### 组内顺序

**窗口内部按 `pair_idx` 时间序输出，不把种子提到最前。** 理由：窗口是一段连续对话，按时间序读才成立；把种子抽到最前会把一段话拦腰截断。而答案阶段按前缀截断——**窗口整体连续，意味着截断点落在窗口边界上**，不会切出半个窗口。

**组间顺序 = 种子名次序。** 最终 `data` 数组是"按种子名次依次排列的各窗口"。答案阶段按这个顺序取前缀，所以**名次最高的种子及其邻域必须排在最前**——这正是 §11.2 开头"按名次依次扩窗"的用意：预算不足时被牺牲的是名次最低的种子，而不是随机某一个。重叠窗口在去重后归属**名次更靠前的那个种子**。

> 组内顺序与组间顺序都是可消融项。若实验显示"种子提到最前"更好，只改组内这一处，不牵动其他环节。

### 11.3 Context Packaging

返回的 `content` 字符串会被**原样注入答案 prompt**（AML 侧仅按 `memory_text()` 做字符串拼接），因此**打包只能做在 content 里**，靠不上任何外部结构。

> ### ⚠ 一条必须在 Smoke 阶段验证的契约细节
>
> `memory_text()` 对**字符串**直接返回、对 **dict 会 `json.dumps`**（同文件第 97-104 行）。也就是说：**AML 从我们的返回项里取哪个字段、取出来是字符串还是整个 dict，决定了模型读到的是记忆正文、还是一串原始 JSON。**
>
> 本地归档里只有"给定 memories 之后怎么渲染"这一半逻辑——**检索结果 → memories 字段的映射在 AML 那一侧，归档里看不到**。而且**这些字段在数据集里根本不存在**：`lme_s_cleaned.json` / `lme_test.json` 中 `speaker_1_memories` / `retrieved_context` / `memories` 的命中数都是 0，两份 readme 也从未定义它们——**只能由检索方在运行时注入，但用什么键名注入没有明文**。因此这条**只能靠 Smoke 用真实响应验证**（Smoke 不返回答案文本，只能用"加一条只有它能回答的记忆、看分数是否变化"这类端到端信号判别）。
>
> 两个**不报错**的失败模式要单独警惕：
>
> 1. AML 按 dict 处理我们的返回项 → 模型读到 `{"id": ..., "content": ...}` 形式的原始 JSON，正文被淹在结构里
> 2. CL-Bench 那条路径读的是 `text` 字段，**且缺 `text` 的项会被直接跳过**（`benchmark_data/clb_pipeline.py` 第 109-111 行的 `if not text: continue`）。若 AML 不做 `content → text` 映射，我们的记忆会被**整批丢弃**，退化成字面量 `(no memories)`
>
> 若到 Step 6 仍无法确认，低风险的兜底是**返回项里同时带 `content` 与 `text`**（AML 规范要求 `content`，多余的键通常被忽略）——但这属于契约外的赌注，**要等 Smoke 的实测结果再决定，不要提前上**。

`content` = 该 QA 对的**渲染文本**，且**同一份渲染必须同时用作 embedding 的输入**（§7.2）——两处一旦不一致，"检索命中的是什么"与"模型读到的是什么"就会漂移，而且这种漂移不会报错。渲染规则：

**渲染模板（v1 初值，Step 3 定稿）：**

```text
Q: {question}
A: {answer}
```

- `question` 为空的对（§6.2 的"批次以 assistant 开头"）只输出 `A:` 那一行
- `answer` 暂缺的 `pending` 对只输出 `Q:` 那一行
- **模板本身是"贵"消融项**：改模板就等于改变 embedding 输入，**整个向量索引要重建**（§7.2 的缓存全部失效）。所以要趁早定稿，别拖到 Step 5 之后

**一个对里有多条非 user 消息时，每条带 role 标记**（§6.2 明确承认"一条 user 后跟多条 assistant 消息（工具调用等）"，CL-Bench 的数据里还有 `system`）：

```text
Q: What time does the train leave?

A: [assistant] Let me check the schedule.
[system] {"tool": "schedule.lookup", "result": "09:42"}
[assistant] It leaves at 09:42.
```

- 不加标记的话，这几段在模型眼里连成一片，**分不清哪句是助手说的、哪句是工具输出**
- **标记会一并进 embedding 输入**（§7.2 要求同一份渲染），所以它同时影响检索——这是有意的，不是副作用
- **`role` 的取值域以 AML 实际传入为准，不要硬编码只认 `user` / `assistant`**：配对判据只依赖一个判断——"这条是不是 `user`"（§6.2）

**content 必须自定界。** AML 把多个返回项拼进答案 prompt 时只做 `"\n".join(...)`（见 `benchmark_data/pipeline_locomo-refined.py` 的 `memory_text()`，第 97-104 行）——**不插分隔符、不编号、不重复 `id`**。因此：

- 每项要能独立读懂，**不要把跨项才成立的结构（如"接上条"）放进 content**
- 首尾不要留空白，否则拼接处会粘连
- `Q:` / `A:` 这类行首标记是必要的——没有它们，两个相邻项在模型眼里就是一段连续文本

**打包策略：**

```text
1. 去重
2. 原文优先，不以任何摘要或合成文本替代（v1 不产生任何合成文本，此项自动满足；留待 v2）
3. 同一 session 相邻信息尽量连续
4. 保留时间信息（原始表述）
```

**关于时间——一条必须遵守的约束：**

裁判 prompt 的 TIME 块里有**两条独立规则**。原文回查于 `benchmark_data/pipeline_locomo-refined.py`（2026-09-22 更正：本 PRD 早先把两条规则的例子错配在了一起）：

**规则一（粒度严匹配）——`July 26, 2019` 那个示例属于这一条，与"相对/绝对"无关：**

> Granularity must match exactly: HOUR↔HOUR, DAY↔DAY, MONTH↔MONTH, YEAR↔YEAR. Do not answer a gold at a different time unit — even if the numeric value overlaps. Do not answer a month-level gold with a specific day, nor a year with a specific month/day/hour, etc.
> (e.g., gold = "July 26, 2019" [DAY]; generated = "2019-07-26 08:09:17" [includes Second] → WRONG)

它惩罚的是**粒度变细**（DAY → Second），gold 的粒度就是回答的粒度上限——与 gold 是相对还是绝对无关。

**规则二（禁止相对 ↔ 绝对互转）——原文没有配任何示例，不要自己编：**

> Do NOT convert relative ↔ absolute. If the gold uses a relative time expression, the generated answer must also use a relative form (or a clear paraphrase of that same form), not a computed date/range.

（配套宽容条款：`the/last/previous/just prior` 这类修饰词，在锚点日期与时间单位都相同时视为等价。）

而答案 prompt 第 7 条：

> Convert relative times like "yesterday", "last month", and "last year" into dates, months, or years when the memory timestamp makes it clear. **Keep week-based expressions relative.**

注意第 7 条**自带触发条件**（"when the memory timestamp makes it clear"），且明文要求周级表述保持相对。因此：

> **不要在 content 里注入绝对时间戳前缀。** 把时间做得越清晰，会同时踩中两条规则：
>
> 1. **粒度变细**（规则一）——模型抄走注入的精确时间戳，而 gold 是 DAY 级 → WRONG。**这条与 gold 是否相对无关，也是此前被漏掉的那条。**
> 2. **相对 → 绝对**（规则二）——gold 是 "yesterday" 这类相对表述时，模型算成日期 → WRONG。
>
> 两条独立机制指向同一个动作。推论：§13 的 T1（时间戳前缀 带/不带）若测出差异，归因时不要默认只来自其中一条。

**但这条规则的适用范围必须限定。** 它来自 LoCoMo-Refined / LongMemEval 共用的那一套契约（`pipeline_locomo-refined.py` 头部注释明说二者契约相同——**但本地归档里只有这一份，LongMemEval 侧的对应文件没有归档，故该断言目前是单边来源**）。**它不是全赛道的规则**：

- **BEAM 的裁判正好相反**——它明文把 `$68,000` / `68k` / `sixty-eight thousand dollars` 视为等价，即**允许**等价形式。
- **CL-Bench 由 AML 侧主动注入时间戳**——它读你返回项的 `created_at` 字段，渲染成 `- [timestamp] text` 前缀。

所以：**代理评测（LoCoMo-Refined + LongMemEval）按"不加时间前缀"执行；但不要把这个结论推及全赛道。** 另外这推出一条硬性要求：**返回项的 `created_at` 字段不要省略**，CL-Bench 那条路径靠它渲染时间。

**`created_at` 的格式与粒度**：归档里的渲染代码只做 `str(...).strip()` 后原样插入，不解析（`clb_pipeline.py:108-112`），所以任何人类可读格式都能过——但**粒度会被模型看见**：给秒级时间戳，模型更容易按秒级回答，从而踩中上面那条"粒度变细"规则。因此 **`created_at` 只给到日粒度**（如 `2026-07-26`），**不要用裸 Unix 毫秒**（渲染出来是 `- [1753512557000] ...`，对模型无意义）。

**`event_time` 为 NULL 时（消息未带 `timestamp`，该字段是可选）发空串 `""`**——渲染代码 `str(item.get("created_at") or "")` 之后就是假值，会退化成 `- {text}`，是一个**有定义的降级路径**。**不要拿 Add 的到达时间兜底**：那是"何时写入"而不是"何时发生"，会给模型错误信息。

> 这与"保留原始表述、不加时间前缀"不矛盾：省略的是**我们自己拼进 content 的那段前缀**，而 `created_at` 是契约字段，**用不用它由 AML 决定**——我们只负责提供，且提供得越粗越安全。（粒度这一条属**待验证**，与 T1 同批测。）

**关于 `score`：返回一个随名次单调递减的占位值，不要返回原始 RRF 分数。** 融合分数不是校准量（§8），对我们没有意义；但**万一 AML 按 `score` 重排，省略 `score` 就有风险**。取最安全的一侧：返回 `1/(rank+1)` 这类单调值——无论 AML 是否重排，顺序都不变。

同时裁判/指令又要求"If memories conflict, prefer the most recent supported memory"——冲突消解需要模型看得出哪个更新。所以时间的处理是：

> **筛选走 `event_time` 列；正文保留原始时间表述，不做任何格式改写。**

这一条是**待验证假设**（见 §13 实验 T1），但在验证之前按上述执行——它是安全的那一侧。

---

## 12. 评测方案

### 12.1 三层结构

由于 AML 不提供本地评测（§2.4），评测必须分三层：

| 层 | 用途 | 次数限制 |
| -- | ---- | -------- |
| **代理评测**（本地） | 全部迭代、消融、调参 | 无限 |
| **Smoke** | 验证契约合规、端到端连通 | 每轨道 ≤30 次，每小时 1 次 |
| **Full** | 最终定稿 | 每 Key 每轨道 **2 次**，版本冻结 |

**代理评测的模型原则上应与提交一致**（`text-embedding-v4` + `gpt-4o-mini`），否则本地调出的阈值与权重不迁移。**R1 记录了我们主动接受的一次偏离**——开发期先用本地模型，理由是 API 成本。

> ### 已知风险 R1（团队已接受，2026-09-22）
>
> **v1 开发期使用本地 BGE-M3 + qwen3.5-9b 跑代理评测以控制 API 成本，提交前切换到 `text-embedding-v4` + `gpt-4o-mini`。**
>
> **代价**：embedding 与 agent 循环两处的模型都被替换，因此**本地标定出的所有阈值、权重、排序策略**在切换后都不保证成立。
>
> **必须做的四条对冲：**
>
> 1. **embedding 按内容哈希缓存**（§7.2）——切换时缓存全部失效，但此后不再重复付费
> 2. **切换后立刻重跑 §13 的 T2**—— T2 是**对外部模型依赖最小**的一组（纯 BM25 检索 + 人工判读，不调用任何模型），先用它确认切换没引入系统性偏移，再去信其他实验
> 3. **所有阈值必须是配置项、不得硬编码**（§15），否则切换等于重写代码
> 4. **切换要单独占一个阶段**，不与任何设计改动同时进行——否则分数变化无法归因

### 12.2 代理数据集

**使用 LoCoMo-Refined + LongMemEval。** 二者均在 AML 官方数据集清单内、均可公开下载，且覆盖"饱和"与"未饱和"两端。

- **LongMemEval**：**许可待确认**（早先记为 MIT，但归档的 `lme_readme.md` 没有 License 章节，属单边来源，见 §12.5 的清单），500 题 / 6 类。**64.8% 的问题需要 ≥2 个 session 的证据**（324/500，平均 1.896 个），其中 multi-session 类 133 题 **100%** 跨 session——这是本项目唯一的靶子。注意其检索召回已接近天花板，因此**代理评测的重点指标是端到端，不是 Recall@K**。
  - **用哪个变体**：`lme_s_cleaned.json`。归档另有 `lme_test.json`，与它**同题、同证据，只在 haystack 上不同**——后者多出 1,230 个空 session 与 15 个干扰 session。**空 session 会污染按"20 条消息"切批的埋点逻辑**（§6.5），不要用后者
  - **题量分布**（Step 0 建 harness 用）：single-session-user 70 / single-session-assistant 56 / single-session-preference 30 / temporal-reasoning 133 / knowledge-update 78 / multi-session 133
- **LoCoMo-Refined**：CC BY-NC 4.0，未饱和，1,382 题。**计数类问题集中在多跳类**：21/213 = 9.9%，单跳类 27/802 = 3.4%。
  - **"计数类"的口径必须先在 harness 里固定**：上列数字对应 `how many|how much|how often|number of|count|how long`。只算 `how many` 时单跳类是 0.25%，多跳类 9.39%——**换口径数字就变**，而它正是附录 A"实体层做不做"的依据
  - 真正的"v1 不做实体层"理由是**绝对占比**：正文口径下计数类共 21 题，**只占全数据集（1,382 题）的约 1.5%**
  - **归档里有两个 LoCoMo-Refined 文件，别假设用一个就够**：`questions.jsonl`（官方公开 QA 格式，1,382 题，含 `evidence_messages` 已解析的**证据轮**）与 `locomo_refined.json`（含 `conversation` **全文**）。二者逐题对齐（仅 6 处答案 int/str 差异）。**Step 0 先确认"喂给 Add 的对话全文"从哪个文件取**——`evidence_messages` 只有证据轮，不是整段对话

### 12.3 已知陷阱（必须规避）

1. **LoCoMo 发布版的分类 ID 与论文顺序不一致。** 实测映射为 `1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop, 5=adversarial`。依据不是猜题面，而是 **evidence 跨度**：ID 1 有 95% 跨 ≥2 个 session，ID 4 有 94.5% 只有单条 evidence，ID 5 则是 446/446 全带 `adversarial_answer` 且 `answer=null`。而**论文 §4.1 的顺序是 `1=single-hop, 2=multi-hop, 3=temporal, 4=open-domain, 5=adversarial`**——按论文顺序映射会让 **5 类里的 4 类**被错标（只有 adversarial 恰好对上），**且不会报错**。
2. **ScriptMem 做不了代理评测**——对话原文因版权原因未发布。
3. **不要用 MemoryAgentBench 当代理**——它不在 AML 的数据集清单里，在其上调优未必迁移。
4. **不要相信自报数字。** Mem0 自报 93.4%、第三方复现 29.07% 是常态。**只有自己 harness 里跑出来的数才算数。**
5. **切分边界**：AML 按"20 条消息或 2,000 个 Adapter 计数的词"切分，而 **"Adapter" 官方从未定义**——本地只能按 20 条复现，词数密集的会话会与线上不一致。**这会让本地测出的 `pending` 埋点数（§6.5）与线上对不上**，解读那三个计数器时必须记住这一点。
6. **LongMemEval 的拒答题是横切标记，不是第 7 类。** `question_id` 以 `_abs` 结尾的共 **30 道**（multi-session 12 / single-session-user 6 / temporal-reasoning 6 / knowledge-update 6）。**§13 的 T2 要人工给 133 道 multi-session 题分三类，其中 12 道是拒答题，行为与其他题不同，必须单独拎出来。**
7. **PersonaMem 三个 split 的 schema 不统一。** `question_type` 的词表在 32k / 128k / 1M 之间互不相同；`correct_answer` 在 32k 里是 `(c)` 这类选项字母、在 128k/1M 里是整段选项文本。**§12.4 说的"MCQ 精确文本匹配"必须先做归一化**，harness 不要硬编码单一词表或单一答案格式。
8. **BEAM 的数据实际上不在归档里。** `beam.json` / `beam_rows.json` 是失败下载的残留（`Entry not found` / `{"error":"Unexpected error."}`）；`beam_100k.json` 是 HuggingFace datasets-server 的**分页响应**（顶层键为 `features`/`rows`/`num_rows_total`，且 `num_rows_total=20`、实际只取到 1 行），**不是数据集**。所以 §12.4 的 BEAM 一行**只有 `pipeline_beam.py` 单方依据，没有数据可交叉核对**——真要覆盖 BEAM，须先把数据取回来。
9. **归档里的数据集与 pipeline 之间存在 schema 落差，Step 0 必须写一层预处理。** 已核实两处：**(i)** PersonaMem 的 CSV **没有 `chat_history` 列、也没有 `incorrect_answers` 列**，而 `pipeline_v2_personamem.py:147-148` 缺 `incorrect_answers` 直接 `raise TypeError`，且 `:158` 用整段选项文本匹配——**该 pipeline 不能直接吃归档 CSV**。**(ii)** 两套契约的答案字段名不同：readme 写 `predicted_answer`（LoCoMo）/ `hypothesis`（LME），而归档 pipeline 实际读写的是 `generated_answer`（`pipeline_locomo-refined.py:178`）——**harness 的 I/O 以 pipeline 代码为准，不要照 readme**。

### 12.4 各数据集的契约并不统一（代理评测的外推边界）

**代理评测只覆盖 LoCoMo-Refined + LongMemEval，而这两份恰好是全部六份里唯一共用同一套契约的。** 其余四份各不相同：

| 数据集 | 记忆注入字段 | 裁判 |
| ---- | ---- | ---- |
| LoCoMo-Refined / LongMemEval | 主字段 `speaker_1_memories` / `speaker_2_memories`。**退化不对称**：`speaker_1_memories` → `retrieved_context` → `memories`，而 `speaker_2_memories` **无退化，缺即空串**（`pipeline_locomo-refined.py:108-113`） | 二元 CORRECT/WRONG，含严格时间粒度规则（完整三条见 §11.3） |
| BEAM | `context` / `retrieved_context` / `memories` **优先**，speaker 块只是兜底（优先级与上面相反） | 逐条 rubric 三点制（0 / 0.5 / 1.0）取均值；**时间规则与上面相反** |
| ScriptMem | **只用** `speaker_*_memories`，缺失时静默渲染成空串，不报错 | **无 LLM 裁判**。选项精确匹配：multi_select 要求集合完全相等，ordering 要求序列完全相等 |
| CL-Bench | 嵌套的 `retrieval.selected` / `msp_retrieval.selected`，每项含 `created_at` + `text`，AML 渲染成 `- [timestamp] text`；缺失退化成字面量 `(no memories)` | 严格全有全无 rubric（任一条不满足即 0），**且 API/JSON 失败一律记 0** |
| PersonaMem v2 | **没有记忆注入**——消费 `chat_history` / `messages`，忽略全部检索字段 | MCQ 精确文本匹配；narrow 走 0.0–1.0 连续分 |

**三条必须记住的推论：**

1. **代理评测的分数不能线性外推到全赛道。** 你在 LoCoMo-Refined 上调出的注入格式、时间处理、排序偏好，到了 PersonaMem 可能完全不生效——那份 pipeline 根本不读检索字段。
2. **`created_at` 不要省**（见 §11.3）。CL-Bench 靠它渲染时间戳，不返回就等于放弃该数据集的时间信息。
3. **"给更多上下文"在不同数据集上风险方向相反。** ScriptMem 与 CL-Bench 是精确匹配 / 全有全无，多给的直接判 0；BEAM 与 PersonaMem 的裁判则宽容。**没有一个"更丰富总是更好"的统一策略。**

> **另有一条 AML 自身的可复现性隐患**：PersonaMem 用 `random.shuffle` 决定选项顺序，种子取自 Python 内置 `hash()`——未固定 `PYTHONHASHSEED` 时逐进程加盐，**同一份输入在不同进程里选项顺序会变**。这不在我们控制范围内，但意味着该数据集的 MCQ 分数自带 run-to-run 噪声，解读时必须知道。
>
> （**证据范围**：这条在归档里只核到**复现代码**如此（`pipeline_v2_personamem.py:151-154`），上游实现不在归档里，无法核实。属单边来源。）

### 12.5 法律约束

**数据不得用于训练。** PRD 早先记录：AML 数据条款明文"Use this data only for the evaluation. Do not train on it"，CLBench 许可证同样禁止训练/微调/蒸馏。**这是双重封死，"微调模型"不是暂不实现，是不允许。**

> **但这两条在 `benchmark_data/` 里都没有出处。** 全库检索 `only for the evaluation` 等措辞零命中；`aml_readme.md` 只说到各数据集"remain subject to their respective upstream licenses and usage terms"，**没有任何"不得训练"的措辞**；归档里也没有 CLBench 的许可证文本。**它们来自归档之外的来源，属单边来源，引用前须回原始页面/许可证文件复核。**
>
> **结论不变**——即便只按 AML 的通用条款，也不该拿这些数据训练——但**不要把它当成已归档的实证**。

**许可证清单**（来自归档，2026-09-22 核对）：

| 数据集 | 许可 | 出处 |
| ---- | ---- | ---- |
| LoCoMo-Refined | **CC BY-NC 4.0** | `locomo_refined_readme.md:9`、`:296` |
| ScriptMem | **CC BY-NC 4.0** | `scriptmem_readme.md:9`、`:183` |
| PersonaMem v2 | **CC BY 4.0** | `pmv2.md:2` |
| LongMemEval | **待确认**——PRD 早先记为 MIT，但 `lme_readme.md` **没有 License 章节**，归档内无依据 | —— |
| BEAM / CL-Bench | 归档内无许可证文本 | —— |

> **NC（非商业）这一列值得注意**：LoCoMo-Refined 与 ScriptMem 都是 CC BY-NC 4.0。这不影响参赛，但**它意味着这两份数据不能进任何商业用途的产物**——如果后续想把系统或其中组件开源/商用，这两份数据的评测结果是引用不了的。

---

## 13. 对照实验

**只做能改变下一步动作的对照。** 不做"为了讲清模块贡献"的完整消融表——不写论文，那张表没有用。

| 对照 | 回答的问题 | 决定什么 |
| ---- | ---------- | -------- |
| **B1. ReFind 原版**（MIT，不修改代码） | 我们赢了吗 | 是否需要继续投入 |
| **A3. Rerank 开 / 关** | 排序值不值（§11 主线的验证） | 若没用，把 L20 的算力挪去别处 |
| **A4. Agent 开 / 关** | Agentic Search 值不值 | 若没用，砍掉核心 claim 之一 |
| **T1. 时间戳前缀 带 / 不带** | §11.3 那条约束对不对 | content 的渲染方式 |
| **T2. 跨 session 失败归因** | 失败是"找不到"还是"留不下" | v2 实体层做不做（附录 A） |

**B1 必须在我们自己的 harness 里重跑**——ReFind 公开的 58.2 / 93.2 是它自己的 harness、可能不同的问题子集和 prompt 下得到的，直接抄来当基线等于拿它的 harness 和我们的比。

> **另外记一笔工作量**：B1 不是"克隆下来跑一下"。ReFind 是一个方法实现，要让它进我们的 harness，得**给它包一层 Add/Search 服务**（或把它的检索器接到我们的 harness 接口上）。**这部分工作量目前没计入任何 Step**——真要跑 B1 之前先估一下，别把它当零成本。

**主路径（混合检索）必须能通过 Smoke 契约校验**（200 响应、`data` 数组、不超 `top_k`）。**它是所有对照的参照点。**

**T2 是半天工作量的前置实验**，做法：`lme_s_cleaned.json` 的 133 道 multi-session 题（**其中 12 道是拒答题，单列一类**，见 §12.3 第 6 条），纯 BM25 检索后人工把失败样本分三类——(i) 没召回 (ii) 召回了但被 top-100 截断 (iii) 在里面但排序靠后。

**从零搭建特有的陷阱**：开关必须只影响它命名的那一件事。如果关掉 rerank 顺带改变了候选数量、或关掉 agent 顺带改变了打包顺序，这个对照就不成立——**而结果看起来完全正常，只是结论错了**。（这也是 §15 要求所有消融项可配的原因。）

> **可选的一次性 sanity check**：把主路径换成"不检索，直接按时间倒序返回最近 N 对"。如果它逼近全系统，说明 §4 那条核心判断要重写。成本约等于零，做了不亏。

**记录**：每个对照记端到端总分 + 各维度子分。latency / 成本只在明显变差时才追。

---

## 14. 监控指标

**只留会触发动作的指标**，其余不看。

| 指标 | 为什么看它 | 异常时触发什么 |
| ---- | ---------- | -------------- |
| 端到端总分 + 各维度子分 | 唯一的目标 | 决定下一步改哪 |
| **Agent Trigger Rate** | 核心 claim 的开关频率 | 太高 → Checker 太保守；接近 0 → agent 没起作用 |
| **latency/query** | 契约允许 30 分钟，但 Full run 要连续跑 0.5–2 天 | 接近上限就削减 agent 轮数 |
| Agent 平均轮数 / Rewrite 次数 | prompt 健康的探针 | 异常升高说明 prompt 崩了 |
| embedding API 调用数 | 缓存命中率 | 不下降说明缓存键写错了 |

**不看 Recall@K**——本任务上它已接近天花板，涨跌都不代表什么。也不做按问题类型的收益归因，那是论文的事。

---

## 15. 工程与部署

- **服务**：单进程 HTTP 服务，暴露 `Add` / `Search`。必须能 24×7 连续运行（Full run 持续 0.5–2 天）
- **存储**：SQLite（真源，文件随 run 归档）+ Qdrant server（派生索引）
- **写入路径**：

```text
1. 幂等守卫：查 applied_batches，本 request_id 已应用过 → 直接返回 200，不写任何东西（§6.5）
2. 恢复 session 上下文：next_idx = MAX(pair_idx) + 1；定位 pending 对（§6.5）
3. 挂接本批消息：前导非 user 消息追加到 pending 对；首个 user 消息关闭 pending 对；
   其余按 §6.2 切成 QA 对，pair_idx 从 next_idx 起连续赋值
4. **同一事务内**：upsert 到 qa_pairs（填空 + 追加，绝不覆盖）+ 向 applied_batches 插入本批记录
5. 对新增或内容变更的对，同步 upsert Qdrant（wait=true）；补全时必须重算 embedding
6. 返回 200
```

- **重跑安全性**：由第 1 步的**批次级守卫**保证——同一 `request_id` 至多被应用一次，因此第 4 步的"追加"也安全。第 5 步的 upsert 同 `id` 覆盖，同样幂等

- **错误响应**：契约只规定了 200 的成功形状（§2.1），**非 200 的行为未定义，因此必须假设 AML 会重试**——这正是第 1 步幂等守卫的动机。**内部异常应让本批保持"可重试"**（事务未提交），而不是返回一个"部分成功"

- **并发**：Add 必须按 `(user_id, session_id)` 串行化——第 2 步读位置、第 4 步写位置是"读-改-写"，两个并发批次会拿到同一个 `next_idx`。单进程下用一把按 session 的锁即可；SQLite 的写事务不足以单独解决它（两次事务读到的 `MAX(pair_idx)` 会相同）

- **运行时开关**：所有消融项（dense / **checker** / rrf / neighbor / rerank / packaging / agent）必须是配置项，否则 §13 无法执行。**注意 `checker` 是 `dense` 的下游**——两者不是独立开关，退化路径见 §8

---

## 16. 路线图

| 阶段 | 交付 |
| ---- | ---- |
| **Step 0** | 代理评测 harness（LoCoMo-Refined + LongMemEval）。开发期用本地 BGE-M3 + qwen3.5-9b |
| Step 1 | 存储层 + Add/Search 服务 + **混合检索**（BM25 + Dense + RRF，含 T2 实验） |
| Step 2 | **Neighbor Expansion + 双预算截断** |
| Step 3 | Rerank + Context Packaging（含 T1 实验） |
| Step 4 | Conditional Agentic Search |
| **Step 5** | **切换到提交模型**（`text-embedding-v4` + `gpt-4o-mini`），重标定全部阈值，重跑 T2 确认无系统性偏移 |
| Step 6 | 对照实验（§13）+ Smoke 验证 + Full 定稿 |

**Step 0 不可跳过**——没有它，后面每一步都是盲调，而 Full 只有 2 次。**Step 5 也不可与任何设计改动合并**（见 §12.1 风险 R1）。

---

## 17. 悬而未决清单（实现期必须盯着）

前面各节的"待验证"散落在各处，这里集中成一张表。**每一类都标了它由谁消除、以及不消除会在哪一步变成阻塞**——最贵的错误是拖到 Step 5 才发现某个未知没清。

### 17.1 只能靠 Smoke 消除（本地无从验证）

| # | 未知 | 不消除的后果 | 何时必须清掉 |
| -- | ---- | ------------ | ------------ |
| **S1** | AML 从返回项里取哪个字段（`content`？`text`？），取出的是字符串还是整个 dict | 记忆可能被**整批丢弃**或退化成原始 JSON，**两种都不报错** | Step 6 之前，越早越好（§11.3） |
| **S2** | 线上是否真按 20 条 / 2,000 词切批，"Adapter 计的词"怎么算 | `pending` 三个埋点与线上对不上，误判配对质量（§6.5） | Step 6 |
| **S3** | `created_at` 是否被 CL-Bench 那条路径消费 | 白白放弃该数据集的时间信息（§11.3） | Step 6 |

### 17.2 靠代理评测消除（§13 的实验回答）

| # | 未知 | 由哪个对照回答 | 牵连什么 |
| -- | ---- | -------------- | -------- |
| E1 | content 里加不加时间戳前缀 | **T1** | content 渲染方式；改了要重建索引 |
| E3 | rerank 值不值 | **A3** | 不值的则 L20 算力改投他处 |
| E4 | agent 值不值 | **A4** | 不值的则砍掉核心 claim 之一 |
| E5 | 跨 session 失败是"找不到"还是"留不下" | **T2** | 决定附录 A 的实体层做不做 |
| E6 | 渲染模板与组内顺序 | Step 3 定稿 | 改模板 = 重建索引（§11.3） |

### 17.3 规则解释风险（不可控，只能对冲）

| # | 风险 | 对冲 |
| -- | ---- | ---- |
| **P1** | `/rules` 有"不限定模型"的字面表述，与 §2.3 的模型规定冲突；issue #19 零回复 | 按模型规定执行——两种读法下都合规（§2.3） |
| **P2** | 数据禁止训练（AML 条款 + CLBench 许可证双重封死） | 不微调，**无对冲余地**（§12.5） |
| **P3** | 代理评测 → 全赛道不可线性外推 | 只在代理上做**相对**比较，不做绝对外推（§12.4） |

> 这里用 `P` 编号，是为了不与 §12.1 的"**风险 R1**"撞名——那个 R1 说的是模型替换，两回事。

> **S1 是这张表里唯一"信息不足且后果最重"的一条**——其余各项要么能用实验回答，要么已经对冲。所以 Smoke 第一次跑通后，第一件事就是设计一个能判别 S1 的最小实验。

---

# 附录 A：实体 / 别名归并层（待判定）

**粒度**：实体与提及挂在 **QA 对**上，与 §6.1 的存储单元一致，不是单条消息。**补全一对时必须重跑该对的实体抽取**——否则实体会挂在一段残缺文本上。这一步与 embedding 的重算同步发生（§6.5）。

**先做 §13 的 T2 实验再决定是否投入。** 若 T2 显示失败主因是"措辞不同导致漏召"，则别名归并值得做；若主因是"召回但被截断/排序靠后"，则实体层解决的不是本项目的瓶颈。

**若要引入打分公式（例如"实体重叠度"加成），先确认它作用在哪一层**——参与 RRF 融合会破坏"只用排名不用分数"的前提（§7.3）；只在融合后重排则安全，但必须证明它优于单纯按 RRF 名次排序。而 Qdrant 层面**"主查询不能同时是 fusion 和 formula"**，故这类公式只能拆成两次查询、或放在客户端做后处理。

---

# 附录 B：事实核查记录

本 PRD 中所有关于 AML 与三个参考系统的陈述，核实于 2026-09-22，来源如下：

| 陈述 | 来源 |
| ---- | ---- |
| AML 存在、主办方、赛道结构、Add/Search 契约、Top-K=100、重试与窗口规则 | agentmemoryleaderboard.ai 的 api-guide / rules（同页） |
| 七个评分维度 | AML README |
| 模型规定（text-embedding-v4 / gpt-4o-mini / Reranker 不限） | agentmemoryleaderboard.ai/competition/ FAQ 05 |
| 与 `/rules` 的冲突未裁决 | GitHub issue #19，开、零回复。**其主体是超时/重试策略，模型规定只是第三问**（§2.3） |
| 三系统分数与做法 | AML 榜单与官方技术解读（dev.to/aml- 的 Deep Dive #1、ActiveMemoryIndex 篇、Beyond Retrieval 篇）。**注意 AML 自己的两篇末位差 0.04**（45.06/44.97/44.84 vs 45.10/45.00/44.80），本 PRD 取前者 |
| ReFind 论文、MIT 许可、消融数据 | arXiv:2608.12888；github.com/imlrz/ReFind。**消融数字出自 GPT-5-mini backbone 的 matched 子集**，勿与 §9 的 gpt-4o-mini 混引 |
| ActiveMemoryIndex 的排序杠杆（.6333 / .5887） | github.com/linxuhao/ActiveMemoryIndex 的 README——**自述为本地 harness + 本地 judge 的 LoCoMo 结果，非平台分**。该仓库 **MIT 许可，且 README 有完整披露章节（含 Zenodo DOI）**，不是"无溯源指针" |
| InvMem 无论文、无第一方文档 | AML 官方 Deep Dive #1 未给出其仓库或论文链接。**"对应 repo 无 LICENSE、代码不出现 InvMem"是以第三方策展的映射为前提**——映射指向 github.com/wenxiaof345-ctrl/vanilla-rag-memory（`/license` 端点 404），但**该映射不是官方指认** |
| Qdrant 的全部行为断言 | qdrant.tech：concepts/hybrid-queries、concepts/indexing、inference/inference-bm25、guides/multiple-partitions、articles/how-to-tune-hybrid-search，以及 **articles/before-tuning-a-qdrant-collection**——§8 的 `score_threshold` 警告与"分片数改变排名且无报错"，**出处是后者，不在 hybrid-queries 页** |
| BGE-M3 sparse vs BM25 on MLDR | BGE-M3 论文 Table 11。**结论对分词器高度敏感**（Analyzer 64.1 vs XLM-R 53.6）；本环境 arxiv.org 被拦截，数值经三个独立二手来源交叉一致（§7.1） |
| 各数据集问题类型分布、跨 session 比例、许可证 | 见 `benchmark_data/` 下的原始数据与 AML pipeline 源码 |

**原始数据与 AML pipeline 源码已归档在 `benchmark_data/`。**

**第二轮核对记录（2026-09-22 同日）**：上表每一行的"来源"都回原文复核过一遍，并对文档正文做了一次全文体检。更正的要点：

- 裁判 TIME 块实为**三条独立规则**，"粒度严匹配"的示例此前错挂在"禁止相对↔绝对"上（§11.3）
- **§6.5 的幂等论证被推翻**：位置分配不幂等，必须新增 `applied_batches` 旁表做批次级守卫（§6.1 / §6.5 / §15）
- §7.3 的 `k` 由 60 更正为 **61**（Qdrant 秩 0-based），并补 `using` / 根级 `limit` / 版本门槛
- LoCoMo/LME 的记忆字段**退化不对称**，`speaker_2_memories` 无退化（§12.4）
- LongMemEval 的 MIT 许可、AML 数据条款原文、CLBench 许可证**在归档内均无出处**，已标为单边来源（§12.5）
- LongMemEval 统计复算通过（64.8% / 1.896 / 133 / 100%）；LoCoMo 分类 ID 映射复算通过；**"单跳类计数问题占 3.1%"复算不通过**，已改为可复现口径（§12.2）
- BEAM 数据不在归档内，§12.4 的 BEAM 一行只有代码单方依据（§12.3 第 8 条）
- 新增 **§17 悬而未决清单**，把散落的未知按"谁消除、何时阻塞"集中列出
