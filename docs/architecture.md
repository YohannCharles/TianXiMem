# 架构：数据流与模块边界

> 最后核对：2026-09-25，对应 PRD v1 规格。**冲突以 PRD 为准。**

---

## 1. 两条路径

系统的全部行为都可以拆成两条路径。**Add 路径不调用任何 LLM**（§7.2）——这是 v1 的一个刻意属性：Add 侧的模型成本只有 embedding，且只与内容有关，缓存后即成为一次性成本，与迭代次数无关。

### Add 路径（§15 的六步，顺序不可换）

```text
1. payload 指纹      canonical(user_id, session_id, messages) → sha256（**D28**）
2. 幂等守卫         查 applied_batches；命中且指纹相同 → 直接 200，不写任何东西
                                 命中但指纹不同 → **409**（同 id 不同 payload，不是重放）
3. 组合 + 连链       compose_memory_blocks(messages)   ← 只看本批，无跨 Add 状态（**D24**）
                    link_blocks(blocks)               ← Add 内邻接，只连完整 QA（**D28**）
4. 同一事务         INSERT qa_pairs（每块一行、`(request_id, local_index=i)`、`prev` / `next`、status 恒为 complete）
                    + 向 applied_batches 插入本批记录（含指纹）
5. 同步 upsert      Qdrant（wait=true）
6. 返回 200
```

**第 3 步没有跨批分支**——组合与邻接的边界都是**一次 Add**（D24 / D28）。
**第 2 步不读库**——位置是**请求的纯函数**（D28）：`request_id` 原样（**不解析**）、
`local_index` 是本批块序号，所以既没有读-改-写，也没有业务锁；同 session 的 Add 可以并发，
而"到达顺序"**根本没有被表达过**（跨 Add 不存在顺序，也不建立邻接）。
**第 1 步不能省**：位置虽然幂等了，但 AML 的重试是**正常行为**，不能每次都靠撞 `UNIQUE` 兜
（那会把正常重试变成 500）；而"同 id 不同 payload"**不是重试**，它必须响亮地冲突
（静默挑一份落库 = 另一份记忆凭空消失）。

**第 4 步与第 5 步的边界**：SQLite 是事务的，Qdrant 不是。因此第 4 步先落真源、第 5 步再同步派生索引；若第 5 步失败，Qdrant 可以从 SQLite 重建（它是派生读存储），而反过来不行。

### Search 路径（§5 流程图）

```text
query → BM25 ┐
             ├→ Weighted RRF → memory_id 稳定去重 → Evidence Checker
     dense  ─┘                              ├ 证据足够 → 直接返回
                                            └ 证据不足 → Agentic Search
                                                          ↓
                                          （候选**合并**，不是替换）
                                                          ↓
                                              Rerank（远端 cross-encoder）
                                                          ↓
                                              Neighbor Expansion（按名次扩窗）
                                                          ↓
                                              Context Segment Merge
                                                          ↓
                                              Token Budget + Context Packaging
                                                          ↓
                                              ≤ top_k（精确计数，**数的是段**）
```

> **`top_k` 数的是段、不是 raw memory**：扩窗会把 raw 条数抬到 `top_k` 之上，
> 而相邻块的合并又把它降回来 ⇒ **截断只能在合并之后做**。

> ⚠ **"相邻"是存储里的显式指针 `prev_memory_id` / `next_memory_id`**（D28），
> 不是算出来的下标：扩窗**沿链跳**，合并**看指针相接**。
> ⇒ 跨 Add **结构上不可能**相邻（指针只在一次 Add 内连过），
> 而链上缺一块也**不会**被"下一块"顶上来。

**"合并而非替换"**（§9）：Agent 的产出与初始候选按 `id` 去重后一起进 Rerank。理由——初始那一路往往已经有正确答案，Agent 的价值是**补上它找不到的那部分**，而不是推翻它。

---

## 2. 模块 → PRD 映射

| 目录 | 负责什么 | PRD |
| --- | --- | --- |
| `service/` | HTTP 层：`POST /add`、`POST /search`（+ `GET /health` 探活）；请求/响应模型 | §2.1、§15 → **D25** |
| `pairing/` | **记忆块组合（一次 Add = 唯一边界，D24）**；幂等守卫 + 位置分配 + 落库；**§15 写入路径的唯一所有者** | §6.2、§6.5（D24 后的口径）、§15 |
| `store/` | SQLite 真源（`qa_pairs` + `applied_batches`）；Qdrant 派生索引；邻域查询的 SQL | §6.1、§6.3 |
| `embed/` | `Embedder` 协议 + 两个实现；**落盘的向量缓存** | §7.4、§7.2 |
| `retrieve/` | BM25；dense；**混合检索的策略与参数所有权**；Evidence Checker（**到 rerank 为止**） | §7.1–§7.3、§8 |
| `rank/` | rerank → **Neighbor Expansion（§10）** → Context Packaging —— "排序 → 扩窗 → 打包"三连环 | §10、§11 |
| `agent/` | Conditional Agentic Search 循环与三个工具 | §9 |
| `llm/` | LLM 后端抽象（`gpt-4o-mini` / qwen3.5-9b） | §2.3、§12.1 |
| `common/` | **渲染模板的唯一实现**；token 计数；配置加载 + **开关校验** | §11.3、§6.4、§15 |
| `observability/` | §14 指标（**各层发射、本层聚合**） | §14 |

### 四处容易摆错的位置

| 东西 | 该在哪 | 为什么不在别处 |
| --- | --- | --- |
| **Neighbor Expansion** | **`rank/`** | §10 明确它跑在 rerank **之后**，属"排序 → 扩窗 → 打包"；且它是 SQLite 读，不是 Qdrant 操作。**在 §5 的流程图里它位置很靠上，极易被误读成检索的一环**（§10 开头的顺序说明是权威的） |
| **向量缓存** | **`embed/`** | 缓存键是"渲染后文本的哈希"（§7.2），属渲染 + embedding 的关注点。且 §6.3 把这层定义为**恰好两样东西**（SQLite 真源 + Qdrant 派生索引，**明确"不可互换"**）——塞进第三个存储会削弱那条规则 |
| **§15 的六步写入路径** | **`pairing/`** | 它是一个事务，跨"幂等守卫 → 取位置 → 组合 → 写正文 → 记旁表"。若 `service/` 也碰它，**两边会长出半个事务**。`service/` 只负责 HTTP 形状与校验，**不碰 Qdrant** |
| **检索的"参数"与"执行"** | **参数在 `retrieve/`，执行在 `store/qdrant_store.py`** | `prefetch_limit` / `weights` / `k` / `top_k` 的**取值、校验、标定**是检索策略（要能被配置驱动、能被 ablation 检验）；而把它们翻成 `prefetch` + `rrf` 的**Qdrant 语法**是 `store/` 的事。**分界线**：`retrieve/` 说"用什么参数"，`store/` 说"怎么发给 Qdrant"。⚠ 一份实现**不许两边都写**——`k=61` 的由来见 D5，校验只在 `retrieve/` |

---

## 3. 依赖方向

```text
        service/                     ← 只依赖 pipeline，不直接碰 store
            │
            ↓
        pipeline（编排）              ← §5 流程图的字面实现
            │
   ┌────────┼────────┬──────────┐
   ↓        ↓        ↓          ↓
pairing  retrieve   rank      agent
   │        │        │          │
   │        ├→ embed │          ├→ llm
   │        │        │          │
   └────────┴────────┴──────────┘
            ↓
          store/                     ← 唯一触碰 SQLite / Qdrant 的地方
            ↑
        common/                      ← 被所有层依赖，**自己不依赖任何业务层**
        observability/               ← 被所有层调用（埋点），不反向依赖
```

**三条硬性边界：**

1. **`store/` 是唯一接触 SQLite 与 Qdrant 的目录。** §6.3 的分工表（真源 vs 派生索引、谁能做什么谁不能做什么）只有在所有读写都收口到一处时才守得住。上层拿到的是领域对象，不是 `sqlite3.Row` 或 Qdrant `ScoredPoint`。
2. **`common/render` 是渲染的唯一实现。** §7.2 与 §11.3 要求 **embedding 的输入**与**返回给 AML 的 `content`** 是**同一份渲染**——两处一旦不一致，"检索命中的是什么"与"模型读到的是什么"就会漂移，**而且这种漂移不会报错**。这是 `common/` 存在的全部理由：它不是工具箱，是一条不变式的落地点。
3. **`observability` 不反向依赖业务层。** 埋点是被调用的，不是去拉取的。**每个指标由产生它的那一层发射**（embedding 调用数由 `embed/` 发、Agent Trigger Rate 由 `agent/` 发、rerank 的 `degraded`/`disabled` 与 latency/query 由 `service/` 在 Search 的请求边界发——**计数住在 `SearchPipeline`**，因为"降级 / 没开"这个判定就在那里做），本目录只聚合。**接了哪些、没接哪些**逐项见 [`../src/tianximem/observability/CLAUDE.md`](../src/tianximem/observability/CLAUDE.md)。

---

## 3.5 七个评分维度 → 机制映射（§3.2）

**官方维度逐字采用**（§3.2）。归档 `aml_readme.md` 的原文（"Textual memory capabilities" 表）：

| # | 维度 | 官方描述 |
| --- | --- | --- |
| 1 | Explicit fact recall | Can the system retrieve the right stated facts? |
| 2 | Relational and multi-hop reasoning | Can it connect evidence distributed across memories? |
| 3 | Temporal and event understanding | Can it distinguish order, updates, and the latest valid state? |
| 4 | **Memory governance** | Can it update, retain, and use memory appropriately over time? |
| 5 | Personalization and care | Can it preserve preferences, identity, and user-specific context? |
| 6 | Rules and process execution | Can it recall and follow established constraints and procedures? |
| 7 | **Epistemic safety and privacy** | Can it respect evidence boundaries, uncertainty, and sensitive information? |

其中 **第 4、7 两维**在早期设计里**完全未被覆盖**，是 §3.2 明确要求**在设计里显式回应**的——**所以下表不是装饰，它是那两个维度的唯一落点。**

| # | 维度 | 由什么机制回应 | 在哪 |
| --- | --- | --- | --- |
| 1 | Explicit fact recall | 索引单元 = **完整 QA 对**（不是 message）；**原文优先**，v1 不产生任何合成文本 | `store/` · `rank/packaging` |
| 2 | Relational and multi-hop reasoning | `agent/` 的多轮检索（**合并而非替换**）；`retrieve/` 的 hybrid + RRF | §9 · §7.3 |
| 3 | Temporal and event understanding | `event_time` 列负责**筛选**；正文**保留原始时间表述**；`created_at` 日粒度 | §11.3 |
| 4 | **Memory governance** | ① **批次级幂等守卫**（`applied_batches`）② **块写下即最终**、此后再无改写路径（**D24**）③ 索引损坏时可从真源全量重建（`tools/reindex.py`） | §6.1 · §6.5（**D24** 后的口径） |
| 5 | Personalization and care | 按 `user_id` 严格隔离；`rank/packaging` 的**原文优先**与窗口内**时间序** | §2.2 · §11.2 |
| 6 | Rules and process execution | 契约合规本身：精确 `top_k`、双预算截断、原样回显；以及 agent 的**受控工具面**（只能给关键词与日期范围） | `service/` · §9 |
| 7 | **Epistemic safety and privacy** | ① **`user_id` 是唯一检索隔离字段**，跨 user 检索被禁止 ② **`Search` 不生成答案、不把答案伪装成记忆记录** ③ **冲突消解靠时间可见性**（保留原始时间表述让模型看得出哪个更新） | §2.2 · §2.1 · §11.3 |

### 第 4 与第 7 维值得单独说

**这两维是"不报错就丢分"的类型**，而且**都不体现在检索质量上**：

- **第 4 维（Memory governance）**——官方描述问的是"**能否随时间正确地更新、保留、使用记忆**"。它测的是**写入是否诚实**：重复应用、覆盖旧数据、`status` 回退，**这些都不会让检索变差**，只会让记录与事实不符。本项目的落地就是 §6.5 那套"**两层幂等 + 块写下即最终**"，**其中批次级守卫是唯一防"位置重分配"的机制**（见 [`decisions.md`](./decisions.md) D4、D24）。
- **第 7 维（Epistemic safety and privacy）**——官方描述问的是"**能否尊重证据边界、不确定性与敏感信息**"。它测的是**边界是否守住**：跨 user 泄漏、越权生成答案、在冲突记忆里选错。**①的落地是"每条路径都带 `user_id` 条件"**（[`../tests/CLAUDE.md`](../tests/CLAUDE.md) §3 要求**按路径逐个测**，因为邻域扩展是 SQL 查询、最容易漏条件）；**③的落地是"时间可见性"**——正文带日粒度锚点、相对表述就地注解成绝对日期（D21 / D22），让模型看得出**哪条更新**，而不是由我们去替它裁决。

> **报告纪律**：`eval/reports/` 的记录 schema **要求七个维度逐维记录**。**这两维为空的 run，等于没测。**

---

## 4. 不变式清单

写代码时若发现某处违反下列任一条，**先停下来确认，不要就地绕过**。

| # | 不变式 | 出处 |
| --- | --- | --- |
| I1 | `content` 与 embedding 输入来自**同一次渲染调用**（**一个声明式例外**：只改 `content` 的确定性注解，见下） | §7.2 / §11.3 |
| I2 | 任何返回值都精确 ≤ `top_k`，且**同时按槽位数与 token 数双预算截断** | §2.2 / §6.4 |
| I3 | 一个 `request_id` 至多被应用一次——守卫查 `applied_batches`，**不是查 `qa_pairs.request_id`** | §6.5 |
| I4 | 相邻性走**存储里的显式指针** `prev` / `next`（只在一次 Add 内连过）；链上缺一块就断开，**不会被顶上来** | §6.1 → **D28** |
| I5 | `id` 位置派生；缓存键内容哈希。**两者不能互换** | §6.1 / §7.2 |
| I6 | **`index_pairs` 失败会留下「SQLite 有、Qdrant 没有」的行，而检索不报错**——修复路径见 `service/pipeline.py` | §6.5 → **D24** |
| I7 | `Search` 不生成答案、不把答案伪装成记忆记录；reranker 只重排证据 | §2.1 / §11.2 |
| I8 | 所有消融开关只影响它命名的那一件事 | §13 / §15 |
| I9 | 每个返回项单件**带 `created_at`**（日粒度；`event_time` 为 NULL 时发 `""`） | §11.3 |
| I10 | 正文里的绝对时间**只到日粒度**（秒级绝不许出现），且**相对表述原样保留**（日期是额外锚点，不是替换） | §11.3 / **D21** |

> ### ⚠ I1 的**唯一例外**：只改 `content` 的注解（`packaging.annotate_relatives`）
>
> `true` ⇒ 正文里的相对时间**就地注解**成绝对日期（`last Tues (July 18, 2023)`），
> 原文一字不动。它**只走 `content` 这一条路**——索引侧与精排输入仍然是 `render_pair`，
> **一个字都不改** ⇒ 开它**不用重建索引、不用换集合**（这一点与 T1 的
> `inject_abs_time` 正相反，那个改的正是 embedding 输入）。
>
> **例外到什么程度**（这句话是 I1 松动的边界，别再往外推）：
> `content` = 被索引的文本 **+ 一层纯函数注解**，剥掉注解后**逐字相同**——
> 不是"两份渲染各写各的"，而是**一份渲染 + 一个可逆的确定性后处理**。
> 实现 [`src/tianximem/common/annotate.py`](../src/tianximem/common/annotate.py)；
> 投影性质由 [`tests/test_annotate.py`](../tests/test_annotate.py) 的两条断言钉住
> （"开关不碰索引" + "剥掉注解逐字相同"）。
>
> ⚠ **"正文与索引必须同形"是纪律，不是物理定律**：I1 的例外与 I10 的日期前缀是**仅有的两处**，
> 且都以"说清例外到哪为止"为代价换来——**别把松动本身当成先例**。

> **I8 单独强调**：§13 末尾指出，若关掉 rerank 顺带改变了候选数量、或关掉 agent 顺带改变了打包顺序，该对照**就不成立**——"而结果看起来完全正常，只是结论错了"。这是从零搭建特有的陷阱，比复用既有代码时更容易犯。

---

## 5. 为什么正文不复制进 Qdrant payload

（§6.3）那样会有两份正文，一旦不一致**就无法判断该信哪一份**。而按主键批量取正文是微秒级操作，没有性能理由去复制。

因此 `Search` 的时序是：**Qdrant 出 `id` → 回 SQLite 按 `id` 批量取正文 → 渲染 → 打包**。"取正文"这一步没有缓存层，也不应该有：正文的可信来源只有一处。
