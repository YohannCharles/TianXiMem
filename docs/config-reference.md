# 配置项参考

> 最后核对：2026-09-24，对应 PRD §6–§11、§15。**冲突以 PRD 为准。**

## 为什么这份清单存在

§12.1 风险 R1 的第 3 条对冲：**"所有阈值必须是配置项、不得硬编码，否则切换等于重写代码。"**

R1 是团队**主动接受**的一次偏离——开发期用 Qwen3-Embedding-8B + qwen3.5-9b，提交前才换回 `text-embedding-v4` + `gpt-4o-mini`。代价是**本地标定出的所有阈值、权重、排序策略在切换后都不保证成立**。因此本清单不是"配置文档"，是**切换可行性**的前提：凡是这里漏掉的量，Step 5 都会变成改代码。

同理 §15：**所有消融项必须是配置项**，否则 §13 的对照实验无法执行。

---

## 1. 三个 profile

| 文件 | 用途 | 模型 | 状态 |
| --- | --- | --- | --- |
| `configs/default.yaml` | 基线值，所有 profile 的父级 | Qwen3-Embedding-8B（**开发期唯一有活端点的**） | ✅ |
| `configs/local.yaml` | **开发期**：代理评测、迭代、消融 | 同上 | ✅（**今天只改集合名**） |
| `configs/submit.yaml` | **提交期**：Full 定稿 | `text-embedding-v4` + `gpt-4o-mini` | ⬜ Step 5 |

`local.yaml` 与 `submit.yaml` 只覆盖**模型与由模型派生的量**（向量维度、token 预算实测量、全部标定阈值），其余继承 `default.yaml`。

> ⚠ **`local.yaml` 今天只有一项**：`storage.qdrant.collection: memories_dev`——开发期的集合与提交期分开，
> 避免 §6.3 的 upsert 把上一套实验的 point **静默留给下一套**（`open-questions.md` 的 **V9**）。
> **基线今天放的是开发期模型**（唯一有活端点的那个）；Step 5 建 `submit.yaml` 时再覆盖它。
> **profile 之间真正的差异要等到那时才出现**——现在硬凑一份"两套完整配置"只会让差异看不出来。

**选 profile 用 `TIANXI_PROFILE`**（默认 `default`）；**指向另一份配置目录用 `TIANXI_CONFIG_DIR`**（替代集合的 arm 快照就靠它，见 `configs/CLAUDE.md` 的 `runs/`）。

> **切换 profile 不是一次配置改动，是一个独立阶段**（§12.1 R1 对冲 4 / §16 Step 5）：**不可与任何设计改动同时进行**，否则分数变化无法归因。

---

## 1.2 每个键**住在哪**：`.env` 还是 `configs/*.yaml`

> 加载器是 [`../src/tianxi_am/common/config.py`](../src/tianxi_am/common/config.py)——**全包唯一读环境变量的地方**（2026-09-24，③-d）。

| 层 | 拥有哪些键 | 例子 |
| --- | --- | --- |
| **`.env`**（环境变量） | 密钥、端点、**路径**、进程形态（worker 数） | `AML_EMB_BASE_URL`、`TIANXI_SQLITE_PATH`、`TIANXI_QDRANT_URL`、`TIANXI_EMBED_CACHE_DIR`、`TIANXI_WORKERS` |
| **`configs/<profile>.yaml`** | 阈值、权重、模型名、集合名 | `retrieval.*`、`pairing.*`、`models.embedder`、`storage.qdrant.collection`、`storage.sqlite.busy_timeout_ms` |

**每个键只有一个家，两边不重叠也不许重叠。** 在 yaml 里写一个 env 拥有的键会**直接报错**
（反之亦然）——两处都能设的值，最终会变成"跑出来的结果和 yaml 里写的不一样，而没人知道为什么"。

**为什么阈值不能藏在环境变量里**（§12.1 R1 对冲 4）：那会让 Step 5 的模型切换变成"改 shell 变量"——
**改了什么无法 diff、无法评审**，而归因恰恰是那一步唯一的目的。

**下面各节的标题会标出本节是否已落地**：
**✅ 已落地**（键已进 `configs/*.yaml` 或 `.env`，有代码消费方）· **⬜ 待接线**（落点已定，消费方未接）。

> ⚠ **③-d 只落地了"今天有代码消费方"的键。** 本文件里 `checker.*` / `neighbor.*` / `rerank.*` /
> `agent.*` / `budget.*` / 消融开关的**落点已经在这里声明**，但它们**还没有进 `configs/*.yaml`**——
> 消费方未接线时收进配置等于预留字段（§6.1 对 DDL 的同一条纪律）。
> ⇒ **别以为现在改 yaml 就能开关 `rerank`**：那个开关还没接线。

---

## 1.5 常量 vs 阈值 —— 读这份清单前必须先分清

> **"配置化"不等于"可调"。** §12.1 R1 对冲 3 要求"所有阈值必须是配置项、不得硬编码"，**这句话常被读成"什么都能调"——那是错的。**

写进配置有两个不同的理由，**混淆它们会导致有人去调不该调的东西**：

| 类 | 是什么 | 例子 | 能不能调 |
| --- | --- | --- | --- |
| **A. 外部契约常量** | AML 定的，**我们无权决定** | `top_k = 100`、`117,760` token、重试 32 次、≤20 条 / 2,000 词 | ❌ **不能调**——写进配置只为**追溯**（"这个数字从哪来的"） |
| **B. 正确性常量** | 我们定的，但**算错就静默出错** | **`rrf.k = 61`**（Qdrant 默认是 2，文献是 60） | ❌ **不能调**——它是换算结果，不是旋钮 |
| **C. 自设阈值** | 需要**在代理评测上标定**的 | checker 的三个判据、`N = 200`、种子 20、窗口 ±1、`weights` | ✅ **可调**，但**任何调整都必须有 ablation 数据支撑**（§7.3） |

**A 类不许动**：它们改了就违约（`top_k` 尤其——写死成 100 会在 AML 传更小值时变成**契约错误**）。

**B 类不许"顺手调一调"**：`k=61` 是 0-based 换算的结果。有人看到"文献用 60"会想改成 60——**那会让融合行为偏离所有参考实现，且极难排查**（`decisions.md` D5）。

**C 类可以调，但要留痕**：§7.3 明确"**权重的任何调整都必须有 ablation 数据支撑**"。

> **推论**：`docs/config-reference.md` 里每个配置项都应当标出它属于 A / B / C 哪一类。**新增配置项时一并标。**

---

## 2. 消融开关（§15）—— ⬜ **待接线**

**这是开关的唯一声明处**——其它文档提到开关时一律指回这里，不要各自列一份。

| 开关 | 关掉它意味着 | 依赖谁 | **关掉时不得改变什么** | 出处 |
| --- | --- | --- | --- | --- |
| `dense` | 检索退化为 BM25-only | —— | —— | §7.2 |
| `checker` | 一律进 Agentic Search | —— | —— | §8 |
| `rrf` | 不融合，只用单路 | —— | —— | §7.3 |
| `neighbor` | 不扩窗 | —— | **种子集合不变** | §10 |
| `rerank` | 直接用融合名次 | —— | **候选数量不变**（只是顺序变了） | §11.2 |
| `packaging` | 不做打包策略 | —— | —— | §11.3 |
| `agent` | 一律不走 Agentic Search | `checker`（门控时） | **打包顺序不变**（只是候选少了 agent 补的那部分） | §9 |

> **最后一列是 §13 的纯度规则**：**开关必须只影响它命名的那一件事。** 否则对照不成立——**而结果看起来完全正常，只是结论错了**。
> 对应测试见 [`../tests/CLAUDE.md`](../tests/CLAUDE.md) §5——**那组测试很便宜，而它保护的是整个 §13 实验计划。**

### ⚠ 两条**已作废**的依赖边（D15，2026-09-23）

上面那张表的"依赖谁"一列原先是 `dense → checker` 与 `dense → rrf`。**两条都已删除**，理由是 D15：

**混合检索是既定的检索形态**（BM25 + Dense 两路无条件跑），**没有裸 BM25 模式**。
于是 `dense: false` 不再有下游 ⇒ **`checker` 与 `rrf` 都不再依赖它**。

连带消失的三项（**不要再实现**）：

| 消失的 | 它曾是什么 |
| --- | --- |
| §8 的 **Checker 退化路径** | "dense 关掉时 Checker 按退化判据走"——dense 永不关 ⇒ 那条路径永远不触发 |
| §13 的 **A1 / A2** 两个对照 | 参照点改由**混合主路径自身**承担 |
| [`open-questions.md`](./open-questions.md) 的 **E2** 与"待决事项 2" | 同上 |

**`dense` 开关仍然保留**（§15 要求所有消融项可配），但**它今天没有下游依赖**——
关掉它只改变"检索走了几路"这一件事。

**`checker.enabled` 必须可配**（§15 的开关清单里原本漏了它）——§13 的 A4 要关它做对照。

**两个不在 §15 清单里、但同样要可配的对照项**：T1 的时间戳前缀渲染变体（§11.3）与 A0 的 recency-only（§13）。

**为什么不用融合分数做判据**（§8）：RRF 融合后的分数**不是校准量**。Qdrant 官方明确警告不要把单路阈值用到根级 `score_threshold`——"照搬 dense-only 的阈值会静默截断结果"。因此判据只能用**名次**。

---

## 3. 检索（§7）—— ✅ **已落地**

| 配置项 | 初值 | 类 | 说明 | 出处 |
| --- | --- | --- | --- | --- |
| `retrieval.prefetch_limit` (`N`) | **200** | **C** | 每路进入 RRF 的候选池大小。**必须显式给定** | §7.3 |
| `retrieval.rrf.k` | **61** | **B** ⛔ | **不是可调项。** Qdrant 默认是 `2`，文献是 `60`；Qdrant 秩 0-based，故 `61` 才等价 | §7.3 |
| `retrieval.rrf.weights` | `[0.5, 0.5]` | **C** | **任何调整都必须有 ablation 数据支撑** | §7.3 |
| `retrieval.bm25.model` | `qdrant/bm25` | A | 服务端推理，真 BM25，`modifier: idf` | §7.1 |
| `retrieval.dense.rewrite` | `false` | A | v1 **没有**查询改写（属 v2）。每查询恰好 1 次调用 | §7.2 |
| `retrieval.query_instruction` | `""` | **C** | **查询侧**的 instruction 前缀（Qwen3-Embedding 的推荐输入格式，不是"改写"）。`""` = 原样送（v1 现状）。模型卡称不加会掉约 1%–5%；**要不要开是待定的规格问题**，实现见 [`../src/tianxi_am/embed/query_instruction.py`](../src/tianxi_am/embed/query_instruction.py) | §7.2 |
| `retrieval.query_top_k` | 请求里的 `top_k` | **A** ⛔ | **不要写死 100** | §7.3 |

**类**：**A** = 外部契约常量（**无权改**）· **B** = 正确性常量（**算错就静默出错，不许"顺手调"**）· **C** = 自设阈值（**可调，但要有 ablation 数据**）。见 §1.5。

**三个不同的量，不要混用**（§7.3）：`prefetch_limit`（200）、种子数（20，§10）、Top-K（100，§2.2）。

**融合池上界**：`N=200` × 2 路 = 至多 400 条，要给 100 个名额留出邻域折叠与去重的余量。

**`k=61` 的两个坑叠在一起**（§7.3）：

- Qdrant 的默认值是 `k = 2`（官方文档逐字："k is a constant (set to 2 by default)"）。**不设就会得到一个与所有参考实现都不同的融合行为，且极难排查。**
- `k = 61` 才等价于 RRF 文献里的 `60`：Qdrant 的秩是 **0-based**（"the top result has r_d = 0"），公式为 `1/(rank + k)`；文献是 1-based 的 `1/(60 + rank)`，首位即 `1/(0+61) = 1/(1+60)`。

**版本门槛**：`k` 参数自 **v1.16.0** 起可用，`weights` 自 **v1.17.0** 起可用，故要求 **≥ v1.17.0**。**部署时把 Qdrant 版本钉死**。

**要实测的一条**（§7.1）：选 `qdrant/bm25` 时，**它的分词行为要实测一次**（拿一两个多语言长文档的 case 对照），别默认它等价于 Lucene Analyzer——**BM25 的分数对分词器高度敏感**，而 `qdrant/bm25` 用哪个分词器是我们没有文档的一条。

---

## 4. Evidence Checker（§8）—— ⬜ **待接线**

| 配置项 | 初值 | 说明 |
| --- | --- | --- |
| `checker.enabled` | `true` | §13 的 A4 要关它做对照 |
| `checker.top1_identical` | `true` | 两路 top1 相同 → 足够 |
| `checker.top5_overlap_min` | `3` | 两路 top-5 重叠 ≥ 3 → 足够 |
| `checker.bm25_top1_in_dense_top` | `3` | bm25 top1 的名次在 dense 结果中位于前 3 → 足够 |

阈值**需在代理评测上标定**，初值如上。

**副产品**：这两次分离查询的结果**就是判据本身所需的数据**，不额外花成本。
**⚠ 但它不为任何实验留档**（§8）——早先写的"正是 §13 中 hybrid-vs-BM25-only 对照所需的数据"
**已随 D15 作废**（那个对照已删除）。

---

## 5. Agentic Search（§9）—— ⬜ **待接线**

**参数沿用 ReFind 的形状**——这套参数是在真实评测上跑出来的，**改它需要理由**。

| 配置项 | 初值 | 说明 |
| --- | --- | --- |
| `agent.max_rounds` | **4** | **是"检索轮数"上限，不是"LLM 调用次数"上限** |
| `agent.per_round_top_k` | **5** | 每轮内部检索 |
| `agent.session_rank_fusion` | `true` | 会话级 rank fusion |
| `agent.neighbor_expansion` | `true` | 邻域扩展 |
| `agent.temporal_narrowing` | `true` | 时间收窄 |
| `agent.dedup_seen_sessions` | `true` | 已见 session 去重 |

> **`4` 不是 LLM 调用次数上限。** ReFind 实测每 query **平均约 5.0 次 LLM 调用**（论文 Table 11：4.99）——一轮里可能既有决策调用也有收尾调用。**§14 要监控的是这个均值，不要拿 4 去卡它。**

**工具集**（§9）：`search_chatrecord(query, date_from?, date_to?)` / `take_note()` / `finish_search()`。Agent 只能给出**关键词与日期范围**，其余控制由服务端自动施加。

> 工具名沿用 ReFind 论文的 `search_chatrecord`。**我们的实现是自有的，改名无妨，但保持与论文同形有利于对照实验时的归因。**

---

## 6. Neighbor Expansion（§10）—— ⬜ **待接线**

| 配置项 | 初值 | 说明 |
| --- | --- | --- |
| `neighbor.enabled` | `true` | —— |
| `neighbor.seed_count` | **20** | 取 rerank 后的名次前 20 作为种子 |
| `neighbor.window` | **±1** | **单位是 QA 对**——`±1` 拿回前后各**一整对**（最多 4 条消息），不是各一条消息 |
| `neighbor.order` | `by_rank_then_pidx` | 见 §8 |

**必须计入 Top-K 预算**：邻域槽位**占 `top_k` 名额**（§10）。

**预算怎么算**：20 个种子 × 每颗最多 3 个槽位 = **最坏 60 个槽位**，给回填留 40 个余量。实际通常远小于此——种子密集时相邻窗口大量重叠，去重会折叠掉大部分；种子分散时才逼近 60。

> **但真正先撞上的限制是 token 而不是槽位**（§6.4）：一对的文本量约为单条消息的两倍，117,760 token 的答案窗口可能比 100 个槽位更早用尽。

> **种子数、窗口大小、Top-K、token 预算，是同一道题，任何一项调整都要重算其余四项**（§10）。

**窗口从 ±1 改 ±2 只需改 `BETWEEN` 的界**（§10）——`UNIQUE(user_id, session_id, pair_idx)` 建出的索引**正好就是这个查询的键**。

---

## 7. Rerank 与打包（§11）—— ⬜ **待接线**

| 配置项 | 初值 | 说明 |
| --- | --- | --- |
| `rerank.enabled` | `true` | §11 是主线 |
| `rerank.model` | **`Qwen3-Reranker-4B`** | 自托管 open weights cross-encoder。**2026-09-24 定**，提交时不得更换（D12） |
| `packaging.render_template` | `Q:/A:` | **"贵"消融项**：改它等于改变 embedding 输入，**整个向量索引要重建** |
| `packaging.role_prefix` | `[assistant]` 等标记 | 一个对里有多条非 user 消息时每条带 role 标记 |
| `packaging.inject_abs_time` | **`false`** | 见 §9 |
| `packaging.group_inner_order` | `pair_idx` 升序 | **不把种子提到最前** |
| `packaging.group_outer_order` | 种子名次序 | —— |
| `packaging.created_at_granularity` | `day` | **只给到日粒度** |
| `packaging.score_mode` | `reciprocal_rank` | `1/(rank+1)` 类单调值，**不是**原始 RRF 分数 |

**组内顺序**（§11.2）：**窗口内部按 `pair_idx` 时间序输出，不把种子提到最前。** 窗口是一段连续对话，按时间序读才成立；把种子抽到最前会把一段话拦腰截断。而答案阶段按前缀截断——**窗口整体连续，意味着截断点落在窗口边界上**，不会切出半个窗口。

**组间顺序 = 种子名次序**（§11.2）：最终 `data` 数组是"按种子名次依次排列的各窗口"。名次最高的种子及其邻域**必须排在最前**——预算不足时被牺牲的是名次最低的种子，而不是随机某一个。重叠窗口在去重后归属**名次更靠前的那个种子**。

> 组内与组间顺序**都是可消融项**。若实验显示"种子提到最前"更好，**只改组内这一处**，不牵动其他环节。

**`inject_abs_time = false` 的原因**（§11.3）：把时间做得越清晰，会**同时踩中裁判 TIME 块的两条独立规则**——粒度变细（与 gold 是否相对无关）与相对→绝对互转。两条独立机制指向同一个动作。

**适用范围必须限定**（§11.3）：这条来自 LoCoMo-Refined / LongMemEval 共用的那套契约，**不是全赛道的规则**——BEAM 的裁判正好相反（明文允许等价形式），CL-Bench 由 AML 侧主动注入时间戳。**代理评测按"不加"执行；但不要把这个结论推及全赛道。**

### token 预算

| 配置项 | 初值 | 说明 |
| --- | --- | --- |
| `budget.max_tokens` | **117,760** | 答案窗口 128k 扣掉输出与安全余量 |
| `budget.tokenizer` | `o200k_base` | **用答案模型自己的分词器** |
| `budget.max_slots` | 请求里的 `top_k` | —— |

> **⚠ R1 类风险（§6.4）**：本地 qwen3.5-9b 的分词器与 `gpt-4o-mini` **不同**，**本地量出的"能装多少对"不能直接搬到线上**。已补入对冲清单：**Step 5 切换后必须重新量一次单请求实际返回的对数**。

---

## 8. 存储（§6）—— ✅ **已落地**

| 配置项 | 值 | 理由 |
| --- | --- | --- |
| `storage.qdrant.mode` | **`server`** | **local 模式会静默丢弃 payload 索引**（`create_payload_index` 只打一行警告就返回），且数据格式与 server 不兼容 |
| `storage.qdrant.image` | **钉死 ≥ v1.17.0** | §7.3 的版本门槛 |
| `storage.qdrant.shard_number` | **1** | 根级融合跨分片合并，**分片数变化会改变排名**——为可复现必须单分片 |
| `storage.qdrant.named_vectors` | `dense` + `bm25` | 稀疏向量的距离固定为 Dot |
| `storage.qdrant.payload_indexes` | `user_id`(**keyword** + `is_tenant`) / `session_id`(keyword) / `event_time`(integer) | **必须在写入数据前建**，否则 HNSW 需要重建才有过滤感知 |
| `storage.qdrant.payload_fields` | `user_id` / `session_id` / `pair_idx` / `event_time` | **不含正文** |
| `storage.qdrant.wait` | **`true`** | 契约要求"响应前立即可搜"；默认异步不保证 |
| `storage.sqlite.path` | `var/tianxi.db`（`.env`） | 真源，文件随 run 归档。⚠ 是 `var/` 不是 `data/`——后者与只读归档 `benchmark_data/` 容易混（见 `var/README.md`） |
| `cache.embed.dir` | `var/embed_cache`（`.env`） | **必须落盘** |
| `cache.embed.key` | **渲染文本哈希** | **不能用 `id`** |

> **`is_tenant` 只支持 keyword / uuid 两种类型**——别把 `user_id` 建成 integer（§6.3）。

> **为什么不用 Qdrant-only**（§6.3）：那会让向量库成为唯一真源，索引不可重建；且 AML 只给 2 次 Full、版本冻结，把提交押在一次不可逆的存储格式迁移上不值得。**SQLite 的成本是一个文件。**

---

## 9. 模型（§2.3）—— ✅ **已落地**（只有 `models.embedder`）

| 配置项 | 提交期 | 开发期 | 说明 |
| --- | --- | --- | --- |
| `models.embedder` | `text-embedding-v4` | Qwen3-Embedding-8B | **只能用前者**（§2.3） |
| `models.llm` | `gpt-4o-mini` | qwen3.5-9b | **只能用前者**（§2.3） |
| `models.reranker` | **`Qwen3-Reranker-4B`**（2026-09-24 定） | 同左 | 整份规则里**唯一不限模型**的组件。**提交时不得更换**（D12） |
| `models.embed_dim` | **由接口提供** | —— | **不能写死** |

**架构必须 embedder-agnostic**（§2.3）：开发期的 Qwen3-Embedding-8B 与提交期的 `text-embedding-v4` **都不提供** sparse 或 ColBERT 输出，因此**任何依赖多向量能力的代码都是死重**。集合的向量维度**必须由 `§7.4` 的接口提供**，Step 5 换模型时按新维度**重建集合**。

**为什么 reranker 是唯一能加码的地方**（§11.1）：§2.3 把 embedding 与 LLM 两处都钉死了，**只有 Reranker 不作规定**——所以这是唯一能自己投入算力的环节，也是 §4 认定"排序是最大杠杆"之后唯一还能加码的地方。


---

## 10. 数据侧常量（§6.2 / §6.5）—— ✅ **已落地**

| 配置项 | 值 | 说明 |
| --- | --- | --- |
| `pairing.batch_max_messages` | **20** | AML 的切批上限之一 |
| `pairing.batch_max_words` | **2000** | **本地复现不了**——"Adapter 计数的词"官方从未定义 |

> **本地复现不了词数那一路**（§6.5 / §12.3 第 5 条）：本地只能按 20 条复现，因此**本地测出的 `pending` 埋点数与线上必然对不上**，解读那三个计数器时必须记住。

**配对判据只做一种判断**（§6.2）：这条消息的 `role` 是不是 `user`。**实现里不要枚举 role 白名单**——AML 传入的取值域没有文档，白名单会在遇到没见过的 role 时**静默丢消息**。

---

## 11. 配置卫生

- **不得硬编码**（§12.1 R1 对冲 3）：凡是本清单里的量，代码里只应有读配置的语句。
- **开关只影响它命名的那一件事**（§13）：关掉 rerank 不得顺带改变候选数量；关掉 agent 不得顺带改变打包顺序。**否则对照不成立，而结果看起来完全正常，只是结论错了。**
- **`k=61` 是正确性常量，不是调参项。** 不要把它放进"可调阈值"那一类。
