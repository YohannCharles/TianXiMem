# 配置项参考

> 对应 PRD §6–§11、§15。**冲突以 PRD 为准。**

## 为什么这份清单存在

§12.1 风险 R1 的第 3 条对冲：**"所有阈值必须是配置项、不得硬编码，否则切换等于重写代码。"**

R1 是团队**主动接受**的一次偏离——开发期用 Qwen3-Embedding-8B + qwen3.5-9b，提交前才换回 `text-embedding-v4` + `gpt-4o-mini`。代价是**本地标定出的所有阈值、权重、排序策略在切换后都不保证成立**。因此本清单不是"配置文档"，是**切换可行性**的前提：凡是这里漏掉的量，Step 5 都会变成改代码。

同理 §15：**所有消融项必须是配置项**，否则 §13 的对照实验无法执行。

---

## 1. 三个 profile

| 文件 | 用途 | 模型 | 状态 |
| --- | --- | --- | --- |
| `configs/default.yaml` | 基线值，所有 profile 的父级 | Qwen3-Embedding-8B（**开发期唯一有活端点的**） | ✅ |
| `configs/local.yaml` | **开发期**：代理评测、迭代、消融 | 同上 | ✅（**覆盖集合名 + `rerank.enabled`**） |
| `configs/submit.yaml` | **提交期**：Full 定稿 | `text-embedding-v4` + `gpt-4o-mini` | 🟡 **已建**（2026-09-28，**只换了模型名**；由模型派生的阈值待 Step 5 重标定） |

`local.yaml` 与 `submit.yaml` 只覆盖**模型与由模型派生的量**（向量维度、token 预算实测量、全部标定阈值），其余继承 `default.yaml`。

> ⚠ **`local.yaml` 只有两项**：`storage.qdrant.collection: memories_dev`（开发期的集合与提交期分开，
> 避免 §6.3 的 upsert 把上一套实验的 point **静默留给下一套**（`open-questions.md` 的 **V9**））
> 与 `rerank.enabled: false`（开发期不花那份墙钟）。
> **基线放的是开发期模型**（唯一有活端点的那个）；`submit.yaml` 覆盖它（2026-09-28 起只有 `models.embedder` 一行）。
> ⚠ **它不是"两套完整配置"**——`local.yaml` 只有两项、`submit.yaml` 现在只有一项，
> 差异一眼可见正是这两份文件存在的理由。
> ⛔ `submit.yaml` 里**别写 `storage.qdrant.collection`**（提交期就是基线那个 `memories`；
> 覆盖它的是 `local.yaml`）。有回归位：`tests/test_config.py::test_submit_profile_changes_only_the_embedder`。

**选 profile 用 `TIANXI_PROFILE`**（默认 `default`）；**指向另一份配置目录用 `TIANXI_CONFIG_DIR`**（arm 快照就靠它，见 `configs/CLAUDE.md` 的 `runs/`）。

> **切换 profile 不是一次配置改动，是一个独立阶段**（§12.1 R1 对冲 4 / §16 Step 5）：**不可与任何设计改动同时进行**，否则分数变化无法归因。

---

## 1.2 每个键**住在哪**：`.env` 还是 `configs/*.yaml`

> 加载器是 [`../src/tianxi_am/common/config.py`](../src/tianxi_am/common/config.py)——**全包唯一读环境变量的地方**（③-d）。

| 层 | 拥有哪些键 | 例子 |
| --- | --- | --- |
| **`.env`**（环境变量） | 密钥、端点、**路径**、进程形态（worker 数） | `AML_EMB_BASE_URL`、`TIANXI_SQLITE_PATH`、`TIANXI_QDRANT_URL`、`TIANXI_EMBED_CACHE_DIR`、`TIANXI_METRICS_PATH`、`TIANXI_CAPTURE_PATH`、`TIANXI_WORKERS` |
| **`configs/<profile>.yaml`** | 阈值、权重、模型名、集合名 | `retrieval.*`、`neighbor.*`、`models.embedder`、`storage.qdrant.collection`、`storage.sqlite.busy_timeout_ms` |

**每个键只有一个家，两边不重叠也不许重叠。** 在 yaml 里写一个 env 拥有的键会**直接报错**
（反之亦然）——两处都能设的值，最终会变成"跑出来的结果和 yaml 里写的不一样，而没人知道为什么"。

**为什么阈值不能藏在环境变量里**（§12.1 R1 对冲 4）：那会让 Step 5 的模型切换变成"改 shell 变量"——
**改了什么无法 diff、无法评审**，而归因恰恰是那一步唯一的目的。

**各节标题标出是否已落地**：**✅ 已落地**（键已进 `configs/*.yaml` 或 `.env`，有代码消费方）· **⬜ 待接线**（落点已定，消费方未接）。

> ⚠ **只收"有代码消费方"的键。** 本文件里 `checker.*` / `agent.*` 与部分消融开关的**落点已经声明**，
> 但它们**还没有进 `configs/*.yaml`**——消费方未接线时收进配置等于预留字段（§6.1 对 DDL 的同一条纪律）。
> **v2 再接**的是 `checker.*` 与 `agent.*`（**D26** / D13——理由见 §2 那句"共用一个死锁"），
> `rrf` 与 `dense` 则是**无下游依赖**（D15）：它们仍在表里，但**不要**当成待办。

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

> **推论**：本文件里每个配置项都应当标出它属于 A / B / C 哪一类。**新增配置项时一并标。**

---

## 2. 消融开关（§15）—— **逐行看"接线"列**

**这是开关的唯一声明处**——其它文档提到开关时一律指回这里，不要各自列一份。

| 开关 | 关掉它意味着 | 依赖谁 | **关掉时不得改变什么** | 出处 | 接线 |
| --- | --- | --- | --- | --- | --- |
| `dense` | 检索退化为 BM25-only | —— | —— | §7.2 | ⛔ **无下游依赖**（D15：混合检索无条件跑） |
| `checker` | 一律进 Agentic Search | —— | —— | §8 | ⛔ **v2 再接**（D26 / D13——见下） |
| `rrf` | 不融合，只用单路 | —— | —— | §7.3 | ⛔ **无下游依赖**（D15 删掉了 A1/A2 ⇒ **没有实验需要它**） |
| `neighbor` | 不扩窗 | —— | **种子集合不变** | §10 | ⬜（只能关 `radius`） |
| `rerank` | 直接用融合名次 | —— | **候选数量不变**（只是顺序变了） | §11.2 | ✅ `rerank.enabled` |
| `packaging` | 不做打包策略 | —— | —— | §11.3 |
| **T1** | 正文**不带**日期前缀（消融臂） | —— | **名次 / 段数 / `created_at` / `score` 全不变**（只有 `content` 多一段 `[YYYY-MM-DD] `） | §11.3 / **D21** | ✅ `packaging.inject_abs_time`（**默认 `true`**） |
| **相对时间注解** | 正文里的相对表达**照原样**留着（`last Tues`） | —— | **名次 / 段数 / `id` / `created_at` / `score` / token 口径全不变**（只有 `content` 多出括号注）；**embedding 输入也不变**（它只走 `content` 这一条路） | D21 的延伸（原文保留、日期是**额外锚点**） | ✅ `packaging.annotate_relatives`（**默认 `true`**——`configs/default.yaml` 里写死；⚠ **加载器对"键缺失"取 `false`**，与 `inject_abs_time` 的缺失约定**相反**，所以 D22 之前冻结的快照仍按"无注解"跑） |
| `agent` | 一律不走 Agentic Search | `checker`（门控时） | **打包顺序不变**（只是候选少了 agent 补的那部分） | §9 | ⛔ **v2 再接**（`agent/` 还没有代码，D13） |

> **最后一列是 §13 的纯度规则**：**开关必须只影响它命名的那一件事。** 否则对照不成立——**而结果看起来完全正常，只是结论错了**。
> 对应测试见 [`../tests/CLAUDE.md`](../tests/CLAUDE.md) §5——**那组测试很便宜，而它保护的是整个 §13 实验计划。**

> ### ⚠ `checker` / `agent` 为什么标成"**v2 再接**"而不是"待接线"（2026-09-28）
>
> 两者**共用一个死锁**：`checker` 关掉的意思是"**一律进 Agentic Search**"，而
> [`../src/tianxi_am/agent/`](../src/tianxi_am/agent/) **只有 `CLAUDE.md`、没有任何代码**
> ⇒ 那个"关"分支**无处可去**（`EvidenceChecker.decide()` 的返回值此刻在
> `service/pipeline.py` 里被丢弃，那是 D13 的**刻意**形态）。
>
> 而**接线的唯一理由本来是 A4**（§13 要关掉 checker 做对照），**A4 已移出 v1**（**D26**）
> ⇒ 在 Step 4 之前，这两个开关接了也没有消费者，而接它们要**跨层**（配置管道 + 让
> `pipeline` 真的按判定分支 + 给 `store/` 加单路查询）。**故 v1 不接。**
>
> **`checker.enabled` 仍必须可配**（§15 的开关清单里漏了它）——但那是 v2 落地时的事，
> 落点与三个判据阈值见 §4。

**两个不在 §15 清单里、但同样要可配的对照项**：T1 的时间戳前缀渲染变体（§11.3，已接线）与 A0 的 recency-only（§13）。

### ⚠ 三条**不要实现**的东西（D15）

混合检索是既定的检索形态（BM25 + Dense 两路无条件跑），**没有裸 BM25 模式**，
所以 `dense: false` **没有下游依赖**——关掉它只改变"检索走了几路"这一件事。⛔ **连带的这三项不存在，也不要再实现**：

| 不要实现 | 为什么它会「看起来该有」 |
| --- | --- |
| §8 的 **Checker 退化路径** | "dense 关掉时 Checker 按退化判据走"——dense 永不关 ⇒ 那条路径**永远不触发** |
| §13 的 **A1 / A2** 两个对照 | 参照点改由**混合主路径自身**承担 |
| [`open-questions.md`](./open-questions.md) 的 **E2** | 同上——**该编号不再启用** |

**`dense` 开关仍然保留**（§15 要求所有消融项可配），但**它没有下游依赖**。

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

## 4. Evidence Checker（§8）—— ⛔ **v2 再接**（D26 / D13）

| 配置项 | 初值 | 说明 |
| --- | --- | --- |
| `checker.enabled` | `true` | §13 的 A4 要关它做对照 |
| `checker.top1_identical` | `true` | 两路 top1 相同 → 足够 |
| `checker.top5_overlap_min` | `3` | 两路 top-5 重叠 ≥ 3 → 足够 |
| `checker.bm25_top1_in_dense_top` | `3` | bm25 top1 的名次在 dense 结果中位于前 3 → 足够 |

阈值**需在代理评测上标定**，初值如上。

**副产品**：这两次分离查询的结果**就是判据本身所需的数据**，不额外花成本。
**⚠ 但它不为任何实验留档**（§8）——没有对照消费它。

---

## 5. Agentic Search（§9）—— ⛔ **v2 再接**（D13：`agent/` 还没有代码）

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

## 5.5 接入口径（D25）—— ✅ **已落地**

| 配置项 | 初值 | 类 | 说明 |
| --- | --- | --- | --- |
| `ingest.chunk_ordinal_pattern` | `(?:chunk-\|\|)(\d+)\s*$` | **A** | 从 `request_id` 里**取 chunk 序号**的正则（`re.search`，恰好 1 个捕获组）。D25 把位置改成 `(chunk_ordinal, local_index)`，**chunk 序号只能从这个 id 里取** ⇒ 取不到就**非 200**（没有回退，见 [`../src/tianxi_am/pairing/pairing.py`](../src/tianxi_am/pairing/pairing.py) 的 `parse_chunk_ordinal`） |

> ⚠ **格式假设的来源是团队告知，不是一手文档**（D25 的"关键依据"一节）——
> 与 S2/S5 同类。**做成配置项**是为了让"平台换了个形状"不必改代码；
> **响亮失败**是为了让"我们猜错了"立刻暴露，而不是安静地跑完一整场。
>
> 默认值同时认两种形态（`re.search`）：
> 平台实发 `eval:<run_id>:locomo_refined:conv-0:chunk-3` · 本仓 harness `<user>|<session>|3`
> ——[`../eval/harness/batching.py`](../eval/harness/batching.py) 的 `request_id_for()` 就是后者。

---

## 6. Neighbor Expansion + 段合并（§10 / §11.2）—— ✅ **已落地**

| 配置项 | 初值 | 类 | 说明 |
| --- | --- | --- | --- |
| `neighbor.expansion_seed_limit` | **1000**（= 全部候选） | C | 只对**名次前 N 位**的候选主动扩窗；其余候选**仍保留，只是不扩展**。✅ 2026-09-25 由 30 提到 1000：3 段 346 题的 A/B **+3.5pt 且三段无倒退**（[`../eval/reports/ledger.md`](../eval/reports/ledger.md) 的 N1） |
| `neighbor.seed_placement` | **`keep`** | C | **段内顺序**（§11.2 的组内顺序，明文可消融）：`keep`（纯时间序）/ `front`（种子提到段首，⚠ **断时序**）/ `echo`（段首重复种子，时间序块原样保留）。✅ 2026-09-25 加，**三档正在对照中** |
| `neighbor.radius` | **2** | C | 扩窗半径，**单位是记忆块**（D24 前叫 QA 对）——`±2` 拿回会话里前后各**两整块**（最多 8 条消息），不是各一条消息。✅ 同日 1 → 2（同一条证据链） |

> ⚠ **PRD §10 的"20 种子 / 60 槽位"是一道示例算术**（用来演示预算怎么算），
> 与这里的实际取值**不是一回事**，别互相替换。
> （已实现的名字才是唯一声明：`expansion_seed_limit` / `radius`。在 yaml 里写一个不存在的键
> ——比如 PRD 的示例口径——服务照常启动、行为一点不变。）

**`top_k` 约束的是段数，不是 raw memory 数**：100 个候选 + 40 个邻居
= 140 条 raw，**相邻块合并**之后可能只剩 60 段。⇒ **`top_k` 只能在合并之后生效**。

**预算怎么算**：**真正先撞上的限制是 token 而不是段数**（§6.4）——一对的文本量约为单条消息的两倍，117,760 token 的答案窗口可能比 100 个段更早用尽。

> **种子数、窗口大小、Top-K、token 预算，是同一道题，任何一项调整都要重算其余三项**（§10）。
> ✅ **2026-09-25 那次调整有数据**（N1，346 题）；坑不在「能不能调」，在**别拿单段对比当依据**——
> conv-26 一段上看着是 +5.8pt，三段 346 题上只有 **+3.5pt**（对话之间的方差自己就有 ±5pt）。
> ⚠ 代价：每题上下文 ~34k → ~57k 字符 ⇒ **判分侧 token 涨约 60%**。

**窗口从 ±1 改 ±2 只需改 `neighbor.radius`**（§10）——扩窗是
**每 session 取一次整段有序列表、在 Python 里切 `[i-radius, i+radius]`**（D25），
所以改半径只动这一个键、不牵动查询。⚠ 它**同时是邻域判据的输入**：
同一 session 内相邻块会被并进同一段，**半径越大、能并进来的一串越长**。

> **`neighbor.enabled` / `neighbor.order` 这两个键没有落地**，而且是**有意的**：
> 前者是 §15 的消融开关（**待接线**，见 §2），后者在 v1 里**不是旋钮**——
> 段内顺序（**`seq` 升序**，D25）与段间顺序（`best_rank` 升序）都是 §11.2 的规格，
> 不是可以各调各的初值。

---

## 7. Rerank 与打包（§11）—— ✅ **已落地**

| 配置项 | 初值 | 说明 |
| --- | --- | --- |
| `rerank.enabled` | **`true`** | ✅ §15 的消融开关。`false` ⇒ **不构造 reranker**，Search 直接用融合名次（记 `rerank_disabled`）。**`true` 是提交口径**（2026-09-28）：D8 把排序定为主线，默认关着等于放弃它（A3 在 conv-26 上 +4.3pt）。⚠ 开发期在 `configs/local.yaml` 里**显式关掉**（墙钟约 3×，直接决定迭代速度）——**那是覆盖，不是默认值**。**它值不值是 A3 要回答的**（两臂快照 `configs/runs/a3-{on,off}/`），**别拿默认值当结论** |
| `rerank.timeout_seconds` | **30.0** | **C 类**。实测 100 篇 ≈ 2.2s、200 篇 ≈ 5.3s ⇒ 约 10 倍余量。**太紧 ⇒ 伪降级**（网关排队被报成"reranker 坏了"）；**太松 ⇒ Search 被拖住** |
| `packaging.inject_abs_time` | **`true`**（2026-09-25，**D21**） | ✅ §2 表里 **T1** 的开关。`true` ⇒ 每对正文前加 `[YYYY-MM-DD] `（与 `created_at` **同一口径、同一 `event_time`**）。⚠ **「贵」消融项**：正文改了 embedding 输入也改 ⇒ **改它要重建索引**、两臂必须分集合 |
| `packaging.annotate_relatives` | **`true`**（2026-09-26） | ✅ **只改 `content`、不碰 embedding**（不变式 **I1 的一个声明式例外**）：`true` ⇒ 正文里的相对时间**就地注解**成绝对日期（`last Tues (July 18, 2023)`），原文一字不动。⇒ **不用重建索引、不用换集合**，随时可开关。实现与"推不出就不动"那条硬纪律在 [`../src/tianxi_am/common/annotate.py`](../src/tianxi_am/common/annotate.py)；开关买到的东西见 [`../eval/reports/ledger.md`](../eval/reports/ledger.md)。⚠ **C 类**：任何调整都要有 ablation 数据 |
| `budget.max_tokens` | **117,760** | ⛔ **A 类**（AML 定的答案窗口余量）。答案窗口 128k 扣掉输出与安全余量 |
| `budget.tokenizer` | **`o200k_base`** | ⛔ **A 类**。必须是**答案模型自己的**分词器 |
| `budget.max_slots` | 请求里的 `top_k` | —— |
| `packaging.created_at_granularity` | `day` | **只给到日粒度**，且**固定 UTC、无旋钮** |
| `packaging.score_mode` | `reciprocal_rank` | `1/(rank+1)`，**按输出位置**、**不是**原始 RRF 分数 |
| `packaging.render_template` | `Q:/A:` | **"贵"消融项**：改它等于改变 embedding 输入，**整个向量索引要重建**。唯一实现是 `common/render.py` |
| `packaging.role_prefix` | `[assistant]` 等标记 | 一个对里有多条非 user 消息时每条带 role 标记 |
| `TIANXI_RERANKER_BASE_URL` | `https://memory3.021130.xyz/v1` | **env**。主网关（**不是 memory2**，D18） |
| `TIANXI_RERANKER_API_KEY` | —— | **env**。与 `AML_EMB_*` 是同 host、不同 key |
| `TIANXI_RERANKER_MODEL` | `qwen3-reranker-4b` | **env**。⚠ **主网关会校验它**（vllm 直服，2026-09-28 迁移后）——填错就是 **404 ⇒ 每次检索都降级**，而服务**不报错**。旧 host（自研封装）是忽略它的。**提交时不得更换**（D12） |

> ⚠ **`rerank.model` 这个键不存在**：模型名是**端点身份**、不是阈值，所以它住在 `.env` 的
> `TIANXI_RERANKER_MODEL`（§1.2 的两层分工）。
> **那三个变量是缺了就降级的**（D12），与 `embed.*` 那种"缺了就拒绝启动"正相反——
> 所以 `validate()` **不**校验它们，改由 `service/app.py` 的 `build_reranker()` 决定接不接。
> ⚠ **"想用却没配全"会打一条 WARNING**：否则它会表现成"每次检索都静默不精排"。

> **`packaging.group_inner_order` / `packaging.group_outer_order` 不是配置项**：
> 它们是 §11.2 的**规格**（段内 **`seq` 升序**、段间 `best_rank` 升序），
> 住在 `rank/neighbor.merge_segments` 里，**不是可以各写各的初值**。
> §11.2 说它们是"可消融项"——真要消融时**改那一个函数**，别先立一个没人读的键。

**组内顺序**（§11.2）：**窗口内部按 `seq` 时间序输出，不把种子提到最前。** 窗口是一段连续对话，按时间序读才成立；把种子抽到最前会把一段话拦腰截断。而答案阶段按前缀截断——**窗口整体连续，意味着截断点落在窗口边界上**，不会切出半个窗口。

**组间顺序 = 种子名次序**（§11.2）：最终 `data` 数组是"按种子名次依次排列的各窗口"。名次最高的种子及其邻域**必须排在最前**——预算不足时被牺牲的是名次最低的种子，而不是随机某一个。重叠窗口在去重后归属**名次更靠前的那个种子**。

> 组内与组间顺序**都是可消融项**。若实验显示"种子提到最前"更好，**只改组内这一处**，不牵动其他环节。

**`inject_abs_time = true` 的原因**（**D21**）：LoCoMo-Refined 的时间 gold 是**锚定式相对**形式（`The Friday before 22 October 2023`），只有把锚点放在**那句话旁边**，模型才会接上去。实测 3 段 346 题：日期写段首无效、写**每对**后整体 0.601 → **0.633**。⚠ **两条边界照旧要守**：只给**日粒度**（秒级会踩「粒度变细」）、**相对表述原样保留**（日期是锚点不是替换）。

**适用范围必须限定**（§11.3）：这条来自 LoCoMo-Refined / LongMemEval 共用的那套契约，**不是全赛道的规则**——BEAM 的裁判正好相反（明文允许等价形式），CL-Bench 由 AML 侧主动注入时间戳。**代理评测按 D21 执行；但不要把这个结论推及全赛道。**

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
| `storage.qdrant.payload_fields` | `user_id` / `session_id` / `chunk_ordinal` / `local_index` / `event_time` | **不含正文**。⚠ **D25** 把 `pair_idx` 拆成了后两个（payload 里没有消费方，只是溯源；过滤只按 `user_id`，扩窗读 SQLite） |
| `storage.qdrant.wait` | **`true`** | 契约要求"响应前立即可搜"；默认异步不保证 |
| `storage.sqlite.path` | `var/tianxi.db`（`.env`） | 真源，文件随 run 归档。⚠ 是 `var/` 不是 `data/`——后者与只读归档 `benchmark_data/` 容易混（见 [`../var/CLAUDE.md`](../var/CLAUDE.md)） |
| `cache.embed.dir` | `var/embed_cache`（`.env`） | **必须落盘** |
| `cache.embed.key` | **渲染文本哈希** | **不能用 `id`** |

> **`is_tenant` 只支持 keyword / uuid 两种类型**——别把 `user_id` 建成 integer（§6.3）。

> **为什么不用 Qdrant-only**（§6.3）：那会让向量库成为唯一真源，索引不可重建；且 AML 只给 2 次 Full、版本冻结，把提交押在一次不可逆的存储格式迁移上不值得。**SQLite 的成本是一个文件。**

---

## 9. 模型（§2.3）—— ✅ **已落地**（`models.embedder` + 三个 env 里的模型名）

| 配置项 | 提交期 | 开发期 | 住哪 | 说明 |
| --- | --- | --- | --- | --- |
| `models.embedder` | `text-embedding-v4` | **`qwen3-embedding-8b`** | **yaml** | **只能用前者**（§2.3）。⚠ 值是**服务端 id**，见下面那条迁移警告 |
| `models.llm` | `gpt-4o-mini` | qwen3.5-9b | **env**（`AML_MODEL`） | **只能用前者**（§2.3） |
| `TIANXI_RERANKER_MODEL` | **`qwen3-reranker-4b`** | 同左 | **env** | 整份规则里**唯一不限模型**的组件。**提交时不得更换**（D12）。⚠ 新网关**校验**它，填错 ⇒ 404 ⇒ 降级 |
| `models.embed_dim` | **由接口提供** | —— | —— | **不能写死** |

> ### ⛔ 2026-09-28：模型 id 随网关迁移**全变了**
>
> embedding 与 rerank 从 `memory.021130.xyz`（自研封装）迁到 `memory3.021130.xyz`（**vllm 直服**），
> 两边的 id **不通用**——它们**是服务端的事**：
>
> | | 旧 host | 新 host |
> | --- | --- | --- |
> | embedding id | `Qwen/Qwen3-Embedding-8B` | **`qwen3-embedding-8b`** |
> | reranker id | `Qwen3-Reranker-4B` | **`qwen3-reranker-4b`** |
> | embedding **维度** | 1024 | **4096** |
> | `/models` 列不列 reranker | 列 | **不列**（但 `/rerank` 可用） |
> | `/rerank` 的请求/响应形状 | `query` → `results[]` | **`queries: [...]` → `data[].score`**（当天网关只改了 nginx.conf：`/v1/rerank` rewrite 到 vLLM 原生 `/v1/score`。**客户端两个信封都收**，逐条实测见 [`../src/tianxi_am/rank/reranker.py`](../src/tianxi_am/rank/reranker.py) 顶部的表） |
>
> ⚠ **维度变了 ⇒ 集合与缓存都要重建**：动作清单在
> [`../deploy/CLAUDE.md`](../deploy/CLAUDE.md) §4 的"网关迁移"一节。
> ⚠ **`configs/runs/` 里那 13 份冻结快照带的是旧 id**——它们是**历史记录**，没改（改了就是伪造当时跑的东西）；
> 但**重放任何 arm 之前必须先覆盖 `models.embedder`**，否则第一步 embedding 就 404。
> ⚠ **`tools/check_env.py` 不再写死 id**：它向 `/v1/models` 问，并把用到的 id 打出来——
> 写死会得到一个"永远红"的探针，而"永远红"最后会被人关掉。

> ⚠ **embedding 走 yaml、另两个走 env，这不是笔误**：`models.embedder` 是 profile 之间
> **唯一真正该变**的东西（`local.yaml` / `submit.yaml` 存在的理由就是让"哪些量随模型变"
> 能被 diff 出来）；而 LLM 与 reranker 的模型名在开发期与提交期**是同一套网关地址上的
> 不同部署**，属于"端点身份"，所以跟 base_url / key 一起住在 `.env`。

**架构必须 embedder-agnostic**（§2.3）：开发期的 Qwen3-Embedding-8B 与提交期的 `text-embedding-v4` **都不提供** sparse 或 ColBERT 输出，因此**任何依赖多向量能力的代码都是死重**。集合的向量维度**必须由 §7.4 的接口提供**，Step 5 换模型时按新维度**重建集合**。

**为什么 reranker 是唯一能加码的地方**（§11.1）：§2.3 把 embedding 与 LLM 两处都钉死了，**只有 Reranker 不作规定**——所以这是唯一能自己投入算力的环节，也是 §4 认定"排序是最大杠杆"之后唯一还能加码的地方。

---

## 10. 数据侧常量（§6.2）

**没有服务端配置项**——切批是 AML 侧的事，本地只在**测试侧**复现。

> ⚠ **"AML 按 20 条消息或 2,000 个 Adapter 计数的词切批"这个事实没有变**，它仍然写在
> [`contract.md`](./contract.md) 里，也仍然由
> [`../eval/harness/batching.py`](../eval/harness/batching.py) 的 `MAX_MESSAGES_PER_BATCH = 20`
> 复现（那是**测试侧自造的切批口径**，从来不是服务端配置）。
> **"Adapter 计数的词"官方从未定义**（S2）这条也仍然未清。

**组合判据只做一种判断**（§6.2）：这条消息的 `role` 是不是 `user`。**实现里不要枚举 role 白名单**——AML 传入的取值域没有文档，白名单会在遇到没见过的 role 时**静默丢消息**。

---

## 11. 配置卫生

- **不得硬编码**（§12.1 R1 对冲 3）：凡是本清单里的量，代码里只应有读配置的语句。
- **开关只影响它命名的那一件事**（§13）：关掉 rerank 不得顺带改变候选数量；关掉 agent 不得顺带改变打包顺序。**否则对照不成立，而结果看起来完全正常，只是结论错了。**
- **`k=61` 是正确性常量，不是调参项。** 不要把它放进"可调阈值"那一类。

---

## 12. 诊断：请求**原文**采集（**S6**）—— ✅ **已落地**（2026-09-29）

**它全是诊断，不是功能**：把官方发来的 `/add` / `/search` **原样**抄一份落盘，
用来核验**官方真实请求的形状**。默认关。

| 配置项 | 初值 | 类 | 说明 |
| --- | --- | --- | --- |
| `capture.enabled` | `false` | **—** | 开关。`false` ⇒ **连中间件都不装**（零开销、零行为差异） |
| `capture.max_bytes` | `67108864` | **—** | **护栏**（不是阈值）：写满就写一行 `truncated` 标记并**停止记录** |

> ⚠ **`—` 不是漏标**：§1.5 的 A / B / C 三类说的都是**影响结果的量**，而本项不改任何行为
> （它的"关"分支与"没有这个功能"逐字等价）。所以它既不是契约常量、也不是阈值、
> 更不需要 ablation 数据——**它是观测**。默认关只是因为其余时间它是净开销。
>
> **路径在 `.env` 的 `TIANXI_CAPTURE_PATH`**（路径归 env，开关归 yaml）——与
> `rerank.enabled` + `TIANXI_RERANKER_*` 同一个拆法。容器形态是
> `/data/capture/requests.jsonl`（在卷里，`docker cp` 取得走）。

**为什么要有它**：**S6** —— 平台的 `request_id` 到底长什么样（是否真带 `chunk-<n>`、
序号在不在末尾）我们**从没见过**，而 D25 的位置模型整个押在这个假设上
（[`open-questions.md`](./open-questions.md) 的 S6 记着"来源是团队告知、不是一手文档"）。
每一行都记下 `request_id` 与 `chunk_ordinal`——**后者为 `null` 就是"服务当前解析不出来"**，
那正是 2026-09-29 那次冒烟失败的形状。

**四条纪律 + 一行记录里有什么**，一处声明在
[`../src/tianxi_am/service/capture.py`](../src/tianxi_am/service/capture.py)：

1. **在解析之前抄** ⇒ 是 ASGI 中间件而不是路由函数（不合 schema 的 422 进不了路由函数）
2. **不改变下游看到的 body**（只复制，不"读掉再重放"）
3. **不吞异常**（只记类型名，照旧往上抛）
4. **写盘失败不影响响应**（只留一行 WARNING，先例是 `SnapshotMetricsSink`）

> ⚠ **打开它要重建镜像**（`configs/` 是烘进镜像的，不是挂载的）⇒ 它是"预先开好、
> 跑完关掉"的开关，不是运行期随手拨的。**跑完记得关回去**——文件里是**官方评测原文**。
