# 参考实现对照

> 最后核对：2026-10-08 增补 MemMachine；ReFind / 候选映射笔记保留原核对范围。对应 PRD §1 / §4 / §11 / §13。**冲突以 PRD 为准。**

**这份文档登记外部参考源码的来源与版本，并保留 ReFind / InvMem 候选映射的对照笔记。**
新增的 MemMachine 源码研读路径与改进方向见 [学习入口](memmachine-reference.md)。

它**不是规格**，也**不改变任何已锁定的决策**——本文里"能改变下一步动作"的结论，要先登记进 [`experiments.md`](./experiments.md) / [`open-questions.md`](./open-questions.md) 才算有身份。

---

## 0. 代码在哪（可复现）

这些仓库以浅克隆落在 [`../eval/baselines/`](../eval/baselines/) 下，各克隆目录分别 **gitignored**（见 [`.gitignore`](../.gitignore) 的"参考实现"一节）：字节从公开源各自取，本地副本不进版本库，来源与学习笔记入库。

| 目录 | 上游 | 克隆时的 commit | 许可 | 是什么 |
| --- | --- | --- | --- | --- |
| `eval/baselines/refind/` | `github.com/imlrz/ReFind` | `a80175c`（2026-08-14） | **MIT** | ReFind 原版（B1） |
| `eval/baselines/invmem-candidate/` | `github.com/wenxiaof345-ctrl/vanilla-rag-memory` | `31ab7bf`（2026-08-11） | **无 LICENSE** | InvMem 的**候选映射**，仅研读 |
| `eval/baselines/memmachine/` | [MemMachine/MemMachine](https://github.com/MemMachine/MemMachine) | `ad8ff24b0b5c73f189eab9bb7342d4655ab85ca6`（tag `v0.4.0`，2026-10-06 上游提交） | [Apache-2.0](https://github.com/MemMachine/MemMachine/blob/ad8ff24b0b5c73f189eab9bb7342d4655ab85ca6/LICENSE) | 源码参考与学习对象（未接入评测）；[学习入口](memmachine-reference.md) |

要换台机器取回同一份：`git clone https://github.com/<上游>.git eval/baselines/<目录>` 然后 `git checkout <commit>`。
MemMachine 的源码获取与版本核对命令见 [学习入口](memmachine-reference.md#获取同一份源码)，不安装依赖。

**⚠ 目录名 `invmem-candidate/` 是刻意的。** 榜单显示名到这份 repo 的映射**是第三方推断**（PRD 附录 B），仓库里**从不出现 "InvMem" 一词**，也**没有 LICENSE**。本次复核的结果是**方法指纹全中**：

| 榜单对 InvMem 的描述 | 候选仓库里的对应物 | 出处 |
| --- | --- | --- |
| 细粒度切分 | `split_memory_text()`，320 token / 40 重叠，**按 token 切** | `src/vanilla_rag/memory_store.py:157` |
| dense / BM25 混合 | `retrieval_mode ∈ {dense, hybrid, enhanced}` | `memory_store.py:216`、`:499` |
| **加权** RRF | `lexical_weight / (rrf_k + rank + 1)`（词法那一路带权重） | `memory_store.py:503-509` |
| 同 session 邻域扩展 | `_expand_result_positions()`，只在同 `session_id` 内扩 | `memory_store.py:555-583` |

> **四条全中是"很可能就是它"的证据，不是指认。** 它足以让这份代码**值得读**，不足以让它**成为基线**——无 LICENSE 意味着不能分发、不能声称、不能依赖（§1）。**读它得到的结论要标记来源，别写成"InvMem 论文说"。**

---

## 1. 两个系统各自是什么

### ReFind（44.97）

**纯 BM25 + ReAct 检索 agent。** 声明配置（[`../eval/baselines/refind/docs/METHOD_CARD.md`](../eval/baselines/refind/docs/METHOD_CARD.md) 逐字）：

```text
RETRIEVAL_MODE=agent · LLM_MODEL=openai/gpt-4o-mini · AGENT_MAX_ITERATIONS=4
SEARCH_TOP_K=5 · CONTEXT_WINDOW=2 · SESSION_RRF=true · RETENTION_DAYS=30
```

三个动作（`search_chatrecord` / `take_note` / `finish_search`），每轮把结果**替换**掉，所以要留证据必须 `take_note` 按 1-based 序号存。已返回过的 chunk **自动排除**，后续搜索必然给新料。空手收场有**两级兜底**：最后一批自动入账 → 仍为空则退回纯 BM25 直检。

### InvMem 候选（自称 0.6.0）

**dense + BM25 加权 RRF + 交叉编码器精排 + 结果窗口扩窗。** 与我们主线（§4 / §11）**逐项同构**，差别在参数：

```text
bge-small-en-v1.5 · chunk 320/40 · hybrid · lexical_weight=0.5 · rrf_k=60
rerank_candidates=200 · reranker_max_length=512 · result_window=1 · seed_k=20
```

它的 `enhanced` 模式另有一套**题干级查询扩展**：把每个选项各当成一路 query（`f"{query}\nCandidate answer: {option}"`），取"与任一选项最像"的分数并加权抬升——**冲选择题去的**（PersonaMem 是 MCQ，[`../eval/CLAUDE.md`](../eval/CLAUDE.md)）。

---

## 2. 与我们逐项对照

### A. 检索与排序

| 维度 | 我们 | ReFind | InvMem 候选 |
| --- | --- | --- | --- |
| **配对时机** | **Add 时**，且**只看本批**（D24） | **Search 时**（每次从原始消息重构） | Add 时按 message 切块，**不再配对** |
| 配对粒度 | 记忆块；一段连续 user 里**只把最后一条**配上回答（**D32**），前面的各自独立 | 相邻 2 turn，**不看 role** | 320 token 块（可切碎一条消息） |
| 记录 id | 位置派生 `hash(user_id,session_id,request_id,local_index)`（**D28**） | 位置派生 `hash(user,session,源位置)` | 位置派生 `hash(request_id:msg:chunk)` |
| 检索后端 | Qdrant 内 RRF 融合 **dense + BM25** | **纯 BM25**（两种粒度） | dense + BM25，**加权** RRF |
| RRF `k` | **61**（Qdrant 0-based，D5） | 代码 `1/(60+rank)` 但 **rank 从 1 起** = `1/(61+rank₀)` | `1/(60+rank+1)`，**同样等价于 61** |
| 融合的是谁 | dense 名次 × BM25 名次 | **对话级名次 × 会话级名次**（会话分 = 组内 BM25 **求和**） | dense 名次 × 词法名次（词法侧乘权重） |
| 精排 | 远端 `Qwen3-Reranker-4B`，**整批一次**，超时 30s，**不重试**，失败退回 RRF 序 | **无** | 交叉编码器，`rerank_candidates=200`，max_len 512 |
| 扩窗 | 按 **pair**、沿显式指针跳 `radius` 跳（默认 **0** = 不扩窗，**D31**）、种子数 `expansion_seed_limit`；**段合并**（严格 `末+1` 才并） | 按 chunk、**±2**、**同 session 内**、锚点 + 邻居当**一整块**返回 | 按 chunk/行序、半径 1、只对前 20 个种子、**同 session 内**、去重后**用原始名次补足 top_k** |
| 名次怎么来的 | rerank 分降序，同分按原名次 | 融合分降序，**同分按时间序下标** | rerank 降序，同分按融合分 |

> **三家的扩窗都遵守同一条**：**扩展出来的邻居占 `top_k` 名额**（§11，PRD 原话也这么写）。

### B. 契约与运维

| 维度 | 我们 | ReFind | InvMem 候选 |
| --- | --- | --- | --- |
| 幂等键 | `applied_batches` 旁表（③-d） | `ingestions.request_id` + `payload_hash` | `add_requests` 表 + `payload_hash` |
| 同 id 不同 payload | **HTTP 409**（**D28**；事务整体回滚，第一次那份原样保留） | **HTTP 409** | **HTTP 409** |
| `created_at` 语义 | **事件时间**，日粒度 `YYYY-MM-DD` | **入库时间**，ISO 秒级 | **事件时间**，ISO 秒级 |
| 正文里放时间戳 | **放日粒度日期前缀**（`inject_abs_time`，**D21**）+ 相对表达**就地注解**成绝对日期（`annotate_relatives`，**D22**） | **放**：`[ISO秒] ROLE: ...` | **放**：`[role \| ISO秒] ...` |
| 返回字段 | `id`/`content`/`created_at`/`score` | 同四字段（`score` 带名次衰减 `/1+0.03r`） | 同四字段 |
| `top_k` | 不硬编码 100，按请求为上限 | 同（`ge=1, le=1000`，**声明建议 100**） | 同 |
| 未知字段 | **忽略**（未开 `extra="forbid"`） | **`forbid`**：多余字段 → 422 | 未核对 |
| 降级 | reranker 失败退回 RRF 序并计数 | agent 无 key → **显式报错**；空结果两级兜底 | reranker 分数非法 → 抛错 |
| 进程模型 | `--workers 1`（锁是进程内的，§15） | 声明**建议并发 16** | `--workers 1`（避免重复加载模型） |
| 数据保留 | **无删除机制** | **自动**：`RETENTION_DAYS ≤ 30`，启动时 + lifespan 各删一次 | **手动**：`scripts/memory_admin.py purge-prefix --user-prefix 'eval:<run_id>:' --confirm` |
| 正文进日志 | 否 | 否（只记聚合计数） | 否 |

---

## 3. 值得抄的细节

**按"能改变下一步动作"的程度排序。** 每条都给落点。

### L1 · S1 的间接证据：AML 大概率读 `content` 字符串

**两个系统返回的都是同样的四字段、都把正文放在 `content`、都进了榜前三。** 如果 AML 读的是 `text` 字段，它们的记忆会被整批丢弃、分数应该接近 0——所以 [`submission.md`](./submission.md) §S1 那张表里的"取 `text` → 整批丢弃"一栏，**大概率不是现实**。

> **这是旁证不是实测。** S1 仍要按原计划用 Smoke 清（§17.1），但**它的赌注变小了**：`content` 是字符串这一支有了两个独立作证者。**"按 dict 处理"那一支也仍未排除。**

### L2 · 正文里的时间：换"谁来算"——我们已测三轮，还有第四种做法没测

这是本次对照里**最值得动手**的一条。

先把三方规则摆在**同一张桌**上（逐字，引自归档的 AML pipeline）：

| 出处 | 原话 |
| --- | --- |
| 答案 prompt 第 7 条 | "Convert relative times like "yesterday", "last month", and "last year" into dates, months, or years **when the memory timestamp makes it clear**. Keep week-based expressions relative." |
| 裁判 TIME · 粒度 | "Granularity must match exactly: HOUR↔HOUR, DAY↔DAY, MONTH↔MONTH, YEAR↔YEAR. Do not answer a gold at a different time unit — even if the numeric value overlaps."（例：gold = "July 26, 2019" [DAY]；generated = "2019-07-26 08:09:17" [includes Second] → WRONG） |
| 裁判 TIME · 换算 | "**Do NOT convert relative ↔ absolute.** If the gold uses a relative time expression, the generated answer must also use a relative form (or a clear paraphrase of that same form), not a computed date/range." |

**这两条规则并不矛盾——它们各自以 gold 的形式为条件**：gold 绝对 ⇒ 必须给绝对（第 7 条让模型去换算）；gold 相对 ⇒ 必须给相对（裁判 TIME 第二条禁止换算）。**而模型看不见 gold** ⇒ 天生是一次下注。

#### 我们已经测过三轮，全部没能改变行为

`--memory-date` 的 `per_item`（逐条 `[YYYY-MM-DD]` 前缀）与 `header`（顶部再写明"每块前的日期是它自己那个会话的日期"）都试过，**"gold 绝对 / 答成相对"的题数几乎没动**；台账的结论是**"瓶颈不在我们怎么放日期，而在答案模型本身"**，指向 Step 5 换 `gpt-4o-mini` 后重测。反方向（gold 相对 / 生成绝对）也已数过，**在这份题集上为 0**。

> **数字、分桶与逐臂分数全在 [`../eval/reports/ledger.md`](../eval/reports/ledger.md)，本文不复制。**

#### 台账的解剖说明了为什么"看得见"≠"用得上"

那些答成相对的题里，**对话正文只有相对说法、没有任何绝对日期可抄 ⇒ 只能靠换算**。而块首挂一个 `created_at`，仍然要求模型自己做"把 `last week` 与那个日期对上"的日历运算——**这恰恰是它做不到的那一步**。

**候选仓库的 `temporal_enrichment` 正是冲着这一步去的**：它不把日期放在块首等模型去算，而是**就地写出解析结果**，且**粒度跟着表达走**（`yesterday`→日、`last month`→月、`last year`→年）：

```text
[user | Message date: August 11, 2026 at 14:12 UTC] last night I…
[Resolved relative dates: last night = August 10, 2026]
```

**⇒ 这一档已经落地**（`--memory-date annotate` / 生产开关 `packaging.annotate_relatives`，**D22**）：保留原文表述，再对正文里出现的相对表达**就地注解**成绝对日期，**粒度跟表达走**（week-based 仍给区间）。实测结果与逐题胜负见 [`../eval/reports/ledger.md`](../eval/reports/ledger.md)。

> ⚠ **它默认是关的，别误读**：候选仓库自报的那份公开数据成绩（**数字在 [`../eval/baselines/CLAUDE.md`](../eval/baselines/CLAUDE.md) 的纪律表里**）是 **0.6.0 冻结配置**跑出来的，而 `temporal_enrichment` **默认 false**，冻结配置里也没开。**那份成绩不是这个开关的功劳**。

#### 两个诚实的限制

1. **week-based 那一支大概救不回来。** 台账记着一批答案是 `last week` 这类，而答案 prompt 第 7 条**明文要求 "Keep week-based expressions relative"**——**那些答案是照规则做的**，却被"gold 是绝对日期"的裁判判错。这是**两条官方 prompt 在子集上的冲突**（§11.3 早记过），**我们的注入改不了它**；做这个实验时**别把那一支算进收益预期**。
2. **粒度只能跟着表达走。** 裁判 TIME 第一条卡的是粒度：把 `last month` 就地写成一个具体日子，会把一道 MONTH gold 变成 DAY 答案 ⇒ 反而判错。候选仓库那套"日/月/年各归各"的写法就是这个道理；**秒级前缀是另一个已知诱因**（§11.3）。

> §11.3 那条"不要在 content 里注入绝对时间戳"**没有被推翻**——它的依据是裁判规则本身，而两个系统的分数是**六数据集混合分**，在 BEAM 上时间规则**恰好相反**（[`../eval/CLAUDE.md`](../eval/CLAUDE.md)）。但两家都这么做且都在前三，说明**代价可能没有 §11.3 预期的那么大**。这条该由代理评测来量，不该由推理来定。

### L3 · 集合的身份：**缓存**那一层我们有了，**存储**那一层没有

**缓存（已有，且比候选仓库细）**：`EmbeddingCoordinate(model, render_template)`，一个坐标系一个缓存文件（`<cache_dir>/<坐标哈希>.db`），文件自描述，另有 `_verify_coordinate()` 校验（[`../src/tianximem/embed/base.py`](../src/tianximem/embed/base.py) 的 `EmbeddingCoordinate` / `DiskVectorCache`）。⇒ **换模型 ⇒ 缓存整体作废是自动的**，不靠人记得。候选仓库的坐标系串（`|context=v1:{n}`）也是这个思路。

**存储（没有）**：Qdrant 集合与 SQLite 真源**都没有身份标记**，看不出这个集合是由哪个模型建的。候选仓库把 `embedding_identity` 写进**记忆库本身**，启动时对不上就 `ValueError` **拒绝打开库**（`memory_store.py:202-212`、`:289-298`）——它护的是**向量库**，不是缓存。

**我们现在靠的是一份 runbook**（[`../deploy/CLAUDE.md`](../deploy/CLAUDE.md) §4：停服务 → 备份 SQLite → 作废缓存 → **按新维度建新集合** → 全量重 embed → 起服务重跑 T2）。**那是流程，不是不变量**——跳过"建新集合"那一步，没有任何东西会拦住你。

**残余风险的确切形状**：维度**不同**时 Qdrant 会拒绝（响）——所以 Step 5 那条路径本身有兜底。真正静默的是**同维度的替换**，或"只把一部分数据用新模型重写进旧集合"：向量空间不同，检索质量下降，**不报错**。⇒ 落点是**给集合一个身份**（集合名带标识，或集合里存一个身份点、启动时核对），不是补缓存校验。

### L4 · 排序不依赖到达顺序

ReFind 与候选仓库都**从数据里重建顺序**：先按源时间戳，再按 **`request_id` 里解析出来的 chunk 序号**（正则见 `store.py:50-58`），插入顺序只是最后兜底。ReFind 的示例 `request_id` 是 `eval:run:dataset:conv-0:chunk-0`——**序号确实在 id 里**。

> ⚠ **"序号确实在 id 里"这条，对我们的平台不成立**（**D28**）。
> 它来自两个**跑过真平台**的实现，比我们自己猜可信——但 2026-09-29 用请求原文采集抓到的
> **官方真实请求**是一个 `r_3115…` 形状的**不透明 id**（见 [`decisions.md`](./decisions.md) 的 D28）。
>
> **它的后半段仍然适用，而 D28 正是照它做的**：顺序要从**存储里的显式信息**拿，
> **不要靠插入顺序兜底**。⇒ D28 把邻接做成**显式链**（`prev_memory_id` / `next_memory_id`），
> 位置 = `(request_id, local_index)`，且**没有**业务串行锁
> （位置不由读-改-写产生 ⇒ 并发与乱序都安全）。

### L5 · 平台可能把整个样本塞进一个 `session_id`

ReFind 的改编说明第 3 条（`METHOD_CARD.md`）**逐字**：

> The paper's seen-session filter is applied at chunk level here. **The platform may assign one `session_id` to all chunks for a sample**; excluding that whole session after the first search would incorrectly hide the remaining memory.

**这是一条平台行为观察，我们没法本地验证。** 对我们的影响面：`session_id` 是我们的**配对分组键**、**位置作用域**（D24 起组合只看一次 Add，但**位置**仍按 session 有序；**D28** 起这个序是 Add 内显式链）、**扩窗作用域**（同 session 才扩）与**段合并分组键**。如果平台上"一个样本 = 一个 session_id"，那么这些机制全都还在跑，但"session"不再等于"一段对话"——**扩窗与段合并的语义会漂**。

> 好消息是**不致命**：三处都只要求"同一个 session 内位置有序"，这个不变量不会因粒度变粗而破。⇒ **已登记为 [S7](./open-questions.md)**（2026-09-25）。

### L6 · 提交申报表的字段清单（可照抄）

ReFind 的 `docs/SUBMISSION.md` 是一份**填好值的申报模板**：系统名 / 版本 / 评测类型 / 分组 / 参赛路线（Code — platform deployment）/ 公开仓库 / 许可 / 鉴权方式 / 三个端点 / API entrypoint / 容器端口 / 规划模型 / 建议并发 / 支持的 `top_k` 范围。

对照 [`submission.md`](./submission.md) §5 的提交包清单：**我们少了"容器端口、entrypoint、建议并发、支持的 `top_k` 范围"这几格**。跑 Smoke 之前要把它们定下来（我们是自托管还是交镜像，决定这一格怎么填）。

### L7 · 三套鉴权 header 都收，两套路径都注册

- ReFind 把 `/v1/memories/add` 与 `/add` **同时注册**（后者 `include_in_schema=False`），注释写明"兼容那些分开请求端点的评测器"；候选仓库的契约文档记着平台接受 `Authorization: Bearer` / `Authorization: Token` / `X-Api-Key` **三种**，它三种都收。
- **我们目前没有任何鉴权**（`service/` 里搜不到），而 ReFind 申报的正是 **"isolated deployment 内无鉴权"**——所以**无鉴权是合规的**，但这属于**申报表里要写明的一格**（L6）。
- **有一个小卫生习惯可以抄**：FastAPI 的**默认 422 会把出错的 `input` 原样回显**（缺字段时是整个父对象）⇒ 报错响应里带着原始正文。候选仓库专门注册了 `RequestValidationError` 处理器，把错误压成 `loc`/`type`/`msg` 三项再返回与记日志（`api.py:142-163`）。我们的**日志本来就不记正文**（`errors.py:56` 只记 method + path），所以这只是"响应体回声"这一处的取舍，**几行的代价，可以顺手做**。

### L8 · 数据保留：两个系统都有删除路径，我们一条都没有

AML 的数据处理要求（候选仓库 `docs/competition-contract.md` 记的，核对日期 2026-08-04）：**跑完 30 天内必须删除，除非主办方书面另许**。两家的形状不同——ReFind 是**自动**的：`RETENTION_DAYS`（代码里**强制 ≤30**，超了拒绝启动）在启动与 lifespan 各删一次；候选仓库是**手动**的：`scripts/memory_admin.py purge-prefix --user-prefix 'eval:<run_id>:' --confirm`（**不给 `--confirm` 就拒删**）。

**我们没有任何删除路径**（只有 preflight 用的 `drop_collection`）。这是**合规项**，不是优化项。

> **候选仓库那种"按 run 前缀删"照抄不到我们身上**：它的 `user_id` 里带 run 前缀（`eval:<run_id>:…`），而我们的 **`user_id` = `sample_id`**、`request_id` = `{user_id}|{session_id}|{index}`，**都不带 run**（[`../eval/datasets/preprocess.py`](../eval/datasets/preprocess.py) 决定 `user_id`，[`../eval/harness/batching.py`](../eval/harness/batching.py) 决定 `request_id`）。⇒ 我们的删除单位是**服务那一整套存储**（SQLite 真源 + Qdrant 集合），不是"某一次 run"。**"run 之间要不要换库"就是那个要和上面那条一起裁决的决定。**

> ⚠ **先别急着写工具，有一处需要团队裁决**：[`submission.md`](./submission.md) §5 把"**Full 的 SQLite 归档**"列为可复现材料，而 AML 的数据条款要求 30 天内删除。两者要么靠"书面另许"打通，要么归档里不能留原始正文。**这是个决定，不是个实现。**

### L9 · 契约有核对日期

候选仓库的契约文档**写明"checked on 2026-08-04"并附一句"正式跑之前回官网重核，运营限制可能变"**。

我们的 [`contract.md`](./contract.md) 是从归档反推的——**建议补一个"最后核对日期"字段**，并把"跑 Smoke/Full 前重核一次"写进 [`submission.md`](./submission.md) 的流程。零成本，且正好覆盖"S2 的 20 条/2,000 词怎么算"这类会变的东西。

### L10 · 无 key 的降级模式（我们已有等价物）

ReFind 留了 `RETRIEVAL_MODE=bm25` 走**不带 LLM 的确定性路径**，专供本地契约校验（`scripts/smoke_test.py`）；并且在 agent 模式下**没有 key 时显式报错**，防止"悄悄跑成了基线"。我们已经有两个等价物：[`../eval/smoke/preflight.py`](../eval/smoke/preflight.py) 与 [`../tools/t2_retrieval_dump.py`](../tools/t2_retrieval_dump.py)。**无须改**，记一笔是为了知道这条已经做到了。

---

## 4. 反向：我们比它们严的地方（别在"对齐"时改坏）

| 项 | 我们 | 它们 |
| --- | --- | --- |
| 未知请求字段 | **忽略**（`options` 因此不会 422，[`contract.md`](./contract.md) 已记） | ReFind 开 `extra="forbid"`，**多余字段直接 422** |
| 真源 / 索引分离 | SQLite 真源 + Qdrant，且有"SQLite 已提交但索引失败"的**修复路径** | ReFind 单库（BM25 现算，无索引可脱钩）；候选仓库向量存库内 |
| `k=61` 校验 | 配置层 + 策略层**两处冗余**，不等于 61 拒绝启动 | 无此校验（写死 60） |
| id 防碰撞 | `netstring` 长度前缀（`len:part`） | NUL 分隔（同样安全） |
| 扩窗可观测性 | 邻居数 / `dropped_missing` 都进返回值 | 未暴露 |

> **两家的 409 我们现在也有**（**D28**）：同 `request_id` 换 payload ⇒ **409**（`PayloadMismatchError`），事务整体回滚 ⇒ 真源里那一批仍是**第一次**投进来的那份（`tests/test_idempotency.py` 有用例钉着）。

---

## 5. 明确不要学的

| 东西 | 为什么 |
| --- | --- |
| **import 它们的代码** | §1：本项目从零搭建。`refind/` 只能被 [`../eval/harness/`](../eval/harness/) 当"另一个 Add/Search 服务"调（[`../eval/baselines/CLAUDE.md`](../eval/baselines/CLAUDE.md) 的隔离边界） |
| **转发它们的分数** | 自报 + 自建 harness + 公开数据子集（[`../eval/baselines/CLAUDE.md`](../eval/baselines/CLAUDE.md) 的纪律表）。**唯一能当基准的数是自己 harness 里跑出来的**（§12.3） |
| **把候选仓库当 InvMem 的证据源** | 无 LICENSE、映射未指认。**读出来的结论要写"某候选实现"，不能写"InvMem 的做法"** |
| **抄 `extra="forbid"`** | 它会因为平台多传一个字段就 422，而 422 在评测里等于这一条记忆全丢 |
| **ReFind 的秒级时间戳原样照抄** | 裁判的粒度规则卡的是**答案**里出现秒；给正文喂秒级前缀是**已知诱因**（§11.3）。候选仓库的**日粒度 + 就地解析**才是该学的那一半 |

---

## 6. 下一步（候选，需登记才有编号）

按"零成本 → 有成本"排：

| # | 动作 | 成本 | 落点 |
| --- | --- | --- | --- |
| 1 | 把 L1 的旁证记进 S1 那一行（**不改变"S1 仍未清"**） | 改几行 | [`submission.md`](./submission.md) §S1 |
| 2 | 补"契约最后核对日期 + 跑前重核" | 改几行 | [`contract.md`](./contract.md) / [`submission.md`](./submission.md) |
| 3 | 申报表补四格（端口 / entrypoint / 并发 / `top_k` 范围） | 一次决定 | [`submission.md`](./submission.md) §5 |
| 4 | 给 **Qdrant 集合**一个身份（**缓存那一层已有**，见 L3） | 几行代码 | `store/` |
| 5 | **`--memory-date annotate` 对照**（做法与前提见 L2） | ✅ **已做**（2026-09-26，**D22**） | [`decisions.md`](./decisions.md) D22 |
| 6 | 30 天保留：先裁决"Full SQLite 归档"与数据条款的冲突 | 一个决定 | [`decisions.md`](./decisions.md) 待决事项 |

> ✅ **原第 2 条已完成**（2026-09-25）：`session_id` 粒度与 Add 到达顺序**已登记为 [S7](./open-questions.md) 与 [S6](./open-questions.md)**——**只登记、未动代码**（口径见 S6 里那段"为什么没顺手加个探测器"）。
>
> **第 5 条的前提已经查过了**：反方向（gold 相对 / 生成绝对）在这份题集上为 **0**（统计在 [`../eval/reports/ledger.md`](../eval/reports/ledger.md)），所以它测的是**净收益**，不是"把错误从一边搬到另一边"。**但这只对 LoCoMo-Refined 成立。**
>
> ⚠ **注意它与"Step 5 换模型"的关系**：台账把日期注入无效归因于**答案模型（9B）本身不会做日历换算**。⇒ 换 `gpt-4o-mini` 之后**这一条要重测**——强模型可能自己就把 `created_at` 用上了，那时"就地解析"的边际收益会小很多。**两件事的顺序会影响归因。**
