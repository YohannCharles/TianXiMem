# 决策日志

> 最后核对：2026-09-24。**冲突以 PRD 为准**——本文件记录的是**决策的来龙去脉**，不是规格本身。

## 用法

这份文件存在的理由：PRD 是**当前状态**的规格，它不记录"哪些结论被推翻过、为什么"。而本项目的几个关键结论**恰好是从被推翻的判断里长出来的**——不知道它们被推翻过，就会有人照旧版重做一遍。

**因此改本文件所记录的任何决策之前**：

1. 先读该条的"理由"与"推翻过什么"
2. 若你打算推翻它，**先给出反例并提问**，不要直接改文档
3. 反例被接受后，**新增一条**记录（不要就地改写旧条）——旧条的价值在于它记录了"当时为什么这么想"

---

## D1 · 赛道与模型规定（2026-09-22 确认，2026-09-23 复议维持）

**决策**：赛道 = **开源方法组**；**模型规定照常适用**——embedding 只能 `text-embedding-v4`、LLM 相关组件只能 `gpt-4o-mini`、**reranker 不作规定**。

**理由**：出处是 `/competition/` FAQ 05。

**这条曾经被读错**：一度被读成"因为我们是开源榜，所以模型规定可能不适用"。**2026-09-23 用户明确确认适用**，该读法作废。

**已知的字面冲突**：`/rules` 另写着"不限定你使用的数据库、索引、向量模型或内部架构"。AML 的 issue #19 触及此事（**主体是超时/重试策略，模型规定只是第三问**）且**零回复**。

**处理**：**不据此放松**——按模型规定执行在**两种读法下都合规**（见 D2）。

---

## D2 · 开发期用本地模型替代（2026-09-22，团队主动接受）

**决策**：开发期用 **Qwen3-Embedding-8B + qwen3.5-9b** 跑代理评测以控 API 成本，**提交前切换到 `text-embedding-v4` + `gpt-4o-mini`**。

**代价**：embedding 与 agent 循环**两处的模型都被替换**，因此**本地标定出的所有阈值、权重、排序策略在切换后都不保证成立**。

**四条对冲**（PRD §12.1 R1）：① 按内容哈希缓存 embedding；② 切换后立刻重跑 **T2**；③ **所有阈值必须是配置项、不得硬编码**；④ **切换要单独占一个阶段**，不与任何设计改动同时进行。

**⚠ 一条 PRD §6.4 要求补入但尚未进对冲清单的**：本地 qwen3.5-9b 的分词器与 `gpt-4o-mini` 不同，**本地量出的"能装多少对"不能直接搬到线上**——**Step 5 切换后必须重新量一次**（见 `docs/open-questions.md` E7）。

**衍生后果**：**架构必须 embedder-agnostic**。开发期的 Qwen3-Embedding-8B 与提交期的 `text-embedding-v4` **都不提供** sparse 或 ColBERT 输出，**任何依赖多向量能力的代码都是死重**。集合的向量维度**必须由接口提供，不能写死**。

---

## D3 · 存储分层：SQLite 真源 + Qdrant 派生（2026-09-22）

**决策**：业务数据一张表 `qa_pairs` + 一张只增不改的旁表 `applied_batches`；Qdrant 为派生索引，**不存正文**。

**为什么不用 Qdrant-only**：那会让向量库成为**唯一真源，索引不可重建**；且 AML 只给 2 次 Full、**版本冻结**，把提交押在一次**不可逆的存储格式迁移**上不值得。**SQLite 的成本是一个文件。**

**为什么正文不复制进 Qdrant payload**：会有两份正文，一旦不一致**就无法判断该信哪一份**。而按主键批量取正文是微秒级操作，**没有性能理由去复制**。

**为什么必须 server 模式**：local 模式会**静默丢弃 payload 索引**（`create_payload_index` 只打一行警告就返回），且数据格式与 server 不兼容——而本项目的三个筛选全部依赖 payload 索引。

**已被删除的字段**：早期版本留过 `kind` / `parent_id`（给 Fact 抽取用）与一个无定义键的杂项袋 `meta`。**三者已一并删除**，理由相同——**预留字段既不入索引也不进 `ORDER BY`，正属于"不该是列"的一类。真要做时再加。**

---

## D4 · 幂等必须分两层（2026-09-22 **二审推翻原论证**）

**决策**：新增 `applied_batches` 旁表做**批次级守卫**。内容层靠"填空 + 追加、绝不覆盖"；**批次层必须查旁表**。

**推翻的是什么**：早期版本认为"只填空不覆盖"**足以**免掉判重。**这个论证是错的。**

**为什么错**：「只填空不覆盖」管得住**内容**，管不住**位置**。`pair_idx` 的分配是**读-改-写**——若服务在事务提交之后、响应发出之前崩溃（或响应丢失），AML 会重试同一批（`request_id` 与 payload 不变），而此时 `MAX(pair_idx)` **已经前移**，重试会把**同一批消息重新分配到新的 `pair_idx` 上**，落成一份重复记录，**且不会报错**。

**为什么不能复用 `qa_pairs.request_id` 做守卫**：那一列会被**后一批覆盖**（后一批触碰过的每一行都会改写它）——若某批唯一触碰过的行随后被下一批改写，**这批的指纹就丢了**，守卫查不到、重试照旧重复应用。**旁表让它永不丢失。**

**连带影响**：`answer` 允许追加（append-only）的**安全性由这条守卫保证**——同一批至多被应用一次，因此不需要额外的判重逻辑。**别把两者混为一谈：它们解决的是两个不同的问题。**

---

## D5 · RRF 的 `k` = **61**，不是 60（2026-09-22 二审更正）

**决策**：`{"rrf": {"weights": [...], "k": 61}}`。

**更正了什么**：PRD 早先写 `k = 60`。

**为什么是 61**：**Qdrant 的秩是 0-based**（官方文档逐字："the top result has r_d = 0"），公式为 `1/(rank + k)`；而 RRF 文献是 1-based 的 `1/(60 + rank)`，首位即 `1/(0+61) = 1/(1+60)`。官方调参文的原话："Qdrant defaults to k=2. The original RRF paper uses 60, **which maps to k=61 in Qdrant's formula**."

**两个坑叠在一起**：
- **Qdrant 的默认值是 `k = 2`**——不设就会得到一个与所有参考实现都不同的融合行为，**且极难排查**
- 文献的 60 **不能直接填**，必须换算成 61

**定位**：`k=61` 是**正确性常量，不是调参项**。不要把它放进"可调阈值"那一类。

**版本门槛**：`k` 自 v1.16.0 起可用，`weights` 自 v1.17.0 起可用 → 要求 **≥ v1.17.0**，部署时**钉死版本**。

**✅ 已在钉死版本上实测确认（2026-09-23，compose 起的 Qdrant v1.17.0 容器）**：

| 请求 | 返回 score | 含义 |
| --- | --- | --- |
| `rrf: {weights:[1.0,1.0], k:61}` | `0.032786883` | = `1/61 + 1/61` ✓ 与文献 `1/(60+1)`（1-based）逐项一致 |
| `rrf: {}`（不设 `k`） | `0.5` / `0.33333334` | = `1/(0+2)` / `1/(1+2)` ✓ **默认 `k` 确实是 2** |

**两种写法的量级差约 30 倍**（单路首位贡献 `1/2 = 0.5` vs `1/61 ≈ 0.016`；本例两路合成后 `0.5` vs `0.0328`）——**而这正是"极难排查"的具体形状**：两种写法产出的**名次都看起来正常**，只有在分数被任何下游逻辑碰到时（阈值、跨方法比较、日志判读）才会暴露。**`k=61` 由此从"引官方文档"升级为"本版本实测"。**

---

## D6 · 裁判 TIME 块是三条独立规则，示例曾错挂（2026-09-22 二审更正）

**更正了什么**：PRD 早先把**两条规则的例子错配在了一起**——`July 26, 2019` 那个示例原本挂在"禁止相对↔绝对"上，**它其实属于"粒度严匹配"那一条**。

**更正后的内容**：

| 规则 | 惩罚什么 |
| --- | --- |
| 规则一：**粒度严匹配** | **粒度变细**（DAY → Second）。gold 的粒度就是回答的粒度上限——**与 gold 是相对还是绝对无关** |
| 规则二：**禁止相对↔绝对互转** | gold 是相对表述时，模型算成日期。**原文没有配任何示例——不要自己编** |
| 规则三：宽容条款 | `the/last/previous/just prior` 这类修饰词，锚点日期与单位相同时视为等价 |

**为什么这条更正重要**：**它让"不要注入绝对时间戳"从一条理由变成了两条独立机制。** 早先只知道规则二，漏掉了规则一——而**规则一与 gold 是否相对无关**，覆盖面更广。

**推论**：T1 实验若测出差异，**归因时不要默认只来自其中一条**。

**适用范围必须限定**：这条来自 LoCoMo-Refined / LongMemEval 共用的那套契约，**不是全赛道的规则**——BEAM 的裁判正好相反（明文允许 `$68,000` / `68k` / `sixty-eight thousand dollars` 等价），CL-Bench 由 AML 侧主动注入时间戳。**代理评测按"不加"执行，但不要把这个结论推及全赛道。**

---

## D7 · 砍 Query Analyzer；实体层收进附录 A（2026-09-22）

**决策**：v1 **不设** Query Analyzer；实体 / 别名归并层属 v2，收在附录 A，**待 T2 判定是否投入**。

**Query Analyzer 的职责去哪了**：它"要不要进 agent"的那部分**由 Evidence Checker 承担**，不另设组件。

**实体层待判定的理由**：真正的理由是**绝对占比**——正文口径下计数类共 **21 题**，**只占全数据集（1,382 题）的约 1.5%**。先做 T2，若主因是"措辞不同导致漏召"才值得做。

**⚠ 一个易混点**：PRD §5 图里的"**关键词重写**"是 **Agent 每轮自己产出的检索关键词**，属于 agent 循环，**不是**被砍掉的 Query Analyzer。§7.2 说的"查询改写属 v2"指的是**检索之前那一次独立的 query 改写**。**两者别混：前者 v1 有，后者 v1 没有。**

> ⚠ **末句已过时（2026-09-23）**：见 D13——关键词重写归 v2，**v1 两条都没有**。上面的区分仍然成立，只是"前者 v1 有"这个标签作废。

---

## D8 · 战略转向：排序是主线，检索降级（2026-09-22）

**决策**：Hybrid Retrieval **降级为待验证假设**（必须有 BM25-only 对照）；**Rerank + Context Packaging 升格为主线**。

> ⚠ **首句已被推翻（2026-09-23）**：见 **D15**——混合检索是**既定的检索形态**，裸 BM25 模式与那条 hedge 都已删除，**"必须有 BM25-only 对照"不再成立**。
> **本条其余部分仍然有效**，且是 D15 想保住的东西：**"排序才是主线"这个判断没有被推翻**，四条证据也仍然成立（其中第 3 条的处境见 D15 的"残余风险"）。
> 保留原文不改，是为了存档"当时为什么这么想"。

**四条独立证据**（详见 PRD §4）：

1. **LongMemEval 的检索召回已接近天花板**（~96–99% R@5/R@10），但端到端 QA 只有约 60–95%——**缺口在"留哪些、按什么顺序留"**
2. **ActiveMemoryIndex 的自测**：排序改动是其找到的**最大单一杠杆**（**限定**：本地 harness + 本地 judge 的 LoCoMo 结果，**非平台分**）
3. **ReFind 的消融**：dense 与 hybrid 都打不过纯 BM25（**限定**：GPT-5-mini backbone 下的 matched 子集消融，**不能直接推及 §9 的 `gpt-4o-mini` 场景**）
4. **AML 契约**：答案阶段按返回顺序取 117,760 token 前缀——**排序直接决定什么被截掉**

**为什么 reranker 是唯一还能加码的地方**：§2.3 把 embedding 与 LLM 两处都钉死了，**只有 Reranker 不作规定**。

---

## D9 · 本地代理数据集的选择（2026-09-22）

**决策**：用 **LoCoMo-Refined + LongMemEval**；LongMemEval 用 **`lme_s_cleaned.json`**。

**理由**：二者均在 AML 官方数据集清单内、均可公开下载，且覆盖"饱和"与"未饱和"两端。且这两份**恰好是全部六份里唯一共用同一套契约的**。

**为什么不用 `lme_test.json`**：与前者**同题、同证据，只在 haystack 上不同**——后者多出 1,230 个空 session 与 15 个干扰 session，**而空 session 会污染按"20 条消息"切批的埋点逻辑**。

**为什么 ScriptMem / MemoryAgentBench / BEAM 不能当代理**：ScriptMem 对话原文**因版权原因未发布**；MemoryAgentBench **不在 AML 的数据集清单里**（在其上调优未必迁移）；BEAM 的**数据实际上不在归档里**（`beam.json` / `beam_rows.json` 是失败下载的残留，`beam_100k.json` 是 HuggingFace 的分页响应，**不是数据集**）。

---

## D10 · "计数类问题占比"改为可复现口径（2026-09-22 复算不通过）

**更正了什么**：早先写的"单跳类计数问题占 **3.1%**"**复算不通过**。

**处理**：改为**可复现口径**，并**在 harness 里先固定口径**——因为**换口径数字就变**，而它正是附录 A"实体层做不做"的依据。

**两个口径的对照**（说明为什么必须写清口径）：

| 口径 | 单跳类 | 多跳类 |
| --- | --- | --- |
| `how many\|how much\|how often\|number of\|count\|how long` | 3.4%（27/802） | **9.9%**（21/213） |
| 只算 `how many` | 0.25% | 9.39% |

---

## D11 · 脚手架技术选型（2026-09-23）

**决策**：Python + **FastAPI/uvicorn**；**uv + pyproject.toml**；**骨架只建目录 + README，不建任何 `.py` 文件**；范围 = 服务骨架 + 评测骨架。

**理由**：FastAPI 自带请求/响应模型校验，§2.1 的契约形状（精确 `top_k`、原样回显 `request_id`）用模型钉死最省事。uv 提供锁文件，而 §7.3 明确要求**把 Qdrant 版本钉死**。骨架深度取最轻，避免在 Step 0 之前就固化了模块签名。

**两个已知环境落差**（本机，需在 Step 0 解决 —— **⚠ 两条均已由 D12 解决，保留原文以存档当时判断**）：

| 落差 | 为什么是问题 |
| --- | --- |
| **无 `docker`** | §6.3 明确**必须用 Qdrant server 模式**，**无法用 local 模式替代**（会静默丢弃 payload 索引） |

**另一项**：本机默认 Python 3.14.4，已按 `requires-python = ">=3.11,<3.14"` 保守钉在 3.12（torch / qdrant-client 的 wheel 覆盖通常滞后）。

---

## D12 · 运行形态：模型远程，服务与 Qdrant 在本机（2026-09-23）

**决策**：**LLM / embedding / reranker 三者全部经自建网关（`memory2.021130.xyz`）远程访问**；**检索服务与 Qdrant（server 模式）都跑在开发机上**。因此本机**不需要 GPU，但仍需要 docker**。

**它解决了什么**：

| 原记录（D11 的两条"环境落差"） | 处置 |
| --- | --- |
| **无 `docker`** | ⬜ **仍在** → ✅ **同日已解决**（见下"同日补充"） |

> **⚠ 更正了什么（同日、条内更正）**：本条**初稿是错的**——当时写成"Qdrant 也跑在服务器上，因此本机不需要 docker"，并据此把 docker 标记为"消失"。**该读法作废**：本机的向量库就在本机。
>
> **连带作废**：期间提出的"用 Qdrant 独立二进制绕开 docker"**也不再需要**——不是绕，是按正常方式装 docker。

**reranker 的部署不由本项目负责**：目前尚未部署，后续部署，**其宿主环境不在本项目范围内**。本项目只依赖它的 HTTP 端点。唯一要守的纪律是 **reranker 选型在提交时保持不变**——开发期选小的、提交期换大的，等于在 Step 5 引入设计改动，**正好违反"Step 5 不可与任何设计改动合并"**（D2 对冲 ④）。

**衍生出两条 v1 必须实现的约束**：

1. **reranker 必须可降级**：它是**唯一位于提交链路关键路径上、又不被规则保证可用**的组件（embedding 与 LLM 由规则规定、由厂商提供）。Full 只有 2 次、第二次隔 30 天，等于一次真机会。**端点失败或超时时，服务退回未重排的 RRF 顺序返回**，而不是报错——契约要求"≤ `top_k` 且精确计数"，超时路径也得在预算内产出合法响应。
2. **开发环路有单点**：answer / judge / embed / rerank **四条**都打同一个网关，而 harness 评测时会同时驱动 Add/Search（embed + rerank）与答案/裁判生成。**必须有客户端并发上限**，否则排队超时会伪装成"模型变差了"（见 `docs/open-questions.md` V8）。

**⚠ 一条被本决定修正的旧记录**：D11 里那两行"本机环境落差"**在当时是真实的**（当时假定模型跑在本机），故保留原文不改——但**不要再按它们去找显卡**；**docker 那条同日已办**（见下）。

**同日补充（2026-09-23 晚）——`docker` 已就位**：上表记为"仍在"的 `docker` 阻塞项**已解决**。本机装了 **Docker Desktop 29.8.0**（`desktop-linux` context，WSL2 后端；`wsl -l -q` 只有 `docker-desktop` 一个发行版，**没有额外用户发行版**）。

已核实的三个含义：

- **daemon 走 Windows 命名管道**（`npipe:////./pipe/dockerDesktopLinuxEngine`）⇒ `docker` 命令**直接从 Git Bash 可用**，**不必进 WSL**
- 容器虽在 Docker Desktop 的 VM 内运行，但**发布端口转发到 Windows 主机** ⇒ 服务在本机访问 `localhost:6333` 成立，`.env` 的默认值**不用改**
- **`qdrant/qdrant:v1.17.0` 的 tag 已核实存在于 registry** ⇒ §7.3 的版本门槛成立（`rrf.weights` 自 v1.17.0 起可用）

⇒ **Step 0 的环境阻塞项只剩 `api_config.py` 一条。**

---

## D13 · v1 不做 agentic：Checker 退化为带日志的空实现（2026-09-23）

**决策**：**v1 默认"证据充足"，不实现任何 Agentic 功能，没有证据补充。** Evidence Checker **存在**，但实现为**恒返回"充足"、且必须记录每轮判定**的空实现。Agentic Search 整块（含**关键词重写**）归 v2。

**它改了什么**：

| 原记录 | 现状 |
| --- | --- |
| D7：「**两者别混：前者 v1 有，后者 v1 没有**」 | **该标签失效**——关键词重写属于 agent 循环，随 agentic 一并归 v2，**v1 两条都没有**。但 **D7 的区分本身仍然成立**（前者是 Agent 自产的检索词、后者是检索前独立的 query 改写），废掉的只是"v1 有"这个标签 |
| `src/tianxi_am/agent/CLAUDE.md`：「这是本项目的核心 claim，**不可砍**」 | v1 **不砍，但也不实现**——见下 |

**为什么 Checker 仍要做成空实现（而不是干脆不写）**：

1. **接缝先存在**，Step 4 到来时是**插入**，不是重写拓扑；
2. **免费拿到反事实分布**——它每轮记下"本来会不会触发多轮搜索"，而这正是回答 **A4「agent 值不值」**（E4）所需的证据。**不记，Step 4 之前就永远无法用数据回答这个问题。**

**v1 的代价（必须正视）**：没有证据补充路径，**召回变成一次性的天花板**——正确的 QA 对不在初始候选里就永久丢了。因此 **v1 的全部重量压在"排序 + token 预算分配"上**。这与 D8 的战略转向一致，但压力更集中。

**⚠ 核心 claim 的处境变了**：§3.1「只有证据不足时才触发多轮搜索」是项目的核心 claim 之一，而 **v1 里它是平凡成立的（因为根本没有 agentic）**，不是被设计出来的。若最终材料要引用这个 claim，**必须由 v2 或 A4a/A4b 实验支撑，不能引用 v1 的分数**。

---

## D14 · 模型部署在服务集群上（2026-09-24）

**决策**：三段模型（qwen3.5-9b / Qwen3-Embedding-8B / Qwen3-Reranker-4B）**部署在另一个服务集群上，不由本项目运维**——本项目只依赖它们的 HTTP 端点。**显存与显存预算不在本项目范围内**，PRD §1 / §11.2 / §12.1 里相关表述已一并删除（本节前身是"显存预算表延后 / 有意偏离"，2026-09-24 作废）。

## D15 · 混合检索无条件使用：删掉裸 BM25 与"验证后再加 dense"的分阶段（2026-09-23）

**决策**：**检索只有一种模式——混合（BM25 + Dense 两路 `prefetch` → Weighted RRF）。不使用裸 BM25。** Dense 与 Weighted RRF 归 **Step 1**，Step 2 只剩 Neighbor Expansion 与双预算截断。

**取代 D8 的那条 hedge**（"Hybrid Retrieval 降级为**待验证假设**，必须有 BM25-only 对照"）——hybrid 是**既定的检索形态**。

**连带消失的三项**：§8 的 Checker 退化路径问题（dense 永不关 ⇒ Checker 永不退化）、§13 的 **A1 / A2 两个对照**、`open-questions.md` 的 **E2** 与"待决事项 2"。**理由是 §13 自己的原则**：「若一个实验的两种结果会导致同一个下一步动作，它不该做」——参照点改由**混合主路径自身**承担。

**保留的部分**：`dense` 开关**仍是配置项**（§15 要求所有消融项可配），只是**不再有下游依赖**；Checker 的开关 `checker.enabled` 也必须可配（§15 的清单里原本漏了它，因为 §13 的 A4 要关它做对照）。

**残余风险（写在这里，以免它消失）**：D8 的四条证据里，**第 3 条**是"ReFind 的消融显示 dense 与 hybrid 都打不过纯 BM25"。**撤销 BM25 对照意味着这条反证不再被本地检验。** 判断依据是：那条证据的限定条件很窄（**GPT-5-mini backbone 下的 matched 子集消融**，不能直接推及 `gpt-4o-mini` 场景），而 D8 的第 1、4 条（LME 召回接近天花板、AML 按 117,760 token 前缀截断）才是主推。**⇒ 若日后分数不及预期，第一个该复检的就是这一条。**

---

## D16 · 数据路径统一走 `TIANXI_BENCHMARK_DIR`；LoCoMo 的 Add 源改用 `conversations.jsonl`（2026-09-23，**决策二 2026-09-24 修订**）

**决策一 —— 路径口径**：**数据路径一律通过 `TIANXI_BENCHMARK_DIR` 读取，代码中不得硬编码 `benchmark_data/` 或 `eval/datasets/`。** 默认值仍是 `benchmark_data/`；本地开发通过 `.env` 指向实际数据目录。

**边界**：`eval/datasets/LoCoMo-Refined/data/` 是**开发与自测用**的数据，**不是最终要跑的数据集**——归档才是。两者不可混为一谈，**也不得让代码依赖任何一边**。

**决策二（2026-09-24 修订）—— LoCoMo 的 Add 对话源**：~~**改用 `data/public/conversations.jsonl`**，不再用 `locomo_refined.json` 的 `conversation`。~~

> ### ⚠ 修订原因：**归档里根本没有 `conversations.jsonl`**
>
> `tools/fetch_benchmark_data.py` 的清单**明写它不收录**（原文："data/public/conversations.jsonl，本清单不收录**——它属 eval/datasets/ 的 clone（D16）"），
> 所以**纯归档路径下这条决策无法执行**。原决策默认了 clone 在场。
>
> **改后的口径**：加载器**两个来源都支持**——`conversations.jsonl` 在场时优先，
> 否则退回 `locomo_refined.json` 的 `conversation` **并逐条 `strip()`**。
> **两者产出的 `Message` 流逐字相同**（见下表的实测，2026-09-24 在**全量**上复核过）；
> 由 `tests/test_datasets.py` 的 `test_locomo_two_source_layouts_are_identical`
> 与 `test_archive_locomo_matches_conversations_jsonl` 钉住（后者在 clone 不在场时 skip）。
>
> **原"采用理由"的落点变了**：它说的是"用 `locomo_refined.json` 会引入 209 条契约违规"——
> **那句的前提是"不 strip"**。归一化层（[`eval/datasets/preprocess.normalize_content`](../eval/datasets/preprocess.py)）
> 对**两条路径都强制 `strip()`**，所以那条违规在新口径下**不可能出现**。
> ⇒ **契约合规不再依赖"选哪个文件"，而依赖"加载层做了归一化"**（后者才是唯一实现处）。
>
> **净效果：不再需要 `eval/datasets/` 的 clone 也能跑代理评测**——归档单独就够。
>
> **⚠ 这条修订没有推翻"落差 #1"**：`locomo_refined.json` 仍是 pretty-printed JSON 数组、
> 而归档 pipeline 的 `rows()` 只解析 JSONL——只是**那层转换不该由 Add 源来承担**：
> harness 自己按 pipeline 的契约构造 JSONL 输入（[`eval/harness/judge.py`](../eval/harness/judge.py) 的 `build_input_items`）。

**已实测两项前置验证（均通过）：**

| 验证项 | 结果 |
| --- | --- |
| **`role` 取值** | 只有两个值——`user` 2,951 / `assistant` 2,931。**且与 `speaker_a`/`speaker_b` 100% 一致**（speaker_a→`user`、speaker_b→`assistant`，零例外、零缺失） |
| **逐条内容等价性** | 5,882 条中：**5,673 完全相同 · 209 条仅首尾空白不同 · 0 条内部空白不同 · 0 条真实内容不同**；且 209 条**全部同向**——`conversations.jsonl` 的 text **恰为** `locomo_refined.json` 的去首尾空白版 |

**⇒ 采用理由不是"格式方便"，而是它同时修掉一类契约违规**：[`tests/CLAUDE.md`](../tests/CLAUDE.md) 要求 content **首尾无空白**（因为 AML 只做 `"\n".join(...)`、不插分隔符）。**用 `locomo_refined.json` 作源会直接引入 209 条契约违规**；`conversations.jsonl` 天然合规。

**顺带消掉的落差**：归档文档记的"落差 #1"是——**`locomo_refined.json` 是 pretty-printed JSON 数组，而 pipeline 的 `rows()` 只解析 JSONL**。改用 `conversations.jsonl`（**本身即 JSONL**，一行一个 conversation）后，**这层格式转换不必写**。

**⚠ 残余注意（不得忽略）**：LoCoMo 是**两个真人在对话**，此处的 `user`/`assistant` 是**数据集对 `speaker_a`/`speaker_b` 的约定标签，不是"用户 vs 助手"**。所以在 LoCoMo 上配对规则实际是"**speaker_a 的一轮 + 对方回应，直到 speaker_a 的下一轮**"。这不构成问题（与 ReFind 的 turn 粒度一致，消融对比才干净），但**不要把它当成真用户会话**——**LongMemEval 的 `role: user` 才是真 user，两边语义不同。**

**量级预期**：LoCoMo 的 `user` 消息 2,951 条 ⇒ 单独的 QA 对数约 **2,951**（实际略少：session 边界与 assistant 开头的 session 会产生无问的对，且一条 user 会被下一条 user 关掉）。

**⚠ 一条边界（不得跨越）：`conversations.jsonl` 只作加载层的输入源，不成为核心系统格式。**

`pairing/` 与 `store/` 只能看见 **AML 契约的形状**——`messages: [{role, content, timestamp?}]` + `user_id` / `session_id` / `request_id`。**知道 `conversations.jsonl` 存在的，只允许是 [`eval/datasets/`](../eval/datasets/)。**

**理由**：同一套 `Add` / `Search` 最终要接 **LongMemEval / PersonaMem / BEAM 等不同数据集，以及未来的真实 API 请求**。LoCoMo 的三点便利（本身即 JSONL / 自带 `role` / 带 `session_date_time`）**全部止步于加载层，不得向上渗透**。

> 这与 D2 推出的 **"架构必须 embedder-agnostic"** 是同一个原则的另一面：**核心系统只依赖契约，不依赖任何具体的输入来源。**

**连带的时间安排**：**先用 LoCoMo 把链路搞通**，LongMemEval 等数据集的适配**放到链路跑通之后**，不阻塞开发。

---

## D17 · `SqliteStore` 的连接模型：**短生命周期连接** + 交给 SQLite 自己串行化（2026-09-24）

**决策**：`SqliteStore` **不长期持有** `sqlite3.Connection`。连接的生命周期恰好是**一次逻辑操作**：

```text
写： connect → BEGIN IMMEDIATE → 读改写 → COMMIT / ROLLBACK → close
读： connect → SELECT                                        → close
```

**不用 thread-local 长连接，也不用单一共享连接。** 本类只保存 `db_path` 与存储逻辑。

**它修的是什么（③-c 发现的硬阻塞）**：`SqliteStore` 原持有**单个** `sqlite3.Connection`
（在启动线程里建），而 FastAPI 的 `def` 路由跑在**线程池**里 ⇒ **每个请求**都抛
`sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that same
thread.`。`TestClient` 也在另一个线程里跑 app，所以**所有 HTTP 级用例**同样跑不了。
⇒ **这不是测试问题：服务层当时根本跑不起来。**

**三条被排除的路**：

| 候选 | 为什么不用 |
| --- | --- |
| `check_same_thread=False` **单独用** | 一个连接上并发 `BEGIN IMMEDIATE` 会抛 "cannot start a transaction within a transaction"；且会让**不同 session 互相阻塞**（那正是 `SessionLocks` 要避免的） |
| **thread-local 长连接** | 能修好线程问题，但把"连接与线程的关系"变成一份**需要被记住**的长期状态：线程池会回收/新建线程，`close()` 的时机就没有主人了 |
| **服务层包一层 per-thread provider** | 要把 `AddPipeline` / `SearchPipeline` / `rank.package` 的依赖从 `store` 改成 provider，**每个上层都要懂这条约束** |

**短生命周期为什么更好**：连接**永远在同一次调用的同一个线程里建与关** ⇒ "连接与线程"
这条约束**不需要泄漏到任何上层**，也不需要谁记得清理。代价是每次操作多一次 `connect`
（约几十微秒）——相对一次远程 embedding（~百毫秒）可忽略。

**PRAGMA 分两层（这是本条最容易写错的地方）**：

| 层 | 项 | 落点 |
| --- | --- | --- |
| **数据库级**（写进库文件、对所有后续连接持续生效） | `journal_mode = WAL` | **只在 `open()` 里执行一次**——重复执行只是白跑一次 IO |
| **每连接级**（新连接**不继承**） | `busy_timeout` / `synchronous` / `foreign_keys` / `row_factory` | **每个新连接都要设**（`_connect()`） |

> ⚠ **不能"启动时配一次、然后假设所有新连接自动继承"**——`foreign_keys` 尤其如此：
> SQLite 的默认值是 **OFF**，而它是每连接生效的。
> （当前 schema 里没有外键 ⇒ 它现在是**空转**，但留着它，将来加 FK 时不会静默失效。）

**并发分工：两层锁，管的是两件不同的事**（不叠加、也不互相替代）

| | `SessionLocks`（`service/`） | SQLite writer 串行化（`store/` + SQLite 自己） |
| --- | --- | --- |
| 键 | `(user_id, session_id)` | 整个库文件 |
| 保证 | 同一 session 的 Add **业务顺序**（§15） | 同时只有一个写事务；输的一方**等** `busy_timeout`，不抛 `SQLITE_BUSY` |
| 范围 | **进程内** ⇒ 必须 `--workers 1` | 跨连接、跨线程（同一进程内） |
| 粒度后果 | 不同 session **可以并发** | 并发进入的多个 session 在 SQLite 处**排队** |

**不新增应用层写锁**：writer 串行化由 SQLite 负责。只有在压力测试**真的**出现
`SQLITE_BUSY` 或高尾延迟时，才考虑加应用层写锁**作为优化**——**不作为正确性基础**。

**`BEGIN IMMEDIATE` 是这条结论的前提（已实测，2026-09-24）**：把 `transaction()` 里那一行
换成默认的 `BEGIN`，`tests/test_store.py` 的两条并发压力用例**双双失败**，报
`OperationalError('database is locked')` ×5。两个**独立**原因：

1. `pair_idx` 的分配是**读-改-写**，deferred 事务在第一次写时才拿写锁 ⇒ 两个事务读到同一个 `MAX(pair_idx)`
2. **升级写锁时 `busy_timeout` 不生效**：已持读锁的事务要升级成写锁、而对方正持写锁时，
   SQLite **立刻**返回 `SQLITE_BUSY`——它宁可立刻报错也不冒死锁的险

> **第 2 条比第 1 条覆盖面大得多**：实测里**跨 session 的那条用例也失败**——即使两个事务
> 写的是**不同的** `(user_id, session_id)`（UNIQUE 键毫不相干）也一样撞。
> ⇒ **"排队而不是互锁"这件事本身就是 `IMMEDIATE` 换来的**，不只是防位置撞车。

**连带修掉的同类问题**：`embed/base.py` 的 `DiskVectorCache` 有**同一个 bug、同一个形状**
（构造时建连接、之后长期复用），只是被 `SqliteStore` 那一份挡住了、测试没抓到
（并发用例当时在 `SqliteStore` 上就抛了）。**两处必须同一套模型**，否则修好一处、
另一处照样炸。它的 `journal_mode=WAL` 也只在构造时落一次。

**回归覆盖**（原先是 6 个 `xfail(strict=True)`，**已全部摘掉**）：

| 用例 | 覆盖 |
| --- | --- |
| `test_contract.py` 三个 HTTP 往返 | 真的 `TestClient` 往返：`/search` 字段集合、`/add` 回显、非法请求 422 |
| `test_service_add.py::test_same_session_is_serialized` | 同 session 串行（peak 并发 = 1）且 `pair_idx` 无重复无空洞 |
| `test_service_add.py::test_different_sessions_do_not_block_each_other` | `Barrier(2)` 证明不同 session **能同时在** `index_pairs` 里 |
| `test_service_add.py::test_different_users_do_not_block_each_other` | 不同 user 不互相阻塞 |
| `test_service_add.py::test_different_sessions_write_to_sqlite_concurrently` | **走完整 Service 路径**的并发真写：无丢批、每 session `pair_idx` 连续、无跨 session 污染 |
| `test_store.py::test_concurrent_write_transactions_across_sessions` | **压力**：6 线程同时 `BEGIN IMMEDIATE`，持写锁 10ms 制造必然排队 ⇒ 无 `SQLITE_BUSY`、无嵌套错误、无丢失、无污染 |
| `test_store.py::test_same_session_concurrent_writers_never_take_the_same_pair_idx` | **压力**：6 线程写**同一个** session ⇒ `pair_idx` 连续无重复。**这条才是 `BEGIN IMMEDIATE` 的必要性用例**（上面那条跨 session 的线程各写各的键，但它实测也会失败——见上文第 2 条原因） |
| `test_store.py::test_store_is_usable_from_another_thread` | 另一个线程里读+写都正常（旧模型的**直接**回归用例） |
| `test_store.py::test_each_operation_gets_its_own_connection` | 两次 `read()` 拿到的不是同一个连接对象 |
| `test_store.py::test_transaction_rolls_back_and_closes_on_exception` | 异常路径：**先回滚**（半批不落库）**再关连接**（不泄漏） |

**一处纪律**：同一事务里调用的每个 helper **必须复用外层传进来的那个 `conn`**——
helper 自己 `connect()` 会落到另一个事务里（拿不到写锁、也看不到未提交的中间态），
helper 自己 `commit()` 则让"半批"落库。**两种情况都不报错。**

---

## D18 · 网关环境变量的语义**以本项目为准**：`AML_EMB_*` = embedding（主网关）（2026-09-24）

**决策**：`.env` 里这几个名字的语义**按本仓代码的读法定死**，不按网关文档的运维命名：

| 变量 | 指向 | 谁读它 |
| --- | --- | --- |
| `AML_EMB_BASE_URL` / `AML_EMB_API_KEY` / `AML_EMB_MODEL` | **主网关** `memory.021130.xyz`（**Embedding**） | [`service/settings.py`](../src/tianxi_am/service/settings.py) 的 `ServiceSettings.from_env()` |
| `AML_BASE_URL` / `AML_API_KEY` / `AML_MODEL` | **memory2** `memory2.021130.xyz`（**LLM 对话**） | harness / 归档 pipeline（经 `api_config.py` 适配器） |
| `TIANXI_RERANKER_BASE_URL` / `_API_KEY` | **主网关**（**Reranker**） | Step 3 的 `rank/reranker.py`（未实现） |

**冲突来自哪**：网关文档（`L20-推理服务API.md` §2.1）写着"**变量名 `AML_EMB_*` 是 memory2 的固定命名，不要改**"——
在**他们那边** `AML_EMB_*` 指内存网关（对话）。而本仓把 `AML_EMB_*` 读成 **embedding 端点**。**语义相反。**

**为什么以本项目为准**（而不是反过来改我们的代码）：

1. **那些变量是我们自己的 `.env` 里的，唯一的消费者是本仓代码。** 网关文档那句话约束的是
   **他们运维侧的 env 文件**（`/data/hechj/AML_Model/llm/etc/llm.env`、`etc/bge-m3.env`）——
   "改名会让已配置的调用方连不上"说的是**他们的**调用方，不是我们的变量名。
2. **反过来做会静默坏掉**：把我们的 `AML_EMB_*` 填成 memory2，服务每次 embedding 都打对话网关
   ⇒ **404**（文档自己写着"调错域名只会拿到 404"），而**服务启动不会失败**——它只校验变量非空。
3. **能力确实不重叠，已实测（2026-09-24）**：memory2 的 `/v1/models` 只列 `Qwen/Qwen3.5-9B`；
   主网关只列 `Qwen/Qwen3-Embedding-8B` 与 `Qwen/Qwen3-Reranker-4B`。

**纪律**：**两个网关的 base_url 与 key 都不同，不能混用**；调错域名拿到的是 404（不是鉴权失败）。

**可选的后续（不是必须）**：③-d 把配置集中到 `common/config.py` 时，可顺手改名
`AML_EMBED_*` / `AML_LLM_*`，与网关文档的运维命名彻底解耦。**语义已由本条锁定，
所以改名是清洁工作，不是修复。**

**复现**：`make check` 的 Embedding 与 LLM 两项分别打两个域名——**都通过才说明没混用**。

---

## 待决事项（尚无决策）

| # | 事项 | 何时必须定 |
| --- | --- | --- |
| 3 | **渲染模板定稿** | Step 3（E6）——**改模板 = 重建索引** |
| 4 | **S1 的判别实验设计** | Smoke 第一次跑通后**立刻**（§17.1） |
| 5 | **B1 包装 ReFind 的工作量估算** | **真要跑 B1 之前**（§13）——目前**未计入任何 Step** |

> **事项 8（`SqliteStore` 的连接与线程模型）已决，2026-09-24 结案 ⇒ 升格为 [D17](#d17--sqlitestore-的连接模型短生命周期连接--交给-sqlite-自己串行化2026-09-24)。**
> 它当时登记的现状是"6 个 `xfail(strict=True)` 记在 `tests/test_service_add.py` 与
> `tests/test_contract.py`"——**那 6 个标记已全部摘掉**，它们是 D17 的回归用例。
