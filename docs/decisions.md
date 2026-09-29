# 决策日志

> **冲突以 PRD 为准**——本文件记录的是**决策的来龙去脉**，不是规格本身。

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

> ⚠ **"前者 v1 有"这个标签已被 D13 作废**——关键词重写随 agentic 一并归 v2。上面的区分本身仍然成立。

---

## D8 · 战略转向：排序是主线，检索降级（2026-09-22）

**决策**：Hybrid Retrieval **降级为待验证假设**（必须有 BM25-only 对照）；**Rerank + Context Packaging 升格为主线**。

> ⚠ **首句已被 [D15](#d15--混合检索无条件使用删掉裸-bm25-与验证后再加-dense-的分阶段2026-09-23) 推翻**——混合检索是**既定的检索形态**，裸 BM25 模式与那条 hedge 都已删除。
> **本条其余部分仍然有效**，且是 D15 想保住的东西：**"排序才是主线"这个判断没有被推翻**，四条证据也仍然成立（其中第 3 条的处境见 D15 的"残余风险"）。

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

**当时记的环境落差（无 `docker`）已由 [D12](#d12--运行形态模型远程服务与-qdrant-在本机2026-09-23) 解决**——留此一行存档当时的判断。

**另一项**：本机默认 Python 3.14.4，已按 `requires-python = ">=3.11,<3.14"` 保守钉在 3.12（torch / qdrant-client 的 wheel 覆盖通常滞后）。

---

## D12 · 运行形态：模型远程，服务与 Qdrant 在本机（2026-09-23）

**决策**：**LLM / embedding / reranker 三者全部经自建网关远程访问**（哪一段走哪个网关见 [D18](#d18--网关环境变量的语义以本项目为准aml_emb_-embedding主网关2026-09-24)）；**检索服务与 Qdrant（server 模式）都跑在开发机上**。因此本机**不需要 GPU，但仍需要 docker**。

> ⚠ **本条初稿把 docker 标成"消失"**（当时写成"Qdrant 也跑在服务器上，因此本机不需要 docker"）——**该读法作废**：本机的向量库就在本机。连带作废的还有"用 Qdrant 独立二进制绕开 docker"那套。
> **docker 同日已就位**：Docker Desktop 29.8.0（`desktop-linux` context，WSL2 后端）。三个已核实的含义：
>
> * **daemon 走 Windows 命名管道**（`npipe:////./pipe/dockerDesktopLinuxEngine`）⇒ `docker` 命令**直接从 Git Bash 可用**，**不必进 WSL**
> * 容器虽在 Docker Desktop 的 VM 内运行，但**发布端口转发到 Windows 主机** ⇒ 服务在本机访问 `localhost:6333` 成立，`.env` 的默认值**不用改**
> * **`qdrant/qdrant:v1.17.0` 的 tag 已核实存在于 registry** ⇒ §7.3 的版本门槛成立（`rrf.weights` 自 v1.17.0 起可用）

**reranker 的部署不由本项目负责**：目前尚未部署，后续部署，**其宿主环境不在本项目范围内**。本项目只依赖它的 HTTP 端点。唯一要守的纪律是 **reranker 选型在提交时保持不变**——开发期选小的、提交期换大的，等于在 Step 5 引入设计改动，**正好违反"Step 5 不可与任何设计改动合并"**（D2 对冲 ④）。

**衍生出两条 v1 必须实现的约束**：

1. **reranker 必须可降级**：它是**唯一位于提交链路关键路径上、又不被规则保证可用**的组件（embedding 与 LLM 由规则规定、由厂商提供）。Full 只有 2 次、第二次隔 30 天，等于一次真机会。**端点失败或超时时，服务退回未重排的 RRF 顺序返回**，而不是报错——契约要求"≤ `top_k` 且精确计数"，超时路径也得在预算内产出合法响应。
2. **开发环路有单点**：answer / judge / embed / rerank **四条**都打同一个网关，而 harness 评测时会同时驱动 Add/Search（embed + rerank）与答案/裁判生成。**必须有客户端并发上限**，否则排队超时会伪装成"模型变差了"（见 `docs/open-questions.md` V8）。

**⚠ 一条被本决定修正的旧记录**：D11 里那条"本机环境落差"**在当时是真实的**（当时假定模型跑在本机）——但**不要再按它去找显卡**。

---

## D13 · v1 不做 agentic：Checker 退化为带日志的空实现（2026-09-23）

**决策**：**v1 默认"证据充足"，不实现任何 Agentic 功能，没有证据补充。** Evidence Checker **存在**，但实现为**恒返回"充足"、且必须记录每轮判定**的空实现。Agentic Search 整块（含**关键词重写**）归 v2。

**它改掉了两处旧记录**：D7 的"关键词重写 v1 有"标签失效（见该条的批注）；`agent/CLAUDE.md` 的"这是本项目的核心 claim，**不可砍**"⇒ v1 **不砍，但也不实现**。

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

## D16 · 数据路径统一走 `TIANXI_BENCHMARK_DIR`；LoCoMo 的 Add 对话源支持两种布局（2026-09-23）

**决策一 —— 路径口径**：**数据路径一律通过 `TIANXI_BENCHMARK_DIR` 读取，代码中不得硬编码 `benchmark_data/` 或 `eval/datasets/`。** 默认值仍是 `benchmark_data/`；本地开发通过 `.env` 指向实际数据目录。

**边界**：`eval/datasets/LoCoMo-Refined/data/` 是**开发与自测用**的数据，**不是最终要跑的数据集**——归档才是。两者不可混为一谈，**也不得让代码依赖任何一边**。

**决策二 —— LoCoMo 的 Add 对话源**：加载器**两个来源都支持**——`data/public/conversations.jsonl` 在场时优先，否则退回 `locomo_refined.json` 的 `conversation` **并逐条 `strip()`**。

> **为什么不是"只用 `conversations.jsonl`"**：`tools/fetch_benchmark_data.py` 的清单**明写归档不收录它**（它属 `eval/datasets/` 的 clone），所以纯归档路径下那条口径**无法执行**。
> 两个来源产出的 `Message` 流**逐字相同**（见下表的实测，在**全量**上复核过），由 `tests/test_datasets.py` 的
> `test_locomo_two_source_layouts_are_identical` 与 `test_archive_locomo_matches_conversations_jsonl` 钉住（后者在 clone 不在场时 skip）。
>
> ⇒ **契约合规不依赖"选哪个文件"，而依赖"加载层做了归一化"**：归一化层
> （[`eval/datasets/preprocess.normalize_content`](../eval/datasets/preprocess.py)）**对两条路径都强制 `strip()`**——
> 不 strip 才会引入那 209 条契约违规（[`tests/CLAUDE.md`](../tests/CLAUDE.md) 要求 content **首尾无空白**，因为 AML 只做 `"\n".join(...)`、不插分隔符）。
> **净效果：只有归档、没有 clone 也能跑代理评测。**
>
> **⚠ "落差 #1"仍然存在**：`locomo_refined.json` 是 pretty-printed JSON 数组，而归档 pipeline 的 `rows()` 只解析 JSONL——
> 只是那层转换**不该由 Add 源承担**：harness 自己按 pipeline 的契约构造 JSONL 输入（[`eval/harness/judge.py`](../eval/harness/judge.py) 的 `build_input_items`）。

**已实测两项前置验证（均通过）：**

| 验证项 | 结果 |
| --- | --- |
| **`role` 取值** | 只有两个值——`user` 2,951 / `assistant` 2,931。**且与 `speaker_a`/`speaker_b` 100% 一致**（speaker_a→`user`、speaker_b→`assistant`，零例外、零缺失） |
| **逐条内容等价性** | 5,882 条中：**5,673 完全相同 · 209 条仅首尾空白不同 · 0 条内部空白不同 · 0 条真实内容不同**；且 209 条**全部同向**——`conversations.jsonl` 的 text **恰为** `locomo_refined.json` 的去首尾空白版 |

**残余注意（不得忽略）**：LoCoMo 是**两个真人在对话**，此处的 `user`/`assistant` 是**数据集对 `speaker_a`/`speaker_b` 的约定标签，不是"用户 vs 助手"**。所以在 LoCoMo 上配对规则实际是"**speaker_a 的一轮 + 对方回应，直到 speaker_a 的下一轮**"。这不构成问题（与 ReFind 的 turn 粒度一致，消融对比才干净），但**不要把它当成真用户会话**——**LongMemEval 的 `role: user` 才是真 user，两边语义不同。**

**量级预期**：LoCoMo 的 `user` 消息 2,951 条 ⇒ 单独的 QA 对数约 **2,951**（实际略少：session 边界与 assistant 开头的 session 会产生无问的对）。

> ⚠ **按 D20 更正后的口径**：连续 user 并入同一个 `question` ⇒ 上面这个数**不变**（LoCoMo 5,882 条消息严格交替、同 role run ≥2 共 **0** 处），变的是理由。
> 同理"**与 ReFind 的 turn 粒度一致**"**不再逐字成立**（有连续 user 时我们产出的对更少）——引用"同粒度"那条消融前提时要加上这个限定。

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

**回归覆盖**：

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

> 📌 **2026-09-28 附记：主网关的 host 从 `memory.021130.xyz` 迁到 `memory3.021130.xyz`**
> （embedding + rerank 两个服务一起搬）。**本条决定不受影响**——它管的是"**哪个变量名指哪个服务**"，
> 与 host 叫什么无关，所以下表的语义**一字未改**、只换了 host 名。
> ⚠ 但迁移**带了一件不是"改地址"的事**：embedding 维度 1024 → **4096** ⇒ 集合与缓存都要重建，
> 见 [`../deploy/CLAUDE.md`](../deploy/CLAUDE.md) §4 的"网关迁移"一节。
> ⚠ 新 host 上 reranker **可用**，但 **id 换了**（`qwen3-reranker-4b`），而且**不在 `/v1/models` 里**；
> 同一天网关侧**又只改了一次 nginx.conf**（`/v1/rerank` rewrite 到 vLLM 原生 `/v1/score`）
> ⇒ **请求与响应两个信封都换了**（`queries: [...]` / `data[].score`）。
> 三版信封与逐条实测见 [`../src/tianxi_am/rank/reranker.py`](../src/tianxi_am/rank/reranker.py) 顶部的表。

| 变量 | 指向 | 谁读它 |
| --- | --- | --- |
| `AML_EMB_BASE_URL` / `AML_EMB_API_KEY` | **主网关** `memory3.021130.xyz`（**Embedding**） | [`common/config.py`](../src/tianxi_am/common/config.py) 的 `load_config()`（**全包唯一读环境变量的地方**，③-d） |
| `AML_BASE_URL` / `AML_API_KEY` / `AML_MODEL` | **memory2** `memory2.021130.xyz`（**LLM 对话**） | harness / 归档 pipeline（经 `api_config.py` 适配器） |
| `TIANXI_RERANKER_BASE_URL` / `_API_KEY` / `_MODEL` | **主网关**（**Reranker**） | [`rank/reranker.py`](../src/tianxi_am/rank/reranker.py) 的 `RemoteReranker`（**已接线**，2026-09-24） |

> ⚠ **`AML_EMB_MODEL` 这个变量不存在**：embedding 的**模型名**住在
> `configs/*.yaml` 的 `models.embedder`——它是 profile 之间唯一真正该变的东西，
> 所以归配置文件（能 diff、能评审）。**把它和端点写在一起是错的**：那会让 Step 5 的切换
> 变成"改 shell 变量"，改了什么无法评审。

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
   > ⚠ **2026-09-28 起这条观察过期**（host 已迁到 `memory3.021130.xyz`）：现在
   > **memory2 仍只列 `Qwen/Qwen3.5-9B`**（结论不变），而新主网关**只列 `qwen3-embedding-8b`**
   > ——**reranker 不在那份清单里，但 `/rerank` 可用**（实测 200，模型 `qwen3-reranker-4b`）。
   > ⇒ "两个网关能力不重叠"**仍然成立**，它要证的那件事（填错会打到对话网关）不受影响。

**纪律**：**两个网关的 base_url 与 key 都不同，不能混用**；调错域名拿到的是 404（不是鉴权失败）。

**可选的后续（不是必须）**：③-d 把配置集中到 `common/config.py` 时，可顺手改名
`AML_EMBED_*` / `AML_LLM_*`，与网关文档的运维命名彻底解耦。**语义已由本条锁定，
所以改名是清洁工作，不是修复。**

**复现**：`make check` 的 Embedding 与 LLM 两项分别打两个域名——**都通过才说明没混用**。

---

## D19 · 打包的单位是 **Context Segment**，不是单条记忆（2026-09-24）

**背景**：§10 的扩窗与 §11.2 的"组内/组间顺序"落地时，必须回答一个此前被绕开的问题——
**`top_k` 数的到底是什么**。三种答案都"能跑"，而只有一种是对的。

**决定**：

```text
Hybrid → 去重 → Checker → rerank（可降级）
       → 扩窗（全部候选保留，只对前 N 条扩 ±radius）
       → 按 (user_id, session_id) 分组、连续 pair_idx 合并成 **Context Segment**
       → 段按 best_rank 升序 → **token 预算** → ≤ top_k **个段**
```

| # | 锁定的口径 | 换一种写法的后果（**都不报错**） |
| --- | --- | --- |
| 1 | **`top_k` 约束段数**，且**只能在合并之后生效** | 在扩窗阶段按 raw 条数截断 ⇒ 本该成段的邻居被砍掉，**返回的每一段看起来都合法，只是少了一大截** |
| 2 | **段是原子单位**：不截半个段、不拆回单条、不跳过装不下的段再塞后面的 | 截半个段 = 模型读到**缺了一环的连续对话**，而 `content` 里看不出来（§11.2） |
| 3 | **邻居不参与排名**：`best_rank = min(段内真实 rerank 名次)`，`anchor` 是那条候选 | 给邻居编个人造名次 ⇒ 段优先级指向**从没被检索选中过**的记忆，锚点跟着错 |
| 4 | **`id` / `created_at` 取自锚点**，`score = 1/(输出位置+1)` | 取段里第一条 ⇒ `created_at` 来自一条**没被选中**的记忆；照抄 rerank 名次 ⇒ `score` 在预算跳段时出现**空洞**（而"单调递减"照样成立） |
| 5 | **v1 不做第二次 rerank、不做 rerank tail refill、不做段内截断** | 都会让"预算不足时被牺牲的是名次最低的种子"这条（§11.2）**不再成立** |

**为什么不把 `top_k` 卡在 raw 条数上**：`neighbor.expansion_seed_limit = 30`、`radius = 1`
时，100 个候选 + 邻居可达 140 条 raw，而合并之后可能只剩 60 段。
（⚠ 这两个取值 **2026-09-25 已改为 1000 / 2**——有 346 题的 A/B 数据，见
[`../eval/reports/ledger.md`](../eval/reports/ledger.md) 的 N1。**上面这句论证不依赖取值**，故原文保留。）
**中间量（raw 条数）与最终量（段数）根本不是同一个量**，拿前者卡后者必然少给。

**代价（必须正视）**：

* **一项可能很长**。一段连续对话可以是几十对，全进同一个 `content`——于是
  **一旦某段装不下预算，就整段丢掉**，而丢掉的可能正是最相关的那一段。
  v1 接受这个代价（§11.2 明确"截断点必须落在窗口边界上"）；
  **段内截断 / 拆回单条属于 v2**。
* **合并会减少可返回的项数**。10 条相邻的候选现在只占 1 个 `top_k` 名额——
  好处是省预算，坏处是**`top_k=100` 很可能填不满**（而填不满**完全合法**，§2.1）。
* **`neighbor.expansion_seed_limit = 30` 只是 v1 初值**（PRD §10 的"20 种子 / 60 槽位"是一道
  **示例算术**，不是这个值）。⚠ **该初值已于 2026-09-25 改为 1000（= 全部候选）、`radius` 改为 2**，
  依据是 3 段 346 题的 A/B（+3.5pt、无倒退）。它与 `radius`、`top_k`、`budget.max_tokens` 是**同一道题**
  （§10 原文）——**改任何一个都要重算其余三个，且必须有 ablation 数据**。

**复现 / 回归**：`tests/test_neighbor.py`（扩窗、合并、锚点、优先级共 37 条）+
`tests/test_packaging.py`（段级打包 + 双预算）+ `eval/smoke/preflight.py` 的 14 条契约检查。

> ⚠ **段模型还改掉了四条"旧假设"**，它们全都会让**旧用例静默空过**——
> 清单在 [`../tests/CLAUDE.md`](../tests/CLAUDE.md) 的"段模型改了四件事"一节。
> 落地时实测到过一例：预检里"`top_k` 真的会截断"那条**一度变成必然成立**。

---

## D20 · 配对规则：连续 user 消息**并入同一个 `question`**（2026-09-24）

> ⚠ **2026-09-27 起部分被 [D24](#d24--一次-add--记忆块组合的唯一边界推翻-65-的跨批续接2026-09-27) 取代。**
> **配对规则本身仍然有效**（连续 user 消息并入同一个 `question`），但它**不再跨批**——
> 作用域从"整个 session"收窄成"一次 Add"。
> **下面这几样连同 D24 一起删除了**：`open_pair()`、`append_question()`、
> 「`answer IS NULL` 的对必在末尾」这条不变式、`pending` 三计数器。**保留原文以存档当时的判断。**

**决定**：

> **一个对的 `question` = 一段连续 user 消息，直到第一条非 user 消息到达为止。**

§6.2 的**原**字面曾是"一条 user 消息关闭前一个对"。两者**只在出现连续 user 消息时不同**：

| 输入 | §6.2 字面 | 本决定 |
| --- | --- | --- |
| `q1 q2 q3 a` | 3 个对：`(q1, ∅)` `(q2, ∅)` `(q3, a)` | **1 个对**：`(q1\nq2\nq3, a)` |
| `q1 a1 q2 a2` | 2 个对 | **不变** |
| `q a1 a2 a3` | 1 个对 | **不变**（这一侧本来就对） |
| `a q a`（以非 user 开头） | 2 个对 | **不变** |

### 为什么改

**AML 可能把一条超长 user 消息按句边界物理切开**，于是 Add 层看到的是
`user: Q-part1 / user: Q-part2 / …`。这种形状在逻辑上仍是一问一答，
但 §6.2 的字面会把它切成若干个"有问无答"的对，并把助手回复只挂到最后一段上。

### ⚠ 证据状态：这条风险**未被证实**，官方文档里根本没有

| 查了什么 | 结果 |
| --- | --- |
| [官方 api-guide](https://agentmemoryleaderboard.ai/api-guide) 的切批规则 | 「Ordinary Textual splits deterministically at either 20 messages or 2,000 Adapter-counted words.」 |
| 该页是否提到 `sentence boundary` / `complete message` | **两个短语整页都不存在** |
| 单条 >2,000 词 message 怎么办 | **未规定**；官方口径是当作 open question、**去问主办方** |
| fragment 元数据（原消息 id / 序号 / 段数） | **契约里不存在**，且「undeclared fields such as `metadata` are ignored」 |

**同时**：[`contract.md`](./contract.md) §7.1 曾据此推出"**消息一般不会被从中间切开**"——那句推论**建立在二手转述上，且该转述在一手来源里无法复现**，现已修正（见该处）。超长单条的行为登记为 §17.1 的 **S5**。

⇒ **两个方向都没有依据。** 本条决策是在"风险未证实"的前提下权衡后的选择，不是对已知事实的响应。

### 代价与影响面（全量实测，2026-09-24）

| 量 | 值 |
| --- | --- |
| 处在"同 role run 长度 ≥2"中的消息 | LoCoMo **0** / 5,882；LongMemEval **48** / 246,750（最长 run = 4） |
| 本规则改变的对数 | LoCoMo **0** / 3,075；LongMemEval **24** / 124,358（**0.019%**） |
| 跨批接缝触发合并（20 条切批路径） | **0**（两份数据集都是 0） |
| 若碎片真存在，修复率 | 批内版约 45% → **本决定 100%** |

**三条结论**：① 在正常数据上它几乎是 no-op（这正是它便宜的原因）；
② 它**不引入** `state` / `fragment_count` / `updated_at` 之类的字段——`pending` 已经是状态位；
③ **无 schema migration**。

### 判据：`open_pair` 而不是 `pending_pair`

"本批要续写哪一对"的判据从**状态位**换成**内容事实**：

```text
旧：open  ⟺  status = 'pending'                    ← 派生自"本批是否命中上限"
新：open  ⟺  status = 'pending' OR answer IS NULL   ← 第二个析取项是内容事实
```

**为什么第二个析取项不能省**：`pending` 是从"本批是否命中上限"推出来的，
而上限里的**词数计数我们复现不了**（S2）。若 AML 按它的口径切出接近但不足 2,000 词的碎片、
我们算出更少，那一对会被标成 `complete`，下一批的碎片便再也接不上——**静默退化**。
`answer IS NULL` 与任何计数无关。⇒ **这是"跨批合并"能宣称 100% 的唯一理由**
（`test_fragments_merge_even_when_the_previous_batch_marked_the_pair_complete` 钉住它）。

**顺带修掉的一类退化**：回复落在下一批、而我们上一批判成了 `complete` 时，
旧行为会建一个 `question` 为空的"无问的对"——**回复与问题彻底脱钩**。
新判据把它接回原对。这条**与碎片是否存在无关**，是 S2 风险的一个独立收敛。

### 三个必须一起改的地方

| 改动 | 理由 |
| --- | --- |
| `SqliteStore.pending_pair()` → **`open_pair()`** | 名字必须诚实：续写的对象不一定是 `pending` 状态 |
| `ResumeActions.pending_id` → **`open_pair_id`** + 新增 `append_question` | 同上；3a 原来只覆盖 `answer` 侧 |
| **删除 `fill_question_if_null()`**，改为 **`append_question()`** | 旧规则的前提是"一个 `question` 只来自一条 user 消息"——**那正是本决定推翻的**。删除它没有推翻任何有论证的不变式：它自己的 docstring 就写着"① 里没有任何代码路径会用到它" |

**`question` 的拼接不加 role 标记**（与 `encode_answer` 相反）：这段文本是用户的原始发言，
role 均一；标记会一并进 embedding（§7.2 同一份渲染），而这是**拼接**而非**渲染**——
要让被切开的一条原消息能逐字拼回去，就不能注入源文本里没有的 token。
§11.3 要求逐条带标记的是 `answer` 侧（那里可能混着 `assistant` / `system` / 工具输出）。

### 仍然未知 / 未做

| # | 事项 |
| --- | --- |
| **S2（与本条耦合）** | **`answer` 侧仍依赖词数计数**："这一对的答话还没写完"**没有任何内容依据**可判（一个已有 answer 的对和写完的对长得一模一样）⇒ 想彻底摆脱 S2 只能**问出"Adapter 计的词"的口径**。本条**不引入**也不消除这个风险 |
| **S5** | 超长单条 message 到底拆不拆——**只能靠 Smoke 观测或问主办方**（§17.1） |
| **PRD 已同步** | §6.2（定义句 / 三种真实情况 / ReFind 粒度）与 §6.5（第 2 步 / 3a′ / 3b / 写入规则）的**正文已按本条更正**——规格与代码一致，**不存在"以本条为准"的临时口径** |
| **旧库不满足新不变式** | `open_pair` 依赖"`answer IS NULL` 的对必在末尾"，而**按旧规则写过的库不满足它**（`q1 q2 a` 会产出非末尾的 `(q1, NULL)`）⇒ 跨 arm 必须用干净库（**V9** 已经要求这件事） |

### 复现

```bash
.venv/bin/python -m pytest tests/test_pairing.py tests/test_continuation.py -q
```

关键三条：`test_consecutive_users_then_assistant_is_one_pair`（批内）、
`test_cross_batch_fragments_merge_into_one_question`（跨批）、
`test_fragments_merge_even_when_the_previous_batch_marked_the_pair_complete`（不依赖计数）。

---

## D21 · **正文里带日粒度会话日期**（推翻 §11.3 的"不加"）（2026-09-25）

**决策**：`packaging.inject_abs_time` **默认改为 `true`** —— 每一对的渲染文本前加 `[YYYY-MM-DD] `
（与响应里的 `created_at` **同一个格式、同一个 `event_time`**）。

**它推翻了什么**：根 `CLAUDE.md` 那条「**不要在 content 里注入绝对时间戳前缀**」，
以及 [`rank/CLAUDE.md`](../src/tianxi_am/rank/CLAUDE.md) §4 的"v1 定稿＝正文不含任何绝对时间戳"。
§11.3 的理由是：裁判 TIME 块有两条独立规则（**粒度变细** / **相对↔绝对互转**）都会因此判负。

**反例（3 段 346 题，`eval/reports/ledger.md`）**：

| 日期写在哪 | overall | multi-hop | temporal | single-hop |
| --- | --- | --- | --- | --- |
| 不写 | 0.566 | 0.482 | 0.071 | 0.808 |
| 每**段**一次（段首） | 0.601 | 0.482 | 0.165 | 0.829 |
| **每对一次（本次决策）** | **0.633** | 0.554 | 0.176 | 0.850 |

**机制（不是调参，是可解释的）**：LoCoMo-Refined 的时间题 gold 是**锚定式相对**形式
（`The friday before 15 July 2023`）。对话正文说的是 `Last Friday`；只有把锚点放在**那句话旁边**，
模型才会把两者接起来（实测它答成 `Last Friday (relative to March 6, 2023)`）——而那正是 gold 的形状。
**同一份日期放在段首不起作用**（"看得见"≠"用得上"），**重复或重排段落反而有害**（N2 的两个变体都输）。

**§11.3 那两条规则仍然成立、仍然要守**：
* **粒度**：只发**日粒度**——秒级会诱发"粒度变细"判负（`preflight` 那条检查管的就是它，**不许放宽**）；
* **相对↔绝对**：日期是**锚点**，不是把相对表述替换掉——`Last week` 仍在正文里，模型自己决定要不要接锚点。

**代价（必须正视）**：正文变了 ⇒ embedding 输入也变 ⇒ **整个向量索引要重建**（`tools/reindex.py`）、
embedding 缓存整体失效；**以后每次改渲染模板都要再付一次这个钱**。

**适用边界（不要外推）**：证据来自 **LoCoMo-Refined 3 段 346 题**；LongMemEval 未测，
其余四份数据集**契约不同**（§12.4，BEAM 的时间规则甚至相反）。⇒ Step 5 切模型后**必须重测这一类**。

**⛔ 已试过、更差的变体（不要再试）**：正文前缀带**星期**（`[2023-10-22 Sun]`）——
整体 0.633 → **0.601**、三段全降（−2.2 / −4.2 / −3.7），而它对「某星期几之前」那 29 道 gold
**一道都没救回来** ⇒ 已撤回。那条推理（「给星期 = 免去查日历」）**被数据否掉了**。

**怎么回退**：把这一行改回 `false`，重建一次索引（同一件工具）。两条路都有据可查。

## D22 · **正文里的相对时间就地注解成绝对日期**（只改 `content`、不碰索引）（2026-09-26）

**决策**：`packaging.annotate_relatives` **默认 `true`**。正文里的相对表达**原文一字不动**，
后面跟一个括号注：`last Tues` → `last Tues (July 18, 2023)`，`Last weekend` →
`Last weekend (July 15 to 16, 2023)`。锚点是**那一对自己的会话日期**（`event_time`，与
`created_at` / D21 的日期前缀**同一口径**、同一处换算 `render.event_day`）。

**它与 D21 的 `inject_abs_time` 有一处根本区别**：那个改的是**正文**，而正文就是 embedding 的
输入（⇒ 要重建索引、要分集合）；**本决策只改 `content`**——索引侧与精排输入仍然是 `render_pair`，
所以**不用重建索引、不用换集合**，随时可开关。

**实测（3 段 346 题，同一索引、背靠背两臂，唯一变量是注解）**：

| 指标 | 不加（`ann-base`） | **加（`ann-on`）** | Δ |
| --- | --- | --- | --- |
| **overall** | 0.6387 | **0.7948** | **+15.6pt** |
| **temporal (85)** | 0.200（17 题） | **0.800（68 题）** | **+60pt** |
| multi-hop (56) | 31 | 35 | +7pt |
| single-hop (193) | 164 | 163 | −0.5pt |
| open-domain (12) | 9 | 9 | 0 |
| 逐段 | 0.565 / 0.611 / 0.728 | **0.768 / 0.736 / 0.853** | 三段全涨、无倒退 |
| 逐题翻转 | —— | —— | **赢 58 / 输 4** |

数字的家在 [`../eval/reports/ledger.md`](../eval/reports/ledger.md)，本处只留结论。

**机制（可解释，不是调参）**：LoCoMo-Refined 的时间 gold **93% 带绝对日期**（其中 30 道是
`The <weekday/周/周末> before <date>` 这种锚定式），而对话正文只说 `Last Friday`。
答案 prompt **第 7 条明文要求模型做这个换算**——9B 做不到（实测 gold 含绝对日期的 79 道里，
模型只对 9 道给出日期）。注解把这一步**确定性地替它做掉**，模型只需"选对句子 + 抄"。

**为什么是"注解"而不是"替换"**：原文保留 ⇒ ① 仍是 D21 的形状（日期是**额外锚点**，不是替换）；
② 那 2 道纯相对 gold 仍有救；③ 换到 BEAM（时间规则相反）不会反向；④ 错了能看出是哪条推错。
**已有先例支持这个形状**：2026-09-25 测过"重复/重排以增显著性"（N2 的两个变体）**都输**，
但那是重复**整对文本**；这里是**就地加 14 个字符**。

**⛔ 硬纪律：推不出就不动。** `a few years ago` / `several weeks ago` 这类量词**没有数字**，
换算它等于**编一个日期**——而编错的日期比不换算更糟（模型会自信照抄，裁判是精确比值的）。

**代价**：正文体积 +1.1%（346 题实测，中位 1.07%、最大 1.40%）；
`content` 不再逐字等于被索引的文本 —— 这是不变式 **I1 的声明式例外**，边界的写法与测试见
[`architecture.md`](./architecture.md) §4。

**⚠ 最大的外部效度风险（Step 5 必须重测）**：**这一步是在替答案模型做它本该自己做的算术**
（prompt 第 7 条要求的正是它）。本地基线是 **9B**，日历算术正是它最弱的一环；
换到 `gpt-4o-mini` 后**基线会抬高，本决策的增益可能显著缩水**。
⇒ 与 D21 同样的处置：Step 5 切换后**先重测 temporal 这一类**，再决定它是否留在默认值里。

**适用边界（不要外推）**：证据来自 **LoCoMo-Refined 的 3 段 346 题**。其余五份数据集契约不同
（§12.4：BEAM 的时间规则**相反**，CL-Bench 由 AML 侧主动注入时间戳，PersonaMem 根本不读检索字段）。

**怎么回退**：把 `configs/default.yaml` 那一行改回 `false`——**不用重建索引**（这正是它与 D21 的区别）。

---
## D23 · **跑 InvMem 候选仓库作参考**（**一次性显式覆盖**，不是新政策）（2026-09-26）

**决定（由项目负责人做出）**：把 `invmem-candidate/` 跑一轮，**作为参考**。
原话："请你继续跑 InvMem 吧，毕竟也是作为一个参考"。

**它覆盖了什么**：[`../eval/baselines/CLAUDE.md`](../eval/baselines/CLAUDE.md) 里那条
"**只研读、不是基线、不要给它包 Add/Search 服务**"。⚠ **覆盖面仅限"跑一次拿个数"**——
那条规则的四条事实（**无 LICENSE**、无第一方文档、名称到代码的映射是**第三方推断**、
**不可作为参考实现依赖**）**一条都没变**：

| 仍然不许 | 理由 |
| --- | --- |
| **把它写成"InvMem 的做法"/"InvMem 的分数"** | 映射是第三方推断 ⇒ 只能写"**某候选实现**" |
| **分发它的代码 / 依赖它** | 无 LICENSE |
| **把它当基线**（进 §13 的对照表当参照点） | 基线要有确凿的身份与许可；它没有 ⇒ 台账里身份是"**参考**" |

**跑法**：**改了它的 embedding**（`bge-small-en-v1.5` 33M → 我们的 `Qwen3-Embedding-8B`），
用户要求——否则"embedding 强弱"会混进"管线设计"。做法是**不改 vendor 代码**：
[`../eval/baselines/serve_invmem_qwen.py`](../eval/baselines/serve_invmem_qwen.py)
在进程内只替换 `build_embedder` 一个函数。
⇒ **该行的口径是「它的管线 + 我们的 embedding」**，不是"候选仓库的分数"；
**它的原生口径没测**。

**结果与读法**：数字在 [`../eval/reports/ledger.md`](../eval/reports/ledger.md)。
一句话：**同一个 embedding 下我们只领先它 +5.8pt，且全在 multi-hop / single-hop；
temporal 反而略输** ⇒ 在"检索 + 排序"这一层两家大体同级，**拉开差距的是打包**。

**后续（未做，登记在案）**：它的 `MEMORY_TEMPORAL_ENRICHMENT`（默认关）**就是 D22 的同一件事**，
但它没覆盖 `last <星期几>` ⇒ "它能不能拿到同样的 +50pt"是一个**可证伪的预测**，值得单独跑一臂。

---
## D24 · **一次 Add = 记忆块组合的唯一边界**（推翻 §6.5 的跨批续接）（2026-09-27）

**决定（由项目负责人做出）**：把"记忆块生成"改成**每次 Add 独立处理**——不跨 Add 合并任何
message。组合规则只剩三步：

```text
一次 Add 的消息（按原序）
  → ① 连续同 role 合并成一个 RoleBlock
  → ② 一个 UserBlock + 紧随其后的【全部】非 UserBlock 配成一个 MemoryBlock
  → ③ 开头的非 UserBlock（前面没有 user）独立成块
  → 每个 MemoryBlock 恰好一次 embedding
```

**不同 Add 之间永不拼接**：不拼 QA、不合并连续 assistant、不等下一个 chunk、不 repair、
不重新 embedding——**即使两个 Add 属于同一 session**。

### 它推翻了什么

| 推翻 | 原文 |
| --- | --- |
| **§6.2 / §6.5 的作用域** | 「配对的作用域是整个 session，不是一个 Add 批次。一个 QA 对**可以跨批次**」 |
| **§6.5 的批次续接三步** | `3a′ / 3a / 3b / 3d` 与 `pending` 判定 |
| **D20 的落点** | D20 的**配对规则本身保留**（连续 user 并入同一个 `question`），但它**不再跨批**——「`answer IS NULL` 的对必在末尾」这条不变式连同 `open_pair()` 一起删除 |

**没有推翻的**：`pair_idx` 仍**在 session 内连续、不按 Add 重置**（`id` 由它派生，§10 的
±1 邻域按它取窗口）。**组合的边界**与**位置的作用域**是两件事，别混。

### 代价（**全量实测，2026-09-27**）

被批界切开的 QA 对会变成两个半块（一个只剩 question、一个只剩 answer）：

| 数据 | session-scoped 对数 | 被切开的对数 | 占比 |
| --- | --- | --- | --- |
| LoCoMo-Refined 全量（10 段 / 272 session / 399 批） | 3,075 | **63** | **2.0%** |
| LongMemEval（40 题子集 / 1,952 session / 1,970 批） | 9,929 | 6 | 0.06% |

> ⚠ **D20 的表格里那行「跨批接缝触发合并（20 条切批路径）｜0」与本次实测不符**——
> 接缝实际有 63 处（批尾是 user、批首是 assistant）。那行大概率是在 **§6.2 字面规则下**
> 量的（旧规则下接缝两侧本来就各自成对，所以"没有触发合并"）。**以本次实测为准。**

**缓解**：两个半块在 `pair_idx` 上**相邻**，而 §10 的扩窗（`radius` 默认 2）会把邻居一起取回
⇒ 检索时多半仍能一起进上下文。**这条没测过**，是推断。

### 端到端精度：**没有可测的影响**（干净 A/B，2026-09-27）

从 `HEAD`（本决定之前）开 **worktree** 起旧服务，与改后代码**同一天背靠背**跑同一批题，
配置逐项相同（主线那套：宽扩窗 + 日期前缀 + 相对时间注解、`rerank` 关、conv-26 的 138 题），
各自**干净库与干净集合** ⇒ **唯一变量就是这个决定**。

| arm | overall | multi-hop (24) | temporal (36) | open (9) | single-hop (69) |
| --- | --- | --- | --- | --- | --- |
| 旧（跨批续接）`ab-old` | **0.7609** | 0.5833 | 0.8889 | 0.6667 | 0.7681 |
| 新（每 Add 独立）`ab-new` | **0.7536** | 0.4583 | 0.8889 | 0.7778 | 0.7826 |

**Δ = −0.7pt（**−1 题**）**；判定翻转 **5 / 138**（新赢 2 · 新输 3）。

> ⚠ **一个必须说清的性质**：两臂的**注入文本 138/138 道都不同**（中位长度只差 0.1%）。
> 多一个 point 会改变 RRF 里各路的相对名次，名次一动、段的选择与边界跟着动
> ⇒ 这个 Δ 是"**整个系统换了组合规则**"的 delta，**不是"那 1 对被切开"的 delta**。
> 两者在端到端上不可分——而端到端正是要答的那个问题。
>
> ⇒ **读法：−1 题就是这次改动的代价上界**，与"什么都没发生"看不出区别。
> 详细逐题与语料点数见 [`../eval/reports/ledger.md`](../eval/reports/ledger.md) 的 D24 一节。

### 为什么这么做（收益）

1. **删除全项目最绕、最容易静默出错的一层**（`pairing/CLAUDE.md` 原文："出错的表现是某些
   记忆永远检索不到，而不会有任何报错"）——`open_pair` / `append_*` / `mark_complete` /
   三个 `pending` 计数器 / 词数计数（S2 里我们复现不了的那个量）**整体消失**。
2. **组合变成纯函数**：deterministic、无数据库依赖、无跨 Add 状态 ⇒ 乱序到达不影响结果，
   且**可单测**。
3. **`status` 恒为 `complete`**：块在写下那一刻就是最终形状，**没有任何后台任务会回头改它**。

### 没动的（**有意**）

幂等守卫（`applied_batches`）、SQLite 事务、`next_pair_idx` 的读-改-写、按 session 的
进程内锁——**一个都没动**。跨 Add 合并的有无与"丢消息"的风险正交（D4）。

### Schema

**DDL 一个字没改**（§6.1 的列全保留）。变化只在**语义**：

* `question` / `answer` 不再跨批追加
* `status` 写入恒为 `'complete'`（`'pending'` 只在按旧规则写过的库里还读得到）
* **没有加任何列**——来源消息序号（`source_message_ids` 之类）**不落库**：§6.1 的纪律是
  "没有消费方的列不加"，而 `MemoryBlock` 已在内存里带着 `source_idxs` / `roles` /
  `timestamps`（单测与调试可见），`event_time` 给出块首时间。

### 怎么回退

`git revert` 这个改动。**但注意它改的是"哪些文本被索引"** ⇒ 回退后必须**重建索引**
（[`../tools/reindex.py`](../tools/reindex.py)），且**换干净的库**（V9）。

### 测试

组合规则 6 个形状 + 跨 Add 的三条（不拼接 / 连续 assistant 不合并 / 乱序等价）住在
[`../tests/test_pairing.py`](../tests/test_pairing.py) 与 [`../tests/test_apply.py`](../tests/test_apply.py)；
幂等（含"事务已提交、响应未发出"的中间态）在 [`../tests/test_idempotency.py`](../tests/test_idempotency.py)。

---
## D25 · **位置由 `request_id` 的 chunk 序号派生**（去锁；推翻 §6.5 的位置分配）（2026-09-27）

**决定（由项目负责人做出）**：`pair_idx` 不再是"写时 `MAX+1` 算出来的 session 内连续整数"，
改成**请求的纯函数**：

> **一块的位置 = `(chunk_ordinal, local_index)`**
> —— `chunk_ordinal` 从 `request_id` 解析，`local_index` 是这一批组合出的块的 0-based 序号。

⇒ 同一 `(user_id, session_id)` 的 Add **不再需要串行化**：删掉
`src/tianxi_am/service/locks.py` 的 `SessionLocks`（**该文件已随本条删除**，所以这里是字面路径、不是链接）。

### 它推翻了什么

| 推翻 | 原文 |
| --- | --- |
| **§6.5 / §6.1 的位置分配** | "`pair_idx` = `MAX(pair_idx)+1`，session 内连续、新批次接着数" |
| **§15 的串行化要求** | "**Add 按 `(user_id, session_id)` 串行化**，且必须 `--workers 1`"（**本次只去掉锁；`--workers 1` 见下**） |
| **D17 的一部分** | "位置分配是读-改-写"——**不再是了** |

### 为什么值得

1. **修一个静默的错误**（这才是主要理由）：旧方案下 `pair_idx` 由**到达顺序**决定。
   若 `chunk-1` 先于 `chunk-0` 成功提交（流水线发送、或重试与首次重叠），`chunk-1` 会拿到
   **更小**的 `pair_idx` ⇒ §10 扩窗与 §11.2 段合并把 `pair_idx` 相邻当成"会话相邻"，
   于是**把对话顺序读反了，且不报错**。位置改为请求的纯函数后，**到达顺序不再影响顺序**。
2. **同 session 的 Add 可以真正并发**——锁原本一直包到远程 embedding 结束，是最贵的一段。
3. 位置分配不再是读-改-写 ⇒ **重试天然幂等**（D4 的"位置重分配"那一半从根上消失；
   批次级守卫 `applied_batches` 仍然保留，它管的是"这批被应用过吗"）。

### 关键依据：`request_id` 里带着 chunk 序号

平台实发的形如 `eval:<run_id>:locomo_refined:conv-0:chunk-0` —— `chunk-<N>` 即该批在
`(conv, session)` 空间里的批次序号。（与 §17.1 的 **S5** 同类：**这是团队告知的平台行为**，
本仓无一手出处；[`../docs/contract.md`](../docs/contract.md) 此前把这种 `eval:` 命名列为
**本仓 harness 的本地约定**——**冲突记在此处，以团队说法为准**。）

> ⚠ **解析失败一律响亮失败**（非 200），**没有 `MAX+1` 回退**。
> 回退会把上面第 1 条那个静默错误偷偷带回来。格式住在
> `ingest.chunk_ordinal_pattern`（正则，可配置），所以真变了不必改代码——
> 这正是 §12.1 R1 对冲 3 要的形态。**格式假设的验证是 Smoke 该做的事**。

### 相邻性怎么保持（`seq`）

`pair_idx` 的**相邻**语义换成**读时的稠密序**：

```sql
ROW_NUMBER() OVER (PARTITION BY user_id, session_id ORDER BY chunk_ordinal, local_index) AS seq
```

⇒ ① chunk 序号里的空洞**不破坏相邻**；② 跨 chunk 边界（chunk c 的末块 ↔ chunk c+1 的首块）
**天然相邻** ⇒ **扩窗与段合并的行为与旧方案逐字相同**（顺序 ingest 下 `seq` 与旧 `pair_idx` 逐一相等）。
实现：`fetch_session_ordered()` 取整段有序列表，`expand_neighbors` 在 Python 里按 `seq` 切窗口。

### 代价

- **`id` 的取值变了**（`hash(user, session, chunk, local)` ≠ 旧的 `hash(user, session, pair_idx)`）
  ⇒ **所有 Qdrant point id 变化 ⇒ 必须重建索引 / 重新 ingest**。
- 引入一条对 `request_id` 格式的**运行时依赖**（见上）。
- 检索侧的并列 tie-break 次级键是 `memory_id`（V13）⇒ id 变了，同分候选的并列次序可能微动
  ⇒ **对照可能落在 ±1–2 题的已知噪声里**。

### 没动的（**有意**）

- **`applied_batches` 幂等守卫、`BEGIN IMMEDIATE`、SQLite 事务边界**——一个都没动。
  `BEGIN IMMEDIATE` 是**数据库级写者串行**（所有 session 共用一把），与位置方案**正交**；
  换成 deferred 会踩 `SQLITE_BUSY`（已实测，见 `test_store.py` 的两条压力用例）。
- **`--workers 1` 保留**：位置虽已进程无关、多 worker 从此在原理上安全，但放开 worker 是
  **另一处 §15 偏离**，**单列为后续**，不在本次范围内。⇒ `service/CLAUDE.md` 那条纪律仍生效。

### 怎么回退

`git revert` 这个改动，并**换干净的库**（位置模型变了，旧库的行不在新坐标系里）。

### 测试

- 位置解析：`tests/test_apply.py` 的 chunk 解析用例（含**格式不符 ⇒ 响亮失败**）。
- **乱序到达与顺序到达结果逐字一致**（`{id 集合, 每块 content, seq 序}`）——本次真正要买的性质。
- 同 session 并发 Add **互不相交地落位**（`tests/test_service_add.py`）。
- 邻域与段合并（`tests/test_neighbor.py`）在 `seq` 上跑同一批断言。

### 实测（2026-09-27）—— **行为保持，且"乱序/并发安全"已验**

| 这条决定要回答的 | 结果 | 数字在哪 |
| --- | --- | --- |
| 端到端**行为保持**？ | ✅ **`generated_answer` 138/138 逐字相同**（不是"分数相近"——生成链路产出完全同一份东西） | [`../eval/reports/ledger.md`](../eval/reports/ledger.md) 的「D25」一节 |
| **乱序 + 并发**投喂 = 顺序投喂？ | ✅ 216 行**逐字一致**（位置集合 / `id` 集合 / 正文 / `seq` 序全同，0 行不同） | 同上 |
| §6.5 的两条守卫还在？ | ✅ `applied_batches` + `BEGIN IMMEDIATE` 一条没动，全量测试 570 通过、契约预检 14/14 | —— |

> ⚠ 两点别读过头：① 精度结论**只到 conv-26 一段**（乱序一致性是结构性的、不依赖数据集，精度不是）；
> ② **`--workers 1` 没放开**——位置虽已进程无关，但多进程的并发写压力一件都没验证过。

---

## D26 · **A4 移出 v1**（它要 agent 真的存在，而 v1 不做 agentic）（2026-09-28）

**决定（由项目负责人做出）**：§13 的 **A4「agent 值不值」不在 v1 的交付里**。

**它只是一处排序，不是一次推翻**：D13 已经定了 v1 不做 agentic，而 A4 的两种读法
（**A4a** 受 Checker 门控 / **A4b** always-on，见 [`experiments.md`](./experiments.md)）
**都要 agent 真的在跑**才成立 ⇒ **A4 在 v1 里跑不了**。但三份文档此前默认它在
（`roadmap.md` 的 Step 6 写着"还剩 A0/A4/T2"），所以这条要把口径固定下来。

| 项 | 处置 |
| --- | --- |
| **arm 定义** | **留着**（`experiments.md` 的 A4a / A4b 两条读法逐字不动）——Step 4 落地时直接用 |
| **v1 的那条核心 claim** | **不得引用 A4 的分数**（D13 已经这么要求）。v1 里 §3.1「只有证据不足才触发多轮搜索」是**平凡成立**的——因为根本没有 agentic |
| **Step 6 的对照清单** | v1 只剩 **A0 / T2**（B1 已跑） |

### 它与 D13 的一处出入（**记在这里，别让它消失**）

D13 说 Checker 的空实现能"**免费拿到反事实分布**"，从而"在 Step 4 之前就能用数据回答 A4"。
**那份分布现在恒为 `None`**：判据要的是 `(bm25-only 名次, dense-only 名次)`，而
[`roadmap.md`](./roadmap.md) 的"Checker 的两路分离查询"一节记着——`QdrantStore.hybrid_search`
**只做融合**，两路排名算完就丢，所以 v1 的调用方不传。**该切片已登记为"不阻塞、现在不做"。**

⇒ 准确的表述是：**D13 的接缝是对的，但它的收益还没通**。A4 的提前回答要等那条切片，
而 A4 本身要等 Step 4。**两者都不在 v1。**

### 代价（要认）

**v1 没有任何机制检验"agent 值不值"**，而这是风险最高的一次对照。缓解只有一条：
**Step 4 到来时 A4a/A4b 的定义是现成的**（它们已经写清楚了"只做 A4a 无法区分
『agent 没用』与『Checker 卡太严』"），不用临场设计。

### 为什么现在定

`experiments.md` 的 A4 小节写着"**做之前先在 `eval/experiments/` 里把 arm 定死**"——
而 v1 的 arm 脚手架（[`../eval/experiments/arms.py`](../eval/experiments/arms.py)）是
"**两臂 = 两份冻结配置 + 两个服务**"的形状，A4 的两种读法都能塞进去；**但 A4 还需要一个
agent 才存在**，那是另一个 Step 的事。⇒ 把它标出 v1，免得 Step 6 的清单上挂着一个做不了的项。

> ⛔ **别据此推进 `checker.enabled` 的接线或单路查询**：前者仍按 §15 的要求可配
> （消费方未接，见 [`config-reference.md`](./config-reference.md) §2），后者仍**不做**。

---

## D27 · 提交期**不设专用的环境变量对**：`AML_EMB_*` 就是 embedding 端点，值随部署变（2026-09-28）

**决定**：删掉 `.env.example` 里骨架期留下的两组"提交期专用"变量（`TIANXI_EMBED_API_KEY` /
`TIANXI_EMBED_BASE_URL`、`TIANXI_LLM_API_KEY` / `TIANXI_LLM_BASE_URL`）。
**embedding 的端点与密钥只有一个家**：`AML_EMB_*`——那两个名字是**"embedding 端点"这个位置**的名字，
**值随部署而变**（开发期 = 自建主网关，提交期 = DashScope 的 OpenAI 兼容端点）；**模型名**仍住
`configs/submit.yaml`（D18 的既定口径）。

**为什么是删，而不是把它们接上**：

| # | 理由 |
| --- | --- |
| 1 | **同一个东西能两处设 ⇒ 填错的那一处不报错。** 填了 `TIANXI_*` 而忘了 `AML_EMB_*`，服务照旧连开发网关、**跑得好好的**——"我明明填的是 DashScope"要到很后面才以别的形状暴露。这正是 [`configs/CLAUDE.md`](../configs/CLAUDE.md) 点名要避免的那类歧义 |
| 2 | **它们从来没有消费方**：09-23 的骨架提交就把它们写进 `.env.example` 了，**早于 D18 锁定 `AML_EMB_*`**。是"收一个没有消费方的键等于预留字段"那条纪律的现成反例 |
| 3 | `TIANXI_LLM_*` 更没有存在理由：**v1 没有 LLM 调用点**（`Add` 侧完全不调用；`Search` 只在 `agent/` 里调用，而 v1 不做 agentic，**D13**）|

**⚠ 由此暴露的一条静默风险（要认，还没对冲）**：`AML_EMB_BASE_URL` 一换，**声明与事实就可能不符**——
`text-embedding-v4` 与开发期的 Qwen3-Embedding-8B **同为 1024 维**（前者是官方默认值，
后者是 `make check` 每次实测的那个范数），**维度一致 ⇒ 换错端点不会被任何一层拦住**：

1. **缓存会串**：缓存坐标是"**配置里的模型名** + 渲染模板版本"（`embed/base.py`），
   而键是文本哈希。拿提交 profile 去连一个只服务 Qwen3 的端点 ⇒ Qwen3 的向量**挂在
   `text-embedding-v4` 这个坐标下** ⇒ 之后真的接上 v4 端点时**全部命中缓存**、根本不调它。
2. **集合也会串**：同维度 ⇒ Qdrant 不拒绝，只表现为"检索质量差"。

> 📌 **2026-09-28 附记：上面那条"同为 1024 维"的前提当天就变了。** 网关迁移
> （`memory.021130.xyz` → `memory3.021130.xyz`）之后**开发期是 4096 维**
> （`make check` 实测 `dim=4096 L2=1.000000`）⇒ "**维度一致 ⇒ 拦不住**"只剩
> "**恰好又同维**"这一种情形（而 `text-embedding-v4` 的维度**可选**，别指望它
> 永远不等于我们在用的那个）。**决定与纪律一字不变**，只是现在多半会
> **响亮失败**（`embed/base.py` 的 `DimensionMismatchError`）而不是静默串——
> **那是运气，不是保障**：门禁仍然没做。
> ⚠ 顺带一条**部分解决**：`tools/check_env.py` 现在**不写死 id**，它向 `/v1/models`
> 问并把用到的 id 打出来（`model=… dim=… L2=…`）⇒ "声明 vs 事实"从**看不见**变成
> **人眼可核**；但**自动比对仍未做**（声明住在 `configs/*.yaml`，而这个工具刻意不读配置层）。
> 那个不存在的变量名 `AML_EMB_MODEL` 也还在，只是如今退化成"想指定就指定"的可选项。

**目前的处置只有纪律**（提交 profile 必须配 DashScope 端点；真要换端点就按
[`../deploy/CLAUDE.md`](../deploy/CLAUDE.md) §4 的 runbook 连缓存一起作废）。
⛔ **门禁还没做**：`tools/check_env.py` 的 embedding 探针验的是"端点可达 + L2 范数"，
**不核对"端点在服务的模型是不是我们声明的那一个"**（它还传着一个不存在的变量名 `AML_EMB_MODEL`）。
⇒ **这条是未办事项**，等 v4 端点到位、能看到真实响应形状时再定怎么做（`GET /models` 比对？
还是让服务启动时核对？）。**在那之前别把"`make check` 过了"读成"端点对了"。**

**为什么现在定**：`configs/submit.yaml` 与 `embed/text_embedding_v4.py` 在这一天落地（Step 5 的前半），
而"提交期端点填哪儿"是它们绕不开的问题。**留着两组可填的变量，等于把 Step 5 唯一要评审的那处差异
（`submit.yaml` + `.env`）藏进一个没有消费方的名字里。**

---
## 待决事项（尚无决策）

| # | 事项 | 何时必须定 |
| --- | --- | --- |
| 4 | **S1 的判别实验设计** | Smoke 第一次跑通后**立刻**（§17.1） |
| 5 | ~~**B1 包装 ReFind 的工作量估算**~~ → **已估完，见下** | —— |

> ### 事项 5 · 已估（2026-09-26）：**不需要"包一层服务"，因为它本身就是兼容服务**
>
> [`eval/baselines/CLAUDE.md`](../eval/baselines/CLAUDE.md) 原先的假设是
> "ReFind 是方法实现，要给它包一层 Add/Search 服务"——**核过代码后发现不成立**：
> [`refind/app/main.py`](../eval/baselines/refind/app/main.py) 已经是一个
> **AML 兼容的 Add/Search 服务**（`/add` `/search` 短别名 + `/v1/memories/*`，
> `description="Agent Memory Leaderboard-compatible Add/Search service."`），
> 请求/响应形状与我们的 driver **逐字段对得上**（`AddRequest{request_id,messages[{role,content,timestamp}],
> user_id,session_id}`、`SearchResponse{data:[{id,content,score,created_at}]}`）。
>
> ⇒ **包装工作量 ≈ 0**：起它的服务、把 harness 的 `--base-url` 指过去即可
> （`--limit` 控制规模）。依赖很轻（fastapi / httpx / pydantic / uvicorn），
> 用独立 venv 装它钉死的版本。
>
> ⚠ **剩下的成本在别处**，跑之前要知道：它是 **agentic**（`RETRIEVAL_MODE=agent`，
> `AGENT_MAX_ITERATIONS=4`、`SEARCH_TOP_K=5`），ReFind 论文实测**每 query 约 5 次 LLM 调用**
> ⇒ 346 题 ≈ **1700 次调用**，是**小时级**的一轮；且它跑在**同一个网关**上。
> ⇒ 先跑 1 段（138 题）量速率，再决定要不要铺到 3 段。
>
> ⚠ **隔离边界照旧**：`refind/` 只被 `eval/harness/` 当"另一个服务"调，
> **`src/` 一行都不 import 它**，而且**不改 vendor 代码**（改了就不再是 B1）。
>
> ⚠ **模型口径**：用我们网关的 `Qwen/Qwen3.5-9B`（与主系统同期同模型才算对照）；
> 它自报的 58.2 / 93.2 是**它自己的 harness**跑出来的，**不能拿来当基线**（§13）。

> **事项 8（`SqliteStore` 的连接与线程模型）已决 ⇒ 升格为 [D17](#d17--sqlitestore-的连接模型短生命周期连接--交给-sqlite-自己串行化2026-09-24)。**

---

## D28 · **`request_id` 回到 opaque string；邻接改成 Add 内显式链**（推翻 D25）（2026-09-29）

**决定（由项目负责人做出）**：

> 1. **`request_id` 只做三件事**：**幂等键**、**溯源**、**成功响应原样回显**。
>    ⛔ **一律不解析**——不取 chunk 序号、不取顺序、不取任何业务语义。
> 2. **位置 = `(request_id, local_index)`**（`id = hash(user_id, session_id, request_id, local_index)`）。
>    `request_id` 作为**一个整体**参与哈希。
> 3. **邻接 = 存储里的显式指针**：`qa_pairs.prev_memory_id` / `next_memory_id`，
>    **只连同一次 Add 内的完整 QA**（A-only / Q-only 两侧恒为 `NULL`，不进链）。
> 4. **扩窗与段合并都只在这条链上走**：`pair_idx ± 1` / 会话稠密序 `seq` 那类
>    **推断出来的**相邻性**整个删掉**（`fetch_session_ordered` / `_SEQ_COLUMN` 已删）。
> 5. **幂等守卫加一半新职责**：`applied_batches.payload_hash`——同 `request_id`
>    **不同 payload** ⇒ **409**（不是 500、更不是静默重放）。

### 它推翻了什么

| 推翻 | 原文 |
| --- | --- |
| **D25 的位置来源** | "`chunk_ordinal` 从 `request_id` 解析（`ingest.chunk_ordinal_pattern`）" |
| **D25 的 fail-loud 形态** | "解析失败一律响亮失败（非 200），没有回退"——**现在不再需要解析，也就没有这条失败** |
| **不变式 2（D25 版）** | "邻域由读时稠密序 `seq` 现算" → 现在由**存储里的指针**给出 |
| **§10 扩窗的"整段取回、按下标切窗口"** | 现在**沿指针跳**（一跳 = 一个完整 QA） |
| **§11.2 段合并的"连续 `seq`"** | 现在**只在同一次 Add 内、且 `前一条.next == 这一条.id`** 才合并 |
| **配置项 `ingest.chunk_ordinal_pattern`** | **整个 `ingest` 段已删**（`common/config.py` 留了一行墓碑注释，防止有人加回来） |

### 为什么（全部有实测依据，不是推测）

| # | 事实 | 出处 |
| --- | --- | --- |
| 1 | **平台实发的 `request_id` 是 `r_31156f4174…`（不透明，没有 `chunk-`、没有序号）** | 2026-09-29 的请求原文采集（`service/capture.py`，S6 那条路的落地）抓到的**真实外部请求**，`client=221.194.152.241`；那一条的 `error=ValueError`、服务返回 500 |
| 2 | ⇒ **D25 的解析在真实流量上 100% 失败** | 也就是 S6 登记的"表现是 **Add 全挂**"——**它发生了** |
| 3 | 而 S6 的依据是"**团队告知 + ReFind 的示例**"，**不是一手文档** | [`open-questions.md`](./open-questions.md) 的 S6（现已收口） |
| 4 | 跨 Add 的"顺序"本来就没有可信来源：`request_id` 不透明之后，**它从哪里来都没有** | 本条的直接推论 |

⇒ 两条路摆在面前：**(a) 继续猜 `request_id` 的形状**（把正则改宽——但它对**任意**形状都不成立，
而"任意形状"正是官方示例之外的全部空间）；**(b) 承认"跨 Add 没有顺序"**，
把位置与邻接都收进**一次 Add 内**。选 (b)：**代价是跨 Add 的 continuation 与邻接，
买到的是协议兼容与"顺序无关"**。

### 代价（要认账）

| 代价 | 说明 |
| --- | --- |
| **跨 Add 不再有邻接** | 两次 Add 的块**永不合并成同一段**（§11.2 的窗口变短）。同一 session 的对话被切成 N 段，**段内是连续的，段间不是** |
| **扩窗变小** | 只能沿本次 Add 的链走 ⇒ 长 session 上"看到整段对话"的能力**变弱**。token 预算的占用也会随之下降（段短了） |
| **`id` 全变** | 旧库的 `id`（D25 公式）与新公式不同 ⇒ Qdrant 里的 point 全成孤儿。`SqliteStore.open()` 会**自动搬一次**（保正文、重算 `id`、重连链），**但必须跑一次 `tools/reindex.py --drop`**（embedding 缓存按内容哈希 ⇒ **不会重付**） |
| **旧 evals 数字不可比** | ledger 里所有以"段"为单位的结论（N1 的扩窗收益、A3 的段级消融）都建立在**会话级邻接**上，D28 之后**同一份数据会切出更多段**。**重跑之前不要引用旧数字** |

### 与 D24 的关系（容易混）

| | D24 | D28 |
| --- | --- | --- |
| 改什么 | **组合**（哪些消息进同一个块） | **位置与邻接**（块与块之间怎么排） |
| 作用域 | 一次 Add | 一次 Add |
| 留下什么 | 块写下即最终形状（无 pending / 无 repair） | 位置与邻接都是**请求的纯函数**（无 `MAX+1`、无读-改-写） |

两条是**同一个方向上的两步**：D24 把"组合"收进一次 Add，D28 把"顺序与邻接"也收进去。
⇒ 现在**一条 Add 的全部行为只由它自己的 payload 决定**，与到达顺序、与其他 Add、
与 `request_id` 的形状**全都无关**。

### 落地清单

* `store/schema.sql`：`(request_id, local_index)` + `prev_memory_id` / `next_memory_id` +
  `applied_batches.payload_hash`；`chunk_ordinal` **删列**。
* `store/sqlite_store.py`：`make_pair_id`、`insert_pair`（一次 INSERT 写死邻接）、
  `fetch_by_request`、`migrate()`（旧库→D28，事务内、保正文、重算 id、重连链）。
* `pairing/pairing.py`：`link_blocks()`（"只连完整 QA"那条规则的**唯一实现**）；
  `parse_chunk_ordinal` **删除**。
* `pairing/apply.py`：`payload_fingerprint()` + 守卫的 payload 校验 + `PayloadMismatchError`。
* `rank/neighbor.py`：扩窗改**沿指针跳**、合并改**看指针相接**。
* `service/errors.py`：`PayloadMismatchError → 409`。
* `service/capture.py`：**不再解析** `request_id`（它已经不解析了——记录里保留那一列只是因为
  "看一眼官方发的形状"仍然有用）。

> ⚠ **不要再加回任何"从 `request_id` 里取东西"的逻辑**——那是本条要根治的病。
> 想表达顺序，就把它放进 payload（`messages` 里本来的次序），或者放一个新的**显式**字段。
