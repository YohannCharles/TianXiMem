# agent/ — 多跳检索执行器（`MULTI_HOP` 那条路）

**PRD**：§9（Conditional Agentic Search）—— ⚠ **本目录只实现其中一小块，见下**
**决策**：**D41**（2026-10-10）· 键与口径：[`../../docs/config-reference.md`](../../docs/config-reference.md) **§5**

## 两种模式（`agentic.mode`，2026-10-12 起）

| 模式 | 文件 | 怎么推进 |
| --- | --- | --- |
| `react`（**代码内置缺省**） | `coq_prompt.py` · `parse.py` · `coq.py` | 每轮重新决定"充分了吗 / 下一跳问什么"（原版 Chain-of-Query） |
| `plan`（**`default.yaml` 即提交口径**） | `plan_prompts.py` · `plan_parse.py` · `plan.py` | **先拆一份固定计划**（1~4 步，含 `{{sN}}` 占位），之后每轮只回答"刚检索完的那一步"，**不重规划** |

⚠ **两者故意不同**（2026-10-14 起）：`configs/default.yaml` 写的是 `plan`，代码常量
`DEFAULT_AGENTIC_MODE` 仍是 `react`——后者是给**不填这个键的旧配置**留的"逐字不变"出口。
⛔ 不要为了让两者看起来一致就把常量也翻掉。口径见
[`../../docs/config-reference.md`](../../docs/config-reference.md) **§5**。

⚠ **两套提示词、两套解析、两张停止原因表**——**互不引用、互不改写**：
`coq_prompt.py` 的原版提示词是**冻结**的，⛔ 绝不把 Plan 的提示词塞进去（或反过来）。
共享的只有三个纯工具（`_ms` / `_usage` / `_render_document`，由 `plan.py` 从 `coq.py` 导入，
⛔ 别抄第二份）与"从一段文本里挖出 JSON"那一件与业务无关的事（`extract_json_object`）。

⚠ **谁选模式**：装配期（[`../service/app.py`](../service/app.py) 的 `build_agentic`）
读一次 `config.agentic.mode`、构造对应的执行器。⛔ **编排层不认识模式**——两条路
在 `SearchPipeline._search` 那里是同一条（它只调交上来的那个执行器）。

## 要写什么

```text
coq_prompt.py      原版 Chain-of-Query 提示词（⛔ 逐字照用，**只有一版**）
parse.py           两阶段 JSON 提取 + 严格字段校验（缺字段 ⇒ 异常，**不补默认值**）
coq.py             `react` 的状态机：多轮召回 → 每轮精排 → 一次 LLM → 停/继续 → 合并候选
plan_prompts.py    `plan` 的两条提示词（拆解 / 逐轮**只回答**）——**独立于上面那份**
plan_parse.py      `plan` 的严格校验（计划结构、依赖以占位为准、逐轮三字段、引句逐字）
plan.py            `plan` 的状态机：拆解 → 逐题检索 → 一次 LLM（**只回答**）→ 收口
```

⚠ **`plan` 的程序职责里还有一条**（2026-10-13 起；**2026-10-15 触发面放宽**）：某一步返回
`unresolved` **或** `conflict` 时（**模型明确说"走不通"的那两种**），
**按请求的 `top_k` 补一次原题检索**——先把那 `top_k` 条**精排一次**
（挑"哪几条有资格进池"），再按名次**去重补位**，**补到池子达到 `top_k` 为止；不够不凑**。
⛔ **工具性失败不触发**（`plan_invalid` / `llm_*` / `retrieval_error`，保持原行为）。
候选按稳定 ID 合并、**追加在分步候选之后**；那种题的**最终精排只用原题**、
**答题输入不带执行记录**。
⛔ 它是**候选池的补充**，不是新的多跳路：不计步骤、不加 LLM 调用、不截 Top-20，
**也不走 `fallback_reason`**（那个词的含义是"整条路降级、候选全丢"）。
⚠ 它那一次精排**不受 `agentic.round_rerank_enabled` 管**（那个开关只管"每轮"）——
§13 的开关纯度：一个开关管两件事，那一臂就同时在两个维度上变了。
⚠ 池子上限因此是 **≤ `top_k`**（2026-10-15 之前是"恒 80"）。
见 [`plan.py`](./plan.py) 的 `_recover` / `_rerank_fallback` 与 `PlanRecovery`。

⚠ **`plan` 的模型职责只有两件事**（2026-10-13 起）：**拆解**与**回答当前步**。
"第几步、下一步问什么、什么时候算跑完"**全部在程序里**
（`PlanAgent.search`）——⛔ 逐轮输出里没有步骤编号，也没有 `next_query`
（见 `plan_prompts.py` 的 `PLAN_STEP_PROMPT`：只回 `status` / `answer` / `evidence`）。
这样切的原因：那三件事都是**确定性的程序决策**，交给模型只会多出三种失败模式
（回错步号 / 自由改写下一查询 / 自己宣布跑完），而它们**都不会报错**。

## 它**只有一件事**

```text
原始 Query
   ├─→ ① 混合召回（复用 retrieve/）→ 现有 Reranker → Top-N
   ├─→ ② 攒证据（本轮 Top-N ＋ 此前已选中的，按稳定 ID 去重、重新编号）
   ├─→ ③ 一次 LLM：充分性 ＋ 选证据 ＋ 下一跳 Query（原版 CoQ 提示词）
   └─→ ④ 停？（充分且置信 ≥ 阈值 / 轮数用尽 / Query 重复或为空 / 出错）
         否 ⇒ 换 Query 回 ①；是 ⇒ 各轮 Top-N 去重合并，交回调用方
```

⛔ **它不是一个通用 Agent**：没有工具调用、没有自主规划、没有 SplitQuery 并行子查询、
不生成最终答案。**PRD §9 的那些**（`search_chatrecord` / `take_note` / `finish_search`
三个工具、会话级 rank fusion、时间收窄、已见 session 去重）**一个都不在这里**，
本阶段明确不做。⚠ **别在这个目录里把它们长出来**——那会同时改掉轮数与每轮候选数
两个量，而它们是一道预算题（§10）。

⚠ `plan` 模式会生成**中间跳的答案**（用来填占位）——那**不是**"生成最终答案"：
它不进响应、不进索引、不落库，只以**执行记录**（`PlanResult.plan_history`）的身份
在 Response 之外被交给诊断/评测侧。⛔ 那条记录**必须一直带着"模型生成的执行记录"
这个标注**（见 `tools/coq_eval.py` 的 `PLAN_RECORD_HEADER`）。

---

## 四条边界（越界都**不会报错**）

1. **不生成答案、不改写记忆原文**（§2.1 的红线）。文档正文一律走
   [`../common/render.py`](../common/render.py) 的 `render_pair`（不变式 I1）——
   与索引侧、精排输入是**同一份渲染**。模型给的 `new_query`（`react`）**只当检索词**用，
   ⛔ 绝不把它当成一条记忆内容（§8.3）。
2. **不做最终精排。** 那一步由 [`../service/pipeline.py`](../service/pipeline.py)
   对**合并后的列表**做**恰好一次**；本目录只交出 `rerank_query`
   （原问题 ＋ **实际执行过的**改写，§9.1）。**两种模式同一条**。
   ⚠ **每轮内部**那一次精排是另一件事：产出物不进响应，只为把交给 LLM 的证据筛干净。
3. **不抛异常。** 所有失败都变成结果对象上的一个具名原因
   （`fallback_reason` / `stats.stop_reason`）。理由与路由器同一条：
   一条可选路径的失败**不该让一次正常 Search 变 500**。
4. **不认识 HTTP、不认识路由器。** 接线在 `service/`。
   有 AST 用例钉着（`tests/test_agent.py::test_the_agent_never_imports_the_http_layer_or_the_router`
   与 `tests/test_plan.py::test_the_plan_executor_does_not_import_the_http_layer_or_the_router`）：
   §13 的消融需要这两条路**各自可关**，互相纠缠会让"关掉它"不再是单变量。

---

## 三处"顺序错了也不报错"的地方（两种模式**都**适用）

| 地方 | 错了会怎样 |
| --- | --- |
| **每轮先召回、再精排、然后取 Top-N** | 反过来只会重排一堆**已被提前排除**的候选，**结果看起来完全正常**（§3.2） |
| **每轮重建 `索引 → 稳定 ID` 映射** | 复用上一轮的：模型给的索引指向**别的证据**（§5.2） |
| **文档渲染只做一次** | 把已带索引头的文本再嵌一次 ⇒ 提示词出现 `[1]\n[0]\nQ: …`，**模型读得懂它**，于是对应关系悄悄错位 |

还有一条**可证的不变式**（别试图为它造测试）：`history_tokens ≤ max_prompt_tokens`
恒成立——历史证据只可能由**发出去过的**文档组成，而那些已经过预算。
⇒ `_build_documents` 里"连历史都放不下"那条出口**今天不可达**，证明写在它的 docstring 里。
⚠ **Plan 的那一份 `_build_documents` 没有这条不变式**（它的"历史"是问题/答案/来源 ID，
不进文档预算），所以它**不复制**那条推理——两份 docstring 说的是两件事，别互相对齐。

---

## 提示词是**冻结的**

`coq_prompt.py` 的 `COQ_PROMPT` 是方案第四章的原版，**照用、不改写、不简化、
不补示例、不改 JSON 字段**。要动它之前先回答一个问题：
**"这还是原版吗？"**——不是的话，本阶段"以原版 CoQ 为核心"这个前提就没有了。
⚠ **这一条只约束 `react`**：`plan_prompts.py` 是**另一个模式**的提示词，
它按方案第四节给的原文实现，**不属于**这条冻结。

⚠ 填充用**一遍扫描的 `re.sub`**，不是连续三次 `str.replace`：后者会让一句含字面
`{used_query}` 的 Query 在**第二次替换时被顶掉**，且**没有任何报错**。
⛔ 也不要用 `str.format`（Query 里的 `{` 会让它炸，或者替换出没人预期的输入）。
⚠ Plan 那边**更是如此**：它的模板自己就含 `{{s1}}`、提示词里还有 JSON 示例，
所以槽位匹配器是**按声明过的槽位**构造的（`plan.py` 的 `_DECOMPOSE_FILLER` /
`_STEP_FILLER`）。

⚠ 提示词里 `{used_query}` 与 `{retrieved_episodes}` **各出现两次**
（"Inputs (schemas)" 段与末尾各一次）——那是原版的形状，不是重复。

---

## 配置与开关

⚠ **`agentic.enabled`（执行器在不在）与 `router.enabled`（选不选这条路）是两条
独立的开关**，**两个都开**多跳才真的跑起来。见 [§5](../../docs/config-reference.md)。
⚠ 第三条是 **`agentic.mode`（装哪一种执行器）**——它既不选路、也不开关，
只是"装上的是哪一套机制"。

每轮**召回深度**：两种模式**都取请求的 `top_k`**（AML 固定 100）——**不为多跳另造一套检索参数**
（`prefetch_limit` / `weights` 一个都不在这里）。顺序是"融合池 → 精排 → 取前 `per_round_top_k`
（=20）条"。⚠ `per_round_top_k` 管的是**精排之后取几条**，**不是召回几条**，两者别混
（§3.2 那条"先截 N 再精排**不会报错**"的坑就在这）。
⚠ 2026-10-13 **改过一次**：`plan` 第一版是"每轮只召回 20 条"，于是它与 `react` 在
机制 ＋ 每轮深度上**同时不同**，50 题对照的 7 道差**归不了因**
（[`../../../eval/reports/coq-musique-20261011.md`](../../../eval/reports/coq-musique-20261011.md) §8.5）。
⛔ 别改回去。

## 测试

[`../../tests/test_agent.py`](../../tests/test_agent.py)（`react`）与
[`../../tests/test_plan.py`](../../tests/test_plan.py)（`plan`）——**都不发任何真请求**：
假 LLM（协议级替身）、假 retriever、假 Reranker、鸭子类型的假 store。
⚠ 假 store **只实现执行器真正用到的那两样**（`read()` + `fetch_pairs_by_ids`）——
用到第三样就说明本目录越界了。

⚠ 两边的假协作者**刻意各写一份**（同形但独立）：共享夹具会让"改一边的假件"
悄悄改掉另一边的语义，而两个文件测的是**两份不同的实现**。
