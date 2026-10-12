# agent/ — 多跳检索执行器（`MULTI_HOP` 那条路）

**PRD**：§9（Conditional Agentic Search）—— ⚠ **本目录只实现其中一小块，见下**
**决策**：**D41**（2026-10-10）· 键与口径：[`../../docs/config-reference.md`](../../docs/config-reference.md) **§5**

## 要写什么

```text
coq_prompt.py  原版 Chain-of-Query 提示词（⛔ 逐字照用，**只有一版**）
parse.py       两阶段 JSON 提取 + 严格字段校验（缺字段 ⇒ 异常，**不补默认值**）
coq.py         状态机：多轮召回 → 每轮精排 → 一次 LLM → 停/继续 → 合并候选
```

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
不生成答案。**PRD §9 的那些**（`search_chatrecord` / `take_note` / `finish_search`
三个工具、会话级 rank fusion、时间收窄、已见 session 去重）**一个都不在这里**，
本阶段明确不做。⚠ **别在这个目录里把它们长出来**——那会同时改掉轮数与每轮候选数
两个量，而它们是一道预算题（§10）。

---

## 四条边界（越界都**不会报错**）

1. **不生成答案、不改写记忆原文**（§2.1 的红线）。文档正文一律走
   [`../common/render.py`](../common/render.py) 的 `render_pair`（不变式 I1）——
   与索引侧、精排输入是**同一份渲染**。模型给的 `new_query` **只当检索词**用，
   ⛔ 绝不把它当成一条记忆内容（§8.3）。
2. **不做最终精排。** 那一步由 [`../service/pipeline.py`](../service/pipeline.py)
   对**合并后的列表**做**恰好一次**；本目录只交出 `rerank_query`
   （原问题 ＋ **实际执行过的**改写，§9.1）。
   ⚠ **每轮内部**那一次精排是另一件事：产出物不进响应，只为把交给 LLM 的证据筛干净。
3. **不抛异常。** 所有失败都变成 `AgenticResult` 上的一个具名原因
   （`fallback_reason` / `stats.stop_reason`）。理由与路由器同一条：
   一条可选路径的失败**不该让一次正常 Search 变 500**。
4. **不认识 HTTP、不认识路由器。** 接线在 `service/`。
   有 AST 用例钉着（`tests/test_agent.py::test_the_agent_never_imports_the_http_layer_or_the_router`）：
   §13 的消融需要这两条路**各自可关**，互相纠缠会让"关掉它"不再是单变量。

---

## 三处"顺序错了也不报错"的地方

| 地方 | 错了会怎样 |
| --- | --- |
| **每轮先召回、再精排、然后取 Top-N** | 反过来只会重排一堆**已被提前排除**的候选，**结果看起来完全正常**（§3.2） |
| **每轮重建 `索引 → 稳定 ID` 映射** | 复用上一轮的：模型给的索引指向**别的证据**（§5.2） |
| **文档渲染只做一次** | 把已带索引头的文本再嵌一次 ⇒ 提示词出现 `[1]\n[0]\nQ: …`，**模型读得懂它**，于是对应关系悄悄错位 |

还有一条**可证的不变式**（别试图为它造测试）：`history_tokens ≤ max_prompt_tokens`
恒成立——历史证据只可能由**发出去过的**文档组成，而那些已经过预算。
⇒ `_build_documents` 里"连历史都放不下"那条出口**今天不可达**，证明写在它的 docstring 里。

---

## 提示词是**冻结的**

`coq_prompt.py` 的 `COQ_PROMPT` 是方案第四章的原版，**照用、不改写、不简化、
不补示例、不改 JSON 字段**。要动它之前先回答一个问题：
**"这还是原版吗？"**——不是的话，本阶段"以原版 CoQ 为核心"这个前提就没有了。

⚠ 填充用**一遍扫描的 `re.sub`**，不是连续三次 `str.replace`：后者会让一句含字面
`{used_query}` 的 Query 在**第二次替换时被顶掉**，且**没有任何报错**。
⛔ 也不要用 `str.format`（Query 里的 `{` 会让它炸，或者替换出没人预期的输入）。

⚠ 提示词里 `{used_query}` 与 `{retrieved_episodes}` **各出现两次**
（"Inputs (schemas)" 段与末尾各一次）——那是原版的形状，不是重复。

---

## 配置与开关

⚠ **`agentic.enabled`（执行器在不在）与 `router.enabled`（选不选这条路）是两条
独立的开关**，**两个都开**多跳才真的跑起来。见 [§5](../../docs/config-reference.md)。

每轮**召回深度取请求的 `top_k`**（AML 固定 100）——**不为多跳另造一套检索参数**
（`prefetch_limit` / `weights` 一个都不在这里）。

## 测试

[`../../tests/test_agent.py`](../../tests/test_agent.py)——**不发任何真请求**：
假 LLM（协议级替身）、假 retriever、假 Reranker、鸭子类型的假 store。
⚠ 假 store **只实现执行器真正用到的那两样**（`read()` + `fetch_pairs_by_ids`）——
用到第三样就说明本目录越界了。
