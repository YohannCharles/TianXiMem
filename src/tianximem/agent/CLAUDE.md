# agent/ — Conditional Agentic Search

> ## ⛔ v1 本目录不实现（[`../../../docs/decisions.md`](../../../docs/decisions.md) D13）
>
> **v1 默认"证据充足"，没有 Agentic Search、没有证据补充、没有关键词重写。** 下文的循环、工具集、参数、监控**整块归 v2**。
>
> **但 Evidence Checker 的接缝要留在 v1**：实现为**恒返回"充足"、且必须记录每轮判定**的空实现。记录的理由见 D13——不记，**Step 4 之前永远无法用数据回答 A4「agent 值不值」**。
>
> **v1 的代价必须正视**：没有证据补充路径，**召回是一次性天花板**——正确的 QA 对不在初始候选里就永久丢了。**v1 的全部重量因此压在"排序 + token 预算分配"上**，而这正是 [`../rank/`](../rank/) 与 Neighbor Expansion 的职责。

**PRD**：§9（含 §5 的流程图注释）

## 要写什么（v2）

```text
loop.py     agent 循环：门控、轮数、合并
tools.py    search_chatrecord / take_note / finish_search
```

**只在 Evidence Checker 判定不足时启动。这是本项目的核心 claim，不可砍**（§9）。

> ⚠ **但 v1 里这条 claim 是平凡成立的**（根本没有 agentic），**不是被设计出来的**。若最终材料要引用它，**必须由 v2 或 A4a/A4b 支撑，不能引用 v1 的分数**（D13）。

---

## 为什么"条件"两个字是全部重点

§3.1 的核心原则：

> **不让所有 Query 都进入昂贵的 Agentic Search**，优先使用低成本、高稳定性的检索；**只有证据不足时才触发** Agent 多轮搜索。

因此本目录有**两个同等重要的失败方向**：

| 失败 | 表现 | 触发什么动作（§14） |
| --- | --- | --- |
| **门控太松** | Agent Trigger Rate 偏高 | Checker 太保守 |
| **门控太紧** | Agent Trigger Rate **接近 0** | **agent 没起作用**——核心 claim 事实上没有被验证 |

**只看平均分看不出这两种失败**——必须同时看 Trigger Rate。这也是 §13 要把 A4 拆成 A4a/A4b 两个 arm 的原因（见 [`../../../docs/experiments.md`](../../../docs/experiments.md)）。

---

## 参数：沿用 ReFind 的形状（§9）

> 这套参数**是在真实评测上跑出来的，改它需要理由**。

```text
最多 4 轮
每轮内部检索 top_k = 5
工具集：
  search_chatrecord(query, date_from?, date_to?)  -- 复用 §7 的 hybrid 检索
  take_note()                                     -- 保存上一轮结果原文，存后即不可再访问
  finish_search()
```

- **Agent 只能给出关键词与日期范围**，其余控制由服务端自动施加

### ⚠ `4` 是"检索轮数"的上限，不是"LLM 调用次数"的上限

**ReFind 实测每 query 平均约 5.0 次 LLM 调用**（论文 Table 11：4.99）——一轮里可能**既有决策调用也有收尾调用**。

> **§14 要监控的是这个均值，不要拿 4 去卡它。**

### 复用的四个 chat-native 控制

会话级 rank fusion · 邻域扩展 · 时间收窄 · 已见 session 去重（§9）。这四个**由服务端施加**，不是 agent 的工具。

### 工具命名

> 工具名沿用了 ReFind 论文的 `search_chatrecord`。**我们的实现是自有的，改名无妨，但保持与论文同形有利于对照实验时的归因**（§9）。

**⚠ 一条边界**：`eval/baselines/refind/` 会把 ReFind **原版**（MIT）作为 B1 基线 Vendor 进来，那是**不改动的外部代码**，与"本项目从零搭建、不复用任何既有代码"（§1）**不冲突**——**冲突只会在有人把那份代码 import 进 `src/` 时发生**。**`src/tianximem/agent/` 里的任何一行都必须是我们自己写的。**

---

## 产出与初始候选是"合并"关系，不是替换（§9）

**两者一起进 [`../rank/`](../rank/) 的 Rerank，按 `id` 去重。**

**理由**：初始那一路里**往往已经有正确答案**——Agent 的价值是**补上它找不到的那部分，而不是推翻它**。

> 这也是 §4"选择与排序是瓶颈"的直接应用：**Agent 只负责把候选弄全，排序仍归 Rerank。**

**⚠ 与 §13 纯度规则的关系**：A4 关掉 agent 时，**打包顺序与候选数量不得因此改变**——否则对照不成立，"而结果看起来完全正常，只是结论错了"。

---

## 重点处理的情况（§9）

Multi-hop · Temporal · Knowledge Update · Ambiguous · 信息分散

---

## 监控

本目录是 **Agent Trigger Rate / 平均轮数 / Rewrite 次数**的**发射方**；读法、阈值含义与聚合都在 [`../observability/CLAUDE.md`](../observability/CLAUDE.md)。

---

## 依赖

- 检索：[`../retrieve/`](../retrieve/)（复用 §7 的 hybrid 检索，**不另起一套**）
- LLM：[`../llm/`](../llm/)（**v1 里它是 LLM 的唯一消费者**）
