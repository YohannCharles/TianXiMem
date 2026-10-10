# llm/ — LLM 后端

**PRD**：§12.1 R1（模型替换）

## 要写什么

```text
base.py        LLM 协议（一次调用）+ `LlmUnavailable`      ✅（工具调用**未实现**——没有消费者）
openai_chat.py OpenAI 兼容的 `/chat/completions` 客户端     ✅
```

> ⛔ **原计划的 `gpt_4o_mini.py` + `qwen_local.py` 已并成一个 `openai_chat.py`**：
> 两个模型**说的是同一套线格式**，差别只有 `base_url` / `key` / `model` 三个值，
> 而**那三个值全部来自 `.env`**。按模型分文件会造出两个内容相同的类，
> 而它们的唯一差别只能靠"哪一行 import 了谁"表达——**那种差别看不出来、也 diff 不出来**。
> R1 的接缝由 [`base.py`](./base.py) 的**协议**承担，**不由文件名承担**。
> （`embed/` 那边确实按模型名分实现，因为 `text-embedding-v4` 与 Qwen3-Embedding
> **真的不同**——批大小是端点硬限。**这里没有那类差异**。）

---

## 为什么单独一个目录

**因为它是 R1 的接缝。** §12.1 的偏离是"开发期用 qwen3.5-9b，提交前换回 `gpt-4o-mini`"——⚠ **R1 同时整体替换两处模型**（LLM 与 embedding，见 [`../../../docs/decisions.md`](../../../docs/decisions.md) D2；embedding 那一半的接缝在 [`../embed/`](../embed/)），**LLM 这一半由本目录收口**。若调用点散落在 [`../agent/`](../agent/) 各处，Step 5 就不是"换一个实现"，而是"改一批调用方"——**而 Step 5 明确不可与任何设计改动合并**（R1 对冲 4）。

---

## 使用范围（窄，这是有意的）

| 路径 | 是否调用 LLM |
| --- | --- |
| **`Add` 路径** | **完全不调用**（§7.2）——embedding 是唯一的 Add 侧成本 |
| **`Search` 路径** | **只有一个调用点：[`../retrieve/router.py`](../retrieve/router.py) 的查询路由器**，且仅在 `router.enabled` 时（**默认 `false`**） |

> ### ⚠ 2026-10-10 修正：原写"v1 里 `agent/` 是 LLM 的唯一消费者、Search 路径实际不调用 LLM"
>
> **那已经不对了。** 三分类查询路由器（[`../retrieve/router.py`](../retrieve/router.py)）是
> LLM 的**第一个真实消费者**，而它不在 `agent/` 里（v1 不做 agentic，D13 没变）。
> 接缝本身**没变、也不许变**：**调用点仍然只有一个**，所以 Step 5 换模型依旧是
> "换一个实现"而不是"改一批调用方"。
>
> ⚠ **但有一件事确实变了**：Search 路径**从"零 LLM 成本"变成"至多一次 LLM 调用"**。
> 这是 §7.2 那条成本属性的**唯一一处松动**——所以它必须**默认关着**
> （见 `configs/default.yaml` 的 `router.enabled`），并且**每次调用都要能在 §14 里被数出来**
> （`SearchObservation.route_latency_ms`）。
> ⛔ **Add 路径的零 LLM 属性一字未动**：`facts/` 仍然不调用任何模型（见下）。

> **Add 侧的成本属性**（§7.2）：只与内容有关，缓存后即成为**一次性成本，与迭代次数无关**。若在 Add 路径里加了任何 **LLM 调用**（摘要、实体归并…），这个属性立刻失效。

**因此**：任何"顺手用 LLM 在写入时抽个摘要 / 归并实体"的想法，都属于 v2，**且会同时破坏成本属性与 §11.3 的"原文优先"**。⚠ **Add 路径确实有确定性的共同事实抽取**（[`../facts/`](../facts/)）——它不经过本目录、不调用任何模型，所以上面那条成本属性原样成立；**别把它当"Add 可以调 LLM"的先例**。

---

## 模型规定与开发期替换

**模型规定表、开发期替换的代价、以及四条对冲着陆点在根 `CLAUDE.md` 与 [`../../../docs/decisions.md`](../../../docs/decisions.md) D2。** 与写本目录代码直接相关的只有一条：

> **所有阈值必须是配置项**（对冲 3）——包括 agent 的轮数、工具参数、prompt 相关常量。

端点在 `.env`（`AML_BASE_URL` / `AML_API_KEY` / `AML_MODEL`），阈值在
`configs/*.yaml` 的 `router.*`——**两层分工不重叠**（见 [`../../../configs/CLAUDE.md`](../../../configs/CLAUDE.md)）。
⚠ **本目录不读配置、不读环境变量**（全包只有 [`../common/config.py`](../common/config.py) 能读），
它只接收构造方传进来的 `base_url` / `api_key` / `model`。

> ⚠ **本目录是"顺手微调一下"的两处高发地之一**（另一处是 reranker）。**数据不得用于训练，「微调」不是暂不实现、是不允许**——理由与出处提醒见根 `CLAUDE.md`。

