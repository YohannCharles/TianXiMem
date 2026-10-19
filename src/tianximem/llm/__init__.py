"""LLM 后端的接缝（§12.1 R1）——**本包唯一对外说的两个名字**。

```text
base.py         LLM 协议（一次调用 / 工具调用）+ `LlmUnavailable`
openai_chat.py  OpenAI 兼容的 `/chat/completions` HTTP 客户端（开发期与提交期共用）
```

> ⚠ **三个消费者**（2026-10-14 起全部默认开）：[`../retrieve/router.py`](../retrieve/router.py)
> 的查询路由器、[`../agent/`](../agent/) 的多跳执行器、[`../service/grounded_planner.py`](../service/grounded_planner.py)
> 的受限计划器。接缝没变：调用点仍然只有"一次 `complete()`"（或一次工具调用），
> 所以 Step 5 换模型仍是"换一个实现"，而不是"改一批调用方"（R1 对冲 4）。
"""

from tianximem.llm.base import LlmClient, LlmUnavailable
from tianximem.llm.openai_chat import OpenAiCompatChat

__all__ = [
    "LlmClient",
    "LlmUnavailable",
    "OpenAiCompatChat",
]
