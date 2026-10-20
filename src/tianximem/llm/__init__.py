"""LLM 后端的公共接口。

```text
base.py         LLM 协议（一次调用 / 工具调用）+ `LlmUnavailable`
openai_chat.py  OpenAI 兼容的 `/chat/completions` HTTP 客户端（开发期与提交期共用）
```

> ⚠ **三个消费者**：[`../retrieve/router.py`](../retrieve/router.py)
> 的查询路由器、[`../agent/`](../agent/) 的多跳执行器、[`../service/grounded_planner.py`](../service/grounded_planner.py)
> 的受限计划器。所有调用都通过同一个 `complete()` 协议，因此更换模型只需要替换
> 客户端配置或实现，不需要修改调用方。
"""

from tianximem.llm.base import LlmClient, LlmUnavailable
from tianximem.llm.openai_chat import OpenAiCompatChat

__all__ = [
    "LlmClient",
    "LlmUnavailable",
    "OpenAiCompatChat",
]
