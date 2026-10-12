"""LLM 后端的接缝（§12.1 R1）——**本包唯一对外说的两个名字**。

```text
base.py         LLM 协议（一次调用 / 工具调用）+ `LlmUnavailable`
openai_chat.py  OpenAI 兼容的 `/chat/completions` HTTP 客户端（开发期与提交期共用）
```

> ⚠ **本目录到现在才第一次有代码**，而第一个消费者**不是** `agent/`（v1 不做 agentic，
> D13）——是 [`../retrieve/router.py`](../retrieve/router.py) 的查询路由器。
> 这**推翻**了 `llm/CLAUDE.md` 原先那句"v1 里 `agent/` 是 LLM 的唯一消费者、
> v1 的 Search 路径实际不调用 LLM"——那份文档已按实际改写。
> 接缝本身没变、也不许变：**调用点只有这一个**，所以 Step 5 换模型仍是"换一个实现"，
> 而不是"改一批调用方"（R1 对冲 4）。
"""

from tianximem.llm.base import LlmClient, LlmUnavailable
from tianximem.llm.openai_chat import OpenAiCompatChat

__all__ = [
    "LlmClient",
    "LlmUnavailable",
    "OpenAiCompatChat",
]
