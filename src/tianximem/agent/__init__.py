"""多跳检索执行器（`MULTI_HOP` 那条路）。**D13 说的 `agent/` 就是这里。**

| 文件 | 是什么 |
| --- | --- |
| [`coq_prompt.py`](./coq_prompt.py) | **原版** Chain-of-Query 提示词，⛔ 不许改（只有一版） |
| [`parse.py`](./parse.py) | 两阶段 JSON 提取 + 严格字段校验（缺字段 ⇒ 异常，不补默认值） |
| [`coq.py`](./coq.py) | 状态机：多轮召回 → 每轮精排 → 一次 LLM → 停/继续 |

## 边界（三条，越界都不会报错）

1. **不生成答案**，只产出候选与名次（§2.1 的红线）；
2. **不做最终精排**——那一步由 [`../service/pipeline.py`](../service/pipeline.py)
   对合并后的列表做**恰好一次**，本包只交出"该用哪个 Query 去排"；
3. **不是通用 Agent**：没有工具调用、没有自主规划、没有 SplitQuery 并行子查询。
   ⛔ 那些属于 PRD §9 的**另一件事**，本阶段明确不做——**别在这个包里长出来**。

## 它与 `retrieve/router.py` 的关系

路由器**选路**（三个枚举值之一），本包**执行**其中一条路。两者不互相调用：
路由器不认识本包，本包也不认识路由器。接线在
[`../service/pipeline.py`](../service/pipeline.py) 的 `_search`。

⚠ **本包是 LLM 的第二个消费者**（第一个是路由器，D40）。接缝没变：
调用点仍然只有"一次 `complete()`"，所以 Step 5 换模型依旧是"换一个实现"。
"""

from __future__ import annotations

from tianximem.agent.coq import (
    FINAL_QUERY_MODES,
    STOP_REASONS,
    AgenticResult,
    AgenticStats,
    CoqAgent,
    RoundRecord,
    normalize_query,
    render_coq_prompt,
)
from tianximem.agent.coq_prompt import COQ_PROMPT, COQ_PROMPT_VERSION
from tianximem.agent.parse import CoqDecision, InvalidCoqOutput, parse_coq_decision

__all__ = [
    "COQ_PROMPT",
    "COQ_PROMPT_VERSION",
    "FINAL_QUERY_MODES",
    "STOP_REASONS",
    "AgenticResult",
    "AgenticStats",
    "CoqAgent",
    "CoqDecision",
    "InvalidCoqOutput",
    "RoundRecord",
    "normalize_query",
    "parse_coq_decision",
    "render_coq_prompt",
]
