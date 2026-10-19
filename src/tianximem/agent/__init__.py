"""多跳检索执行器（`MULTI_HOP` 那条路）。**两种模式，一份接口**。

| 模式（`agentic.mode`） | 文件 | 是什么 |
| --- | --- | --- |
| **`react`**（默认） | `coq_prompt.py` · `parse.py` · `coq.py` | **原版** CoQ：每轮决定下一问 |
| **`plan`** | `plan_prompts.py` · `plan_parse.py` · `plan.py` | **先拆固定计划**，再逐题落地 |

⚠ **两份互不引用、互不改写**：原版 CoQ 提示词是**冻结**的，Plan 的两条提示词是**独立**的。
删掉任何一份都不影响另一份。⛔ 共享的只有三个纯工具
（`_ms` / `_usage` / `_render_document`，由 `plan.py` 从 `coq.py` 导入）与
"JSON 从一段文本里挖出来"那一件与业务无关的事（`extract_json_object`）。

## 边界（三条，越界都不会报错）

1. **不生成最终答案**（`plan` 模式会生成**中间跳**的答案，它只用来填占位：
   不进响应、不进索引、不落库）；两条路都**只产出候选与名次**（§2.1 的红线）；
2. **不做最终精排**——那一步由 [`../service/pipeline.py`](../service/pipeline.py)
   对合并后的列表做**恰好一次**，本包只交出"该用哪个 Query 去排"；
3. **不是通用 Agent**：没有工具调用、没有 SplitQuery 并行子查询。
   ⛔ 那些属于 PRD §9 的**另一件事**，本阶段明确不做——**别在这个包里长出来**。

## 它与 `retrieve/router.py` 的关系

路由器**选路**（两个枚举值之一），本包**执行**其中一条路。两者不互相调用：
路由器不认识本包，本包也不认识路由器。接线在
[`../service/pipeline.py`](../service/pipeline.py) 的 `_search`——
⚠ **它按 `route` 选路、不按模式选路**："选哪个执行器"在**装配期**
（[`../service/app.py`](../service/app.py) 的 `build_agentic`）就定死了。

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
from tianximem.agent.plan import (
    PLAN_MAX_STEPS,
    PLAN_RECOVERY_STATUSES,
    PLAN_RECOVERY_TRIGGERS,
    PLAN_STOP_REASONS,
    PlanAgent,
    PlanRecovery,
    PlanResult,
    PlanStats,
    PlanStepRecord,
)
from tianximem.agent.plan_parse import (
    PLAN_STEP_KEYS,
    InvalidPlanOutput,
    Plan,
    PlanStep,
    StepAnswer,
    bind_placeholders,
    parse_plan,
    parse_step_answer,
    placeholders_in,
    render_plan,
)
from tianximem.agent.plan_prompts import (
    PLAN_DECOMPOSE_PROMPT,
    PLAN_DECOMPOSE_PROMPT_VERSION,
    PLAN_STEP_PROMPT,
    PLAN_STEP_PROMPT_VERSION,
)

__all__ = [
    "COQ_PROMPT",
    "COQ_PROMPT_VERSION",
    "FINAL_QUERY_MODES",
    "PLAN_DECOMPOSE_PROMPT",
    "PLAN_DECOMPOSE_PROMPT_VERSION",
    "PLAN_MAX_STEPS",
    "PLAN_RECOVERY_STATUSES",
    "PLAN_RECOVERY_TRIGGERS",
    "PLAN_STEP_KEYS",
    "PLAN_STEP_PROMPT",
    "PLAN_STEP_PROMPT_VERSION",
    "PLAN_STOP_REASONS",
    "STOP_REASONS",
    "AgenticResult",
    "AgenticStats",
    "CoqAgent",
    "CoqDecision",
    "InvalidCoqOutput",
    "InvalidPlanOutput",
    "Plan",
    "PlanAgent",
    "PlanRecovery",
    "PlanResult",
    "PlanStats",
    "PlanStep",
    "PlanStepRecord",
    "RoundRecord",
    "StepAnswer",
    "bind_placeholders",
    "normalize_query",
    "parse_coq_decision",
    "parse_plan",
    "parse_step_answer",
    "placeholders_in",
    "render_coq_prompt",
    "render_plan",
]
