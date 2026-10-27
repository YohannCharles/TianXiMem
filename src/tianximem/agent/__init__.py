"""多跳检索执行器（`MULTI_HOP` 那条路）。**两种模式，一份接口**。

| 模式（`agentic.mode`） | 文件 | 是什么 |
| --- | --- | --- |
| **`react`**（默认） | `coq_prompt.py` · `parse.py` · `coq.py` | CoQ：每轮决定下一问 |
| **`plan`** | `plan_prompts.py` · `plan_parse.py` · `plan.py` | **先拆固定计划**，再逐题落地 |

⚠ **两份互不引用、互不改写**：共享的只有三个纯工具（`_ms` / `_usage` /
`_render_document`）与 `extract_json_object`。

## 边界（四条，越界都不会报错）

1. **不生成最终答案**：`plan` 的中间跳答案只用来填占位，不进响应 / 索引 / 库。
2. **不做最终精排**——那一次由 [`../service/`](../service/) 对合并后的列表做**恰好一次**。
3. **不是通用 Agent**：没有工具调用，也没有并行的子查询执行器。
4. **不认识 HTTP、不认识路由器**：接线在 `service/`（有 AST 用例钉着）。

完整边界和两种模式的差异由各自的模块实现，并由服务的最终排序与打包路径约束。
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
