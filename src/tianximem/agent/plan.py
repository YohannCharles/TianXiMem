"""Plan 模式的多跳执行器（`MULTI_HOP` 那条路的**第二种执行器**，2026-10-12 新增）。

它与 [`coq.py`](./coq.py) 的 **React 模式**并列，由 `agentic.mode` 选择：

| 模式 | 执行器 | 提示词 | 怎么推进 |
| --- | --- | --- | --- |
| `react`（**默认**） | `CoqAgent` | 原版 CoQ（**冻结**） | 每轮重新决定"充分了吗" |
| `plan` | `PlanAgent` | `plan_prompts.py`（**两条，独立**） | **先拆固定计划**，再逐题落地 |

⛔ **本文件不动 React 的任何一行**：`coq.py` / `coq_prompt.py` 与本次改动之前逐字相同。
三个纯工具（`_ms` / `_usage` / `_render_document`）**直接复用**它们那一份——
⛔ **不要在这里再写一遍**：`_usage` 里那条"排除 bool"是踩出来的
（`{"prompt_tokens": true}` 会被读成 1 个 token，而在 §14 的汇总里**看不出来**）。

## 一条请求的完整形状

```text
原始 Query
   │
   ├─→ ① 拆解（一次 LLM，独立提示词）⇒ 1~4 步的有序计划（含 {{sN}} 占位）
   ├─→ ② 用**第一个子问题**检索（⛔ 不先检索原题）
   ├─→ ③ 每一轮：
   │       a. 程序把已知答案填进本步模板 ⇒ 本轮实际执行的 Query
   │       b. 混合召回 **`top_k`（=100）** → 精排这 100 条 → 取前 `per_round_top_k`（=20）
   │          ⇒ 本轮片段（索引从 0 起，**每轮重建**）
   │       c. 一次 LLM：**只回答本轮**（⛔ 不回步骤编号、不回下一查询）
   │       d. 程序校验：索引合法、引句逐字存在；**下一查询由程序按固定模板绑定**
   ├─→ ④ 停？（计划跑完 / 依赖未解 / 绑定冲突 / 输出非法 / 轮数用尽 / 模型或检索失败）
   └─→ ⑤ ⭐ **恢复分支**（`unresolved` **或** `conflict` 触发；**程序侧、一次性**）：
          按**请求的 `top_k`** 补一次**原题**检索并**精排一次**（挑哪几条进池），
          与已有候选按名次去重合并、补到池子满为止（不够不凑）；最终精排**只用原题**
          ⇒ 各轮候选按稳定 ID 去重，交回调用方（最终精排与打包在编排层）
```

⚠ **b 那一步与 React 逐字相同**（2026-10-13 改）：两边都是"融合池（`top_k`）→ 精排 → 取前 N"，
⇒ **两条路的差别只剩"推进机制"**。⛔ 第一版按方案的字面要求让这里**只召回 20 条**，
于是两模式在**两个**维度上同时不同、归因不了——理由与实测见
[`coq-musique-20261011.md`](../../../eval/reports/coq-musique-20261011.md) §8.5。

## 七条边界（越界都**不会报错**）

1. **它不生成最终答案。** 它拿到的 `answer` 是**中间跳的答案**（用来填占位），
   与 §2.1 的红线不冲突：**它不进响应、不伪装成记忆、不写进任何库、不进向量索引**。
   ⚠ 它在响应之外**以"模型生成的执行记录"的身份**被交出去（`plan_history`）——
   标注必须一直带着，⛔ **不许把它当原文**。
2. **不做最终精排。** 与 React 同一个理由：那一次由
   [`../service/pipeline.py`](../service/pipeline.py) 对合并后的列表做**恰好一次**。
3. **不抛异常**（构造参数非法除外）。所有失败都变成 `PlanResult` 上的一个具名原因。
4. **不认识 HTTP、不认识路由器。** 接线在 `service/`（`_search` 那条路不区分执行器）。
5. **不自由重规划、不做补救查询、不跨步执行**：计划一旦拆出来就是**固定的**；
   某一步解不出来就停下并记名，而不是换个问法再试。
6. **每轮只记当前步骤的回答**：v2 起提示词只要 `status` / `answer` / `evidence`
   三个字段 ⇒ "模型回了别的步骤"**结构性地不可能**（步骤编号由程序附加）；
   ⛔ 模型也**不再**产出下一查询，所以"自由改写被采用"这条路整条没有了。
7. ⭐ **"走不通"之后的那一次原题检索也是程序的**（`_recover`；触发面
   `unresolved` **或** `conflict`）：模型**不知道**它发生过、也没有第二个问题要答。
   候选池的补充**不属于任何一步** ⇒ ⛔ 不计步骤、不加 LLM 调用、不截 Top-20。
   ⚠ 它**自己精排一次**（挑哪几条进池）——那不是"第二次最终精排"，见 `_rerank_fallback`。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from time import monotonic
from typing import Final

from tianximem.agent.coq import _ms, _render_document, _usage
from tianximem.agent.plan_parse import (
    InvalidPlanOutput,
    Plan,
    PlanStep,
    StepAnswer,
    bind_placeholders,
    parse_plan,
    parse_step_answer,
    render_plan,
)
from tianximem.agent.plan_prompts import (
    PLAN_DECOMPOSE_PROMPT,
    PLAN_DECOMPOSE_PROMPT_VERSION,
    PLAN_DECOMPOSE_SLOTS,
    PLAN_STEP_PROMPT,
    PLAN_STEP_PROMPT_VERSION,
    PLAN_STEP_SLOTS,
)
from tianximem.common.render import day_granularity, render_pair
from tianximem.common.tokens import TokenCounter
from tianximem.llm import LlmClient, LlmUnavailable
from tianximem.rank import Reranker, RerankUnavailable
from tianximem.retrieve.fusion import Candidate, HybridRetriever
from tianximem.store.sqlite_store import QaPair, SqliteStore

__all__ = [
    "FINAL_QUERY_MODES",
    "PLAN_MAX_STEPS",
    "PLAN_RECOVERY_STATUSES",
    "PLAN_RECOVERY_TRIGGERS",
    "PLAN_STOP_REASONS",
    "PlanAgent",
    "PlanRecovery",
    "PlanResult",
    "PlanStats",
    "PlanStepRecord",
    "render_completed_history",
    "render_plan_decompose_prompt",
    "render_plan_step_prompt",
    "step_line",
]

#: 停止原因。⚠ 与 React 的 `STOP_REASONS` **不是同一张表**：两边的推进机制不同，
#: 共用一张表只会让"哪条路停在哪"变得无法统计。**它们要能在 §14 里分开数。**
PLAN_STOP_REASONS: Final[tuple[str, ...]] = (
    # 计划的最后一步**也被回答之后**才记它
    "plan_complete",
    # 模型说这一步解不出来（`status="unresolved"`）——**不猜、不跳过**
    "dependency_unresolved",
    # 模型说证据互相冲突（`status="conflict"`）
    "binding_conflict",
    # 程序侧的绑定失败：模板里还有没填上的 `{{sN}}` ⇒ **绝不送进检索器**
    "unbound_placeholder",
    # 本轮输出过不了校验（字段不符 / 索引越界 / 引句不在本轮文档里）
    "invalid_output",
    # 拆解那一次的输出过不了校验
    "plan_invalid",
    # 预算用尽：计划还有下一步，但 `agentic.max_rounds` 到了
    "max_rounds",
    "llm_unavailable",
    "llm_error",
    "retrieval_error",
    "prompt_budget_exceeded",
)

#: 计划的步数上限（方案是 1～4）。⚠ **不是** `agentic.max_rounds`，见 `parse_plan` 的说明。
PLAN_MAX_STEPS: Final[int] = 4

#: ⭐ **`unresolved` 之后原题兜底的池子口径**（2026-10-15 改）。
#:
#: ⛔ **不再是"按轮数算的剩余名额"**——旧口径 `max(0, 80 − 已执行轮数 × per_round_top_k)`
#: （⇒ 补 60/40/20/0，且**池子恒为 80**）**已删除**。
#:
#: 现在的口径就是**请求的 `top_k`**：兜底按 `top_k` 召回 → 精排一次 → 按名次去重补位，
#: **补到池子达到 `top_k` 为止**；兜底给不够就**停在有什么算什么**（⛔ 绝不硬凑）。
#:
#: ⇒ 池子上限从"恒 80（= `max_rounds` × `per_round_top_k`）"变成 **≤ `top_k`**。
#: ⚠ 这条把兜底与**每一轮**的召回口径**对齐**了（每一轮也是按请求的 `top_k` 召回），
#: 且与 `per_round_top_k` **不再耦合**（改那个数不再牵动兜底）。
#: 见 `_recover` 与 `docs/config-reference.md` §5。

#: 恢复分支的四种状态。⚠ **"触发但池子已经够了"与"真的跑了兜底"靠它分开**——
#: 只看 `entered` 会把"没检索"读成"兜底过"。
PLAN_RECOVERY_STATUSES: Final[tuple[str, ...]] = (
    "not_triggered",
    "success",
    "error",
    "skipped_pool_full",
)

#: ⭐ **哪些停止原因会触发恢复分支**（2026-10-15 由"只有 `unresolved`"放宽）。
#:
#: 判据是**"模型明确报告了它没走通"**，⛔ 不是"哪一步失败了"：
#:
#: | 停止原因 | 谁报告的 | 触发？ |
#: | --- | --- | --- |
#: | `dependency_unresolved` | **模型**：这一步解不出来 | ✅ |
#: | `binding_conflict` | **模型**：证据互相冲突 | ✅ |
#: | `plan_invalid` / `llm_unavailable` / `llm_error` / `retrieval_error` | **程序或端点**：工具坏了 | ⛔ 不触发 |
#: | `unbound_placeholder` / `invalid_output` / `max_rounds` / `plan_complete` | 程序（绑定 / 校验 / 预算 / 跑完） | ⛔ 不触发 |
#:
#: **为什么把 `conflict` 也收进来**：它与 `unresolved` 是**同一类事**——模型读到手上的证据后
#: **明确说了"我走不通"**。那时候选池里往往只有前几轮攒下的一小撮，直接收口等于
#: **把"模型已经发现有问题"当成"没事"**。补一次原题检索正对应"退回到老实按原题捞一遍"。
#:
#: ⛔ **工具性失败仍不触发**：`plan_invalid` / `llm_*` / `retrieval_error` 是**我们这边坏了**，
#: 它们的处置是"保留已有候选、把原因原样交出去"（`fallback_reason` 只在一条都没攒到时才落）。
#: 把它们也拉进来会让"哪一种失败走了哪条路"再也数不清——**那条理由没变。**
PLAN_RECOVERY_TRIGGERS: Final[frozenset[str]] = frozenset(
    {"dependency_unresolved", "binding_conflict"}
)

#: 最终精排 Query 的两种构造方式。**与 React 同名同义**（取值域定义在 `config.py`）。
FINAL_QUERY_MODES: Final[tuple[str, ...]] = ("combined", "original")

_DECOMPOSE_FILLER: Final[re.Pattern[str]] = re.compile(
    "|".join(re.escape(slot) for slot in PLAN_DECOMPOSE_SLOTS)
)
_STEP_FILLER: Final[re.Pattern[str]] = re.compile(
    "|".join(re.escape(slot) for slot in PLAN_STEP_SLOTS)
)


@dataclass(frozen=True, slots=True)
class PlanStepRecord:
    """一步的**完整过程**（实验模式下要能重建"模型当时看到了什么"）。"""

    index: int
    step_id: str
    #: 计划里的**模板**（可能含占位）。
    template: str
    #: ⭐ **这一轮真正执行的 Query**（占位已由**程序**填好）——不是模板。
    executed_query: str
    recalled_candidates: int
    #: 本轮发给模型的文档 ID，**顺序即索引序**（每轮重建）。
    document_ids: tuple[str, ...]
    #: 本轮进候选池的稳定 ID（各轮并起来就是最终候选，按稳定 ID 去重）。
    candidate_ids: tuple[str, ...]
    #: 模型引用的稳定 ID（索引已换算）与它抄回来的引句（**诊断用**）。
    evidence_ids: tuple[str, ...]
    evidence_quotes: tuple[str, ...]
    #: 引句**没有**逐字出现在对应文档里的条数（被丢弃，但必须能被看见）。
    quotes_unverified: int
    #: `None` = 这一轮没拿到可用的模型输出（停止原因说明为什么）。
    status: str | None
    answer: str | None
    #: ⭐ 程序**自己**用"固定模板 ＋ 已确立答案"算出来的**下一查询**——
    #: **送给检索器的是它**。⚠ v2 起模型**根本不产出**下一查询，所以这里没有"两者比对"
    #: （v1 那个 `next_query_mismatch` 统计随之取消：它量的是"模型有没有照着模板改写"，
    #: 而这件事现在已经不是模型的任务了）。
    canonical_next_query: str | None
    #: 模型这一轮实际看到的"已完成历史"文本（诊断用；含数据集原文 ⇒ 别进 git）。
    history_text: str
    dropped_documents: int
    #: `disabled` / `called` / `degraded`（D12 的三分口径）。
    rerank_status: str
    extraction: str | None
    stop_reason: str | None
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class PlanRecovery:
    """⭐ **恢复分支**（2026-10-13；**2026-10-15 触发面放宽到 `conflict`**）——
    依赖链"走不通"之后那条**程序侧、一次性**的候选池补充（`PlanRecovery`）。

    | `status` | `entered` | 检索了吗 | 什么时候 |
    | --- | --- | --- | --- |
    | `not_triggered` | `False` | 否 | 计划正常跑完 / 停在**工具性失败**或程序性停止上 |
    | `skipped_pool_full` | `True` | **否**（池子已 ≥ 目标） | 请求的 `top_k` 比已有池子还小时 |
    | `success` | `True` | 是 | 兜底检索返回了候选 |
    | `error` | `True` | 是（失败了） | 兜底检索抛异常 ⇒ **已有候选一条不动**，继续最终精排与答题 |

    ⚠ 是否进入由 `PLAN_RECOVERY_TRIGGERS` 决定（`unresolved` **或** `conflict`）。

    ⛔ 它**不是** `fallback_reason`：那个词在本仓的含义是"整条路降级回混合检索、
    执行器的候选全丢"。这里**恰好相反**——候选一条都不丢，只是**补一批**。
    """

    #: 是否进入恢复分支（= 触发条件成立）。
    entered: bool = False
    status: str = "not_triggered"
    #: ⭐ **已执行轮数**（**含刚刚返回 `unresolved` 的那一轮**）——**只作诊断**：
    #: 2026-10-15 起补位名额**不再按它算**（与轮数解耦，见 `_recover`）。
    executed_rounds: int = 0
    #: 补位**目标**（= 请求的 `top_k`）：池子到此为止，⛔ **不够也不凑**。
    target_top_k: int = 0
    #: 兜底**动手前**池子里已有多少条（诊断用；`target − pool_before` 才是要补的名额）。
    pool_before: int = 0
    #: 兜底**实际召回**条数（`success` 时才有意义）。
    recalled: int = 0
    #: 兜底那一次精排的三分状态（`called` / `degraded` / `disabled`）。
    #: ⚠ 它**不受 `agentic.round_rerank_enabled` 管**（那个开关只管"每轮"）——
    #: 兜底精排是**另一个机制**：决定"`top_k` 条里哪几条有资格进池"，**常开**。
    reranked: str = "disabled"
    #: 合并去重后**真正新增**的候选数（全都已在池子里时是 0）。
    added_unique: int = 0
    #: 兜底那一次检索 **＋ 精排** 的耗时（**如实记录**：它是真实发生的用量）。
    latency_ms: float = 0.0
    #: 最终精排用哪一句：`original`（恢复分支，只用最初的原题）／`joint`（其余，现状）。
    rerank_query_mode: str = "joint"


@dataclass(frozen=True, slots=True)
class PlanStats:
    """一次 Plan 多跳 Search 的汇总（§14）。**不进响应。**

    ⚠ 前九个字段**与 `AgenticStats` 同名同义**：编排层的 `_record_agentic` 读的正是
    这几个名字，于是"两条路怎么记账"只有一处实现。
    """

    rounds: tuple[PlanStepRecord, ...]
    stop_reason: str
    fallback_reason: str | None
    candidate_count: int
    unique_candidate_count: int
    selected_evidence_count: int
    joint_query_count: int
    llm_calls: int
    rerank_calls: int
    latency_ms: float
    #: 拆解出来的完整计划（含没跑到的那些步）。
    plan: tuple[PlanStep, ...] = ()
    #: 拆解 / 逐轮**分开计数**——合成一个数就看不出"拆解失败"这一类了。
    decompose_llm_calls: int = 0
    step_llm_calls: int = 0
    #: 计划的步数 vs 真的执行了几步。⚠ 两者不等**不是错误**（`max_rounds` 会截断），
    #: 但"计划 4 步只跑了 2 步"必须能被看见。
    plan_steps: int = 0
    steps_executed: int = 0
    #: ⭐ 计划里有几步的 `depends_on` **声明与占位不一致**（被程序规范化过）。
    #: ⚠ **不是失败**（程序只按占位绑定），但必须能被数出来：v2 提示词下 50 题里 5 题命中。
    dependencies_repaired: int = 0


@dataclass(frozen=True, slots=True)
class PlanResult:
    """执行器交给编排层的全部东西。

    ⚠ **前四个字段与 `AgenticResult` 同名**（`candidates` / `rerank_query` / `stats` /
    `fallback_reason`）⇒ 编排层的 `_search` **一行都不用改**。

    `plan_history` 是**模型生成的执行记录**（要求"另行保存"）。
    ⛔ 它**不是记忆**：没有编造 memory id、不进向量索引、不进响应；
    要进答案上下文时只能**以执行记录的身份前置**（本地评测那边做的事）。
    """

    candidates: tuple[Candidate, ...]
    rerank_query: str | None
    stats: PlanStats
    fallback_reason: str | None = None
    #: 逐步的执行记录（含实际 Query、答案、引句、停止原因）。
    plan_history: tuple[PlanStepRecord, ...] = ()
    #: 拆解出来的计划本身（`plan_history` 只有跑到的那些步）。
    plan: tuple[PlanStep, ...] = ()
    #: ⭐ `unresolved` 恢复分支的决定与结果（没触发时是 `not_triggered`）。
    recovery: PlanRecovery = PlanRecovery()


@dataclass(slots=True)
class _State:
    """一次 Search 的临时状态。**不持久化，也不跨请求共享**（同 React）。"""

    #: 实际执行过的 Query，按顺序。
    executed: list[str] = field(default_factory=list)
    #: `step_id → 本轮实际执行的 Query`（历史那一段要按步号对上问题）。
    executed_for: dict[str, str] = field(default_factory=dict)
    #: 各轮候选的稳定 ID，**保序去重**（第一轮的名次最先）。
    ordered_ids: dict[str, None] = field(default_factory=dict)
    #: 模型引用过的稳定 ID，保序去重。
    evidence: list[str] = field(default_factory=list)
    #: 稳定 ID → 文档正文（`render_pair` 的产物）。
    bodies: dict[str, str] = field(default_factory=dict)
    #: 稳定 ID → 日期串（**只到日**，D21）。
    days: dict[str, str] = field(default_factory=dict)
    #: `step_id → 已确立的答案`。**只放 `answered` 的那一步**（解不出来的不进）。
    answers: dict[str, str] = field(default_factory=dict)
    #: `step_id → 这一步引用过的稳定 ID`（历史那一段的"来源 ID"）。
    sources: dict[str, tuple[str, ...]] = field(default_factory=dict)


# ── 提示词填充（一遍扫描）────────────────────────────────────────────────


def _fill(
    pattern: re.Pattern[str], template: str, slots: Sequence[str], values: dict[str, str]
) -> str:
    """把槽位填进提示词。**不改提示词本身，也不重扫替换结果。**

    ⚠ 与 `render_coq_prompt` **同一条理由、同一套写法**（这里只是把槽位表参数化）：
    值来自**模型/用户文本**，连续替换会让其中的字面槽位被**二次替换**且不报错；
    `str.format` 会被模板里的 JSON 示例和 `{{s1}}` 直接炸掉（或替换出没人预期的输入）。

    ⚠ 替换前先断言槽位都在——**提示词被人动过**要在调用期响亮失败，
    而不是把一份还带着 `{...}` 的提示词发给模型（它会照着抄）。
    """
    missing = [slot for slot in slots if slot not in template]
    if missing:
        raise ValueError(f"提示词缺少占位符：{missing}")
    return pattern.sub(lambda match: values[match.group(0)], template)


def render_plan_decompose_prompt(*, original_question: str) -> str:
    """填充**拆解**提示词。"""
    return _fill(
        _DECOMPOSE_FILLER,
        PLAN_DECOMPOSE_PROMPT,
        PLAN_DECOMPOSE_SLOTS,
        {"{original_question}": original_question},
    )


def step_line(step: PlanStep) -> str:
    """一步在提示词里那一行的**唯一写法**：`s2: Who was born in {{s1}}?`。

    ⚠ **只有这一处**：模型要靠 `sN` 把"现在这一问"与计划里那一步对上
    （它⛔ **不再回步骤编号**——编号由程序附加，但提示词里仍要标明在答哪一步）。
    """
    return f"{step.id}: {step.query}"


def render_plan_step_prompt(
    *,
    original_question: str,
    retrieval_plan: str,
    completed_history: str,
    current_step: str,
    executed_query: str,
    current_documents: Sequence[str],
) -> str:
    """填充**逐轮**提示词（v2，2026-10-13）。

    ⚠ `{current_documents}` 是**已经带索引头**的那份列表，`{current_step}` 是**模板**
    （含 `{{sN}}`）。⛔ **没有 `{next_step}`**：下一查询由**程序**绑定，
    模型既不需要看模板、也不再被要求产出一句 `next_query`（那是 v1 的形状）。
    """
    return _fill(
        _STEP_FILLER,
        PLAN_STEP_PROMPT,
        PLAN_STEP_SLOTS,
        {
            "{original_question}": original_question,
            "{retrieval_plan}": retrieval_plan,
            "{completed_history}": completed_history,
            "{current_step}": current_step,
            "{executed_query}": executed_query,
            "{current_documents}": "\n\n".join(current_documents),
        },
    )


def render_completed_history(plan: Plan, state: _State) -> str:
    """已完成步的**精简**历史。

    ⛔ **只传**过去的问题、答案、来源 ID——**不传旧检索片段全文，也不传旧引用原句**
    （要求里明说了这一条）。旧片段留在 `plan_history` 里供复核（诊断记录），
    **不进提示词**：它是这条路上唯一会长大的东西，而它长大之后的表现是
    "模型被旧上下文带偏"，不是任何报错。
    """
    blocks: list[str] = []
    for step in plan.steps:
        answer = state.answers.get(step.id)
        if answer is None:
            continue
        sources = ", ".join(state.sources.get(step.id, ())) or "(none)"
        blocks.append(
            f"{step.id} question: {state.executed_for.get(step.id, step.query)}\n"
            f"{step.id} answer: {answer}\n"
            f"{step.id} sources: {sources}"
        )
    return "\n\n".join(blocks) if blocks else "(none yet)"


class PlanAgent:
    """`MULTI_HOP` 的 Plan 模式执行器。**同步、无状态：一次 `search()` 一份请求局部状态。**

    它**不抛异常**（构造参数非法除外）：所有失败都变成 `PlanResult` 上的一个具名原因。
    """

    def __init__(
        self,
        *,
        store: SqliteStore,
        retriever: HybridRetriever,
        llm: LlmClient,
        counter: TokenCounter,
        reranker: Reranker | None = None,
        max_rounds: int = 4,
        per_round_top_k: int = 20,
        max_prompt_tokens: int = 6000,
        max_output_tokens: int = 768,
        temperature: float = 0.0,
        round_rerank_enabled: bool = True,
        final_rerank_query_mode: str = "combined",
        inject_abs_time: bool = False,
        max_plan_steps: int = PLAN_MAX_STEPS,
        decompose_prompt_version: str = PLAN_DECOMPOSE_PROMPT_VERSION,
        step_prompt_version: str = PLAN_STEP_PROMPT_VERSION,
    ) -> None:
        # ⚠ 这两个是 `agentic.plan_*_prompt_version`，**不是** `agentic.prompt_version`
        #   （那个是冻结的原版 CoQ 的）。⚠ 与 `CoqAgent` / `QueryRouter` 同一套写法：
        #   **这是一条断言**——"配了哪一版提示词"与"跑的是哪一版"必须永远是同一件事。
        if decompose_prompt_version != PLAN_DECOMPOSE_PROMPT_VERSION:
            raise ValueError(
                f"agentic.plan_decompose_prompt_version = {decompose_prompt_version!r}，"
                f"但代码里只有一版提示词 {PLAN_DECOMPOSE_PROMPT_VERSION!r}"
            )
        if step_prompt_version != PLAN_STEP_PROMPT_VERSION:
            raise ValueError(
                f"agentic.plan_step_prompt_version = {step_prompt_version!r}，"
                f"但代码里只有一版提示词 {PLAN_STEP_PROMPT_VERSION!r}"
            )
        if max_rounds < 1:
            raise ValueError("agentic.max_rounds 必须 ≥ 1")
        if per_round_top_k < 1:
            raise ValueError("agentic.per_round_top_k 必须 ≥ 1")
        if max_prompt_tokens <= 0:
            raise ValueError("agentic.max_prompt_tokens 必须为正")
        if max_output_tokens <= 0:
            raise ValueError("agentic.plan_max_output_tokens 必须为正")
        if temperature < 0:
            raise ValueError("agentic.temperature 不得为负")
        if not 1 <= max_plan_steps <= PLAN_MAX_STEPS:
            raise ValueError(f"计划步数上限必须在 1..{PLAN_MAX_STEPS} 之间")
        if final_rerank_query_mode not in FINAL_QUERY_MODES:
            raise ValueError(
                f"agentic.final_rerank_query_mode 必须是 {FINAL_QUERY_MODES} 之一，"
                f"收到 {final_rerank_query_mode!r}"
            )

        self._store = store
        self._retriever = retriever
        self._llm = llm
        self._counter = counter
        self._reranker = reranker
        self._max_rounds = max_rounds
        self._per_round_top_k = per_round_top_k
        self._max_prompt_tokens = max_prompt_tokens
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._round_rerank = round_rerank_enabled
        self._final_query_mode = final_rerank_query_mode
        self._inject_abs_time = inject_abs_time
        self._max_plan_steps = max_plan_steps

    @property
    def llm(self) -> LlmClient:
        """当前接的对话客户端。只读暴露给**装配侧**（关闭连接池）与**测试**。

        ⚠ 与 `CoqAgent.llm` 同一个理由：**不要让关闭路径自己再存一份引用**——
        两份状态的结果是"装配时接上了、退出时漏关"，而那平时完全看不出来。
        """
        return self._llm

    # ── 对外唯一入口 ────────────────────────────────────────────────────
    def search(self, *, user_id: str, query: str, top_k: int) -> PlanResult:
        """跑完一次 Plan 多跳。

        ⚠ **每轮召回深度取请求的 `top_k`**（AML 固定 100），与 `react` **同一条口径**：
        顺序是"融合池 → 精排 → 取前 `per_round_top_k` 条"，而不是"先只召 20 条再精排这 20 条"。
        ⚠ 这两件事的结果**不一样**，而且反过来写**不会报错**（§3.2 专门点了这个坑）。

        > **改这一条的理由（2026-10-13）**：第一版按方案第二节的字面要求"每轮实际召回 20 条"，
        > 于是它与 React 在**两个**维度上同时不同（推进机制 ＋ 每轮检索深度），
        > 使 50 题对照里的 7 道差**归不了因**（见
        > [`coq-musique-20261011.md`](../../../eval/reports/coq-musique-20261011.md) §8.5）。
        > ⇒ 现在两条路**只在推进机制上不同**，"机制"这一类自变量才成立。

        最终响应的 `top_k` 由编排层在打包那一步收口，两条路共用。
        """
        started = monotonic()
        state = _State()
        records: list[PlanStepRecord] = []
        rerank_calls = 0
        stop_reason: str | None = None
        fallback_reason: str | None = None

        # ── ① 拆解（一次 LLM，**在检索之前**）────────────────────────────
        plan, failure = self._decompose(query)
        if plan is None:
            # 一步都没跑 ⇒ **回退原混合链**（§10.1 第 6 条）：这是"回退"这个动作
            # 唯一的触发条件——手上一条候选都没有，丢掉的东西为零。
            reason = failure or "plan_invalid"
            return self._finish(
                state=state,
                records=records,
                stop_reason=reason,
                fallback_reason=reason,
                rerank_calls=0,
                decompose_llm_calls=1,
                step_llm_calls=0,
                plan=None,
                original=query,
                # ⚠ 拆解就失败 ⇒ **不在 `PLAN_RECOVERY_TRIGGERS` 里** ⇒ 恢复分支
                #   **不触发**（`not_triggered`）。工具性失败一律保持原行为。
                recovery=PlanRecovery(),
                started=started,
            )

        steps = plan.steps[: self._max_rounds]
        # ⚠ **计划被预算截断**：还有下一步，但它不在这轮真的会跑的列表里。
        truncated = len(plan.steps) > len(steps)

        for index, step in enumerate(steps, start=1):
            executed_query, unbound = bind_placeholders(step.query, state.answers)
            if unbound:
                # ⛔ **绝不把未绑定的占位送进检索器**：那会变成一次词面检索
                #    （检索器拿到的是 `{{s3}}` 这串字符），而且**不会报错**。
                stop_reason = "unbound_placeholder"
                records.append(self._bare_record(index, step, executed_query, stop_reason))
                break

            round_started = monotonic()
            try:
                # ⚠ 深度用**请求的 `top_k`**（AML 固定 100）：那是当前默认 Search 的既有口径
                #   （混合链也是先取 top_k 条再精排）⇒ 不为多跳另造一套检索参数（§3.2）。
                #   于是"融合池 → 精排 → 取前 N"这条顺序是**结构性成立**的，
                #   而不是"先 RRF 截 20 再精排 20"——那两件事的结果不一样，
                #   **而且不会报错**。⛔ `per_round_top_k` 管的是**精排之后取几条**，
                #   不是召回几条，两者别混（2026-10-13 的更正，见 `search` 的 docstring）。
                recalled = self._retriever.search(
                    user_id=user_id, query=executed_query, top_k=top_k
                )
            except Exception:  # noqa: BLE001 —— 与 CoqAgent 同一条纪律
                # §10.2：**检索调用失败 ≠ 没有相关记忆**。这里刻意捕得宽：
                # 下游是远端 embedding 与 Qdrant，异常类型不是本模块该知道的。
                # **吞掉它不等于掩盖它**：一条候选都没攒到时 `fallback_reason` 让调用方
                # 去跑原混合链，同一个错误会在那里**以本来的样子**冒出来。
                stop_reason = "retrieval_error"
                if not state.ordered_ids:
                    fallback_reason = "retrieval_error"
                records.append(
                    self._bare_record(
                        index,
                        step,
                        executed_query,
                        stop_reason,
                        retrieval_latency_ms=_ms(round_started),
                    )
                )
                break
            retrieval_ms = _ms(round_started)

            # ── b. 本轮精排（**只为了把交给模型的片段筛干净**）──────────
            rerank_started = monotonic()
            ranked, rerank_status = self._rerank_round(query=executed_query, candidates=recalled)
            if rerank_status == "called":
                rerank_calls += 1
            rerank_ms = _ms(rerank_started)

            top = ranked[: self._per_round_top_k]
            top_ids = tuple(candidate.memory_id for candidate in top)
            self._cache_documents(state, top_ids)
            for memory_id in top_ids:
                # 各轮候选并起来必然产生重复 ⇒ 保序去重（第一轮的名次最先）。
                state.ordered_ids.setdefault(memory_id, None)
            state.executed.append(executed_query)
            state.executed_for[step.id] = executed_query

            # ── c. 组本轮文档（**预算在这里收口**）─────────────────────
            next_step = steps[index] if index < len(steps) else None
            history_text = render_completed_history(plan, state)
            plan_text = render_plan(plan)
            documents, document_ids, dropped, budget_exceeded = self._build_documents(
                state=state,
                original=query,
                plan_text=plan_text,
                history=history_text,
                step=step,
                executed_query=executed_query,
                top_ids=top_ids,
            )
            if budget_exceeded:
                stop_reason = "prompt_budget_exceeded"
                records.append(
                    self._bare_record(
                        index,
                        step,
                        executed_query,
                        stop_reason,
                        recalled=len(recalled),
                        candidate_ids=top_ids,
                        dropped=dropped,
                        rerank_status=rerank_status,
                        history_text=history_text,
                        retrieval_latency_ms=retrieval_ms,
                        rerank_latency_ms=rerank_ms,
                    )
                )
                break

            # ── d. 一次 LLM：**回答本轮** ＋ 填下一查询 ─────────────────
            llm_started = monotonic()
            decision, failure = self._answer_step(
                original=query,
                plan_text=plan_text,
                history=history_text,
                step=step,
                executed_query=executed_query,
                documents=documents,
            )
            llm_ms = _ms(llm_started)
            input_tokens, output_tokens = _usage(self._llm)
            if decision is None:
                # 失败的那一轮**不许报 token 数**：`last_usage` 是客户端上的残留值
                # （成功过一次之后它一直留着），照抄会把上一轮的用量记到这一轮头上。
                input_tokens = output_tokens = None
                stop_reason = failure or "llm_error"

            evidence_ids: tuple[str, ...] = ()
            quotes: tuple[str, ...] = ()
            unverified = 0
            canonical: str | None = None
            if decision is not None:
                # 索引 → 稳定 ID（**每轮重建**）
                evidence_ids = tuple(document_ids[ref.document_index] for ref in decision.evidence)
                quotes = tuple(ref.quote for ref in decision.evidence)
                unverified = sum(1 for ref in decision.evidence if not ref.verified)
                for memory_id in evidence_ids:
                    if memory_id not in state.evidence:
                        state.evidence.append(memory_id)

                if decision.status != "answered":
                    # ⛔ **不猜占位、不执行未来步骤**：这一步的答案进不了 `state.answers`，
                    #    计划就停在这里。
                    stop_reason = (
                        "dependency_unresolved"
                        if decision.status == "unresolved"
                        else "binding_conflict"
                    )
                else:
                    # 只有**过了全部校验的回答**才进历史。
                    state.answers[step.id] = decision.answer or ""
                    state.sources[step.id] = evidence_ids
                    if next_step is None:
                        # ⭐ 最后一步**也要先被回答**，然后才记 `plan_complete`；
                        #    计划被预算截断时改记 `max_rounds`（下一步没跑，不是跑完了）。
                        stop_reason = "max_rounds" if truncated else "plan_complete"
                    else:
                        # ⭐ **下一查询由程序绑定**（v2 起模型根本不产出它）：
                        #   固定模板 ＋ 已确立答案 ⇒ 唯一的那一句。⛔ 这里没有"模型那份"可对照。
                        canonical, still_unbound = bind_placeholders(next_step.query, state.answers)
                        if still_unbound:
                            stop_reason = "unbound_placeholder"

            records.append(
                PlanStepRecord(
                    index=index,
                    step_id=step.id,
                    template=step.query,
                    executed_query=executed_query,
                    recalled_candidates=len(recalled),
                    document_ids=document_ids,
                    candidate_ids=top_ids,
                    evidence_ids=evidence_ids,
                    evidence_quotes=quotes,
                    quotes_unverified=unverified,
                    status=None if decision is None else decision.status,
                    answer=None if decision is None else decision.answer,
                    canonical_next_query=canonical,
                    history_text=history_text,
                    dropped_documents=dropped,
                    rerank_status=rerank_status,
                    extraction=None if decision is None else decision.extraction,
                    stop_reason=stop_reason,
                    retrieval_latency_ms=retrieval_ms,
                    rerank_latency_ms=rerank_ms,
                    llm_latency_ms=llm_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )
            if stop_reason is not None:
                break

        # ── ⑤ 恢复分支（`unresolved` **或** `conflict`；**程序侧、一次性**）────────
        # ⚠ 它**不是**一条新的多跳路，是**对候选池的一次补充**：不计步骤、不加 LLM 调用、
        #   不截 Top-20，按请求的 `top_k` 补到池子满为止（**不够不凑**）。
        # ⚠ 它**自己精排一次**（挑选"哪几条有资格进池"）——那**不算**"第二次最终精排"：
        #   最终精排仍然只有一次，在编排层的 `_rank_and_package`。
        #   也**不走 `fallback_reason`**——那个词的含义是"整条路降级回混合检索"，
        #   而这里**一条候选都不丢**。
        recovery = self._recover(
            state=state,
            user_id=user_id,
            original=query,
            stop_reason=stop_reason or "plan_complete",
            top_k=top_k,
        )
        # ⭐ 兜底那一次精排**也要计入 §14 的 `rerank_calls`**：`_recover` 改不了这个
        #    局部变量，所以在调用点加回去（漏了会让"多跳一共排了几次"少算一次）。
        if recovery.reranked == "called":
            rerank_calls += 1

        return self._finish(
            state=state,
            records=records,
            # `None` 只剩下"一次都没进循环"（`steps` 为空）这一种可能——而计划至少一步，
            # 所以它是**不可达**的兜底，不是正常出口。
            stop_reason=stop_reason or "plan_complete",
            fallback_reason=fallback_reason,
            rerank_calls=rerank_calls,
            decompose_llm_calls=1,
            step_llm_calls=len(records),
            plan=plan,
            original=query,
            recovery=recovery,
            started=started,
        )

    # ── 恢复分支（`unresolved` / `conflict`）─────────────────────────────
    def _recover(
        self, *, state: _State, user_id: str, original: str, stop_reason: str, top_k: int
    ) -> PlanRecovery:
        """**依赖链"走不通"之后**按**请求的 `top_k`** 补一次**原题**检索。**从不抛异常。**

        ⭐ **它做两件事**：
        1. 按 `top_k` 召回原题的候选，并**精排一次**（决定"哪几条有资格进池"）；
        2. 按精排名次**去重补位**，把池子补到 `top_k` 为止。

        ⛔ **兜底给不够就停在有什么算什么**——绝不硬凑。
        后续（那一次最终精排、打包、答题）走的还是原来那一条路。

        ## 与"每轮"的关系（2026-10-15 改）

        旧口径是 `max(0, 80 − 已执行轮数 × per_round_top_k)`（⇒ 补 60/40/20/0，且
        **池子恒为 80**）。现在**与轮数解耦**：兜底和**每一轮**一样按**请求的 `top_k`** 召回。
        ⇒ 池子上限从"恒 80"变成 **≤ `top_k`**。

        ⚠ **那次精排不受 `agentic.round_rerank_enabled` 管**（那个开关只管"每轮"）——
        见 `_rerank_fallback`。理由是 §13 的开关纯度：一个开关管两件事，
        那一臂的消融就同时在两个维度上变了。

        ⚠ **触发面**：只看 `stop_reason in PLAN_RECOVERY_TRIGGERS`
        ——即 `dependency_unresolved` **或** `binding_conflict`，
        **模型明确说"走不通"的那两种**。⛔ 工具性失败（`plan_invalid` / `llm_*` /
        `retrieval_error`）与程序性停止（`unbound_placeholder` / `invalid_output` /
        `max_rounds`）**一律保持原行为**。
        """
        if stop_reason not in PLAN_RECOVERY_TRIGGERS:
            return PlanRecovery()
        # ⚠ **已执行轮数仍如实记下**（诊断用），但它**不再决定补多少**。
        executed_rounds = len(state.executed)
        target = top_k
        pool_before = len(state.ordered_ids)
        if pool_before >= target:
            # ⭐ "**触发恢复但池子已经够了**"（请求的 `top_k` 比已有池子还小）
            #    ——与"真的跑了兜底"分开记（见 `PlanRecovery`）。这里**不检索**。
            return PlanRecovery(
                entered=True,
                status="skipped_pool_full",
                executed_rounds=executed_rounds,
                target_top_k=target,
                pool_before=pool_before,
            )
        started = monotonic()
        try:
            extra = self._retriever.search(user_id=user_id, query=original, top_k=target)
        except Exception:  # noqa: BLE001 —— 与执行器其余各处同一条纪律
            # ⛔ 兜底失败**不是**整条路的失败：已有候选**一条不动**，
            #   继续走原来那一次最终精排与答题。**绝不**退化成"整条路降级"。
            return PlanRecovery(
                entered=True,
                status="error",
                executed_rounds=executed_rounds,
                target_top_k=target,
                pool_before=pool_before,
                latency_ms=_ms(started),
            )
        # ⚠ **补位前先精排**：否则补进去的是"RRF 序的前若干条"，而漏掉的可能是
        #   精排会排到很前的那几条。这一步只决定"谁有资格进池"，不改已有顺序。
        ranked, rerank_status = self._rerank_fallback(query=original, candidates=list(extra))
        added = 0
        for candidate in ranked:
            if len(state.ordered_ids) >= target:
                break  # ⛔ 到目标就停：**不为了凑满而多塞**
            # ⚠ 兜底候选**追加在分步候选之后**（保序去重：分步的名次保持在前）——
            #   最终精排本来就会整体重排，这里不动已有顺序。
            if candidate.memory_id not in state.ordered_ids:
                added += 1
            state.ordered_ids.setdefault(candidate.memory_id, None)
        return PlanRecovery(
            entered=True,
            status="success",
            executed_rounds=executed_rounds,
            target_top_k=target,
            pool_before=pool_before,
            recalled=len(extra),
            reranked=rerank_status,
            added_unique=added,
            latency_ms=_ms(started),
        )

    # ── 拆解与逐轮 ──────────────────────────────────────────────────────
    def _decompose(self, original: str) -> tuple[Plan | None, str | None]:
        """一次 LLM ＋ 严格校验。**失败返回 `(None, 原因)`，从不抛异常。**"""
        prompt = render_plan_decompose_prompt(original_question=original)
        try:
            raw = self._llm.complete(
                prompt, max_tokens=self._max_output_tokens, temperature=self._temperature
            )
        except LlmUnavailable:
            return None, "llm_unavailable"
        except Exception:  # noqa: BLE001 —— 换一个客户端实现就可能抛别的类型
            return None, "llm_error"
        try:
            return parse_plan(raw, max_steps=self._max_plan_steps), None
        except InvalidPlanOutput:
            # ⛔ 绝不"补一份默认计划再当作合法计划"：那等于让一次胡话改变检索。
            return None, "plan_invalid"

    def _answer_step(
        self,
        *,
        original: str,
        plan_text: str,
        history: str,
        step: PlanStep,
        executed_query: str,
        documents: Sequence[str],
    ) -> tuple[StepAnswer | None, str | None]:
        """一次 LLM ＋ 严格校验。**失败返回 `(None, 原因)`。**

        ⚠ 提示词里**没有"下一步的模板"**（v2）：下一查询由程序绑定，
        ⛔ 模型不回步骤编号、也不回 `next_query`——那两件事**不是它的职责**。
        """
        prompt = render_plan_step_prompt(
            original_question=original,
            retrieval_plan=plan_text,
            completed_history=history,
            current_step=step_line(step),
            executed_query=executed_query,
            current_documents=documents,
        )
        try:
            raw = self._llm.complete(
                prompt, max_tokens=self._max_output_tokens, temperature=self._temperature
            )
        except LlmUnavailable:
            return None, "llm_unavailable"
        except Exception:  # noqa: BLE001
            return None, "llm_error"
        try:
            # ⚠ `documents` 是**本轮真正发出去的那份列表**——索引的合法上界由它决定。
            return parse_step_answer(raw, step_id=step.id, documents=documents), None
        except InvalidPlanOutput:
            return None, "invalid_output"

    # ── 每一步 ──────────────────────────────────────────────────────────
    def _rerank_round(
        self, *, query: str, candidates: list[Candidate]
    ) -> tuple[list[Candidate], str]:
        """对**本轮候选**精排一次；返回 `(候选, 状态)`。

        ⚠ 与 `CoqAgent._rerank_round` **同一条语义**（三分状态、降级回原顺序、不改候选集合）。
        两份实现是刻意的：React 那一份**不许被本次改动碰到**。要改这里的语义时，
        先去看那一边的 docstring——那里写全了"为什么降级必须被记下来"。

        ⚠ **它受 `agentic.round_rerank_enabled` 管**（关掉它这一路整块不跑）。
        ⛔ 与**兜底**那一次**不是同一个开关口径**——见 `_rerank_fallback`。
        """
        return self._rerank_candidates(
            query=query, candidates=candidates, respect_round_switch=True
        )

    def _rerank_fallback(
        self, *, query: str, candidates: list[Candidate]
    ) -> tuple[list[Candidate], str]:
        """兜底那一次的**挑选**精排。**常开**——⛔ **不读** `agentic.round_rerank_enabled`。

        与 `_rerank_round` 共用同一个主体（`_rerank_candidates`），**只差"要不要看那个
        '每轮'开关"**。分开的理由（2026-10-15）：兜底精排是**另一个机制**
        （从 `top_k` 条里挑哪几条进池），而 §13 要求"一个开关只影响它命名的那一件事"
        ——让 `round_rerank_enabled` 顺手把它也关掉，"每轮不精排"那一臂就同时在
        **两个维度**上变了，而那正是那条消融要避免的。

        ⇒ 它只受 `reranker is None` 管：没配精排就退回 RRF 序，记 `disabled`。
        """
        return self._rerank_candidates(
            query=query, candidates=candidates, respect_round_switch=False
        )

    def _rerank_candidates(
        self, *, query: str, candidates: list[Candidate], respect_round_switch: bool
    ) -> tuple[list[Candidate], str]:
        """精排一份候选；返回 `(候选, 状态)`，状态取 `called` / `degraded` / `disabled`。

        ⚠ **降级一律退回原顺序**（D12）：端点不可用不是"整条路失败"。
        ⛔ **不改候选集合**——只重排名次（`Candidate` 的 docstring 那条契约）。
        """
        if (
            (respect_round_switch and not self._round_rerank)
            or self._reranker is None
            or not candidates
        ):
            return candidates, "disabled"
        pairs = self._fetch([candidate.memory_id for candidate in candidates])
        documents = [
            render_pair(pair, inject_abs_time=self._inject_abs_time)
            for pair in pairs
            if pair is not None
        ]
        if len(documents) != len(candidates):
            # 真源缺行 ⇒ 拼不出一一对应 ⇒ 不重排（与 `_maybe_rerank` 同一条判断）
            return candidates, "degraded"
        try:
            scores = self._reranker.score(query=query, documents=documents)
        except RerankUnavailable:
            return candidates, "degraded"
        if len(scores) != len(candidates):
            # 形状不符与端点挂了一样不可用——按前缀对齐会让后半段名次错位。
            return candidates, "degraded"
        order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), i))
        return (
            [Candidate(memory_id=candidates[i].memory_id, rank=new) for new, i in enumerate(order)],
            "called",
        )

    def _build_documents(
        self,
        *,
        state: _State,
        original: str,
        plan_text: str,
        history: str,
        step: PlanStep,
        executed_query: str,
        top_ids: Sequence[str],
    ) -> tuple[list[str], tuple[str, ...], int, bool]:
        """本轮发给模型的文档列表 ＋ 索引映射 ＋ 丢弃数 ＋ 是否**因预算整体失败**。

        ⚠ 顺序是**本轮名次**，索引映射随后**一次性**建立、之后**不再改动文档顺序**
        ——顺序一动，模型给的 `document_index` 就指向别的文档。

        ⚠ 预算是"固定部分先占住、剩下的按名次补"：固定部分 = 原问题 ＋ 完整计划 ＋
        已完成历史 ＋ 本轮子问题（v2 起**没有**"下一条模板"）。⛔ **历史不会因此丢掉**
        ——它本来就只有问题/答案/来源 ID（见 `render_completed_history`）。

        ⚠ 固定部分的那几段文字**必须与 `_answer_step` 真正发出去的那一份逐字相同**
        （两边都走 `render_plan_step_prompt` ＋ 同一个 `step_line`）——否则预算算的是
        另一份提示词，而"超了"这件事**不会报错**。
        """
        base = render_plan_step_prompt(
            original_question=original,
            retrieval_plan=plan_text,
            completed_history=history,
            current_step=step_line(step),
            executed_query=executed_query,
            current_documents=(),
        )
        fixed_tokens = self._counter.count(base)
        if fixed_tokens > self._max_prompt_tokens:
            # 连固定部分都放不下 ⇒ **不组织一个残缺的输入**，安全停下。
            return [], (), len(top_ids), True
        budget = self._max_prompt_tokens - fixed_tokens

        documents: list[str] = []
        document_ids: list[str] = []
        used = 0
        dropped = 0
        for memory_id in top_ids:
            body = state.bodies.get(memory_id)
            if body is None:
                dropped += 1  # 真源缺行 ⇒ 它进不了文档，也就不能被引用
                continue
            rendered = _render_document(len(document_ids), body, state.days.get(memory_id, ""))
            cost = self._counter.count(rendered)
            if used + cost > budget:
                dropped += 1
                continue
            used += cost
            documents.append(rendered)
            document_ids.append(memory_id)
        return documents, tuple(document_ids), dropped, False

    # ── 收口 ────────────────────────────────────────────────────────────
    def _finish(
        self,
        *,
        state: _State,
        records: list[PlanStepRecord],
        stop_reason: str,
        fallback_reason: str | None,
        rerank_calls: int,
        decompose_llm_calls: int,
        step_llm_calls: int,
        plan: Plan | None,
        original: str,
        recovery: PlanRecovery,
        started: float,
    ) -> PlanResult:
        candidates = tuple(
            Candidate(memory_id=memory_id, rank=rank)
            for rank, memory_id in enumerate(state.ordered_ids)
        )
        executed = list(state.executed)
        # `rerank_query_mode` 是**两条路径的记法**（`original` / `joint`），
        # ⚠ 它由"有没有进恢复分支"决定，与 `final_rerank_query_mode` **不是**同一个键。
        recovery = replace(recovery, rerank_query_mode="original" if recovery.entered else "joint")
        rerank_query = _final_rerank_query(
            mode=self._final_query_mode,
            executed=executed,
            recovery=recovery,
            original=original,
        )
        steps = () if plan is None else plan.steps
        return PlanResult(
            candidates=candidates,
            rerank_query=rerank_query,
            stats=PlanStats(
                rounds=tuple(records),
                stop_reason=stop_reason,
                fallback_reason=fallback_reason,
                candidate_count=sum(record.recalled_candidates for record in records),
                unique_candidate_count=len(candidates),
                selected_evidence_count=len(state.evidence),
                joint_query_count=len(executed),
                llm_calls=decompose_llm_calls + step_llm_calls,
                rerank_calls=rerank_calls,
                latency_ms=_ms(started),
                plan=steps,
                decompose_llm_calls=decompose_llm_calls,
                step_llm_calls=step_llm_calls,
                plan_steps=len(steps),
                steps_executed=len(executed),
                dependencies_repaired=sum(1 for step in steps if step.dependencies_repaired),
            ),
            fallback_reason=fallback_reason,
            plan_history=tuple(records),
            plan=steps,
            recovery=recovery,
        )

    @staticmethod
    def _bare_record(
        index: int,
        step: PlanStep,
        executed_query: str,
        stop_reason: str,
        *,
        recalled: int = 0,
        candidate_ids: tuple[str, ...] = (),
        dropped: int = 0,
        rerank_status: str = "disabled",
        history_text: str = "",
        retrieval_latency_ms: float = 0.0,
        rerank_latency_ms: float = 0.0,
    ) -> PlanStepRecord:
        """在**还没走到模型那一步**就停下时的记录（绑定失败 / 检索失败 / 预算超限）。

        ⚠ 它必须存在：否则"第 3 步因为占位没绑上而停下"在 `plan_history` 里**什么都不留**，
        读的人只会看到计划跑了 2 步——**原因不见了**。
        """
        return PlanStepRecord(
            index=index,
            step_id=step.id,
            template=step.query,
            executed_query=executed_query,
            recalled_candidates=recalled,
            document_ids=(),
            candidate_ids=candidate_ids,
            evidence_ids=(),
            evidence_quotes=(),
            quotes_unverified=0,
            status=None,
            answer=None,
            canonical_next_query=None,
            history_text=history_text,
            dropped_documents=dropped,
            rerank_status=rerank_status,
            extraction=None,
            stop_reason=stop_reason,
            retrieval_latency_ms=retrieval_latency_ms,
            rerank_latency_ms=rerank_latency_ms,
        )

    # ── 原文 ────────────────────────────────────────────────────────────
    def _cache_documents(self, state: _State, memory_ids: Sequence[str]) -> None:
        """把还没有正文的候选从真源取回来（**一次批量读**）。请求局部缓存，同 React。"""
        missing = [memory_id for memory_id in memory_ids if memory_id not in state.bodies]
        if not missing:
            return
        with self._store.read() as conn:
            pairs = self._store.fetch_pairs_by_ids(conn, missing)
        by_id = {pair.id: pair for pair in pairs}
        for memory_id in missing:
            pair = by_id.get(memory_id)
            if pair is None:
                continue  # 真源缺行：不缓存 ⇒ 组文档时记成 dropped
            state.bodies[memory_id] = render_pair(pair, inject_abs_time=self._inject_abs_time)
            state.days[memory_id] = day_granularity(getattr(pair, "event_time", None))

    def _fetch(self, memory_ids: Sequence[str]) -> list[QaPair | None]:
        with self._store.read() as conn:
            pairs = self._store.fetch_pairs_by_ids(conn, list(memory_ids))
        by_id = {pair.id: pair for pair in pairs}
        return [by_id.get(memory_id) for memory_id in memory_ids]


def _final_rerank_query(
    *, mode: str, executed: Sequence[str], recovery: PlanRecovery, original: str
) -> str | None:
    """**最终精排用哪一句**（两条路径，2026-10-13）。

    | 情况 | 精排词 | 为什么 |
    | --- | --- | --- |
    | ⭐ **进了恢复分支** | **最初的那句原题** | 分步那几跳**没走通**（最后一步 `unresolved`）
      ⇒ 拿它们当精排词等于让"没答出来的中间问题"参与排序；而候选里既有分步的、
      也有原题兜底的 ⇒ 只有原题是它们**共同的**问题 |
    | 没进（`combined`，现状） | 实际执行过的那几句，换行分隔 | §9.1 的联合精排 |
    | 没进（`original`） | `None` | 编排层用原问题（不换） |

    ⚠ **原题在 `combined` 里本来就不出现**：Plan **从不检索原题**（第一轮就查子问题，
    见模块 docstring 的 ②）⇒ 这里**没有"原题被拼两次"的可能**。
    ⛔ 也**不要**顺手把原题补进 `combined`——那是**另一个自变量**，会污染本次对照。

    ⚠ "进了恢复分支"包括**预算耗尽（没真检索）**与**兜底检索失败**两种：
    它们的精排词同样是原题（§新 第四条明说了）。
    """
    if recovery.entered:
        return original if original.strip() else None
    if mode == "combined" and executed:
        return chr(10).join(executed)
    return None
