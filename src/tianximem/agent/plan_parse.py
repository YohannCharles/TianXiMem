"""Plan 模式输出的**严格校验**（拆解 JSON ＋ 每轮 JSON）。

两条与 CoQ 那边**刻意不同**的地方，先说清楚：

| | CoQ（[`parse.py`](./parse.py)） | Plan（本文件） |
| --- | --- | --- |
| JSON 提取 | 两阶段（整串 → 括号平衡） | **同一份实现**（复用 `extract_json_object`） |
| 校验失败 | 抛 `InvalidCoqOutput` ⇒ 本轮无效、停下 | 抛 `InvalidPlanOutput`，**语义相同** |

⚠ **提取复用、校验不共用**：JSON 的"从一堆思考里挖出那个对象"是一件与业务无关的事，
写两份必然漂；而**校验规则两边毫无交集**（CoQ 是四个定长字段，Plan 是
"步骤／依赖／占位引用"这套结构）。⇒ 提取复用 [`parse.py`](./parse.py) 的
`extract_json_object`，校验在本文件重写。

## 三处"看起来能跑、其实错了"的规则

1. **`{{s1}}` 只能指**已经出现的、更早的步骤**。指自己、指后面、指不存在的 id
   ⇒ **整份计划作废**。理由：一个还没绑定的占位被送进检索器，会变成一次
   **词面检索**（模型看到的是 `{{s3}}` 这串字符），而**它不会报错**，
   只会安静地捞回一堆无关记忆。
2. ⭐ **依赖以占位为准**：程序**只按占位绑定**，所以"这一步依赖谁"
   **由查询文本唯一决定**；模型写的 `depends_on` 只是一个**声明**。
   ⇒ 声明与占位不一致时规范化成占位那一份并记录 `dependencies_repaired`，
   不作废计划。占位合法而声明遗漏时，查询文本仍足以确定依赖，因此计划可以执行。
   ⚠ **一条原则性的分界**：判据是**"真相可不可确定"**，不是"错得严不严重"——
   "声明了一个不存在的步骤"猜不出来 ⇒ **作废**；"有占位却没声明"由文本决定 ⇒ **修**。
3. **引句逐字校验是"存在性"，不是"语义正确性"**。⚠ **报告里不要把两者混起来**：
   `quote` 逐字出现在本轮文档里，**不等于**它支持那个答案。后者要人看（或另一个裁判），
   本模块不假装能判。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from tianximem.agent.parse import extract_json_object

__all__ = [
    "EVIDENCE_KEYS",
    "PLAN_DECOMPOSE_KEYS",
    "PLAN_KEY_RE",
    "PLAN_STEP_KEYS",
    "STEP_ANSWER_KEYS",
    "STEP_STATUSES",
    "EvidenceRef",
    "InvalidPlanOutput",
    "Plan",
    "PlanStep",
    "StepAnswer",
    "bind_placeholders",
    "parse_plan",
    "parse_step_answer",
    "placeholders_in",
    "render_plan",
]

#: 拆解输出的顶层：**恰好一个键**。
PLAN_DECOMPOSE_KEYS: Final[frozenset[str]] = frozenset({"steps"})

#: 一步的字段：**恰好三个**。
PLAN_STEP_KEYS: Final[frozenset[str]] = frozenset({"id", "query", "depends_on"})

#: 每轮输出的字段：**恰好三个**（`step_id` / `next_step_id` / `next_query`
#: 不是模型字段——步骤编号**由程序附加**，下一查询**由程序绑定**）。
STEP_ANSWER_KEYS: Final[frozenset[str]] = frozenset({"status", "answer", "evidence"})

#: 一条证据的字段：**恰好两个**。
EVIDENCE_KEYS: Final[frozenset[str]] = frozenset({"document_index", "quote"})

#: 每轮的状态取值域。⚠ 它与停止原因**不是一回事**：`unresolved` 落到
#: `dependency_unresolved`、`conflict` 落到 `binding_conflict`（映射见 `plan.py`）。
STEP_STATUSES: Final[tuple[str, ...]] = ("answered", "unresolved", "conflict")

#: 步骤 id 的形状。**必须钉死**：`{{s1}}` 这套占位语法要能机械地解析出 id，
#: 而"模型自己发明一套 id 命名（`step1` / `S-1`）"会让占位引用对不上——
#: 那正是上面第 1 条要挡的东西。
PLAN_KEY_RE: Final[re.Pattern[str]] = re.compile(r"^s[0-9]+$")

#: 占位符。**只认 `{{sN}}`**——单花括号是 JSON/文本里最常见的字符，
#: 认它会把模型的 JSON 示例当成占位。
_PLACEHOLDER_RE: Final[re.Pattern[str]] = re.compile(r"\{\{(s[0-9]+)\}\}")


class InvalidPlanOutput(ValueError):
    """模型的输出**不能**被当作一份计划／一次步骤回答使用。**调用方必须降级**。

    ⚠ 与 `InvalidCoqOutput` 同一条纪律：异常消息里**只有原因，没有原文**——
    原文是模型读着不可信输入写出来的东西（可能含数据集正文）。
    """


@dataclass(frozen=True, slots=True)
class PlanStep:
    """计划里的一步。**`query` 是模板**（可能含 `{{sN}}`），不是可以直接检索的字符串。"""

    id: str
    query: str
    #: ⭐ **规范化的依赖 = 模板里真正出现的那些占位**（按出现顺序去重）。
    #:
    #: ⚠ 它是**程序按 `query` 算出来的**，不是照抄模型写的那一份——理由见
    #: [`PlanStep.declared_dependencies`][PlanStep] 与模块 docstring 第 2 条的更正：
    #: 绑定只认占位，所以"该依赖谁"这件事**由查询文本唯一决定**；
    #: 模型写的 `depends_on` 只是一个**声明**，写错了不改变计划能不能执行。
    depends_on: tuple[str, ...]
    #: 模板里**真正出现**的占位 id，按出现顺序去重。⚠ 恒等于 `depends_on`（规范化之后）；
    #: 留成两个名字是因为它们在提示词里是**两个概念**（"引用谁"与"依赖谁"）。
    placeholders: tuple[str, ...]
    #: ⛔ **模型实际写的那一份 `depends_on`**（原样保留，只作诊断）。
    declared_dependencies: tuple[str, ...]
    #: 声明了依赖、却没有对应占位的 id。⚠ **不作废计划**（见模块 docstring 第 2 条）。
    unused_dependencies: tuple[str, ...]
    #: **程序把声明规范化成了占位引用**。这不是失败，但会计入诊断统计。
    dependencies_repaired: bool


@dataclass(frozen=True, slots=True)
class Plan:
    """一份通过全部校验的有序计划。**它不给答案**——只给"要去查什么"。"""

    steps: tuple[PlanStep, ...]
    #: `direct` / `recovered`。与 CoQ 那边同一个含义（模型输出稳定性的探针）。
    extraction: str
    raw: str


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """模型指认的一条证据。**`document_index` 是本轮呈现列表的**临时索引**（从 0 起）。

    ⚠ 索引**每轮重建**：拿上一轮的索引去查这一轮的文档，会指到**另一条证据**上，
    而那种错**不会报错**。
    """

    document_index: int
    quote: str
    #: `quote` 是否逐字出现在该文档里。⚠ **只表示"引用存在"，不表示"引用支持答案"**。
    verified: bool


@dataclass(frozen=True, slots=True)
class StepAnswer:
    """一轮的模型输出——**只回答那一步**（`status` / `answer` / `evidence`）。

    ⚠ **`step_id` 是程序附加的**（⛔ 不是模型回的）：从这一版起提示词只要求三个字段，
    所以"模型回了一个对不上的步骤号"这件事**结构性地不可能发生**。
    """

    #: **由程序附加**（`parse_step_answer` 的入参），不是模型输出的一部分。
    step_id: str
    status: str
    #: `answered` 时必为非空的单行串；`unresolved` / `conflict` 时**应当**是 `None`。
    answer: str | None
    evidence: tuple[EvidenceRef, ...]
    extraction: str
    raw: str

    @property
    def is_answer(self) -> bool:
        """这一步**可不可以**拿来绑定占位。⛔ 只有 `answered` 可以。"""
        return self.status == "answered" and bool(self.answer)


# ── 占位符：读与填 ──────────────────────────────────────────────────────


def placeholders_in(text: str) -> tuple[str, ...]:
    """`text` 里出现的占位 id，**按出现顺序去重**。"""
    out: list[str] = []
    for match in _PLACEHOLDER_RE.finditer(text):
        if match.group(1) not in out:
            out.append(match.group(1))
    return tuple(out)


def bind_placeholders(template: str, answers: dict[str, str]) -> tuple[str, tuple[str, ...]]:
    """把已知答案填进模板；返回 `(结果, 没填上的 id)`。

    ⚠ **一遍扫描、绝不重扫替换结果**（`re.sub` 而不是循环 `str.replace`，
    同 [`coq.py`](./coq.py) 的 `render_coq_prompt`）：答案是**模型写的自由文本**，
    它完全可能自己就含一串 `{{s1}}`——连续替换会让它被**二次替换**，
    或者反过来把模板里另一个占位吃掉，而**没有任何报错**。

    ⚠ **填不上的占位原样留着**（不在返回值里悄悄删掉）：调用方据 `没填上的 id`
    决定停下——把 `{{s3}}` 这串字符送进检索器**不会报错**，只会安静地捞回无关记忆。
    """
    missing: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = answers.get(key)
        if value is None:
            if key not in missing:
                missing.append(key)
            return match.group(0)
        return value

    return _PLACEHOLDER_RE.sub(_replace, template), tuple(missing)


# ── 小工具：严格取值 ────────────────────────────────────────────────────


def _single_line(value: str, *, where: str) -> str:
    if "\n" in value or "\r" in value:
        # 提示词要求单行。多行不是"格式小瑕疵"：它会被当成**要执行的检索词**，
        # 而检索侧的每一次调用只接受一条查询。
        raise InvalidPlanOutput(f"`{where}` 必须是单行")
    return value


def _require_str(payload: dict[str, Any], key: str, *, where: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise InvalidPlanOutput(f"`{where}` 必须是字符串，收到 {type(value).__name__}")
    return value


def _require_optional_str(payload: dict[str, Any], key: str, *, where: str) -> str | None:
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidPlanOutput(f"`{where}` 必须是字符串或 null，收到 {type(value).__name__}")
    return value


def _require_keys(payload: dict[str, Any], expected: frozenset[str], *, where: str) -> None:
    keys = set(payload)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise InvalidPlanOutput(f"`{where}` 字段不匹配：缺少 {missing}，多出 {extra}")


# ── 拆解输出 ────────────────────────────────────────────────────────────


def parse_plan(text: str, *, max_steps: int = 4) -> Plan:
    """从模型输出里拿到一份**可执行的**计划，否则抛 `InvalidPlanOutput`。

    `max_steps` 是**计划的步数上限**（当前为 1～4）。⚠ 它**不是** `agentic.max_rounds`
    ——后者是实际执行步数的预算，前者是计划本身的结构约束。
    两者混成一个数，会让"计划有 4 步但只跑 3 步"看起来像"计划不合法"。
    """
    found = extract_json_object(text)
    if found is None:
        raise InvalidPlanOutput("响应里找不到 JSON 对象")
    payload, extraction = found
    _require_keys(payload, PLAN_DECOMPOSE_KEYS, where="plan")

    raw_steps = payload["steps"]
    if not isinstance(raw_steps, list):
        raise InvalidPlanOutput(f"`steps` 必须是数组，收到 {type(raw_steps).__name__}")
    if not 1 <= len(raw_steps) <= max_steps:
        raise InvalidPlanOutput(f"`steps` 必须是 1..{max_steps} 步，收到 {len(raw_steps)} 步")

    steps: list[PlanStep] = []
    seen: list[str] = []
    for position, item in enumerate(raw_steps):
        where = f"steps[{position}]"
        if not isinstance(item, dict):
            raise InvalidPlanOutput(f"`{where}` 必须是对象，收到 {type(item).__name__}")
        _require_keys(item, PLAN_STEP_KEYS, where=where)

        step_id = _single_line(_require_str(item, "id", where=f"{where}.id"), where=f"{where}.id")
        if not PLAN_KEY_RE.match(step_id):
            raise InvalidPlanOutput(f"`{where}.id` 必须是 `s1` 这种形状，收到 {step_id!r}")
        if step_id in seen:
            raise InvalidPlanOutput(f"`{where}.id` 与前一步重复：{step_id!r}")
        query = _single_line(
            _require_str(item, "query", where=f"{where}.query"), where=f"{where}.query"
        )
        if not query.strip():
            raise InvalidPlanOutput(f"`{where}.query` 不能是空串")

        raw_deps = item["depends_on"]
        if not isinstance(raw_deps, list):
            raise InvalidPlanOutput(
                f"`{where}.depends_on` 必须是数组，收到 {type(raw_deps).__name__}"
            )
        declared: list[str] = []
        for entry in raw_deps:
            if not isinstance(entry, str):
                raise InvalidPlanOutput(
                    f"`{where}.depends_on` 里必须是字符串，收到 {type(entry).__name__}"
                )
            if entry not in seen:
                # 指自己 / 指后面 / 指不存在的 id —— 三种都归这里，**整份计划作废**。
                # ⚠ **这一条与下面那条修正是有原则区别的**：
                #   * "声明了一个不存在的依赖" ⇒ 我们**不知道模型想依赖谁**（猜不了）⇒ 作废；
                #   * "有占位却没声明" ⇒ 该依赖谁**由查询文本唯一决定**（可修）。
                #   ⇒ 判据是**"真相是否可确定"**，不是"错得严不严重"。
                raise InvalidPlanOutput(
                    f"`{where}.depends_on` 里的 {entry!r} 不是**此前**出现过的步骤（已有 {seen}）"
                )
            if entry not in declared:
                declared.append(entry)

        placeholders = placeholders_in(query)
        for name in placeholders:
            if name not in seen:
                # ⛔ **这才是真正的安全判据**：占位必须指向**先出现过**的步骤。
                #   指自己 / 指后面 / 指不存在的 id ⇒ 绑不上（或绑出一个循环）⇒ 整份作废。
                raise InvalidPlanOutput(
                    f"`{where}.query` 里的 {{{{{name}}}}} 不是**此前**出现过的步骤（已有 {seen}）"
                )
        # ⭐ **依赖以占位为准**：模型写的 `depends_on`
        #   只是一个**声明**——程序只按占位绑定，所以"声明漏了一个"**不影响这份计划能不能执行**。
        #   ⇒ 规范化成占位那一份并**记一笔**，而不是把一份能用的计划整份丢掉。
        steps.append(
            PlanStep(
                id=step_id,
                query=query,
                depends_on=placeholders,
                placeholders=placeholders,
                declared_dependencies=tuple(declared),
                unused_dependencies=tuple(name for name in declared if name not in placeholders),
                dependencies_repaired=tuple(declared) != placeholders,
            )
        )
        seen.append(step_id)

    return Plan(steps=tuple(steps), extraction=extraction, raw=text)


def render_plan(plan: Plan) -> str:
    """把计划渲染成**给模型看的那一份**（填进 `{retrieval_plan}`）。

    ⚠ 它是**给模型读的文本**，不是"计划的可执行形式"——执行走的是程序里的
    `PlanStep` 对象。两者漂移的表现是"模型以为在答 s2、程序在跑 s3"，
    所以渲染只有这一处实现。
    """
    lines: list[str] = []
    for step in plan.steps:
        suffix = f"  (depends on {', '.join(step.depends_on)})" if step.depends_on else ""
        lines.append(f"{step.id}: {step.query}{suffix}")
    return "\n".join(lines)


# ── 每轮的输出 ──────────────────────────────────────────────────────────


def parse_step_answer(text: str, *, step_id: str, documents: Sequence[str]) -> StepAnswer:
    """从模型输出里拿到**本轮**那一条回答。

    `step_id` 是**程序**告诉它"现在答的是哪一步"（⛔ 模型不回这个字段，见
    `STEP_ANSWER_KEYS`）——它只被**附加**到返回值上，用于记账与诊断。

    `documents` 是**本轮真正发出去的那份列表**——`document_index` 的合法上界由它决定，
    引句的逐字校验也拿它做（⚠ 这个参数刻意必填，同 `parse_coq_decision` 的
    `document_count`：把上界写成常量就等于放弃了那条不变式）。
    """
    found = extract_json_object(text)
    if found is None:
        raise InvalidPlanOutput("响应里找不到 JSON 对象")
    payload, extraction = found
    _require_keys(payload, STEP_ANSWER_KEYS, where="step")

    status = _require_str(payload, "status", where="status")
    if status not in STEP_STATUSES:
        raise InvalidPlanOutput(f"`status` 必须是 {STEP_STATUSES} 之一，收到 {status!r}")

    answer = _require_optional_str(payload, "answer", where="answer")
    if status == "answered":
        if not answer or not answer.strip():
            raise InvalidPlanOutput("`status=answered` 时 `answer` 必须是非空字符串")
        # ⚠ 答案会被填进下一句检索词 ⇒ 多行必须**在进入历史之前**挡下（同 `_require_str`
        #   在 CoQ 那边的理由）。
        _single_line(answer, where="answer")
    # ⚠ `unresolved` / `conflict` 时提示词要求 `answer=null`、`evidence=[]`。
    #   模型偶尔会附上它们 ⇒ **不采用、不作废、照实记下来**（与"依赖以占位为准"
    #   同一条原则：程序的行为不依赖这两个字段，所以没有理由因此丢掉整份计划）。
    #   ⇒ 只有 `status == "answered"` 的那一份会被 `is_answer` 放行。

    raw_evidence = payload["evidence"]
    if not isinstance(raw_evidence, list):
        raise InvalidPlanOutput(f"`evidence` 必须是数组，收到 {type(raw_evidence).__name__}")
    evidence: list[EvidenceRef] = []
    for position, item in enumerate(raw_evidence):
        where = f"evidence[{position}]"
        if not isinstance(item, dict):
            raise InvalidPlanOutput(f"`{where}` 必须是对象，收到 {type(item).__name__}")
        _require_keys(item, EVIDENCE_KEYS, where=where)
        index = item["document_index"]
        # ⚠ **排除 bool**：`True` 是 `int` 的实例，不排除的话 `{"document_index": true}`
        #   会被读成索引 1（同 `parse.py` 的 `_require_indices`）。
        if isinstance(index, bool) or not isinstance(index, int):
            raise InvalidPlanOutput(
                f"`{where}.document_index` 必须是整数，收到 {type(index).__name__}"
            )
        if index < 0 or index >= len(documents):
            # **越界即作废**（不是"忽略这一条"）：忽略会让"模型指着不存在的证据"
            # 变成一个看起来正常的回答。
            raise InvalidPlanOutput(
                f"`{where}.document_index` 越界：{index}（本轮只有 {len(documents)} 篇文档）"
            )
        quote = _require_str(item, "quote", where=f"{where}.quote")
        evidence.append(
            EvidenceRef(
                document_index=index,
                quote=quote,
                # ⚠ `strip()` 之后做包含判定：模型偶尔把引句前后带上换行/空格。
                #   ⛔ **只做这一层归一化**——大小写、标点一概不归一，那会让
                #   "引用存在"这条判据变得比它承诺的松。判据是"逐字"。
                verified=bool(quote.strip()) and quote.strip() in documents[index],
            )
        )

    return StepAnswer(
        # ⭐ **步骤编号由程序附加**（模型这一版不回它）——见 `StepAnswer` 的说明。
        step_id=step_id,
        status=status,
        answer=answer,
        evidence=tuple(evidence),
        extraction=extraction,
        raw=text,
    )
