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

PLAN_DECOMPOSE_KEYS: Final[frozenset[str]] = frozenset({"steps"})

PLAN_STEP_KEYS: Final[frozenset[str]] = frozenset({"id", "query", "depends_on"})

STEP_ANSWER_KEYS: Final[frozenset[str]] = frozenset({"status", "answer", "evidence"})

EVIDENCE_KEYS: Final[frozenset[str]] = frozenset({"document_index", "quote"})

STEP_STATUSES: Final[tuple[str, ...]] = ("answered", "unresolved", "conflict")

PLAN_KEY_RE: Final[re.Pattern[str]] = re.compile(r"^s[0-9]+$")

_PLACEHOLDER_RE: Final[re.Pattern[str]] = re.compile(r"\{\{(s[0-9]+)\}\}")


class InvalidPlanOutput(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PlanStep:
    id: str
    query: str
    depends_on: tuple[str, ...]
    placeholders: tuple[str, ...]
    declared_dependencies: tuple[str, ...]
    unused_dependencies: tuple[str, ...]
    dependencies_repaired: bool


@dataclass(frozen=True, slots=True)
class Plan:
    steps: tuple[PlanStep, ...]
    extraction: str
    raw: str


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    document_index: int
    quote: str
    verified: bool


@dataclass(frozen=True, slots=True)
class StepAnswer:
    step_id: str
    status: str
    answer: str | None
    evidence: tuple[EvidenceRef, ...]
    extraction: str
    raw: str

    @property
    def is_answer(self) -> bool:

        return self.status == "answered" and bool(self.answer)


def placeholders_in(text: str) -> tuple[str, ...]:

    out: list[str] = []
    for match in _PLACEHOLDER_RE.finditer(text):
        if match.group(1) not in out:
            out.append(match.group(1))
    return tuple(out)


def bind_placeholders(template: str, answers: dict[str, str]) -> tuple[str, tuple[str, ...]]:

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


def _single_line(value: str, *, where: str) -> str:
    if "\n" in value or "\r" in value:
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


def parse_plan(text: str, *, max_steps: int = 4) -> Plan:

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
                raise InvalidPlanOutput(
                    f"`{where}.depends_on` 里的 {entry!r} 不是**此前**出现过的步骤（已有 {seen}）"
                )
            if entry not in declared:
                declared.append(entry)

        placeholders = placeholders_in(query)
        for name in placeholders:
            if name not in seen:
                raise InvalidPlanOutput(
                    f"`{where}.query` 里的 {{{{{name}}}}} 不是**此前**出现过的步骤（已有 {seen}）"
                )
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

    lines: list[str] = []
    for step in plan.steps:
        suffix = f"  (depends on {', '.join(step.depends_on)})" if step.depends_on else ""
        lines.append(f"{step.id}: {step.query}{suffix}")
    return "\n".join(lines)


def parse_step_answer(text: str, *, step_id: str, documents: Sequence[str]) -> StepAnswer:

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
        _single_line(answer, where="answer")

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
        if isinstance(index, bool) or not isinstance(index, int):
            raise InvalidPlanOutput(
                f"`{where}.document_index` 必须是整数，收到 {type(index).__name__}"
            )
        if index < 0 or index >= len(documents):
            raise InvalidPlanOutput(
                f"`{where}.document_index` 越界：{index}（本轮只有 {len(documents)} 篇文档）"
            )
        quote = _require_str(item, "quote", where=f"{where}.quote")
        evidence.append(
            EvidenceRef(
                document_index=index,
                quote=quote,
                verified=bool(quote.strip()) and quote.strip() in documents[index],
            )
        )

    return StepAnswer(
        step_id=step_id,
        status=status,
        answer=answer,
        evidence=tuple(evidence),
        extraction=extraction,
        raw=text,
    )
