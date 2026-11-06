from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "COQ_DECISION_KEYS",
    "CoqDecision",
    "InvalidCoqOutput",
    "extract_json_object",
    "parse_coq_decision",
]

COQ_DECISION_KEYS: Final[frozenset[str]] = frozenset(
    {"is_sufficient", "evidence_indices", "new_query", "confidence_score"}
)


class InvalidCoqOutput(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CoqDecision:
    is_sufficient: bool
    evidence_indices: tuple[int, ...]
    new_query: str
    confidence_score: float
    extraction: str
    raw: str


def _balanced_objects(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append((start, index))
    return spans


def extract_json_object(text: str) -> tuple[dict[str, Any], str] | None:
    stripped = text.strip()
    if stripped:
        try:
            direct = json.loads(stripped)
        except ValueError:
            direct = None
        if isinstance(direct, dict):
            return direct, "direct"

    spans = _balanced_objects(text)
    for start, end in reversed(spans):
        try:
            parsed = json.loads(text[start : end + 1])
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed, "recovered"
    return None


def _require_bool(payload: dict[str, Any], key: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise InvalidCoqOutput(f"`{key}` 必须是布尔，收到 {type(value).__name__}")
    return value


def _require_indices(payload: dict[str, Any], key: str, *, document_count: int) -> tuple[int, ...]:
    value = payload[key]
    if not isinstance(value, list):
        raise InvalidCoqOutput(f"`{key}` 必须是数组，收到 {type(value).__name__}")
    indices: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise InvalidCoqOutput(f"`{key}` 里必须是整数，收到 {type(item).__name__}")
        if item < 0 or item >= document_count:
            raise InvalidCoqOutput(f"`{key}` 越界：{item}（本轮只有 {document_count} 篇文档）")
        if item not in indices:
            indices.append(item)
    return tuple(indices)


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise InvalidCoqOutput(f"`{key}` 必须是字符串，收到 {type(value).__name__}")
    if "\n" in value or "\r" in value:
        raise InvalidCoqOutput(f"`{key}` 必须是单行")
    return value


def _require_score(payload: dict[str, Any], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidCoqOutput(f"`{key}` 必须是数字，收到 {type(value).__name__}")
    score = float(value)
    if not math.isfinite(score):
        raise InvalidCoqOutput(f"`{key}` 必须是有限数字")
    if not 0.0 <= score <= 1.0:
        raise InvalidCoqOutput(f"`{key}` 必须在 0..1 之间，收到 {score}")
    return score


def parse_coq_decision(text: str, *, document_count: int) -> CoqDecision:
    found = extract_json_object(text)
    if found is None:
        raise InvalidCoqOutput("响应里找不到 JSON 对象")
    payload, extraction = found

    keys = set(payload)
    if keys != COQ_DECISION_KEYS:
        missing = sorted(COQ_DECISION_KEYS - keys)
        extra = sorted(keys - COQ_DECISION_KEYS)
        raise InvalidCoqOutput(f"字段不匹配：缺少 {missing}，多出 {extra}")

    return CoqDecision(
        is_sufficient=_require_bool(payload, "is_sufficient"),
        evidence_indices=_require_indices(
            payload, "evidence_indices", document_count=document_count
        ),
        new_query=_require_str(payload, "new_query"),
        confidence_score=_require_score(payload, "confidence_score"),
        extraction=extraction,
        raw=text,
    )
