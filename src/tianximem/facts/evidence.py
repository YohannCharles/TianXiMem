"""共同证据：主体、关系、客体、原文限定条件及来源，不保存汇总答案。"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from tianximem.common.render import day_granularity, render_evidence, render_record_evidence

EVIDENCE_VERSION = "memory-facts-v14-event-verbs"
Qualifier = str | int | bool
SourceSide = Literal["question", "answer"]


def key(value: str) -> str:
    """保留标点身份，不能把 C++ 与 C# 合成同一实体。"""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split()).strip(" ?.!。！？")


def words_key(value: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))


def title_key(value: str) -> str:
    return re.sub(r"\s+meeting$", "", words_key(value))


_ALIASES = {
    "work": "employee",
    "works": "employee",
    "working": "employee",
    "employee": "employee",
    "employees": "employee",
    "employer": "employee",
    "study": "student",
    "studies": "student",
    "studying": "student",
    "student": "student",
    "students": "student",
    "enrolled": "student",
    "belong": "member",
    "belongs": "member",
    "belonging": "member",
    "member": "member",
    "members": "member",
    "volunteer": "volunteer",
    "volunteers": "volunteer",
    "volunteering": "volunteer",
    "员工": "employee",
    "雇员": "employee",
    "工作": "employee",
    "上班": "employee",
    "任职": "employee",
    "学生": "student",
    "就读": "student",
    "念书": "student",
    "学习": "student",
    "志愿者": "volunteer",
    "做志愿者": "volunteer",
    "当志愿者": "volunteer",
    "会员": "member",
    "成员": "member",
    "患者": "patient",
    "病人": "patient",
    "老师": "teacher",
    "教师": "teacher",
    "任教": "teacher",
    "教练": "coach",
}


#: 档案字段的同义写法 → 规范字段名（`my personality type` 与题面的 `MBTI personality
#: type` 是同一件事）。只收**同一字段的不同叫法**，不收相邻但不同的字段。
_PROFILE_ALIASES = {
    "mbti personality type": "personality type",
    "myers-briggs personality type": "personality type",
    "myers briggs personality type": "personality type",
    "highest education level": "education level",
    "education": "education level",
    "current job title": "job title",
    "life goal statement": "life goal",
    # 题面的"状态"说法与原文的字段名是同一件事（HaluMem，2026-10-19）。
    "physical health status": "physical condition",
    "physical health": "physical condition",
    "mental health status": "mental health",
}


def relation_key(value: str) -> str:
    value = re.sub(r"\s+(?:of|at|for|in|to|with)$", "", key(value))
    value = " ".join(
        re.sub(
            r"\b(?:the|all|people|everyone|who|are|there|currently|now|still)\b", "", value
        ).split()
    )
    if value in _ALIASES:
        return _ALIASES[value]
    # 档案字段的**同义写法**归一到同一字段名：原文写 `my personality type is INTJ`，
    # 题面问 `MBTI personality type`——不归一就永远匹配不上（2026-10-18 实测）。
    if value in _PROFILE_ALIASES:
        return _PROFILE_ALIASES[value]
    mapped = {_ALIASES[w] for w in value.split() if w in _ALIASES}
    if len(mapped) == 1 and all(w in _ALIASES for w in value.split()):
        return next(iter(mapped))
    if value.endswith("ies"):
        return value[:-3] + "y"
    if value.endswith(("sses", "shes", "ches", "xes", "zes")):
        return value[:-2]
    return value[:-1] if value.endswith("s") and not value.endswith("ss") else value


@dataclass(frozen=True, slots=True)
class EvidenceFact:
    id: str
    parent_memory_id: str
    user_id: str
    subject: str
    relation: str
    object: str
    source_quote: str
    source_side: SourceSide
    statement: str
    content: str
    event_time: int | None
    source_date: str
    qualifiers: tuple[tuple[str, Qualifier], ...] = ()
    version: str = EVIDENCE_VERSION

    def get(self, name: str, default: Qualifier | None = None) -> Qualifier | None:
        return dict(self.qualifiers).get(name, default)


def make_fact(
    *,
    parent_memory_id: str,
    user_id: str,
    subject: str,
    relation: str,
    object: str,
    source_quote: str,
    source_side: SourceSide = "question",
    statement: str,
    event_time: int | None,
    position: str,
    source_date: str | None = None,
    qualifiers: Mapping[str, Qualifier] | None = None,
    record_fields: Mapping[str, str] | None = None,
    **attributes: Qualifier,
) -> EvidenceFact:
    qualifiers = {**(qualifiers or {}), **attributes}
    date = day_granularity(event_time) if source_date is None else source_date
    parts = (EVIDENCE_VERSION, user_id, parent_memory_id, source_side, position)
    identity = "".join(f"{len(part)}:{part}" for part in parts)
    return EvidenceFact(
        id=hashlib.sha256(identity.encode()).hexdigest(),
        parent_memory_id=parent_memory_id,
        user_id=user_id,
        subject=subject,
        relation=relation,
        object=object,
        source_quote=source_quote,
        source_side=source_side,
        statement=statement,
        content=render_record_evidence(
            source_id=parent_memory_id,
            source_date=date,
            source_quote=source_quote,
            fields=record_fields,
        )
        if record_fields is not None
        else render_evidence(
            statement=statement,
            source_date=date,
            source_quote=source_quote,
            quantitative="quantity" in qualifiers,
        ),
        event_time=event_time,
        source_date=date,
        qualifiers=tuple(sorted(qualifiers.items())),
    )
