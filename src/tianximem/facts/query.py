"""问题表面形式 → 共用条件/连接/区间计划；名单与计数使用相同证据计划。"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

from tianximem.facts.evidence import EvidenceFact, key, relation_key, title_key
from tianximem.facts.grammar import MONTH, NAME, month_index
from tianximem.facts.input import input_complete


@dataclass(frozen=True, slots=True)
class FactPattern:
    relations: tuple[str, ...] = ()
    subject: str = ""
    object: str = ""
    qualifiers: tuple[tuple[str, str | int | bool], ...] = ()
    object_terms: str = ""
    container_terms: str = ""
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "relations", tuple(relation_key(r) for r in self.relations))

    def matches(self, fact: EvidenceFact) -> bool:
        return (
            (not self.source_ids or fact.parent_memory_id in self.source_ids)
            and (not self.relations or relation_key(fact.relation) in self.relations)
            and (not self.subject or key(fact.subject) == self.subject)
            and (not self.object or key(str(fact.get("match_object", fact.object))) == self.object)
            and all(
                key(str(fact.get(name, ""))) == key(str(value)) for name, value in self.qualifiers
            )
            and (
                not self.object_terms or re.search(self.object_terms, fact.object, re.I) is not None
            )
            and (
                not self.container_terms
                or re.search(self.container_terms, str(fact.get("container", "")), re.I) is not None
            )
        )


@dataclass(frozen=True, slots=True)
class QueryPlan:
    operator: Literal["filter", "walk", "interval", "current"]
    patterns: tuple[FactPattern, ...] = ()
    projection: Literal["fact", "source"] = "fact"
    resolve_references: bool = False
    query: str = ""
    guard_literal: str = ""
    date_bounds: tuple[str, str] | None = None
    month: int | None = None
    direction: Literal["before", "after"] | None = None
    reference: str = ""
    bind_side: Literal["subject", "object"] = "subject"
    require_complete_walk: bool = False


def period(text: str) -> tuple[str, str] | None:
    names = {name.casefold(): i for i, name in enumerate(calendar.month_name) if name}
    named = re.search(
        r"\b(?:in|during)\s+(" + "|".join(names) + r")(?:\s+of)?\s+(\d{4})\b", text, re.I
    )
    numeric = re.search(r"\b(?:in|during)\s+month\s+(\d{1,2})(?:\s+of)?\s+(\d{4})\b", text, re.I)
    quarter = re.search(
        r"\b(?:in|during)\s+(?:Q|quarter\s+)([1-4])(?:\s+of)?\s+(\d{4})\b", text, re.I
    )
    if named:
        month, year, length = names[named[1].casefold()], int(named[2]), 1
    elif numeric:
        month, year, length = int(numeric[1]), int(numeric[2]), 1
    elif quarter:
        month, year, length = (int(quarter[1]) - 1) * 3 + 1, int(quarter[2]), 3
    else:
        return None
    try:
        return date(year, month, 1).isoformat(), date(
            year + (month + length - 1) // 12, (month + length - 1) % 12 + 1, 1
        ).isoformat()
    except ValueError:
        return None


# 关系同义词集中在问题语义层，不创建该关系自己的执行路径。
_QUERY_RELATIONS = (
    ("author", r"\bauthor\b|\bwrote\b"),
    ("capital", r"\bcapital\b"),
    ("country of citizenship", r"\bcitizenship\b|\bcitizen\b|\bnationality\b"),
    ("creator", r"\bcreator\b|\bcreated\b"),
    ("place of birth", r"\bbirthplace\b|\bborn\b|\boriginate from\b"),
    ("spouse", r"\bspouse\b|\bpartner\b|\bmarried\b|\bhusband\b|\bwife\b"),
    ("sport", r"\bsport\b"),
    ("country of origin", r"\bcountry of origin\b|\bcountry.*produced\b|\bcountry.*created\b"),
    ("founded by", r"\bfound(?:ed|er)\b"),
    ("continent", r"\bcontinent\b"),
    ("position played on team / speciality", r"\bposition\b|\bspeciality\b"),
    (
        "employee",
        r"\bemployer\b|\bworkplace\b|\bwork(?:s)? (?:for|at)\b|"
        r"\bwhere\s+(?:do|does)\s+.+?\s+work$",
    ),
    ("religion or worldview", r"\breligion\b|\bworldview\b|\bfaith\b"),
    ("notable work", r"\bknown for\b|\brenowned\b|\bnotable\b|\bachievement\b"),
    ("director", r"\bdirector\b|\bdirected\b"),
    ("official language", r"\blanguage\b"),
    ("head of state", r"\bhead of state\b|\bcurrent (?:head|leader)\b(?!.*government)"),
    ("head of government", r"\bgovernment\b"),
)
_UNSAFE_SCOPE = re.compile(
    (
        r"\b(?:not|except|before|after|during|since|until|last|previous|former|ever|should|"
        r"could|would|will|plan|hope|joined|started|attended)\b|\d{4}"
    ),
    re.I,
)

_ZH_UNSAFE_SCOPE = re.compile(
    r"不是|没有|不在|不再|除了|除外|排除|之前|之后|以前|曾经|去年|前年|上个月|"
    r"计划|打算|希望|将会|将要|下周|下个月|明年|如果|假如|\d{4}年"
)
_ZH_VERBS = r"工作|上班|任职|就读|念书|学习|做志愿者|当志愿者|任教"


def _chinese_plan(text: str) -> QueryPlan | None:
    """中文只绑定共同字段；同一范围的名单与计数复用一份证据。"""
    text = re.sub(r"^请问\s*", "", text).rstrip("呢")
    listed = bool(re.match(r"^(?:请)?列出", text))
    plain = re.sub(r"^(?:请)?列出", "", text)
    owned = re.fullmatch(rf"(我|{NAME})的(.+?)(?:里|中|内)(?:面)?(.*)", plain)
    if owned:
        owner, container, intent = owned[1], owned[2], owned[3].lstrip("的")
        all_items = bool(
            re.fullmatch(
                r"(?:有)?(?:哪些|什么)(?:物品|东西|物件|物品种类)|"
                r"(?:一共|总共)?(?:有)?(?:多少|几)(?:件|个)?(?:物品|东西|物件)|"
                r"物品(?:种类)?",
                intent,
            )
        )
        quantity = re.fullmatch(
            r"(?:一共|总共)?(?:有)?(?:多少|几)([个本支把件台张只条瓶枚块根顶])(.+)", intent
        )
        if all_items or (listed and intent in {"东西", "物件"}) or quantity:
            return QueryPlan(
                "filter",
                (
                    FactPattern(
                        ("contain",),
                        subject="i" if owner == "我" else key(owner),
                        object=key(quantity[2]) if quantity and not all_items else "",
                        qualifiers=(("container", key(container)),),
                    ),
                ),
                guard_literal=key(container),
            )
    frames = (
        rf"在(.+?)({_ZH_VERBS})的人(?:有多少|有哪些|都有谁|有谁)",
        rf"(?:哪些人|有哪些人|谁|有多少人)在(.+?)({_ZH_VERBS})",
    )
    match = re.fullmatch(rf"在(.+?)({_ZH_VERBS})的人", plain) if listed else None
    if match is None:
        match = next((m for p in frames if (m := re.fullmatch(p, plain))), None)
    if match:
        return QueryPlan(
            "filter",
            (
                FactPattern(
                    (relation_key(match[2]),),
                    object=key(match[1]),
                ),
            ),
            guard_literal=key(match[1]),
        )
    noun_patterns = (
        r"(.+?)(?:的)?(?:有)?(?:哪些|哪几位|多少名|多少位|多少个|几名|几位)(.+)",
        r"(.+?)的(.+?)(?:一共|总共|目前|现在)?(?:有)?"
        r"(?:多少(?:人|名|位)?|几(?:人|名|位)|有哪些|都有谁|有谁|名单(?:是什么)?)",
    )
    for pattern in noun_patterns:
        match = re.fullmatch(pattern, plain)
        if match:
            org, role = match[1].strip(), match[2].strip()
            if org and role:
                return QueryPlan(
                    "filter",
                    (
                        FactPattern(
                            (relation_key(role),),
                            object=key(org),
                        ),
                    ),
                    guard_literal=key(org),
                )
    if listed:
        match = re.fullmatch(r"(.+?)的([^的]+?)(?:名单)?", plain)
        if match:
            return QueryPlan(
                "filter",
                (
                    FactPattern(
                        (relation_key(match[2]),),
                        object=key(match[1]),
                    ),
                ),
                guard_literal=key(match[1]),
            )
    return None


_LIST_PREFIX = re.compile(
    (
        r"^(?:who(?: are)?|which are|list|name|give me|show me|how many|what is the (?:total )"
        r"?(?:number|count) of)\s+(.+)$"
    ),
    re.I,
)


def compile_query(query: str) -> QueryPlan | None:
    if input_complete(query):
        return QueryPlan("current")
    text = query.strip().rstrip("?.!。！？")
    if re.search(r"[\u4e00-\u9fff]", text):
        if _ZH_UNSAFE_SCOPE.search(text):
            return None
        chinese = _chinese_plan(text)
        if chinese is not None:
            return chinese
    # 显式区间问题：只有原文起止月份参与比较，绝不用消息日期补区间。
    at = re.fullmatch(rf"(.+?) in ({MONTH})", text, re.I)
    relative = re.fullmatch(r"(.+?) (before|after) (.+)", text, re.I)
    if at or relative:
        direction: Literal["before", "after"] | None = None
        reference_value = ""
        if at:
            frame, month = at[1], month_index(at[2])
        else:
            assert relative is not None
            frame, month = relative[1], None
            direction = "before" if relative[2].lower() == "before" else "after"
            reference_value = key(relative[3])
        subject_frame = re.fullmatch(
            r"Which (.+?) did (.+?) (?:work for|play for|hold|belong to)", frame, re.I
        )
        inverse = re.fullmatch(r"Who was the (.+?) of (?:the team )?(.+)", frame, re.I)
        pattern = None
        side: Literal["subject", "object"] = "subject"
        if subject_frame:
            pattern = FactPattern((relation_key(subject_frame[1]),), subject=key(subject_frame[2]))
        elif inverse:
            relation = inverse[1].lower().replace("head of the government", "head of government")
            pattern, side = FactPattern((relation_key(relation),), object=key(inverse[2])), "object"
        if pattern and (not at or month is not None):
            return QueryPlan(
                "interval",
                (pattern,),
                "source",
                month=month,
                direction=direction,
                reference=reference_value,
                bind_side=side,
            )
    # 来源标题是字段条件。与关系事实使用同一过滤和原文投影。
    if re.match(
        (
            r"^(?:how many (?:times|occurrences)|what is the total number of times|who "
            r"(?:organized|was the organizer|set up|attended|participated)|which employees "
            r"(?:were present|attended))\b"
        ),
        text,
        re.I,
    ):
        titles = set()
        explicit = re.search(
            r"\b(?:meeting|event|document|session) (?:titled|called|named) (.+)$", text, re.I
        )
        if explicit:
            titles.add(title_key(explicit[1]))
        for match in re.finditer(r"(?=\b(?:the|titled|called)\s+(.+?)\s+meeting\b)", text, re.I):
            titles.add(title_key(match[1]))
        if titles:
            return QueryPlan(
                "filter",
                tuple(FactPattern(("document title",), object=key(t)) for t in sorted(titles)),
                "source",
                date_bounds=period(text),
            )
    # 报告中的状态观察只约束来源日期，不宣称已经知道事件的精确日期。
    started = re.fullmatch(
        (
            r"(?:who (?:began employment|started working|began working|joined)|which employees "
            r"joined)\s+(?:(?:at|for)\s+)?(.+?)\s+(?:in|during)\s+(.+)"
        ),
        text,
        re.I,
    )
    if started:
        bounds = period(text)
        return (
            QueryPlan(
                "filter",
                (
                    FactPattern(
                        ("role commencement", "orientation schedule"), object=key(started[1])
                    ),
                ),
                date_bounds=bounds,
            )
            if bounds
            else None
        )
    action = re.fullmatch(
        rf"(?:Which|What) (events|activities|places) (?:has|did) ({NAME}) "
        r"(?:participated in|participate in|attended|attend)(?: to .+)?",
        text,
    )
    if action:
        terms = (
            (
                r"\b(?:networking|events?|fairs?|expos?|conferences?|exhibitions?|workshops?|"
                r"seminars?|conventions?|festivals?)\b"
            )
            if action[1] == "events"
            else ""
        )
        return QueryPlan(
            "filter", (FactPattern(("participated",), subject=key(action[2]), object_terms=terms),)
        )
    state = re.fullmatch(
        rf"When did ({NAME}) (open|close|get accepted for) (?:his|her|their|the) (.+)", text
    )
    if state:
        relation = {"open": "is open", "close": "is closed", "get accepted for": "accepted"}[
            state[2]
        ]
        noun = re.escape(state[3].split()[-1])
        return QueryPlan(
            "filter",
            (FactPattern((relation,), subject=key(state[1]), object_terms=rf"\b{noun}\b"),),
        )
    reference = re.fullmatch(rf"Where did ({NAME}) move from (\d+ years?) ago", text)
    if reference:
        return QueryPlan(
            "filter",
            (
                FactPattern(
                    ("moved from",),
                    subject=key(reference[1]),
                    qualifiers=(("duration", reference[2]),),
                ),
            ),
            resolve_references=True,
        )
    reference = re.fullmatch(rf"Where has ({NAME}) made friends", text)
    if reference:
        return QueryPlan(
            "filter",
            (FactPattern(("friends with", "friends from"), subject=key(reference[1])),),
            resolve_references=True,
        )
    if not _UNSAFE_SCOPE.search(text):
        # 容器不限业务名；集合对象由所有权和原文字面限定，不做总数计算。
        if re.match(
            (
                r"^(?:what types of items|how many items|list the kinds of belongings|what is "
                r"the (?:total )?number of items)\b"
            ),
            text,
            re.I,
        ):
            owned = re.search(rf"\b(?:in|inside)\s+the\s+(.+?)\s+owned by\s+({NAME})$", text)
            keeps = re.search(rf"\b({NAME})\s+(?:currently\s+)?keeps in the\s+(.+)$", text)
            possessive = re.search(rf"\b(?:in|inside)\s+({NAME})'s\s+(.+?)(?: in total)?$", text)
            if owned or keeps or possessive:
                if owned:
                    bound_subject, container = owned[2], owned[1]
                elif keeps:
                    bound_subject, container = keeps[1], keeps[2]
                else:
                    assert possessive is not None
                    bound_subject, container = possessive[1], possessive[2]
                return QueryPlan(
                    "filter",
                    (
                        FactPattern(
                            ("contain",),
                            subject=key(bound_subject),
                            qualifiers=(("container", key(container)),),
                        ),
                    ),
                    guard_literal=key(container),
                )
        quantities = re.fullmatch(
            (
                r"How many (.+?) (?:are there|do I have)\s*(?:in total\s*)?(?:in\s*(?:(?:both|"
                r"all)\s+of\s+)?)?my\s+(.+)"
            ),
            text,
            re.I,
        )
        if quantities:
            noun = relation_key(quantities[2])
            container_terms = (
                r"\b(?:aquariums?|tanks?)\b"
                if noun in {"aquarium", "tank"}
                else rf"\b{re.escape(noun)}s?\b"
            )
            item_terms = (
                r"\b(?:fish|tetras?|gouramis?|catfish|plecos?)\b"
                if relation_key(quantities[1]) == "fish"
                else ""
            )
            return QueryPlan(
                "filter",
                (
                    FactPattern(
                        ("contain",),
                        subject="i",
                        object_terms=item_terms,
                        container_terms=container_terms,
                    ),
                ),
            )
        listed = _LIST_PREFIX.fullmatch(text)
        if listed and not re.match(r"^who (?:is|was)\b", text, re.I):
            corpus = re.fullmatch(
                (
                    r"(?:distinct\s+)?(.+?)\s+(?:are\s+)?mentioned\s+(?:in|throughout)\s"
                    r"+(?:the\s+)?corpus"
                ),
                listed[1],
                re.I,
            )
            if corpus:
                return QueryPlan("filter", (FactPattern((relation_key(corpus[1]),)),))
            parts = re.fullmatch(r"(.+?)\s+(?:at|for|of|in|to|with)\s+(.+)", listed[1], re.I)
            if parts and not re.search(r"\band\b", parts[1] + " " + parts[2], re.I):
                return QueryPlan(
                    "filter",
                    (FactPattern((relation_key(parts[1]),), object=key(parts[2])),),
                    guard_literal=key(parts[2]),
                )
    relations = tuple(
        sorted(
            {
                relation_key(rel)
                for rel, pattern in _QUERY_RELATIONS
                if re.search(pattern, text, re.I)
            }
        )
    )
    if relations and len(text.split()) <= 64 and not _UNSAFE_SCOPE.search(text):
        return QueryPlan(
            "walk",
            (FactPattern(relations),),
            "source",
            query=text,
            require_complete_walk=bool(
                re.search(r"\bwork(?:s)? at\b|^where\s+(?:do|does)\s+.+?\s+work$", text, re.I)
            ),
        )
    return None
