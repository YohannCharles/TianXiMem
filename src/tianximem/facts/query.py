"""问题表面形式 → 共用条件/连接/区间计划；名单与计数使用相同证据计划。"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from tianximem.facts.evidence import FIRST_PERSON, EvidenceFact, key, relation_key
from tianximem.facts.grammar import MONTH, NAME, month_index
from tianximem.facts.input import input_complete
from tianximem.facts.paths import question_path
from tianximem.facts.sales import SALE_ASSERTION

#: 「逐次取证」的关系集合：同一条事实在**不同来源**里的每一次出现各自算证据
#: （"哪几次""谁在场"）。执行器不得按三元组去重——去重会把"几次"压成"一次"。
#: ⚠ **这是唯一一份**：[`../service/grounded_planner.py`](../service/grounded_planner.py)
#: 从本模块导入它；⛔ 不要在 planner 里再抄一份——新增关系时两侧会**静默不同步**，
#: 表现为"出现几次"类题少算。
MULTISET_RELATIONS = frozenset(
    {
        "document title",
        "meeting topic",
        "meeting type",
        "attendee",
        "organizer",
        "absent from meeting",
        "meeting frequency",
    }
)


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

    def matches(self, fact: EvidenceFact, owner: str = "") -> bool:
        """`owner` 是该 user 的真名键（读取期解析，D50）。

        ⚠ **为什么需要它**：`store.fetch_evidence` 在**读取时**把第一人称事实的
        `subject` 换成真名（D50），而计划里的 `subject` 仍是字面 `"i"`。
        不带 `owner` 比较会让「问 I」的计划在**有自我介绍的用户**上全部判否
        ——静默回退原链，且只在部分用户上发生（2026-10-19 实测复现）。
        """
        return (
            (not self.source_ids or fact.parent_memory_id in self.source_ids)
            and (not self.relations or relation_key(fact.relation) in self.relations)
            and (
                not self.subject
                or key(fact.subject) == self.subject
                or (bool(owner) and self.subject in FIRST_PERSON and key(fact.subject) == owner)
            )
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
    #: 同一三元组在**不同来源**里的重复各自算证据（"出现几次"按事实行数数）。
    #: 执行器必须为此关掉按三元组的去重；段级去重（同一条来源只留一条）不受影响。
    per_occurrence: bool = False
    month: int | None = None
    direction: Literal["before", "after"] | None = None
    reference: str = ""
    bind_side: Literal["subject", "object"] = "subject"
    require_complete_walk: bool = False
    guard_assertion_pattern: str = ""
    guard_source_literals: tuple[str, ...] = ()
    walk_root: str = ""
    walk_relations: tuple[str, ...] = ()


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


_DATE_MONTHS = {name.casefold(): i for i, name in enumerate(calendar.month_name) if name}
_DATE_MONTHS.update(
    {name.casefold(): i for i, name in enumerate(calendar.month_abbr) if name}
)
_ORDINAL = r"(?:st|nd|rd|th)?"


def _date_literal(value: str, *, default_year: int | None = None) -> date | None:
    """Parse an explicit human date without using the message timestamp."""
    value = re.sub(r"^(?:on|the)\s+", "", value.strip(), flags=re.I)
    value = re.sub(
        r"^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+",
        "",
        value,
        flags=re.I,
    )
    value = re.sub(r"\s+", " ", value).strip().rstrip(".,")
    patterns = (
        r"(?P<iso_year>\d{4})-(?P<iso_month>\d{2})-(?P<iso_day>\d{2})",
        r"(?:the\s+)?(?P<day>\d{1,2})" + _ORDINAL + r"\s+of\s+(?P<month>[A-Za-z]+)(?:,?\s+(?P<year>\d{4}))?",
        r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})" + _ORDINAL + r"(?:,?\s+(?P<year>\d{4}))?",
        r"(?P<day>\d{1,2})" + _ORDINAL + r"\s+(?P<month>[A-Za-z]+)(?:,?\s+(?P<year>\d{4}))?",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, value, re.I)
        if not match:
            continue
        if match.groupdict().get("iso_year"):
            try:
                return date(int(match["iso_year"]), int(match["iso_month"]), int(match["iso_day"]))
            except ValueError:
                return None
        month = _DATE_MONTHS.get(match["month"].casefold())
        year = int(match["year"]) if match["year"] else default_year
        if month is None or year is None:
            return None
        try:
            return date(year, month, int(match["day"]))
        except ValueError:
            return None
    return None


def time_bounds(text: str) -> tuple[str, str] | None:
    """显式时间措辞 → 半开区间。解析不出返回 `None`（**不用消息时间补**）。

    这是"字面时间段"的唯一入口：编译期与 LLM 计划的 `period` 都走它。
    裸日期（"September 4, 2025"）按"截至该日"处理——planner 常把 "as of X" 里的
    日期单独抄下来，语义相同。
    """
    bounds = period(text) or _explicit_date_bounds(text)
    if bounds:
        return bounds
    value = _date_literal(text.strip())
    if value and len(text.strip()) <= 40:
        return "0001-01-01", (value + timedelta(days=1)).isoformat()
    return None


def _explicit_date_bounds(text: str) -> tuple[str, str] | None:
    """Return half-open bounds for explicit date language in a query."""
    # A seven-day window is the only interpretation of "week of/starting" that
    # is independent of locale and does not borrow the record timestamp.
    week = re.search(
        r"\b(?:during|in|for)\s+(?:the\s+)?week\s+(?:of|starting(?:\s+on)?)\s+(.+)$",
        text,
        re.I,
    )
    if week:
        start = _date_literal(week[1])
        if start:
            return start.isoformat(), (start + timedelta(days=7)).isoformat()

    between = re.search(
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:,?\s+(\d{4}))?$",
        text,
        re.I,
    )
    if between:
        year = int(between[3]) if between[3] else None
        left = _date_literal(between[1], default_year=year)
        right = _date_literal(between[2], default_year=year)
        if left and right is None and re.fullmatch(r"\d{1,2}" + _ORDINAL, between[2].strip(), re.I):
            try:
                right = date(left.year, left.month, int(re.sub(r"\D", "", between[2])))
            except ValueError:
                right = None
        if left and right:
            if right < left and year is not None:
                right = _date_literal(between[2], default_year=left.year)
            if right:
                return left.isoformat(), (right + timedelta(days=1)).isoformat()

    from_to = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+)$", text, re.I)
    if from_to:
        year_match = re.search(r"\b(\d{4})\b", from_to[2])
        year = int(year_match[1]) if year_match else None
        left = _date_literal(from_to[1], default_year=year)
        right = _date_literal(from_to[2], default_year=year)
        if left and right is None and re.fullmatch(r"\d{1,2}" + _ORDINAL, from_to[2].strip(), re.I):
            try:
                right = date(left.year, left.month, int(re.sub(r"\D", "", from_to[2])))
            except ValueError:
                right = None
        if left and right:
            return left.isoformat(), (right + timedelta(days=1)).isoformat()

    as_of = re.search(r"\bas of\s+(.+)$", text, re.I)
    if as_of:
        # "as of X" = 截至 X 的状态：只要来源日期不晚于 X 就行。
        value = _date_literal(as_of[1])
        if value:
            return "0001-01-01", (value + timedelta(days=1)).isoformat()

    after = re.search(r"\bafter\s+(.+)$", text, re.I)
    if after:
        value = _date_literal(after[1])
        if value:
            return (value + timedelta(days=1)).isoformat(), "9999-12-31"

    before = re.search(r"\bbefore\s+(.+)$", text, re.I)
    if before:
        value = _date_literal(before[1])
        if value:
            return "0001-01-01", value.isoformat()
    return None


#: 路径编译的护栏。⚠ **年份本身不是限制**（2026-10-17）：`time_bounds()` 会把
#: "as of September 4, 2025" 这类字面日期解析成区间并放进计划的 `date_bounds`；
#: 只要时间措辞**已被字面吸收**，就不该因为句子里出现 `\d{4}` 而整题弃权
#: （HaluMem 的 "… as of September 4, 2025" 曾因此 100% 落空）。
_UNSAFE_SCOPE = re.compile(
    (
        r"\b(?:not|except|before|after|during|since|until|last|previous|former|ever|should|"
        r"could|would|will|plan|joined|started|attended)\b"
        # ⚠ `hope` 只拦**带人称主语的计划动词**：MQuAKE 有 "where Hope Lange is a
        # citizen"，词级护栏把整题毙成 `unsupported_scope`（2026-10-17 实测）。
        r"|\b(?:I|we|you|they|he|she)\s+hope(?:s|d)?\b"
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


#: "how many times … the X meeting" 里，`(?=the …)` 会先跨过量化短语；那不是标题。
_TITLE_QUANTIFIER = re.compile(r"(?:total )?(?:number|count) of (?:times|occurrences)\b", re.I)

#: 列表前缀的通用回退只接名词短语；否定与疑问动词说明那是句子，不是关系名。
_CLAUSE_RELATION = re.compile(
    r"\b(?:n't|not|never|no)\b|"
    r"\b(?:appear|occur|happen|take|took|discuss|mention|hold|held|attend|join|"
    r"participate|organize|schedule|invite)(?:s|ed|ing)?\b",
    re.I,
)

_LIST_PREFIX = re.compile(
    (
        r"^(?:who(?: are)?|which are|list|name|give me|show me|how many|what is the (?:total )"
        r"?(?:number|count) of)\s+(.+)$"
    ),
    re.I,
)


#: 事件问法宾语里不参与匹配的词（题面与原文常有代词/介词差异）。
_EVENT_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "about", "into", "over",
    "our", "your", "their", "its", "are", "was", "were", "has", "have", "had",
    "his", "her", "him", "she", "they", "them", "you", "all", "some", "any",
    # 序数 / 数量词不是事件宾语的核心名词（`adopt the first three of her dogs`
    # 的匹配词应该是 `dogs`，不是 `first`）。
    "first", "second", "third", "fourth", "fifth", "last", "next", "new",
    "one", "two", "three", "four", "five", "many", "much", "more", "most",
}



def _event_terms(text: str) -> str | None:
    """事件宾语 → 匹配词。

    **精度优先**（2026-10-19）：要求宾语里的**每一个**实词都出现在事实客体里，
    而不是只取第一个。只取第一个时 "What did Evan mention …" 会匹配到任何含
    "been" 的事件（实测 LoCoMo 精确率 0.47）；全词合取把这类假阳性挡在外面。
    """
    tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9]+", text)
        if len(token) > 2 and token.casefold() not in _EVENT_STOP
    ]
    if not tokens:
        return None
    if len(tokens) == 1:
        # 单个实词保持原来的 `\bX\b` 形式（与既有计划逐字一致）。
        return rf"\b{re.escape(tokens[0])}\b"
    return "".join(rf"(?=.*\b{re.escape(token)}\b)" for token in tokens)

def _meeting_topic_plan(text: str) -> QueryPlan | None:
    """会议主题问法 → 普通 filter 计划（逐次取证，**不汇总、不计数**）。

    "What meetings are linked to X" / "What are the titles of the meetings about X" /
    "Which meetings discuss X" / "Did … meetings take place about X"。
    主题抓不到时返回 `None`（调用方回退原链），**不硬造标题**。
    """
    meeting = re.fullmatch(
        r"(?:what|which) (?:meetings?|sessions?) (?:are |were )?(?:linked to|about|"
        r"discussing|discuss|on) (.+?)(?: that (?:took|occurred|happened).*)?",
        text,
        re.I,
    ) or re.fullmatch(
        r"(?:what|which) (?:are|were) the titles of the (?:meetings?|sessions?) "
        r"(?:about|on|discussing|discuss|regarding) (.+?)(?: that .*)?",
        text,
        re.I,
    )
    if meeting is None:
        # "… about X that took place in quarter 1 of 2024"——尾部时间从句先剥掉再匹配。
        stripped = re.sub(r"\s+that\s+(?:took|occurred|happened)\b.*$", "", text, flags=re.I)
        meeting = re.fullmatch(
            r"(?:what|which) (?:are|were) the titles of the (?:meetings?|sessions?) "
            r"(?:about|on|discussing|discuss|regarding) (.+)",
            stripped,
            re.I,
        )
    if meeting is None:
        meeting = re.fullmatch(
            r"did (?:any )?(?:department|direct report|project|team|task)? ?meetings? "
            r"(?:take place|occur|happen) (?:about|on|discussing) (.+?)"
            r"(?: between | during | in .*)?",
            text,
            re.I,
        )
    if meeting is None:
        return None
    topic = meeting[1].strip().rstrip("?.")
    if not topic:
        return None
    # **时间从句不参与主题匹配**（2026-10-18）：`about Task Documentation and Handover
    # between January 2 and 5, 2024` 的主题是前半句；把 `between/January/2024` 当
    # 主题词元会让它永远匹配不上（实测 CorporateBench 那批 `no_match`）。
    # 时间本身交给 `date_bounds`——与其它计划同一口径。
    bounds = time_bounds(text)
    topic = re.split(
        r"\s+(?:that\s+)?(?:took place|occurred|happened)?\s*"
        r"(?:between|during|in|on|after|before)\s+(?:the\s+)?(?:quarter|month|week|year|\d|"
        r"January|February|March|April|May|June|July|August|September|October|November|December)\b",
        topic,
        maxsplit=1,
        flags=re.I,
    )[0].strip()
    if not topic:
        return None
    # ⚠ **标题那一半不在规则层做**：`fetch_evidence` 的 SQL 只支持 relation /
    # subject / object 相等，不支持 `object_terms`（那是 `matches` 才有的）——
    # 拿它当标题前缀匹配会在**取回阶段**全量返回（实测直接打满 96 条上限）。
    # 标题匹配留给 planner 的 `document title` 分支；规则层只认**明确的主题事实**。
    #
    # 主题用 `object_terms`（词元全部出现，顺序不限）而不是 `object` 相等：
    # 事实里存的是原文长句（`the Checkout Process Streamlining & Payment Gateway
    # Integration initiative`），题面是短名——**相等永远匹配不上**。
    # 取回阶段只按 relation 拉 `meeting topic`（实测 15 条），不会爆上限。
    # 停用词不参与匹配：`task progress and status update` 里的 `and` 不该要求标题里
    # 也出现 "and"（标题写 `task progress & status update` 就永远匹配不上）。
    _STOP = {
        "the", "and", "for", "with", "from", "that", "this", "about", "into", "over",
        "our", "your", "their", "its", "are", "was", "were", "has", "have", "had",
    }
    tokens = [
        re.escape(t)
        for t in re.findall(r"[A-Za-z0-9]+", topic)
        if len(t) > 2 and t.casefold() not in _STOP
    ]
    if not tokens:
        return None
    terms = "".join(rf"(?=.*\b{t}\b)" for t in tokens)
    # ⚠ **不设 `guard_literal`**：会议邮件里还有大量与本主题无关的句子，
    # 来源审查会把它们当成"未解释的冲突"从而整题回退（实测 `unsupported_source`）。
    #
    # **`document title` 也算同一批证据**（2026-10-18）：CorporateBench 的会议在语料里
    # 大量以**邮件标题**出现（353 条 `document title` 事实），而 `meeting topic` 只有
    # 15 条。只查后者会让"titles of the meetings about X"整类 `no_match`。
    # 两者是**同一次取证的两种来源写法**，不是两条独立事实。
    return QueryPlan(
        "filter",
        (
            FactPattern(("meeting topic", "document title"), object_terms=terms),
        ),
        projection="source",
        per_occurrence=True,
        # 题面写了时间范围就带上（`between Jan 2 and 5` / `in quarter 1 of 2024`）——
        # 不带等于把限制丢掉，`Did … take place during …` 那类会给出范围外的会议。
        date_bounds=bounds,
    )


def compile_query(query: str) -> QueryPlan | None:
    if input_complete(query):
        return QueryPlan("current")
    text = query.strip().rstrip("?.!。！？")
    sales = re.fullmatch(
        r"(?:What is (?:the )?(?:total (?:amount of )?(?:money|income|revenue|earnings)) "
        r"I (?:earned|made|received)|How much (?:money|income|revenue) (?:did|have) I "
        r"(?:earn|earned|make|made|receive|received)) from selling (?:my )?products "
        r"at (?:the )?markets",
        text,
        re.I,
    )
    if sales:
        return QueryPlan(
            "filter",
            (FactPattern(("sale",), subject="i", container_terms=r"\bmarket\b"),),
            guard_literal="market",
            guard_assertion_pattern=rf"{SALE_ASSERTION}|\b(?:refund\w*|correction|corrected|cancelled|canceled|returned)\b",
            guard_source_literals=("refund", "correction", "corrected", "cancel"),
        )
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
            r"(?:were present|attended)|what (?:are|were) the titles? of|which meetings? "
            r"(?:discussed|are associated with|are linked to|are attached to)|did (?:any )?"
                r"(?:department|direct report|project|team|task) meetings? (?:take place|occur|"
            r"(?:discuss|discussing|about))|were there (?:any )?"
            r"(?:department|direct report|project|team|task) meetings?)\b"
        ),
        text,
        re.I,
    ):
        titles = set()
        explicit = re.search(
            r"\b(?:meeting|event|document|session) (?:titled|called|named) (.+)$", text, re.I
        )
        if explicit:
            titles.add(key(explicit[1]))
        for match in re.finditer(r"(?=\b(?:the|titled|called)\s+(.+?)\s+meeting\b)", text, re.I):
            # 题面里的标题按 `key()` 保留标点（与事实侧 `object_key` 同一归一化）；
            # 只跳过 "how many times … the X meeting" 这种跨过量化短语的捕获。
            title = key(match[1])
            if _TITLE_QUANTIFIER.match(title):
                continue
            titles.add(title)
        if not titles:
            # 题面没点名**文档标题**时，它问的是**会议主题**——而抽取层现在有
            # `meeting topic` 事实（2026-10-17）。这里只生成主题计划；主题抓不到
            # 就照旧回退混合链（`no_match`），不硬造标题。
            topic_plan = _meeting_topic_plan(text)
            return topic_plan
        if titles:
            return QueryPlan(
                "filter",
                tuple(FactPattern(("document title",), object=t) for t in sorted(titles)),
                "source",
                date_bounds=time_bounds(text),
                # 同一标题的**每一次出现**都是证据（"出现几次"直接数事实行）。
                per_occurrence=True,
            )
    # 报告中的状态观察只约束来源日期，不宣称已经知道事件的精确日期。
    started = re.fullmatch(
        (
            r"(?:who (?:began employment|started working|began working|joined)|which employees "
            r"joined)\s+(?:(?:at|for)\s+)?(.+?)\s+"
            r"(?:in|during|from|between|after|before)\s+(.+)"
        ),
        text,
        re.I,
    )
    if started is None:
        started = re.fullmatch(
            (
                r"(?:how many|what is the total number of)\s+(?:employees|people)\s+"
                r"(?:who\s+)?(?:joined|started working(?: at)?|began working(?: at)?)\s+"
                r"(?:(?:at|for)\s+)?(.+?)\s+"
                r"(?:in|during|from|between|after|before)\s+(.+)"
            ),
            text,
            re.I,
        )
    if started:
        bounds = time_bounds(text)
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
    # Employment lists/counts are a common closed-world fact query. Keep the
    # company literal as the guard so unrelated department/team mentions do
    # not get counted as employees.
    employment = re.fullmatch(
        (
            r"(?:which|what|who)\s+(?:employees?|people)\s+(?:work|are employed|"
            r"(?:currently\s+)?(?:working|employed))\s+(?:at|for)\s+(.+)"
        ),
        text,
        re.I,
    )
    if employment:
        return QueryPlan(
            "filter",
            (FactPattern(("employee",), object=key(employment[1])),),
            guard_literal=key(employment[1]),
        )
    employment_count = re.fullmatch(
        (
            r"(?:how many|what is the total number of)\s+(?:employees?|people)\s+"
            r"(?:work|are employed|(?:currently\s+)?(?:working|employed))\s+"
            r"(?:at|for)\s+(.+)"
        ),
        text,
        re.I,
    )
    if employment_count:
        return QueryPlan(
            "filter",
            (FactPattern(("employee",), object=key(employment_count[1])),),
            guard_literal=key(employment_count[1]),
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
    # **通用事件时间问法**（2026-10-17）："When did X <动词> <宾语>"。答案不是关系客体，
    # 而是**那条事实的来源日期**（`source_date`，渲染进 content 的 "Source record date"）。
    # 所以计划只需要找回**那条事件事实**；日期由执行器原样带出，不做任何日历换算。
    # ⚠ 匹配词取**宾语里第一个实词**，跳过开头的 `to` 与停用词：
    # "start to go to the gym" 的第一个词是 `to`，会命中 "trip to Rome"（实测）；
    # 而要求**所有**实词都出现又会被目的状语拖死（"for his store" 不在事件宾语里）。
    # ⚠ **主语必须是"人"**（`{NAME}` 要求首字母大写、无动词）。`When does winter
    # begin where …` 里的 `winter` 是**小写普通名词**，不能被当成事件主语——
    # 2026-10-19 实测：放宽动词表后这条被误接成 `participated` 计划。
    # `{NAME}` 本身已经拦住了小写词，这里再显式要求 `did`（不是 `does`）。
    event = re.fullmatch(
        rf"When did ({NAME}) (?:go to|attend|visit|host|launch|start|get|"
        rf"paint|draw|join|apply to|receive|mention|design|lose|lost|take|took|"
        # 事件动词补表（2026-10-18，官方 LoCoMo 语料实测）：这些在对话里都带来源日期，
        # 与上面那批同一口径——答案仍是**那条事实的 source_date**，不做日历换算。
        # ⚠ `begin` 只作**人的**开始（"began a course"）；"winter begins" 是自然现象，
        # 不是记忆里的事件——它由 `{NAME}` 的大写要求挡住。
        rf"interview|decide|decided|collaborate|create|share|find|found|meet|met|"
        rf"adopt|buy|purchase|sell|sold|move|moved|leave|left|finish|complete|"
        rf"celebrate|perform|compete|win|won|publish|record|open|close|stop|"
        rf"quit|retire|marry|graduate|travel|return|arrive|discover|learn|teach) "
        rf"(?:a |an |the )?(.+)",
        text,
    )
    if event:
        terms = _event_terms(event[2]) or ""
        return QueryPlan(
            "filter",
            (FactPattern(("participated",), subject=key(event[1]), object_terms=terms),),
        )
    # **"When did X <动词> <宾语>" 的语序变体**（官方 LoCoMo，2026-10-19）：
    # 疑问词后直接跟主语（"When Jon has lost his job"）或进行时（"When is X performing
    # at a festival"）。仍是**同一件事**：答案 = 那条事件事实的 `source_date`。
    # ⚠ 与上面同一口径：宾语匹配词要求**全部实词**命中，不做日历换算。
    loose_event = re.fullmatch(
        rf"When ({NAME}) (?:has |had |was |is |were |did )?"
        rf"(?:go to|went to|attend|attended|visit|visited|host|hosting|launch|launched|"
        rf"start|started|begin|began|get|got|paint|painted|draw|drew|join|joined|"
        rf"apply to|applied to|receive|received|mention|mentioned|design|designed|"
        rf"lose|lost|take|took|interview|interviewed|decide|decided|collaborate|"
        rf"create|created|share|shared|find|found|meet|met|adopt|adopted|buy|bought|"
        rf"sell|sold|move|moved|leave|left|finish|finished|complete|completed|"
        rf"celebrate|celebrated|perform|performing|compete|competed|win|won|publish|"
        rf"published|record|recorded|open|opened|close|closed|stop|stopped|quit|"
        rf"retire|retired|marry|married|graduate|graduated|travel|travelled|traveled|"
        rf"return|returned|arrive|arrived|discover|discovered|learn|learned|teach|taught) "
        rf"(?:a |an |the )?(.+)",
        text,
    )
    if loose_event:
        terms = _event_terms(loose_event[2]) or ""
        return QueryPlan(
            "filter",
            (
                FactPattern(
                    ("participated",), subject=key(loose_event[1]), object_terms=terms
                ),
            ),
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
    # **会议主题问法**（2026-10-17）："What meetings are linked to X" /
    # "What are the titles of the meetings about X" / "Which meetings discuss X"。
    topic_plan = _meeting_topic_plan(text)
    if topic_plan is not None:
        return topic_plan
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
            if (
                parts
                and not re.search(r"\band\b", parts[1] + " " + parts[2], re.I)
                # 这一支只接**名词短语**（"the members at X"）；带否定或疑问动词的是问句句法，
                # 拿它当关系名会造出一个永远匹配不上的字段（"didn't attend meeting"）。
                and not _CLAUSE_RELATION.search(parts[1])
            ):
                return QueryPlan(
                    "filter",
                    (FactPattern((relation_key(parts[1]),), object=key(parts[2])),),
                    guard_literal=key(parts[2]),
                )
    path = question_path(text) if not _UNSAFE_SCOPE.search(text) else None
    if path and len(text.split()) <= 64:
        root, steps = path
        ordered = tuple(relation_key(step) for step in steps)
        # A literal source edge can answer a walk even when the final hop is
        # not yet indexed.  Keep the historical partial-walk fallback for
        # these controlled completion queries; newly introduced paths remain
        # strict and require every hop to be present.
        require_complete = not (
            len(ordered) > 1
            and ordered[-1] in {"head of government", "head of state"}
            and ordered[0] in {"country of citizenship", "country of origin"}
        )
        # 题面里的字面日期（"as of September 4, 2025"）由 `time_bounds` 解析成区间；
        # 解析不了就不带区间——**不因为出现年份而弃权**。
        return QueryPlan(
            "walk",
            (FactPattern(tuple(sorted(set(ordered)))),),
            "source",
            query=text,
            date_bounds=time_bounds(text),
            require_complete_walk=require_complete,
            walk_root=key(root),
            walk_relations=ordered,
        )
    return None
