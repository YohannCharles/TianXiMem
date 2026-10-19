"""来源句法 → 共同事实。语义词只绑定字段，不拥有表、开关或检索路径。"""

from __future__ import annotations

import calendar
import json
import re
from email.utils import parseaddr
from typing import cast

from tianximem.facts.assertions import interrogative
from tianximem.facts.conversation import NAME, SPEAKER, named_utterances, sentences
from tianximem.facts.entities import EntityExtractor, ExtractionCandidates, load_spacy_extractor
from tianximem.facts.evidence import (
    EvidenceFact,
    SourceSide,
    key,
    make_fact,
    relation_key,
)
from tianximem.facts.sales import sale_fields

_UNSAFE = re.compile(
    r'["“”>]|‘[^‘’]*’|\b(?:not|never|no|if|would|might|fictional|imagining|said|says|quoted)\b'
    r"|不是|并非|没有|不在|不再|从未|未曾|如果|假如|假设|虚构|示例|例子|例如|据说|听说"
    r"|说[：:]|写道[：:]",
    re.I,
)
_PLANNED = re.compile(
    r"\b(?:will|hope|plan|want|gonna|going to)\b"
    r"|计划|打算|希望|想要|准备成为|准备去|将会|将要|下周|下个月|明年|曾经|以前",
    re.I,
)
_ROLE = re.compile(
    (
        r"^I(?: am|'m)\s+(?:(?:currently|still|now)\s+)?(?:enrolled as\s+)?(?:a|an)\s"
        r"+([A-Za-z][A-Za-z -]{0,50}?)\s+(at|of|in|with|for)\s+([^,;.!?\n]+)[.!?]?$"
    ),
    re.I,
)
_VERB = re.compile(
    (
        r"^I\s+(?:(?:am\s+)?(?:currently|still|now)\s+)?(work|study|volunteer|belong|live|"
        r"teach|coach|train|enrolled)\s+(at|for|in|to|with)\s+([^,;.!?\n]+)[.!?]?$"
    ),
    re.I,
)
_CONTAINER = re.compile(
    (
        r"\b(?:My|The same)\s+(?P<container>[A-Za-z0-9][A-Za-z0-9 '-]{0,70}?)\s*(?:,\s"
        r"*(?:which|that)\s+)?(?:(?:currently|still|now)\s+)?(?:has|contains)\s"
        r"+(?P<items>[^.!?\n]+)[.!?]?$"
    ),
    re.I,
)
_NUMBERS = dict(
    zip(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        ],
        range(21),
        strict=True,
    )
)
_ITEM = re.compile(r"(\d+|a|an|" + "|".join(_NUMBERS) + r")\s+([A-Za-z][A-Za-z '-]*)", re.I)
_NAMED_ITEM = re.compile(r"my ([A-Za-z][A-Za-z '-]*),\s*([A-Z][A-Za-z'-]*)")
_GROUP_UNITS = re.compile(r"^(?:school|shoal|group|pack|bag|gallons?)\b", re.I)
_FROM = re.compile(r"^From:\s*(.+)$", re.M)
_DATE = re.compile(r"^Date:\s*(\d{4}-\d{2}-\d{2})\b", re.M)
_SUBJECT = re.compile(r"^Subject:\s*(.+)$", re.M)
#: 邮件正文里的会议主题（"our upcoming task meeting to discuss X" / "a meeting about X" /
#: "agenda … focus on X"）。组 2 是主题宾语；**只收同一句里的明确说法**。
_MEETING_TOPIC = re.compile(
    (
        r"\b(?:meeting|session|agenda|discussion)s?\b[^.!?\n]{0,80}?"
        r"\b(?:to discuss|to review|about|regarding|on|covering|focus(?:ing)? on|"
        r"review(?:ing)?)\s+([A-Za-z0-9][^.!?\n]{2,160})"
    ),
    re.I,
)
#: 会议主题的**形状守卫**：主题是名词短语，不是从句。代词/连词/助动词开头的一律不要
#: （实测 `what we'll be covering`、`making sure we're all connected …` 都是从句碎片）。
_TOPIC_REJECT = re.compile(
    r"^(?:what|which|who|how|why|when|where|whether|that|this|these|those|"
    r"making|being|getting|doing|having|to\s|we|you|they|i|it|our|your|their|"
    r"the\s+\w+ing)\b",
    re.I,
)
_SCHEDULED = re.compile(r"^Scheduled for:\s*(\d{4}-\d{2}-\d{2})\b", re.M | re.I)
#: Calendar invite 的**标注字段**（CorporateBench / 真实日历邮件里都长这样）：
#: `Organizer: X (mail)` 与 `Attendees:` 后面的一串 `- 人名 (mail)`。字段是原文写明的，
#: 不需要任何推断 ⇒ 与署名块同一口径（§2.1 允许的"原文明说的独立事实"）。
_ORGANIZER = re.compile(r"(?m)^Organizer:\s*([^(\n]+?)\s*(?:\([^)]*\))?\s*$")
_ATTENDEES = re.compile(r"(?m)^Attendees:\s*\n((?:[ \t]*-[^\n]*\n?)+)")
_ATTENDEE_LINE = re.compile(r"(?m)^[ \t]*-[ \t]*([^(\n]+?)[ \t]*(?:\([^)]*\))?[ \t]*$")
_QUOTED_MAIL = re.compile(r"(?mi)^(?:On .+wrote:\s*|>.*|-+\s*Original Message\s*-+)\s*$")
#: 自我介绍（"My name is X" / "I'm X" / "I am X, and …"）——把后面的第一人称绑定到 X。
#: ⚠ 只认**明确的名字**（首字母大写、无动词），避免把 "I am a doctor" 当成名字。
_SELF_NAME = re.compile(
    r"\b(?:My name is|I(?:'m| am)) ([A-Z][A-Za-z'’-]+(?: [A-Z][A-Za-z'’-]+){0,3})"
    r"(?=\s*[,.;!?]|\s+and\b|\s*$)",
)
#: `I am <状态词>` **不是**自我介绍——`I am Stable and Normal`（HaluMem 的心理健康句）
#: 曾被当成名字 `Stable`，进而把同一 Add 里所有第一人称事实的主语都改成了它
#: （2026-10-18 实测：`monthly income` 的主语变成 `Stable`）。这里只拦**状态/形容
#: 词**这一小类：真正的名字不会出现在这张表里，而状态词几乎只以首字母大写出现在
#: 句首或 `I am` 之后。
_NOT_A_NAME = frozenset(
    {
        "stable", "normal", "fine", "good", "great", "okay", "ok", "well", "happy",
        "sad", "tired", "busy", "ready", "sure", "glad", "sorry", "afraid",
        "excited", "worried", "confident", "optimistic", "determined", "grateful",
        "single", "married", "divorced", "employed", "unemployed", "retired",
        "healthy", "sick", "ill", "free", "here", "there", "back", "done",
    }
)
#: 出生日期（"I was born on 1996-08-02" / "born on 2 August 1996"）。**日期原样保留**，
#: 不换算格式——答案侧要按原写法比对。
_BIRTH_DATE = re.compile(
    r"\bborn on (\d{4}-\d{2}-\d{2}|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}"
    r"|[A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})",
    re.I,
)
_JOB_WORDS = re.compile(
    (
        r"\b(?:chief|CEO|manager|specialist|engineer|officer|director|analyst|consultant|"
        r"scientist|developer|associate|president|architect|intern|employee)\b"
    ),
    re.I,
)
_ORG_WORDS = re.compile(
    (
        r"\b(?:labs?|inc|corp(?:oration)?|ltd|LLC|LLP|company|technologies|research|university|"
        r"foundation|institute)\b"
    ),
    re.I,
)
_COMMENCED = re.compile(
    r"(?<![\"'“‘\w])\bI(?:'ve| have)\s+(?:commenced|begun|started)\s+my\s+role\s+at\s+([^,;.!\n]+)",
    re.I,
)
_ORIENTATION = re.compile(
    (
        r"(?:^|(?<=[.!?]))\s*(?:I(?:'m| am|'ve been| have been)\s+)?going through "
        r"([^,;.!?\n]+?)['’]s orientation schedule\b"
    ),
    re.I,
)
#: 本人自述的**标注字段**（`My highest education level is PhD` / `my personality type
#: is INTJ` / `my major is Law`）。⚠ **字段名必须在白名单里**：泛化的 `my X is Y`
#: 会把叙述句（"my journey is values like empathy"）也变成事实——那是推断，不是原文
#: 字段。白名单只收**问句里能问、原文里真写**的档案字段（HaluMem 那批 + 通用档案）。
#: 规范化交给 `relation_key`（与查询侧 `relation_key(字段名)` 对齐）。
_PROFILE_FIELDS = frozenset(
    {
        "highest education level",
        "education level",
        "major",
        "personality type",
        "mbti personality type",
        "monthly income",
        "job title",
        "blood type",
        "physical condition",
        "mental health",
        "middle name",
        "life goal",
        "life goal statement",
        "weight",
        "height",
    }
)
#: 自我介绍写下的名字本身也是**原文写明的字段**（`My name is X` ⇒ `X` 的 `name` 是 `X`）。
#: 它是跨 Add 绑定第一人称的唯一可信来源——不抽它，后面那些块就没有可查的姓名事实。
#: 本人明说的偏好（`contemporary is my top pick` / `my favorite style is X`）。
#: 两条形状都**必须是原文写明的**：前者是"X 是我的首选"，后者是"我的最爱是 X"。
#: 本人明说的偏好。两种形状，各自一条正则（**不合并**：合并后分组编号会错位，
#: 且 `, but X is my top pick` 里的连词会被当成宾语的一部分——2026-10-18 实测踩到）。
#: 只收原文写明的"X 是我的首选 / 我的最爱是 X"，不做推断。
_PREFERENCE_FIRST = re.compile(
    r"(?:^|[,;.]\s*(?:but|and|so)?\s*)([A-Za-z][A-Za-z0-9 '&/-]{1,40}?) is my "
    r"(?:top pick|favourite|favorite|number one|go-to)\b",
    re.I,
)
_PREFERENCE_MINE = re.compile(
    r"\bmy (?:favourite|favorite) ([A-Za-z][A-Za-z0-9 '&/-]{1,60}?) is "
    r"([A-Za-z][A-Za-z0-9 '&/-]{1,60})",
    re.I,
)
_SELF_INTRO = re.compile(
    r"\bMy name is ([A-Z][A-Za-z'’-]+(?: [A-Z][A-Za-z'’-]+){0,3})"
    r"(?=\s*[,.;!?]|\s+and\b|\s*$)"
)
_PROFILE_FIELD = re.compile(
    r"\bmy ([A-Za-z][A-Za-z '-]{2,40}?) (?:is|was) ([A-Za-z0-9][^.,;!?\n]{0,60})",
    re.I,
)
_JSON_TRIPLE = re.compile(r"\{.*\}", re.S)


def _json_triple(text: str) -> tuple[str, str, str] | None:
    """`{"subject": …, "relation": …, "object": …}` ⇒ 三个字段（原文逐字给出）。

    官方采集的 MQuAKE 语料用这种写法；`UPDATE: …` 前缀已由调用方剥掉。
    ⚠ 只用 `json.loads`，不猜字段：三个键**必须都存在且是字符串**，否则返回 `None`
    交回原检索（与所有其它抽取分支同一口径——解析不出就不产出事实）。
    """
    match = _JSON_TRIPLE.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match[0])
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    subject, relation, obj = (payload.get(name) for name in ("subject", "relation", "object"))
    if not isinstance(subject, str) or not isinstance(relation, str) or not isinstance(obj, str):
        return None
    if not (subject.strip() and relation.strip() and obj.strip()):
        return None
    return subject.strip(), relation.strip(), obj.strip()


_LABEL = re.compile(r"^[\w -]{1,40}:\s*")
_REPLACEMENT = re.compile(r"\s*\(this replaces the earlier value\)\.?$", re.I)
#: 官方采集的更新句前缀（MQuAKE）：`UPDATE: replace the prior value of this subject and
#: relation with the following fact.` 后面跟一行 JSON 三元组。语义与末尾那句
#: `(this replaces the earlier value)` 相同——都是**原文明写的替换**。
_UPDATE_PREFIX = re.compile(
    r"^\s*UPDATE:\s*replace the prior value[^\n]*\n?", re.I
)
MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec),\s*\d{1,4}"
_INTERVAL = re.compile(rf"(.+?) from ({MONTH}) to ({MONTH})\.?", re.I)

_ZH_MODIFIER = r"(?:(?:目前|现在|仍然|仍|现在仍然)\s*)?"
_ZH_ROLE = re.compile(
    rf"^我{_ZH_MODIFIER}是(?:一[名个位])?([^,，;；。！？\n]+)的([^的,，;；。！？\n]{{1,24}})$"
)
_ZH_VERB = re.compile(
    rf"^我{_ZH_MODIFIER}在([^,，;；。！？\n]+?)(工作|上班|任职|就读|念书|学习|"
    r"做志愿者|当志愿者|任教)$"
)
_ZH_CONTAINER = re.compile(
    rf"^我的(?P<container>[^,，;；。！？\n]{{1,90}}?)(?:里|中|内)?{_ZH_MODIFIER}"
    r"(?:有|装着|包含|放着)(?P<items>[^。！？\n]+)[。！？]?$"
)
_ZH_ITEM = re.compile(
    r"(?P<amount>\d+|[零〇一二两三四五六七八九十百千万]+)\s*"
    r"(?P<unit>[个本支把件台张只条瓶枚块根顶])\s*(?P<object>[^,，、;；。！？\n]+)"
)

_MODEL_EXTRACTOR_UNSET = object()
_MODEL_EXTRACTOR: EntityExtractor | None | object = _MODEL_EXTRACTOR_UNSET


def _default_entity_extractor() -> EntityExtractor | None:
    """Load the optional English NER sidecar once; rules remain the fallback."""
    global _MODEL_EXTRACTOR
    if _MODEL_EXTRACTOR is _MODEL_EXTRACTOR_UNSET:
        try:
            _MODEL_EXTRACTOR = load_spacy_extractor("en_core_web_sm")
        except (ImportError, RuntimeError, OSError):
            _MODEL_EXTRACTOR = None
    if _MODEL_EXTRACTOR is _MODEL_EXTRACTOR_UNSET:
        return None
    return cast(EntityExtractor | None, _MODEL_EXTRACTOR)


def _model_fact_candidates(
    *,
    source: str,
    extractor: EntityExtractor,
) -> tuple[tuple[str, str, str, str], ...]:
    """Return only conservative model relation candidates with exact quotes."""
    candidates: list[tuple[str, str, str, str]] = []
    result: ExtractionCandidates = extractor.extract(source)
    for relation in result.relations:
        if relation.source not in {"surface", "structured-rule"}:
            continue
        if relation.object is None or not relation.sentence.strip():
            continue
        sentence = relation.sentence.strip()
        if (
            interrogative(sentence)
            or _UNSAFE.search(sentence)
            or _PLANNED.search(sentence)
            or sentence not in source
        ):
            continue
        subject = relation.subject.text.strip()
        object_ = relation.object.text.strip()
        # ``Corpus:`` is a dataset marker, not part of the knowledge entity.
        corpus_subject = re.match(
            r"(?is)^Corpus:\s*(.+?)\s+(?:is|was|were|has|have)\s+", sentence
        )
        if corpus_subject:
            subject = corpus_subject[1].strip()
        else:
            subject = re.sub(r"^Corpus:\s*", "", subject, flags=re.I)
        if key(subject) in {"let", "the", "a", "an", "this", "that"}:
            continue
        candidates.append(
            (
                subject,
                relation_key(relation.relation),
                object_,
                sentence,
            )
        )
    return tuple(candidates)


def _model_sources(side: SourceSide, source: str | None) -> tuple[str, ...]:
    """Keep the sidecar inside the same speaker boundary as rule extraction.

    Answer blocks without a named speaker are generated assistant content and
    must not become durable facts. Explicit non-assistant speakers remain
    eligible, matching the existing deterministic extraction path.
    """
    if not source:
        return ()
    if side == "question":
        return (source,)
    fragments: list[str] = []
    for _, utterance in named_utterances(source):
        speaker = SPEAKER.fullmatch(utterance.strip())
        if not speaker:
            continue
        if key(speaker["actor"]) in {"assistant", "system", "助手", "系统"}:
            continue
        fragments.append(speaker["body"])
    return tuple(fragments)


def chinese_quantity(text: str) -> int | None:
    """只接受确切整数；拒绝两三、十几、一百二等模糊或省略写法。"""
    text = key(text)
    if text.isdigit():
        value = int(text)
        return value if 0 <= value <= 10000 else None
    digits = {word: i for i, word in enumerate("零一二三四五六七八九")}
    normalized = text.replace("两", "二").replace("〇", "零")
    if normalized == "一万":
        return 10000
    value, digit = 0, 0
    for word in normalized:
        if word in digits:
            digit = digits[word]
        elif word in "十百千":
            value += (digit or 1) * {"十": 10, "百": 100, "千": 1000}[word]
            digit = 0
        else:
            return None
    value += digit
    if not 0 <= value <= 9999:
        return None
    if value == 0:
        canonical = "零"
    else:
        remaining, parts, needs_zero = value, [], False
        for base, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
            amount, remaining = divmod(remaining, base)
            if amount:
                if needs_zero:
                    parts.append("零")
                parts.extend(("零一二三四五六七八九"[amount], unit))
                needs_zero = False
            elif parts and remaining:
                needs_zero = True
        canonical = "".join(parts)
        if canonical.startswith("一十"):
            canonical = canonical[1:]
    alternatives = {canonical, canonical.replace("零一十", "零十")}
    if 10 <= value <= 19:
        alternatives.add("一" + canonical)
    return value if normalized in alternatives else None


def _chinese_role(sentence: str, actor: str) -> tuple[str, str, str, str] | None:
    text = sentence.strip().rstrip("。！？")
    if actor == "I":
        # 您 / 咱们 等不归"我"；漏掉它们会让 "您觉得这是正常的" 被拆成
        # 主体="您觉得这"、关系="正常"这样的假事实。
        if text.startswith(("我", "你", "您", "他", "她", "它", "咱", "俺", "大家", "我们")):
            return None
        named = re.fullmatch(rf"({NAME})((?:目前|现在|仍然)?(?:是|在).+)", text)
        if not named:
            return None
        actor, text = named[1], "我" + named[2]
    role = _ZH_ROLE.fullmatch(text)
    verb = _ZH_VERB.fullmatch(text)
    if role:
        org, surface = role[1].strip(), role[2].strip()
    elif verb:
        org, surface = verb[1].strip(), verb[2]
    else:
        return None
    if re.search(r"和|与|或|以及|、", surface):
        return None
    statement = f"{actor}是{org}的{surface}。" if role else f"{actor}在{org}{surface}。"
    return actor, relation_key(surface), org, statement


# 同一关系的名词/动词表面形式。任意显式三元组的关系仍直接来自原文。
PREFIX_FORMS = ("author", "capital", "official language", "head of state", "head of government")
INFIX_FORMS = (
    ("country of citizenship", "is a citizen of"),
    ("creator", "was created by"),
    ("place of birth", "was born in the city of"),
    ("spouse", "is married to"),
    ("sport", "is associated with the sport of"),
    ("country of origin", "was created in the country of"),
    ("founded by", "was founded by"),
    ("continent", "is located in the continent of"),
    ("position played on team / speciality", "plays the position of"),
    ("employee", "works for"),
    ("employee", "is employed by"),
    ("religion or worldview", "is affiliated with the religion of"),
    ("notable work", "is known for"),
    ("team", "plays for"),
    ("position", "holds the position of"),
    ("head coach", "is the head coach of"),
    ("head of government", "is the head of the government of"),
    ("chair", "is the chair of"),
    ("political party", "is a member of the"),
)

#: 显式更新句（`… (this replaces the earlier value)`）里出现、但上面的前后缀表没覆盖的
#: 来源句法。**只在更新句上尝试**（普通陈述仍走原有守卫），每条都是「主语在组 1、宾语在
#: 组 2」的整句匹配。这些写法不是 MQuAKE 专属：更新流量里常见的 `The name of the
#: current head of state in X is Y` 与 `X speaks the language of Y` 都在内。
#: ⚠ `univeristy` / `origianl` 是**数据源里的拼写**，不是笔误——照抄才能匹配上。
_UPDATE_FORMS = (
    (
        r"The name of the current head of state (?:in|of) (.+?) is (.+)",
        "head of state",
    ),
    (r"The name of the current head of the (.+?) government is (.+)", "head of government"),
    (r"The name of the current head of the government of (.+?) is (.+)", "head of government"),
    (r"(.+?) speaks the language of (.+)", "languages spoken, written or signed"),
    (r"(.+?) died in the city of (.+)", "place of death"),
    (r"The chairperson of (.+?) is (.+)", "chairperson"),
    (r"The univer\w* where (.+?) was educated is (.+)", "educated at"),
    (r"The director of (.+?) is (.+)", "director / manager"),
    (r"The company that produced (.+?) is (.+)", "manufacturer"),
    (r"The origi[a-z]*l broadcaster of (.+?) is (.+)", "original broadcaster"),
    (r"The headquarters of (.+?) is located in the city of (.+)", "headquarters location"),
    (r"(.+?) was founded in the city of (.+)", "location of formation"),
    (r"The type of music that (.+?) plays is (.+)", "genre"),
    (r"(.+?)'s child is (.+)", "child"),
    (r"(.+?) was developed by (.+)", "developer"),
    (r"(.+?) was performed by (.+)", "performer"),
    (r"The chief executive officer of (.+?) is (.+)", "chief executive officer"),
    (r"(.+?) works in the field of (.+)", "occupation"),
)

#: `The <职务> is <人名>` 形式的在任者更新（`The United Nations Secretary-General is …`）。
#: 职务词表与 [`paths.py`](./paths.py) 的 `officeholder` 同一口径——泛化的 `The X is Y`
#: 会把 "The capital of … is …" 这类也吞掉，所以只认明确职务。
_OFFICEHOLDER_FORM = re.compile(r"The (.+?) is (.+)", re.I)
_OFFICE_TITLE = re.compile(
    r"\b(?:secretary[- ]general|president|governor|chairperson|ceo|chief .+ officer|"
    r"prime minister|monarch|king|queen|mayor|director-general)$",
    re.I,
)


def month_index(value: str) -> int | None:
    match = re.fullmatch(r"([A-Za-z]{3}),\s*(\d{1,4})", value.strip())
    months = {name.casefold(): i for i, name in enumerate(calendar.month_abbr) if name}
    if not match or match[1].casefold() not in months or not 1 <= int(match[2]) <= 9999:
        return None
    return int(match[2]) * 12 + months[match[1].casefold()] - 1


def bind_first_person(sentence: str, actor: str) -> str:
    for pattern, replacement in (
        (r"\bI'm\b", f"{actor} is"),
        (r"\bI've\b", f"{actor} has"),
        (r"\bI am\b", f"{actor} is"),
        (r"\bI have\b", f"{actor} has"),
        (r"\bmy\b", f"{actor}'s"),
        (r"\bI\b", actor),
    ):
        sentence = re.sub(pattern, replacement, sentence, flags=re.I)
    return sentence


def _organization(line: str) -> str | None:
    if re.match(r"^(?:no|not|none|unknown|n/a)\b", line, re.I) or re.search(
        r"\d|@|https?://|INSERT_", line, re.I
    ):
        return None
    return next(
        (
            part.strip()
            for part in re.split(r"[|,]", line)
            if _ORG_WORDS.search(part) and not _JOB_WORDS.search(part)
        ),
        None,
    )


def extract_evidence(
    *,
    parent_memory_id: str,
    user_id: str,
    question: str | None,
    answer: str | None,
    event_time: int | None,
    entity_extractor: EntityExtractor | None = None,
) -> tuple[EvidenceFact, ...]:
    facts: list[EvidenceFact] = []

    def emit(
        subject: str,
        relation: str,
        obj: str,
        quote: str,
        statement: str,
        *,
        side: SourceSide = "question",
        date: str | None = None,
        attributes: dict[str, str | int | bool] | None = None,
        record_fields: dict[str, str] | None = None,
        **attrs: str | int | bool,
    ) -> None:
        source = question if side == "question" else answer
        if quote and quote in (source or ""):
            facts.append(
                make_fact(
                    parent_memory_id=parent_memory_id,
                    user_id=user_id,
                    subject=subject,
                    relation=relation,
                    object=obj,
                    source_quote=quote,
                    statement=statement,
                    source_side=side,
                    event_time=event_time,
                    source_date=date,
                    position=f"{len(facts)}:{relation}:{obj}",
                    qualifiers={**(attributes or {}), **attrs},
                    record_fields=record_fields,
                )
            )

    # 显式边、显式月份限定或显式替换。只用于取回原文。
    # ⚠ `"\n" not in question` 曾经把**多行记录**整条挡在外面；官方采集的 MQuAKE
    # 更新句就是 `UPDATE: …\n{json}` 两行（2026-10-18 实测：1124 题 0 命中）。
    # 放宽到"首行是标签/短前缀 + 余下是一条显式声明"，长度上限照旧。
    if question and not answer and len(question) <= 2048:
        text = _LABEL.sub("", question.strip(), count=1)
        edited = bool(_REPLACEMENT.search(text))
        body = _REPLACEMENT.sub("", text).strip() if edited else text
        # 官方的更新句前缀（`UPDATE: replace the prior value …`）与末尾那句
        # `(this replaces the earlier value)` 是同一件事——**更新语义由原文写明**。
        update_prefix = _UPDATE_PREFIX.match(body)
        if update_prefix:
            edited = True
            body = body[update_prefix.end():].strip()
        interval = _INTERVAL.fullmatch(body)
        attrs: dict[str, str | int | bool] = {"projection": "source"}
        if interval:
            start, end = month_index(interval[2]), month_index(interval[3])
            if start is not None and end is not None and start <= end:
                attrs.update(start_month=start, end_month=end)
                body = interval[1]
            else:
                body = ""
        if edited:
            attrs["replaces_earlier"] = True
            attrs["observation"] = True
        fields = None
        if not interval and not edited:
            pieces = body.split(" — ")
            if len(pieces) == 3:
                fields = tuple(pieces)
        # **JSON 三元组**（官方采集的 MQuAKE 语料就是这种）：
        # `{"subject": "...", "relation": "...", "object": "..."}`。与 `A — r — B`
        # 是同一件事的另一种写法——**原文逐字给出的三个字段**，不是推断。
        # 额外字段（`subject_id` / `relation_id`）忽略：它们是 Wikidata id，不是事实内容。
        if fields is None:
            fields = _json_triple(body)
        if fields is None:
            for rel in PREFIX_FORMS:
                match = re.fullmatch(r"The " + re.escape(rel) + r" of (.+?) is (.+)", body, re.I)
                if match:
                    fields = (match[1], rel, match[2])
                    break
            if fields is None:
                for rel, phrase in INFIX_FORMS:
                    if phrase == "is employed by" and (
                        _UNSAFE.search(body) or _PLANNED.search(body)
                    ):
                        continue
                    match = re.fullmatch(r"(.+?)\s+" + re.escape(phrase) + r"\s+(.+)", body, re.I)
                    if match:
                        fields = (match[1], rel, match[2])
                        break
            if fields is None and edited:
                # 更新句的额外句法：主语在组 1、宾语在组 2。否定/计划/引用整句仍拒绝
                # （下面那道统一守卫照旧跑，这里只补"能认出来的写法"）。
                for pattern, rel in _UPDATE_FORMS:
                    match = re.fullmatch(pattern, body, re.I)
                    if match:
                        fields = (match[1], rel, match[2])
                        break
                if fields is None:
                    match = _OFFICEHOLDER_FORM.fullmatch(body)
                    if match and _OFFICE_TITLE.search(match[1]):
                        fields = (match[1], "officeholder", match[2])
        if (
            fields
            and all(part.strip() for part in fields)
            and not re.search(r'["“”]', fields[0] + fields[2])
            and not re.match(r"(?:not|unknown|none|no)\b", fields[2], re.I)
        ):
            emit(
                fields[0].strip(),
                fields[1].strip(),
                fields[2].strip(),
                question.strip(),
                question.strip(),
                attributes=attrs,
            )
        elif edited:
            emit(
                parent_memory_id,
                "explicit replacement",
                text,
                question.strip(),
                question.strip(),
                projection="source",
                unparsed=True,
            )

    # 元数据和署名：主体绑定自己的 From，拒绝收件人或引用邮件的署名。
    if question and not answer:
        own = _QUOTED_MAIL.split(question, maxsplit=1)[0]
        own = re.sub(r"^[\w -]{1,40}:\s*(?=(?:From|Date|Subject|Message-ID):)", "", own, count=1)
        dated, title = _DATE.search(own), _SUBJECT.search(own)
        date = dated[1] if dated else ""
        if dated and title:
            scheduled = _SCHEDULED.search(own)
            emit(
                parent_memory_id,
                "document title",
                title[1].strip(),
                title[0],
                title[0],
                date=date,
                scope_date=scheduled[1] if scheduled else date,
                projection="source",
                # 与查询侧同一个归一化：标题按 `key()` 保留标点身份
                # （`title_key` 会抹掉 `-`、`<>`、`1:1`，那些标题将永远匹配不上）。
                match_object=key(title[1]),
            )
            # **会议讨论主题**（2026-10-17）：正文里的 "our upcoming task meeting to
            # discuss X" / "meeting about X" / "agenda … focus on X" 是**原文明说的主题**，
            # 与 Subject 是两回事（`What meetings are linked to X?` 问的正是它）。
            # ⚠ 只认**明确的讨论动词 + 同一句的宾语**，不做推断；计划/愿望句不产生事实。
            body_text = own[title.end() :]
            for match in _MEETING_TOPIC.finditer(body_text):
                topic = match[1].strip().rstrip(".")
                # 名词短语守卫：从句碎片（`what we'll be covering`）不是主题。
                if not topic or len(topic) > 160 or _TOPIC_REJECT.search(topic):
                    continue
                # 只截掉**从句**（"X that we discussed"），不截并列名词（"risk and issues"）。
                topic = re.split(
                    r"\s+(?:that|which|who)\s+(?:we|you|they|i|he|she|it|is|was|are|were|"
                    r"will|would|can|could)\b.*$",
                    topic,
                )[0].strip()
                if not topic or len(topic) < 3:
                    continue
                sentence = match[0].strip()
                if _UNSAFE.search(sentence) or _PLANNED.search(sentence):
                    continue
                emit(
                    parent_memory_id,
                    "meeting topic",
                    topic,
                    sentence,
                    sentence,
                    date=date,
                    scope_date=scheduled[1] if scheduled else date,
                    projection="source",
                    match_object=key(topic),
                )
        # **Calendar invite 的出席者 / 组织者**（2026-10-17）：标注字段，逐字可核。
        # 主体是会议名（与 `document title` 同一字面），客体是人名——
        # `Who organized X?` / `Which employees were present at X?` 靠这两类事实。
        if title:
            meeting_name = title[1].strip()
            organizer = _ORGANIZER.search(own)
            if organizer and organizer[1].strip():
                emit(
                    organizer[1].strip(),
                    "organizer",
                    meeting_name,
                    organizer[0].strip(),
                    organizer[0].strip(),
                    date=date,
                    scope_date=scheduled[1] if scheduled else date,
                    projection="source",
                )
            block = _ATTENDEES.search(own)
            if block:
                for line in _ATTENDEE_LINE.finditer(block[1]):
                    person = line[1].strip()
                    if not person:
                        continue
                    emit(
                        person,
                        "attendee",
                        meeting_name,
                        line[0].rstrip("\n"),
                        line[0].rstrip("\n"),
                        date=date,
                        scope_date=scheduled[1] if scheduled else date,
                        projection="source",
                    )
        sender = _FROM.search(own)
        actor, address = parseaddr(sender[1]) if sender else ("", "")
        if actor.strip() and "@" in address:
            actor = actor.strip()
            signatures = list(re.finditer(r"(?m)^" + re.escape(actor) + r"\s*$", own))
            found_signature = False
            if signatures:
                signature = signatures[-1]
                lines = [m for m in re.finditer(r"[^\n]+", own[signature.end() :]) if m[0].strip()]
                if lines and _JOB_WORDS.search(lines[0][0]):
                    for line in lines:
                        org = _organization(line[0])
                        if org:
                            quote = own[signature.start() : signature.end() + line.end()].strip()
                            emit(
                                actor,
                                "employee of",
                                org,
                                quote,
                                f"{actor} is an employee of {org}.",
                                date=date,
                            )
                            found_signature = True
                            break
                        if re.search(r"\d|@|https?://|INSERT_", line[0], re.I):
                            break
            body = own.split("\n\n", 1)[1] if "\n\n" in own else own
            for paragraph in re.split(r"\n\s*\n", body):
                if '"' in paragraph or "“" in paragraph or "”" in paragraph:
                    continue
                for rel, role_pattern in (
                    ("role commencement", _COMMENCED),
                    ("orientation schedule", _ORIENTATION),
                ):
                    for match in role_pattern.finditer(paragraph):
                        org = match[1].strip()
                        if not _organization(org):
                            continue
                        statement = (
                            f"{actor} reports having commenced their role at {org}."
                            if rel == "role commencement"
                            else f"{actor} reports going through {org}'s orientation schedule."
                        )
                        emit(
                            actor,
                            rel,
                            org,
                            paragraph.strip(),
                            statement,
                            date=date,
                            observation=True,
                            direct=rel == "role commencement",
                        )
                        if rel == "role commencement" and not found_signature:
                            emit(
                                actor,
                                "employee of",
                                org,
                                match[0],
                                f"{actor} is an employee of {org}.",
                                date=date,
                            )

    # 每个明确具名的发言独立处理；无姓名用户陈述保留第一人称所有权。
    sources: tuple[tuple[SourceSide, str | None], ...] = (
        ("question", question),
        ("answer", answer),
    )
    for side, source in sources:
        for _, utterance in named_utterances(source):
            speaker = SPEAKER.fullmatch(utterance.strip())
            if speaker:
                actor, body = speaker["actor"], speaker["body"]
                if key(actor) in {"assistant", "system", "助手", "系统"}:
                    continue
                if key(actor) in {"user", "用户"} and side == "question":
                    actor = "I"
            elif side == "question":
                actor, body = "I", utterance
            else:
                continue
            # **自我介绍把第一人称绑定到真名**（2026-10-17）：HaluMem / MemTrap 一类记录
            # 以 "My name is X, and I am …" 开头，后面所有 "I / my" 都属于 X。不绑定的话
            # 事实主语会停在字面 "I"，查询永远匹配不上。
            introduced = _SELF_NAME.search(body)
            if introduced and key(actor) in {"i", "user", "用户"}:
                actor = introduced[1].strip()
            safe_attributes = not (_UNSAFE.search(body) or _PLANNED.search(body))
            # **省略主语的后续句仍归该说话人**（2026-10-17）：英语对话里主语只在第一句
            # 出现（"Been hectic lately. Started hitting the gym last week."）——
            # 拆句后第二句没有 "I"，事件就丢了。判据：说话人**明确**（不是字面 "I"）
            # 且整条发言里出现过第一人称。
            for sentence in sentences(body):
                sentence = sentence.strip()
                # 只对**明确的过去时事件动词**开头补主语——"Been hectic" 这类状态句
                # 补上会造出 "I been hectic" 这种假事实，宁可不抽。
                # 说话人有名字时不需要先出现 "I"：对话里主语常常整段省略。
                # "Since I lost my job …" / "Unfortunately, I also lost …" 这类状语开头
                # 的句子同样带第一人称，事件就在从句里，交给下面的谓词表处理。
                if actor != "I" and re.match(
                    r"^(?:Started|Began|Took|Went|Launched|Opened|Designed|Hosted|Visited|"
                    r"Attended|Lost|Got|Received|Applied|Painted|Drew|Interviewed|"
                    r"Participated)\b",
                    sentence,
                ):
                    sentence = f"I {sentence[0].lower()}{sentence[1:]}"
                # **第一人称复数也归说话人**（2026-10-19）：对话里 "We adopted another
                # puppy called Pixie" 的主语是说话人这一方；不绑定就整条事件丢失
                # （实测 LoCoMo 的 "When did Audrey adopt Pixie?" 因此 `no_match`）。
                # ⚠ 只改**明确的过去时事件句**，且说话人有名字——状态句、计划句照旧不抽。
                if actor != "I" and re.match(
                    r"^We\s+(?:adopted|bought|got|met|moved|left|finished|completed|"
                    r"created|shared|won|sold|started|began|took|went|launched|opened|"
                    r"hosted|visited|attended|received|applied|painted|drew|interviewed|"
                    r"lost|participated|traveled|travelled|returned|arrived|discovered|"
                    r"decided|collaborated|celebrated|performed|competed|published|"
                    r"recorded|stopped|quit|retired|married|graduated|found|learned|taught)\b",
                    sentence,
                ):
                    sentence = f"I {sentence[3:]}"
                # 疑问句不是事实：MedMemoryBench 的 "您觉得这是正常的吗？" 曾被拆成
                # 主体="您觉得这"、关系="吗"、客体="正常"。
                # （全库对拍：净删 11 条假事实、零附带删除。）
                if interrogative(sentence) or _UNSAFE.search(sentence) or _PLANNED.search(sentence):
                    # ⚠ 计划/否定/引用会作废整句，但**事件可能在前半句已经说完了**：
                    # "Lost my job as a banker yesterday, so I'm gonna take a shot at …"
                    # 的后半句是计划，前半句是已发生的过去时陈述。只保留**不含标记的
                    # 前半句**（它自己就是完整陈述），其余事实照旧整句不产出。
                    head = (
                        ""
                        if interrogative(sentence)
                        else re.split(r",\s*(?:so|but)\b|;\s*", sentence, maxsplit=1)[0]
                    )
                    if (
                        not head
                        or head == sentence
                        or _UNSAFE.search(head)
                        or _PLANNED.search(head)
                    ):
                        continue
                    sentence = head
                sale = sale_fields(sentence) if side == "question" and actor == "I" else None
                if sale:
                    emit(
                        actor,
                        "sale",
                        sale["product"],
                        sentence,
                        sentence,
                        side=side,
                        record_fields=sale,
                        attributes={
                            "sold_quantity": sale["quantity"],
                            "container": sale["market"],
                            "event": True,
                            "observation": True,
                            **{k: v for k, v in sale.items() if k != "quantity"},
                        },
                    )
                chinese_role = _chinese_role(sentence, actor) if safe_attributes else None
                if chinese_role:
                    subject, relation, org, statement = chinese_role
                    emit(subject, relation, org, sentence, statement, side=side)
                role = (
                    (_ROLE.fullmatch(sentence) or _VERB.fullmatch(sentence))
                    if speaker and safe_attributes
                    else None
                )
                if role:
                    relation = relation_key(role[1])
                    prep = "of" if relation in {"employee", "member"} else role[2].lower()
                    emit(
                        actor,
                        f"{relation} {prep}",
                        role[3].strip(),
                        sentence,
                        f"{actor} is a {relation} {prep} {role[3].strip()}.",
                        side=side,
                    )
                # **"work as <职位> at <组织>" / "promoted to <职位> at <组织>"**
                # （HaluMem 的通用问法，2026-10-19）：职位与组织都是原文逐字给出的
                # 独立字段。⚠ 不收推断——只认这两种明确句式。
                worked_as: re.Match[str] | tuple[str, str, str] | None = re.fullmatch(
                    r"I(?:'m| am)? (?:currently |still |now )?work(?:ing)? as (?:a |an )?(.+?) "
                    r"(?:at|for|in) ([^,;.!?\n]+)[.!?]?",
                    sentence,
                    re.I,
                )
                if worked_as is None:
                    worked_as = re.fullmatch(
                        r"I(?:'m| am)? (?:currently |still |now )?work(?:ing)? (?:at|for) "
                        r"([^,;.!?\n]+?) as (?:a |an )?(.+?)[.!?]?",
                        sentence,
                        re.I,
                    )
                    if worked_as:
                        worked_as = (worked_as[0], worked_as[2], worked_as[1])
                if worked_as:
                    emit(
                        actor,
                        "job title",
                        worked_as[1].strip(),
                        sentence,
                        f"{actor}'s job title is {worked_as[1].strip()}.",
                        side=side,
                    )
                    emit(
                        actor,
                        "employee of",
                        worked_as[2].strip(),
                        sentence,
                        f"{actor} is an employee of {worked_as[2].strip()}.",
                        side=side,
                    )
                promoted = re.fullmatch(
                    r"I(?:'ve| have)? (?:just |recently )?(?:been )?promoted to (?:a |an )?(.+?) "
                    r"(?:at|in|by) ([^,;.!?\n]+)[.!?]?",
                    sentence,
                    re.I,
                )
                if promoted:
                    emit(
                        actor,
                        "job title",
                        promoted[1].strip(),
                        sentence,
                        f"{actor}'s job title is {promoted[1].strip()}.",
                        side=side,
                    )
                birth = _BIRTH_DATE.search(sentence)
                if birth:
                    # 出生日期是**原文说过的独立事实**（主语 = 说话人），日期原样保留。
                    emit(
                        actor,
                        "date of birth",
                        birth[1],
                        sentence,
                        f"{actor}'s date of birth is {birth[1]}.",
                        side=side,
                    )
                # **标注字段**（2026-10-18）：`My highest education level is PhD` /
                # `my personality type is INTJ` / `my major is …` —— 与 `I was born on`
                # 同一口径（原文写明的字段就是事实，D46 的邻居）。只收**本人自述**
                # （说话人是 I / 具名发言）且值是短名词短语；计划/否定整句早已被上面挡掉。
                field = _PROFILE_FIELD.search(sentence) if actor != "I" else None
                if field is None and key(actor) in {"i", "user", "用户"}:
                    field = _PROFILE_FIELD.search(sentence)
                if field and key(field[1]) not in _PROFILE_FIELDS:
                    field = None
                if field:
                    emit(
                        actor,
                        relation_key(field[1]),
                        field[2].strip(),
                        sentence,
                        f"{actor}'s {field[1].strip()} is {field[2].strip()}.",
                        side=side,
                    )
                # **自我介绍本身**（`My name is X`）——名字是后续 Add 绑定第一人称的
                # 唯一可信来源，所以它也要落成事实（2026-10-18）。
                intro = _SELF_INTRO.search(sentence)
                if intro:
                    emit(
                        intro[1].strip(),
                        "name",
                        intro[1].strip(),
                        sentence,
                        f"{intro[1].strip()}'s name is {intro[1].strip()}.",
                        side=side,
                    )
                # **本人明说的偏好**（`contemporary is my top pick` /
                # `my favorite style of dance is X`）。原文写明的偏好就是事实——
                # 与档案字段同一口径，不做推断（LoCoMo 那批 "What is X's favorite …"）。
                # `X is my top pick` ⇒ 偏好是 X；`my favorite Y is X` ⇒ 偏好是 X。
                preference_match = _PREFERENCE_FIRST.search(sentence)
                if preference_match is None:
                    mine = _PREFERENCE_MINE.search(sentence)
                    preference = mine[2] if mine else None
                else:
                    preference = preference_match[1]
                if preference and preference.strip():
                    emit(
                        actor,
                        "favorite",
                        preference.strip(),
                        sentence,
                        f"{actor}'s favorite is {preference.strip()}.",
                        side=side,
                    )
                inventory = _CONTAINER.search(sentence)
                if inventory:
                    container, items = inventory["container"].strip(), inventory["items"].strip()
                    named = _NAMED_ITEM.fullmatch(items)
                    entries: list[tuple[str, str, int, str]] = []
                    if named:
                        entries.append((named[1], named[2], 1, "which has " + items))
                    else:
                        for part in re.split(r",\s*(?:and\s+)?|\s+and\s+", items, flags=re.I):
                            item = _ITEM.fullmatch(part.strip())
                            if not item:
                                entries = []
                                break
                            amount = item[1].casefold()
                            quantity = (
                                int(amount)
                                if amount.isdigit()
                                else 1
                                if amount in {"a", "an"}
                                else _NUMBERS[amount]
                            )
                            if not 0 <= quantity <= 10000 or _GROUP_UNITS.search(item[2]):
                                entries = []
                                break
                            entries.append((item[2].strip(), "", quantity, part.strip()))
                    for obj, name, amount, _quote in entries:
                        owner = "My" if actor == "I" else f"{actor}'s"
                        emit(
                            actor,
                            "contains",
                            obj,
                            inventory[0],
                            f"{owner} {container} contains {amount} {obj}"
                            + (f" named {name}" if name else "")
                            + ".",
                            side=side,
                            container=container,
                            quantity=amount,
                            name=name,
                            snapshot=True,
                        )
                chinese_inventory = _ZH_CONTAINER.fullmatch(sentence)
                if chinese_inventory:
                    container = chinese_inventory["container"].strip()
                    entries_chinese: list[tuple[str, int, str, str]] = []
                    for part in re.split(r"[、，,]|以及|和|及", chinese_inventory["items"]):
                        item = _ZH_ITEM.fullmatch(part.strip())
                        chinese_amount = chinese_quantity(item["amount"]) if item else None
                        if item is None or chinese_amount is None:
                            entries_chinese = []
                            break
                        entries_chinese.append(
                            (item["object"].strip(), chinese_amount, item["unit"], part.strip())
                        )
                    for obj, quantity, unit, item_quote in entries_chinese:
                        owner = "我" if actor == "I" else actor
                        emit(
                            actor,
                            "contains",
                            obj,
                            chinese_inventory[0],
                            f"{owner}的{container}包含{obj}，数量为{quantity}{unit}。",
                            side=side,
                            container=container,
                            quantity=quantity,
                            unit=unit,
                            item_quote=item_quote,
                            name="",
                            snapshot=True,
                        )
                if not speaker or actor == "I":
                    continue
                # 状态/已完成动作保留原文限定语，不生成事件日期。
                for match in re.finditer(
                    (
                        r"\bmy (?P<object>(?:[A-Za-z]+ ){0,4}[A-Za-z]+) is (?P<state>open|"
                        r"closed|ready|available)\b"
                    ),
                    sentence,
                    re.I,
                ):
                    emit(
                        actor,
                        "is " + match["state"].lower(),
                        match["object"],
                        body,
                        f"{actor}'s {match['object']} is {match['state'].lower()}.",
                        side=side,
                        observation=True,
                    )
                for match in re.finditer(
                    r"\bI (?:(?:just|recently) )?got accepted for (?:a |an )?([^.!?]+)",
                    sentence,
                    re.I,
                ):
                    emit(
                        actor,
                        "accepted for",
                        match[1].strip(),
                        body,
                        f"{actor} just got accepted for a {match[1].strip()}.",
                        side=side,
                        observation=True,
                    )
                completed = re.search(
                    (
                        r"\bI (?:(?:just|recently) )?(?:went to|attended|participated in|took "
                        r"part in|chose to go to) (.+?)(?: to |[,.!?]|$)"
                    ),
                    sentence,
                    re.I,
                )
                # **通用事件谓词**（2026-10-17）：上面那张表只覆盖"参加"，而对话里
                # 大量事件是 "started hitting the gym" / "took a short trip to Rome" /
                # "launch an ad campaign"。它们与 `participated in` 是同一件事：
                # **原文说过的、有来源日期的事件**。
                # ⚠ 两个边界：① `got accepted for` 已有专门分支，这里排除；
                # ② 事件句常以状语开头（"Started hitting the gym last week"），
                # 所以用**整条 utterance** 找第一人称，而不是拆句后的片段。
                if completed is None:
                    completed = re.search(
                        (
                            r"\bI(?:'ve| have)? (?:(?:just|recently|already|also|then|finally) )?"
                            r"(?:started|began|took|went on|hosted|launched|opened|designed|"
                            r"interviewed|received|attended|visited|hitting|"
                            r"applied|painted|drew|drew up|"
                            # 失去工作/机会也是带日期的事件（LoCoMo "Lost my job as a banker
                            # yesterday"），与上面那批同一口径。
                            r"lost|participated in|"
                            # 领养 / 收养（LoCoMo "We adopted another puppy called Pixie"）：
                            # 与查询侧 `When did X adopt Y` 同一件事，答案仍是来源日期。
                            r"adopted|adopt|"
                            # 其他对话里带来源日期的已完成动作（2026-10-19 官方语料实测）。
                            # ⚠ **`joined` 刻意不在表里**（既有测试钉着）：`I joined a
                            # church and a gym` 这类并列句不适合当单一事件。
                            r"bought|purchased|sold|met|moved|finished|completed|"
                            r"created|shared|won|decided|collaborated|celebrated|performed|"
                            r"competed|published|recorded|stopped|quit|retired|married|"
                            r"graduated|travelled|traveled|returned|arrived|discovered|"
                            r"learned|taught|found|signed up for|enrolled in) "
                            r"(.+?)(?:[,.!?]|$)"
                        ),
                        sentence,
                        re.I,
                    )
                networking = re.search(r"\bI(?:'ve| have) been networking\b", sentence, re.I)
                if completed or networking:
                    obj = completed[1].strip() if completed else "networking events"
                    emit(
                        actor,
                        "participated in",
                        obj,
                        body,
                        f"{actor} participated in {obj}.",
                        side=side,
                        observation=True,
                    )
                for match in re.finditer(
                    r"\bmy home country,\s*([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*)", sentence
                ):
                    emit(
                        actor,
                        "home country",
                        match[1],
                        match[0],
                        f"{actor}'s home country is {match[1]}.",
                        side=side,
                    )
                for match in re.finditer(
                    (
                        r"\bI(?:'ve| have) known (.+?) for (\d+ years?), since I moved from my "
                        r"home country\.?"
                    ),
                    sentence,
                    re.I,
                ):
                    emit(
                        actor,
                        "moved from",
                        "my home country",
                        match[0],
                        f"{actor} has known {match[1]} for {match[2]}, "
                        f"since moving from {actor}'s home country.",
                        side=side,
                        duration=match[2],
                        reference_relation="home country",
                        unique_reference=True,
                    )
                peer = re.search(
                    r"\bI(?: am|'m)(?: now)? friends with one of my fellow volunteers\b",
                    sentence,
                    re.I,
                )
                if peer:
                    emit(
                        actor,
                        "friends with",
                        "fellow volunteers",
                        body,
                        bind_first_person(peer[0], actor) + ".",
                        side=side,
                        reference_relation="volunteer",
                    )
                for match in re.finditer(
                    r"\bmy ([A-Za-z][A-Za-z '-]{0,60}?)\s+friends\b", sentence, re.I
                ):
                    emit(
                        actor,
                        "friends from",
                        match[1].strip(),
                        body,
                        f"{actor} has {match[1].strip()} friends.",
                        side=side,
                        match_object=key(
                            re.sub(r"^(?:a|an|the)\s+", "", match[1].strip(), flags=re.I)
                        ),
                    )
                for match in re.finditer(
                    (
                        r"\b(my|some|a few) friends from ([A-Za-z][A-Za-z '-]{0,70}?)(?= and "
                        r"I\b|[,.!?]|$)"
                    ),
                    sentence,
                    re.I,
                ):
                    if (
                        match[1].lower() == "my"
                        or re.search(r"\bI\b.*\bwith\b", sentence[: match.start()], re.I)
                        or re.search(r"\band I\b", sentence[match.end() :], re.I)
                    ):
                        emit(
                            actor,
                            "friends from",
                            match[2].strip(),
                            body,
                            bind_first_person(sentence, actor),
                            side=side,
                            direct=match[1].lower() == "my",
                            match_object=key(
                                re.sub(r"^(?:a|an|the)\s+", "", match[2].strip(), flags=re.I)
                            ),
                        )
                for place_pattern in (
                    r"\b(?:at|to) (?:the |a )?(.+?) I volunteer at\b",
                    r"(?:^|(?<=[.!?]))\s*I volunteer at (.+?)(?=[,.!?]|$)",
                ):
                    if not safe_attributes:
                        continue
                    for match in re.finditer(place_pattern, sentence, re.I):
                        obj = match[1].strip()
                        if place_pattern.startswith(r"\b(?:at|to)") and re.search(
                            r"\ba " + re.escape(obj), match[0], re.I
                        ):
                            obj = "a " + obj
                        emit(
                            actor,
                            "volunteer at",
                            obj,
                            body,
                            f"{actor} volunteers at {obj}.",
                            side=side,
                            match_object=key(re.sub(r"^(?:a|an|the)\s+", "", obj, flags=re.I)),
                        )
    if entity_extractor is not None:
        relation_surface = {
            "employee": "employee of",
            "student": "student at",
            "volunteer": "volunteer at",
            "lives in": "lives in",
            "spouse": "spouse",
            "creator": "creator",
            "place of birth": "place of birth",
            "team": "team",
        }
        for side, source in sources:
            for model_source in _model_sources(side, source):
                for subject, relation, object_, quote in _model_fact_candidates(
                    source=model_source, extractor=entity_extractor
                ):
                    surface_relation = relation_surface.get(relation, relation)
                    relation_conflict = any(
                        key(fact.subject) == key(subject)
                        and relation_key(fact.relation) == relation_key(surface_relation)
                        and key(fact.object) != key(object_)
                        and fact.source_side == side
                        and fact.source_quote == quote
                        and not fact.get("model_entity")
                        for fact in facts
                    )
                    if relation_conflict:
                        continue
                    duplicate = any(
                        key(fact.subject) == key(subject)
                        and relation_key(fact.relation) == relation_key(surface_relation)
                        and key(fact.object) == key(object_)
                        and fact.source_side == side
                        and (
                            fact.source_quote == quote
                            or fact.source_quote in model_source
                            and quote in model_source
                            and (
                                fact.source_quote in quote
                                or quote in fact.source_quote
                            )
                        )
                        for fact in facts
                    )
                    if duplicate:
                        continue
                    emit(
                        subject,
                        surface_relation,
                        object_,
                        quote,
                        f"{subject} is {surface_relation} {object_}.",
                        side=side,
                        model_entity=True,
                    )
    unique: list[EvidenceFact] = []
    seen: set[tuple[str, str, str, str, SourceSide]] = set()
    for fact in facts:
        marker = (
            key(fact.subject),
            relation_key(fact.relation),
            key(fact.object),
            fact.source_quote,
            fact.source_side,
        )
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(fact)
    return tuple(unique)
