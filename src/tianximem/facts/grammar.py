"""来源句法 → 共同事实。语义词只绑定字段，不拥有表、开关或检索路径。"""

from __future__ import annotations

import calendar
import re
from email.utils import parseaddr

from tianximem.facts.conversation import named_utterances
from tianximem.facts.evidence import (
    EvidenceFact,
    SourceSide,
    key,
    make_fact,
    relation_key,
    title_key,
)

NAME = r"[A-Z][A-Za-z'-]*(?: [A-Z][A-Za-z'-]*){0,3}"
_SPEAKER = re.compile(rf"^(?:\[assistant\]\s*)?(?P<actor>{NAME}):\s*(?P<body>.+)$", re.S)
_UNSAFE = re.compile(
    r'["“”>]|\b(?:not|never|no|if|would|might|fictional|imagining|said|says|quoted)\b', re.I
)
_PLANNED = re.compile(r"\b(?:will|hope|plan|want|gonna|going to)\b", re.I)
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
_SCHEDULED = re.compile(r"^Scheduled for:\s*(\d{4}-\d{2}-\d{2})\b", re.M | re.I)
_QUOTED_MAIL = re.compile(r"(?mi)^(?:On .+wrote:\s*|>.*|-+\s*Original Message\s*-+)\s*$")
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
_LABEL = re.compile(r"^[\w -]{1,40}:\s*")
_REPLACEMENT = re.compile(r"\s*\(this replaces the earlier value\)\.?$", re.I)
MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec),\s*\d{1,4}"
_INTERVAL = re.compile(rf"(.+?) from ({MONTH}) to ({MONTH})\.?", re.I)

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
    ("religion or worldview", "is affiliated with the religion of"),
    ("notable work", "is known for"),
    ("team", "plays for"),
    ("position", "holds the position of"),
    ("head coach", "is the head coach of"),
    ("head of government", "is the head of the government of"),
    ("chair", "is the chair of"),
    ("political party", "is a member of the"),
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
                )
            )

    # 显式边、显式月份限定或显式替换。只用于取回原文。
    if question and not answer and "\n" not in question and len(question) <= 2048:
        text = _LABEL.sub("", question.strip(), count=1)
        edited = bool(_REPLACEMENT.search(text))
        body = _REPLACEMENT.sub("", text).strip() if edited else text
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
        fields = None
        if not interval and not edited:
            pieces = body.split(" — ")
            if len(pieces) == 3:
                fields = tuple(pieces)
        else:
            for rel in PREFIX_FORMS:
                match = re.fullmatch(r"The " + re.escape(rel) + r" of (.+?) is (.+)", body, re.I)
                if match:
                    fields = (match[1], rel, match[2])
                    break
            if fields is None:
                for rel, phrase in INFIX_FORMS:
                    match = re.fullmatch(r"(.+?)\s+" + re.escape(phrase) + r"\s+(.+)", body, re.I)
                    if match:
                        fields = (match[1], rel, match[2])
                        break
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
                match_object=title_key(title[1]),
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
                for rel, pattern in (
                    ("role commencement", _COMMENCED),
                    ("orientation schedule", _ORIENTATION),
                ):
                    for match in pattern.finditer(paragraph):
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
            speaker = _SPEAKER.fullmatch(utterance.strip())
            if speaker:
                actor, body = speaker["actor"], speaker["body"]
                if actor == "Assistant":
                    continue
                if actor == "User" and side == "question":
                    actor = "I"
            elif side == "question":
                actor, body = "I", utterance
            else:
                continue
            safe_attributes = not (_UNSAFE.search(body) or _PLANNED.search(body))
            for sentence in re.split(r"(?<=[.!?])\s+", body):
                if _UNSAFE.search(sentence) or _PLANNED.search(sentence):
                    continue
                role = (
                    (_ROLE.fullmatch(sentence) or _VERB.fullmatch(sentence))
                    if speaker and actor != "I" and safe_attributes
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
    return tuple(facts)
