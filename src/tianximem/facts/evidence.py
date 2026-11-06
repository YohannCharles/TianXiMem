from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from tianximem.common.render import day_granularity, render_evidence, render_record_evidence

EVIDENCE_VERSION = "memory-facts-v17-supersession"
Qualifier = str | int | bool
SourceSide = Literal["question", "answer"]


def key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split()).strip(" ?.!。！？")


FIRST_PERSON = frozenset({"i", "user", "用户", "我"})


_TRAILING_PREPOSITIONS = frozenset({"of", "at", "for", "in", "to", "with", "by"})


def relation_key(value: str) -> str:

    words = key(value).split()
    while len(words) > 1 and words[-1] in _TRAILING_PREPOSITIONS:
        words.pop()
    value = " ".join(words)
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

    @property
    def supersedes(self) -> bool:
        value = self.get("supersedes", False)
        if isinstance(value, str):
            return value.strip().casefold() in {"true", "yes", "1"}
        return bool(value)


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
