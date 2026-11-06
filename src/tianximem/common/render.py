from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Final

TEMPLATE_VERSION: str = "v1"

QUESTION_PREFIX: str = "Q: "
ANSWER_PREFIX: str = "A: "

QUESTION_ANSWER_SEP: str = "\n"

SEGMENT_SEP: str = "\n"


_DATE_FORMAT: Final[str] = "%Y-%m-%d"

DATE_PREFIX: Final[str] = "[{date}] "


def event_day(event_time: int | None) -> date | None:
    if event_time is None:
        return None
    return datetime.fromtimestamp(event_time / 1000, tz=UTC).date()


def day_granularity(event_time: int | None) -> str:
    day = event_day(event_time)
    return "" if day is None else day.strftime(_DATE_FORMAT)


def render_date(event_time: int | None, *, inject_abs_time: bool) -> str:
    return day_granularity(event_time) if inject_abs_time else ""


def template_version(*, inject_abs_time: bool) -> str:
    return f"{TEMPLATE_VERSION}+abs_time" if inject_abs_time else TEMPLATE_VERSION


def render(question: str | None, answer: str | None, *, date: str = "") -> str:
    lines: list[str] = []
    if question:
        lines.append(f"{QUESTION_PREFIX}{question}")
    if answer:
        lines.append(f"{ANSWER_PREFIX}{answer}")
    text = QUESTION_ANSWER_SEP.join(lines).strip()
    if not text or not date:
        return text
    return DATE_PREFIX.format(date=date) + text


def render_pair(pair: object, *, inject_abs_time: bool = False) -> str:
    return render(
        getattr(pair, "question", None),
        getattr(pair, "answer", None),
        date=render_date(getattr(pair, "event_time", None), inject_abs_time=inject_abs_time),
    )


def render_evidence(
    *, statement: str, source_date: str, source_quote: str, quantitative: bool = False
) -> str:
    lines = ["Memory fact", f"Statement: {statement}", f"Source record date: {source_date}"]
    if not quantitative:
        lines.append(f"Source quotation: {source_quote}")
    return "\n".join(lines)


def render_record_evidence(
    *, source_id: str, source_date: str, source_quote: str, fields: Mapping[str, str]
) -> str:
    return json.dumps(
        {
            "source_id": source_id,
            "source_record_date": source_date,
            "user_quotation": source_quote,
            **fields,
        },
        ensure_ascii=False,
    )


def render_segment(pair_texts: Sequence[str]) -> str:
    return SEGMENT_SEP.join(text for text in pair_texts if text).strip()


__all__ = [
    "ANSWER_PREFIX",
    "DATE_PREFIX",
    "QUESTION_ANSWER_SEP",
    "QUESTION_PREFIX",
    "SEGMENT_SEP",
    "TEMPLATE_VERSION",
    "day_granularity",
    "event_day",
    "render",
    "render_date",
    "render_pair",
    "render_segment",
    "template_version",
]
