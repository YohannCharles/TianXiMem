from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, timedelta

__all__ = ["MARKS", "annotate"]

MARKS: tuple[str, ...] = ("paren", "tag")

_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_WEEKDAYS = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "wed": 2,
    "weds": 2,
    "wednesday": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}

_QUANT = r"a few|several|a couple of|couple of|" + "|".join(_NUMBER_WORDS)
_COUNT = rf"({_QUANT}|\d+)"

Rule = tuple[re.Pattern[str], Callable[[re.Match[str], date], "str | None"]]


def _day(day: date) -> str:
    return f"{day:%B} {day.day}, {day.year}"


def _range(start: date, end: date) -> str:

    if (start.year, start.month) == (end.year, end.month):
        return f"{start:%B} {start.day} to {end.day}, {end.year}"
    if start.year == end.year:
        return f"{start:%B} {start.day} to {end:%B} {end.day}, {end.year}"
    return f"{start:%B} {start.day}, {start.year} to {end:%B} {end.day}, {end.year}"


def _month(year: int, month: int) -> str:
    return date(year, month, 1).strftime("%B %Y")


def _quantifier(raw: str) -> int | None:

    if raw.isdigit():
        return int(raw)
    return _NUMBER_WORDS.get(raw.lower())


def _shift_month(anchor: date, months: int) -> tuple[int, int]:

    year, month = anchor.year, anchor.month - months
    while month < 1:
        year, month = year - 1, month + 12
    return year, month


def _saturday_of(anchor: date) -> date:

    monday = anchor - timedelta(days=anchor.weekday())
    return monday + timedelta(days=5)


def _ago_rule(match: re.Match[str], anchor: date) -> str | None:
    count = _quantifier(match.group(1))
    if count is None:
        return None
    unit = match.group(2).lower()
    if unit == "day":
        return _day(anchor - timedelta(days=count))
    if unit == "week":
        return _day(anchor - timedelta(days=7 * count))
    if unit == "weekend":
        saturday = _saturday_of(anchor) - timedelta(days=7 * count)
        return _range(saturday, saturday + timedelta(days=1))
    if unit == "month":
        year, month = _shift_month(anchor, count)
        return _month(year, month)
    return str(anchor.year - count)


def _duration_rule(match: re.Match[str], anchor: date) -> str | None:

    count = _quantifier(match.group(1))
    if count is None:
        return None
    unit = match.group(2).lower()
    if unit == "year":
        return f"since {anchor.year - count}"
    if unit == "month":
        year, month = _shift_month(anchor, count)
        return f"since {_month(year, month)}"
    return None


def _weekday_rule(match: re.Match[str], anchor: date) -> str | None:

    back = (anchor.weekday() - _WEEKDAYS[match.group(1).lower()]) % 7 or 7
    return _day(anchor - timedelta(days=back))


def _weekend_rule(match: re.Match[str], anchor: date) -> str | None:
    saturday = _saturday_of(anchor) - timedelta(days=7)
    return _range(saturday, saturday + timedelta(days=1))


def _week_rule(match: re.Match[str], anchor: date) -> str | None:

    monday = anchor - timedelta(days=anchor.weekday()) - timedelta(days=7)
    return _range(monday, monday + timedelta(days=6))


def _month_rule(match: re.Match[str], anchor: date) -> str | None:
    year, month = _shift_month(anchor, 1)
    return _month(year, month)


def _year_rule(match: re.Match[str], anchor: date) -> str | None:
    return str(anchor.year - 1)


def _fixed(days: int) -> Callable[[re.Match[str], date], str | None]:

    return lambda match, anchor: _day(anchor + timedelta(days=days))


_RULES: tuple[Rule, ...] = (
    (re.compile(r"\byesterday\b", re.I), _fixed(-1)),
    (re.compile(r"\b(?:last|yesterday) night\b", re.I), _fixed(-1)),
    (re.compile(r"\bthis morning\b", re.I), _fixed(0)),
    (re.compile(r"\btoday\b", re.I), _fixed(0)),
    (re.compile(r"\btonight\b", re.I), _fixed(0)),
    (re.compile(r"\bthe other day\b", re.I), _fixed(-2)),
    (re.compile(r"\btomorrow\b", re.I), _fixed(1)),
    (re.compile(r"\b(?:last|this past|past)\s+weekend\b", re.I), _weekend_rule),
    (re.compile(r"\b(?:last|this past|past)\s+week\b", re.I), _week_rule),
    (re.compile(r"\b(?:last|past|earlier this)\s+month\b", re.I), _month_rule),
    (re.compile(r"\b(?:last|past|earlier this)\s+year\b", re.I), _year_rule),
    (re.compile(rf"\b{_COUNT}\s+(day|week|weekend|month|year)s?\s+ago\b", re.I), _ago_rule),
    (re.compile(rf"\b(?:for|since)\s+{_COUNT}\s+(day|week|month|year)s?\b", re.I), _duration_rule),
    (re.compile(rf"\b(?:last|this past|past)\s+({'|'.join(_WEEKDAYS)})\b", re.I), _weekday_rule),
)


def annotate(text: str, anchor: date, *, mark: str = "paren") -> str:

    if mark not in MARKS:
        raise ValueError(f"未知的注解记号 {mark!r}——只有 {' / '.join(MARKS)}")

    out = text
    for pattern, build in _RULES:

        def replace(match: re.Match[str], build: Callable = build) -> str:
            body = build(match, anchor)
            if body is None:
                return match.group(0)
            return (
                f"{match.group(0)} ({body})" if mark == "paren" else f"{match.group(0)} [= {body}]"
            )

        out = pattern.sub(replace, out)
    return out
