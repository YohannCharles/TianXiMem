"""正文里的相对时间 → **就地注解**绝对日期（**只加不改**）。

```text
last Tues               →  last Tues (July 18, 2023)
last weekend            →  last weekend (July 15 to July 16, 2023)
I've taught for 7 years →  I've taught for 7 years (since 2016)
```

## 为什么是"注解"而不是"替换"（2026-09-25）

两者都能让模型**不用做日历算术**（`t1-dated` 的 temporal 79 道里 27 道卡在这一步），但注解多三件事：

| | 替换 | **注解** |
| --- | --- | --- |
| 相对形式还在吗 | ❌ 没了 ⇒ 纯相对 gold 必错、BEAM（规则相反）反向 | ✅ 在，模型按 gold 选形式 |
| 与 **D21** 的关系 | 跨过它（"原文保留、日期是额外锚点"） | **就是 D21**，从段级挪到短语级 |
| 失败可逆吗 | 错日期顶替了原文，看不出来 | 原文在，错了能看出是哪条推错 |

代价只是正文变长——实测 346 题**平均 +1.1%**（中位 1.1%、最大 1.4%）。

## 锚点 = 该段自己的会话日期

`SearchHit.created_at`（段级）——与 `DATE_PREFIX` 那套**同一口径**。

## 一条硬纪律：**推不出就不动**

`a few years ago` / `several weeks ago` 这类量词**没有数字**——换算它等于编一个日期，
而编错的日期比不换算更糟：**模型会自信地照抄**，而裁判是精确比值的。
⇒ 查不到数字的一律原样保留（`_quantifier` 返回 `None`）。

## ⚠ 这是**原型**，不是生产实现（S1 那条代理假设的又一次使用）

生产形态要落在 `src/tianximem/common/render.py`（`render_pair` 的入口 + 一个 `annotate` 开关），
而且**只在 `content` 侧调用、索引侧一个字不改**——理由与实测（`per-pair` 0.627 vs
`t1-dated` 0.633，差在噪声内 ⇒ 日期进向量**没有贡献**）见
[`../reports/ledger.md`](../reports/ledger.md)。
搬过去时要一并改不变式 **I1 的表述**：从"embedding 输入 == content"改成
"**索引侧渲染是唯一真源，content = 索引文本 + 一个纯注解函数**"。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, timedelta

__all__ = ["MARKS", "annotate"]

#: 注解的两种记号——**同一个换算逻辑，只换写法**。
#:
#: * `paren`：`last Tues (July 18, 2023)`——读起来最自然，但**可能被模型整段抄进答案**，
#:   而裁判对"半相对半绝对"的混合形态**没有明文规则**（TIME 块只写了不许相对↔绝对互转）
#: * `tag`：`last Tues [= 2023-07-18]`——更像"附注"、不像句子的一部分，抄进答案的概率更低
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

#: 量词的候选写法——**长的必须排在短的前面**（`a few` 要在 `a` 之前，否则只吃到 `a`）。
_QUANT = r"a few|several|a couple of|couple of|" + "|".join(_NUMBER_WORDS)
_COUNT = rf"({_QUANT}|\d+)"

#: 一条规则 = `(正则, 由命中算出注解正文的函数)`；函数返回 `None` ⇒ **不动**。
Rule = tuple[re.Pattern[str], Callable[[re.Match[str], date], "str | None"]]


def _day(day: date) -> str:
    # ⚠ **不要写 `%-d`**：那是 glibc 的扩展，Windows 的 `strftime` 直接抛
    # `ValueError: Invalid format string`——而本函数在**每次 Search** 上都会跑
    # （`packaging.annotate_relatives` 默认 true）⇒ 那等于"Windows 上服务不能用"。
    # `{day.day}` 是不带前导零的十进制，与 `%-d` 输出**逐字节相同**。
    return f"{day:%B} {day.day}, {day.year}"


def _range(start: date, end: date) -> str:
    """`July 15 to July 16, 2023`——两端都在同月时省掉前半的年月。"""
    if (start.year, start.month) == (end.year, end.month):
        return f"{start:%B} {start.day} to {end.day}, {end.year}"
    if start.year == end.year:
        return f"{start:%B} {start.day} to {end:%B} {end.day}, {end.year}"
    return f"{start:%B} {start.day}, {start.year} to {end:%B} {end.day}, {end.year}"


def _month(year: int, month: int) -> str:
    return date(year, month, 1).strftime("%B %Y")


def _quantifier(raw: str) -> int | None:
    """`"7"` / `"seven"` → `7`；**查不到返回 `None`**（`a few` 这类没有数字）。"""
    if raw.isdigit():
        return int(raw)
    return _NUMBER_WORDS.get(raw.lower())


def _ago_rule(match: re.Match[str], anchor: date) -> str | None:
    count = _quantifier(match.group(1))
    if count is None:
        return None  # 「a few years ago」——没有数字，**不猜**
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
    """`for 7 years` → 起点（`since 2016`）。**这是"多久"而不是"多久以前"**。"""
    count = _quantifier(match.group(1))
    if count is None:
        return None
    unit = match.group(2).lower()
    if unit == "year":
        return f"since {anchor.year - count}"
    if unit == "month":
        year, month = _shift_month(anchor, count)
        return f"since {_month(year, month)}"
    return None  # 天/周粒度当起点对问题没意义（LoCoMo 没有这类题），**不碰**


def _weekday_rule(match: re.Match[str], anchor: date) -> str | None:
    """`last Tues` = 锚点**之前**最近的那个周二（当天不算——"last" 是往回的）。"""
    back = (anchor.weekday() - _WEEKDAYS[match.group(1).lower()]) % 7 or 7
    return _day(anchor - timedelta(days=back))


def _saturday_of(anchor: date) -> date:
    """锚点**所在那一周**的周六。周的起点是**周一**（见 `_week_rule` 的核对）。"""
    monday = anchor - timedelta(days=anchor.weekday())
    return monday + timedelta(days=5)


def _weekend_rule(match: re.Match[str], anchor: date) -> str | None:
    saturday = _saturday_of(anchor) - timedelta(days=7)
    return _range(saturday, saturday + timedelta(days=1))


def _week_rule(match: re.Match[str], anchor: date) -> str | None:
    """`last week` = 锚点所在周**之前那一整周**（周一→周日）。

    **口径是核出来的，不是猜的**——用三条 gold 的并列形式反推：

    | 会话日期 | gold 的相对说法 | gold 的并列绝对形式 |
    | --- | --- | --- |
    | 2023-06-09（周五） | `The week before 9 June 2023` | `From May 29, 2023 to June 4, 2023` |
    | 2023-07-06（周四） | `The week before 6 July 2023` | `From June 26, 2023 to July 2, 2023` |
    | 2023-07-17 | `The weekend before 17 July 2023` | `From July 15 to 16, 2023` |

    三条都指向**周一为界**（按周日为界三条全部差一天）。
    """
    monday = anchor - timedelta(days=anchor.weekday()) - timedelta(days=7)
    return _range(monday, monday + timedelta(days=6))


def _shift_month(anchor: date, months: int) -> tuple[int, int]:
    """往前推 `months` 个月——返回 `(年, 月)`。"""
    year, month = anchor.year, anchor.month - months
    while month < 1:
        year, month = year - 1, month + 12
    return year, month


def _month_rule(match: re.Match[str], anchor: date) -> str | None:
    year, month = _shift_month(anchor, 1)
    return _month(year, month)


def _year_rule(match: re.Match[str], anchor: date) -> str | None:
    return str(anchor.year - 1)


def _fixed(days: int) -> Callable[[re.Match[str], date], str | None]:
    """固定偏移的那些（`yesterday` / `today` / `last night`…）——只差几天。"""
    return lambda match, anchor: _day(anchor + timedelta(days=days))


#: **顺序有意义**：先具体后笼统，先长后短。`last weekend` 必须在 `last week` 之前
#: （虽然后者因 `\b` 不会误吃，但把更具体的规则放前面更不容易出错）。
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
    """把 `text` 里的相对时间**就地**注解成以 `anchor` 为基准的绝对日期。**只加不改。**

    `anchor` 是这段文本自己的会话日期（`SearchHit.created_at`）。
    推不出来的（`a few years ago`）原样保留——见模块 docstring 的硬纪律。
    """
    if mark not in MARKS:
        raise ValueError(f"未知的注解记号 {mark!r}——只有 {' / '.join(MARKS)}")

    out = text
    for pattern, build in _RULES:

        def replace(match: re.Match[str], build: Callable = build) -> str:
            body = build(match, anchor)
            if body is None:
                return match.group(0)
            # 两种记号**只差外壳**（换算结果一字不改）：圆括号读起来最自然，
            # 方括号更像"附注"、更不容易被整段抄进答案。
            return (
                f"{match.group(0)} ({body})" if mark == "paren" else f"{match.group(0)} [= {body}]"
            )

        out = pattern.sub(replace, out)
    return out
