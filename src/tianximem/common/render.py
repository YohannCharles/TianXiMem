"""QA 文本和 grounded fact 文本的唯一渲染实现。

embedding 输入与返回给 AML 的 `content` 必须使用同一份基础渲染，否则检索命中的文本
与模型读到的文本会静默漂移。本模块不依赖业务层，只接收调用方提供的 QA 字段和日期。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Final

# ── 模板版本 ────────────────────────────────────────────────────────────
# 改模板必须同时改这个版本号。它会进入 embedding 缓存的坐标系；模板变化会改变
# embedding 输入并要求重建向量索引，版本号可防止不兼容缓存被静默命中。
TEMPLATE_VERSION: str = "v1"

QUESTION_PREFIX: str = "Q: "
ANSWER_PREFIX: str = "A: "

# Q 与 A 之间使用单个换行；修改模板分隔符也必须更新 TEMPLATE_VERSION。
QUESTION_ANSWER_SEP: str = "\n"

#: 一个 segment 内各 QA 对之间的分隔。每一对以 `Q:` / `A:` 自定界，
#: 因此单个换行足够，并避免为额外空行消耗 token。
SEGMENT_SEP: str = "\n"


# ── 时间 ───────────────────────────────────────────────────────────────
#: 日期格式：`YYYY-MM-DD`。不要使用裸 Unix 毫秒；`content` 前缀与响应中的
#: `created_at` 必须使用同一个日期格式。
_DATE_FORMAT: Final[str] = "%Y-%m-%d"

#: 启用绝对日期注入时加在正文前的日期前缀。
DATE_PREFIX: Final[str] = "[{date}] "


def event_day(event_time: int | None) -> date | None:
    """Unix 毫秒 → `date`（**UTC 口径**）；`None` → `None`。

    **时间换算只此一处**：`day_granularity`（要字符串）与
    [`annotate`](./annotate.py)（要 `date` 做日历算术）都从这里走。
    两条各写一份换算，早晚会在时区或取整上分叉——而**分叉不报错，只会差一天**。
    """
    if event_time is None:
        return None
    return datetime.fromtimestamp(event_time / 1000, tz=UTC).date()


def day_granularity(event_time: int | None) -> str:
    """Unix 毫秒 → `YYYY-MM-DD`（**UTC 口径**）；`None` → **空串 `""`**。

    响应里的 `created_at` 和可选的 `content` 日期前缀共用这一实现。

    * `event_time` 是该对首条消息的 timestamp，可为空
    * 空值渲染为 `""`，保证响应字段始终存在
    * 不使用 Add 到达时间兜底：那是写入时间，不是事件时间
    * 只输出日粒度；时区固定为 UTC
    * UTC 口径必须与加载 `event_time` 时一致，否则日期可能静默偏移一天
    """
    day = event_day(event_time)
    return "" if day is None else day.strftime(_DATE_FORMAT)


def render_date(event_time: int | None, *, inject_abs_time: bool) -> str:
    """正文前缀里的日期串。

    * `inject_abs_time=False` ⇒ `""`：正文不带绝对日期
    * `True` ⇒ 日粒度日期，如 `2023-10-22`

    `created_at` 使用相同的日粒度。前缀不包含星期或秒级时间；`preflight` 固定该形状。
    """
    return day_granularity(event_time) if inject_abs_time else ""


def template_version(*, inject_abs_time: bool) -> str:
    """缓存坐标里的模板标识（`embed/base.EmbeddingCoordinate.render_template`）。

    改模板必须更新版本号。绝对日期注入会改变正文，因此是否注入日期必须产生不同的缓存
    坐标，防止不同渲染变体静默复用向量。
    """
    return f"{TEMPLATE_VERSION}+abs_time" if inject_abs_time else TEMPLATE_VERSION


def render(question: str | None, answer: str | None, *, date: str = "") -> str:
    """把一个 QA 对渲染成最终文本。**这是唯一一处拼装。**

    渲染规则见 [`../rank/CLAUDE.md`](../rank/CLAUDE.md)：

    | 输入 | 输出 |
    | --- | --- |
    | 都有 | `Q: {question}\\nA: {answer}` |
    | `question` 为空 | 只输出 `A:` 那一行 |
    | `answer` 为空 | 只输出 `Q:` 那一行 |

    返回值首尾无空白，保证 AML 用换行拼接多个 `content` 时不会粘连。

    默认不注入时间戳；是否注入由 `inject_abs_time` 控制。时间筛选由 `event_time` 负责，
    正文默认保留原始表述。`date` 非空时在正文最前面加 `[date] `。索引、精排输入和
    `content` 必须使用同一套日期口径。

    ⚠ 空文本**不加前缀**：一个只有日期的 `content` 会让模型读到一条不存在的记忆。
    """
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
    """**一个 QA 对（鸭子类型）→ 它的渲染文本**——索引侧与精排输入的**唯一入口**。

    它存在的理由是那几个调用点（索引 / 精排输入 / content）必须用**同一条日期口径**：
    只要有一处取错（比如拿 Add 的到达时间兜底），"命中的是什么"与"模型读到的是什么"
    就分叉了——**而分叉不报错**（不变式 I1）。

    `content` 路径由 `rank/neighbor.py` 使用 `render()` 和 `render_date()` 拼出，以便接入
    相对时间注解；规则见 [`../rank/CLAUDE.md`](../rank/CLAUDE.md)。

    只要求对象有 `question` / `answer` / `event_time` 三个属性
    （[`../store/qdrant_store.py`](../store/qdrant_store.py) 的 `index_pairs` 本来就是这个立场：
    "像 QA 对一样可读"，不要求是哪一类对象）。
    """
    return render(
        getattr(pair, "question", None),
        getattr(pair, "answer", None),
        date=render_date(getattr(pair, "event_time", None), inject_abs_time=inject_abs_time),
    )


def render_evidence(
    *, statement: str, source_date: str, source_quote: str, quantitative: bool = False
) -> str:
    """来源支持的一条事实。数量不在正文重复，完整引句仍存于共同事实索引。"""
    lines = ["Memory fact", f"Statement: {statement}", f"Source record date: {source_date}"]
    if not quantitative:
        lines.append(f"Source quotation: {source_quote}")
    return "\n".join(lines)


def render_record_evidence(
    *, source_id: str, source_date: str, source_quote: str, fields: Mapping[str, str]
) -> str:
    """Literal observation fields plus the unchanged quotation; no arithmetic."""
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
    """把若干**已渲染**的 QA 对拼成一个 Context Segment 的 `content`。

    ⚠ **它必须是拼串的唯一去处**：`rank/` 不许自己 `"\\n".join(...)`——理由与 `render`
    一样：两处拼法一旦不同，"模型读到的"就与"能被复现的"漂移，而**漂移不报错**。

    * 空输入 ⇒ **空串**（调用方不该产出空段，但空串是唯一安全的降级）
    * 结果**首尾无空白**、内部不留空行（见 `SEGMENT_SEP`）
    * **不改写任何一对的内容**——它们已经是 `render()` 的产物
    """
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
