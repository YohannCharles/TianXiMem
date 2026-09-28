"""Schema 落差预处理层（§12.3 第 9 条）——**归一化成 AML 契约形状的唯一实现**。

> **边界（D16）**：本模块**只知道 AML 的形状**（`role` / `content` / 可选 `timestamp`），
> 不知道任何数据集文件名。文件名的知识止步于 [`locomo.py`](./locomo.py) /
> [`longmemeval.py`](./longmemeval.py)。**两个加载器都产出这里定义的形状**，
> 于是 `pairing/` 与 `store/` 只需面对一种形状。

## 两条不可协商的形状规则

1. **`content` 首尾无空白**——AML 只做 `\"\\n\".join(...)` 拼接、**不插分隔符**，
   所以首尾空白会直接进模型读到的正文
   （[`../../../docs/contract.md`](../../../docs/contract.md) §4）。
2. **`role` 必须存在**——§6.2 的配对判据只依赖一个字段（"这条是不是 `user`"）。
   缺 `role` 时 `pairing/` **不会报错**，它会把整个 session 归成一个对
   （[`../datasets/CLAUDE.md`](./CLAUDE.md) 的"两个数据集 turn schema 不一样"）。

## 时间：一律合成到消息级，但**没有 session 内区分度**

两个数据集都**没有** per-turn 时间戳（LoCoMo 是 session 级、LongMemEval 在
`haystack_dates` 里）⇒ 同一 session 内所有消息拿到**同一个时间**
⇒ `event_time` 在 session 内必然没有区分度，**位置才是唯一能保证
±1 邻域稳定的东西**（§6.1；D25 之后位置 = `(chunk_ordinal, local_index)`，
邻域用读时稠密序 `seq` 现算）。这不是近似，是数据集的真实属性。

**时区口径**：两份数据的时间串都是无时区的本地时间。这里按 **UTC** 解释
（不做任何偏移）。理由：§11.3 只让 `created_at` 发到**日粒度**，±时区偏移最坏
把日期挪一天，影响面远小于"自己发明一个时区"。这是**已定口径，不是待验证项**——
真要改，改这一处即可，全链路只有这一个解释点。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

__all__ = [
    "Message",
    "Sample",
    "Session",
    "Question",
    "parse_locomo_time",
    "parse_lme_time",
    "normalize_content",
    "to_epoch_ms",
]

Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    """一条消息——**Add 载荷的最小单位**（§2.1）。"""

    role: str
    content: str
    timestamp_ms: int | None = None

    def to_add_payload(self) -> dict:
        """转成 §2.1 的 `messages` 项。`timestamp` 缺省时**不出现**该键（不是 `null`）。"""
        payload: dict = {"role": self.role, "content": self.content}
        if self.timestamp_ms is not None:
            payload["timestamp"] = self.timestamp_ms
        return payload


@dataclass(frozen=True, slots=True)
class Session:
    """一个 session——**Add 的一次投喂范围**（§6.5 的切批只在 session 内发生）。"""

    session_id: str
    messages: tuple[Message, ...]


@dataclass(frozen=True, slots=True)
class Question:
    """一道题——**Search 的输入与裁判的输入**。

    `gold` 保留原始形态（LoCoMo 是 `list[str]`，LongMemEval 是 `str`）：
    归档 pipeline 的 `gold_answer()` 走 `memory_text()`，**列表会被它自己用 `\\n` 拼**——
    在这里预先拼会改变那条链路的行为。
    """

    qid: str
    question: str
    gold: object
    category: str
    #: **数据集自带的证据标记，只用于诊断，不参与评分**——LoCoMo 是 turn 级 `dia_id`
    #: （`"D1:3"`），LongMemEval 是 session 级 id。两者不可比，故不统一成一种语义。
    #: §13 的 T2 要人工判"没召回 / 召回但被截断 / 在里面但排序靠后"，靠的是它。
    evidence: tuple[str, ...] = ()
    #: LongMemEval 的拒答题标记（`question_id` 以 `_abs` 结尾，全库 30 道）。
    is_abstention: bool = False


@dataclass(frozen=True, slots=True)
class Sample:
    """**一个 `user_id` 的全部内容**——一次 Add 序列 + 跑在它上面的一组题。

    > ⚠ **两份数据集的基数不一样，这是真实差异不是实现细节**：
    > - **LoCoMo-Refined**：`N 题 : 1 段对话`（`conv-26` 一段对话上有 138 题）
    > - **LongMemEval**：`1 题 : 1 个 haystack`（每题自带一套 session，`user_id` 按题派生）
    >
    > 所以 `user_id` **由加载器决定**，harness 不得自己拼。
    """

    user_id: str
    dataset: str
    sessions: tuple[Session, ...]
    questions: tuple[Question, ...]
    #: 对话里两个说话人的显示名（LoCoMo 有真名）——注入 `speaker_*_name` 用。
    speaker_names: tuple[str, str] = ("speaker 1", "speaker 2")

    @property
    def message_count(self) -> int:
        return sum(len(s.messages) for s in self.sessions)


# ── 时间解析 ───────────────────────────────────────────────────────────────
# 两份数据各只有一种格式（在全量上枚举过，见 tests）。**不做宽松匹配**：
# 解析失败就抛，别让它静默退化成 None（那会让 event_time 全 NULL 且不报错）。

_LOCOMO_TIME = re.compile(
    r"^(?P<h>\d{1,2}):(?P<m>\d{2})\s*(?P<ap>am|pm)\s+on\s+(?P<d>\d{1,2})\s+(?P<mon>[A-Za-z]+),\s*(?P<y>\d{4})$"
)
_LME_TIME = re.compile(
    r"^(?P<y>\d{4})/(?P<mon>\d{2})/(?P<d>\d{2})\s*\([A-Za-z]{3}\)\s*(?P<h>\d{2}):(?P<m>\d{2})$"
)

_LOCOMO_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def parse_locomo_time(value: str) -> datetime:
    """`'1:56 pm on 8 May, 2023'` → naive `datetime`（UTC 口径，见模块 docstring）。"""
    m = _LOCOMO_TIME.match(value.strip())
    if not m:
        raise ValueError(f"LoCoMo 时间串无法解析：{value!r}")
    hour = int(m["h"]) % 12
    if m["ap"] == "pm":
        hour += 12
    month = _LOCOMO_MONTHS.get(m["mon"].lower())
    if month is None:
        raise ValueError(f"LoCoMo 时间串月份无法解析：{value!r}")
    return datetime(int(m["y"]), month, int(m["d"]), hour, int(m["m"]), tzinfo=UTC)


def parse_lme_time(value: str) -> datetime:
    """`'2023/05/20 (Sat) 02:21'` → naive `datetime`。

    ⚠ 格式里的 `(Sat)` 星期几**故意不校验**——它与日期不总是一致（官方数据如此），
    校验它只会把能跑的数据变成跑不了的。
    """
    m = _LME_TIME.match(value.strip())
    if not m:
        raise ValueError(f"LongMemEval 时间串无法解析：{value!r}")
    return datetime(int(m["y"]), int(m["mon"]), int(m["d"]), int(m["h"]), int(m["m"]), tzinfo=UTC)


def to_epoch_ms(moment: datetime) -> int:
    """§2.1 的 `timestamp` 是 **Unix 毫秒**（不是秒）。"""
    return int(moment.timestamp() * 1000)


# ── 正文归一化 ─────────────────────────────────────────────────────────────


def normalize_content(raw: str, *, where: str) -> str:
    """`strip()` 后校验非空——**AML 侧不报错，所以必须在这里报**。

    `where` 只用于把出错位置说清楚（数据集 + turn 定位）。
    """
    text = raw.strip()
    if not text:
        raise ValueError(f"{where}：content 为空（AML 只做 join、不插分隔符，空正文会污染拼接）")
    return text
