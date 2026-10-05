"""渲染 —— **唯一实现**（不变式 I1）。

> §7.2 与 §11.3 要求：embedding 的输入 与 返回给 AML 的 `content`，必须是**同一份渲染**。

两处一旦不一致，"检索命中的是什么"与"模型读到的是什么"就会**漂移**——**而且这种漂移不会报错**。
检索照常返回值，分数照常算，只是**命中的是另一个版本**。

**因此本模块是唯一拼 QA 对文本的地方**：[`../embed/`](../embed/) 不自己拼字符串，
[`../rank/`](../rank/) 的 packaging 也不自己拼。

**本模块不依赖任何业务层**，只是 `(question, answer) -> str` 的纯函数（附加一个可选的
`date`，见 §T1）——**不认识 SQLite、不认识 Qdrant、不认识任何数据集**，调用方自己取
`pair.question` / `pair.answer` / `pair.event_time` 传进来。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Final

# ── 模板版本 ────────────────────────────────────────────────────────────
# ⚠ 改模板【必须】同时改这个版本号：它会进 embedding 缓存的坐标系
#   （embed/base.py 的 EmbeddingCoordinate），而 §11.3 明确"改模板 = 改变 embedding
#   输入 = 整个向量索引要重建"。版本号是让"旧缓存静默命中"变成不可能的那把锁。
TEMPLATE_VERSION: str = "v1"

QUESTION_PREFIX: str = "Q: "
ANSWER_PREFIX: str = "A: "

# Q 与 A 之间的分隔。PRD §11.3 的【模板块】是一个换行、同节的示例里画成了空行
# （两处文档有这个矛盾）——这里取模板块的字面（规范表述优先于示意），并把"改它"集中到这一行。
QUESTION_ANSWER_SEP: str = "\n"

#: **一个 Context Segment 里各 QA 对之间的分隔**（§10 / §11.3 的扩窗之后才有段的概念）。
#: 取单个换行：每一对都以 `Q:` / `A:` 开头，那已是自定界的行首标记（§11.3），再加空行只是
#: 白花 token，而 token 要按**真实字符串**算进 117,760 的预算里（§6.4）。
SEGMENT_SEP: str = "\n"


# ── 时间（T1 在正文前加日期时用的那一套）────────────────────────────────
#: 日期格式：`YYYY-MM-DD`（§11.3 的示例是 `2026-07-26`）。
#: ⚠ **不要用裸 Unix 毫秒**（渲染出来是 `- [1753512557000] ...`，对模型无意义）；且
#: **`content` 的前缀与响应里的 `created_at` 必须是同一个格式**，否则 T1 测的就不是
#: "加不加日期"而是"加哪种日期"（两件事，混在一起就归因不了）。
_DATE_FORMAT: Final[str] = "%Y-%m-%d"

#: T1 的"带"臂在正文最前面加的那一段。**前缀形状只此一处**。
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

    两个消费方：响应里的 `created_at`（[`../rank/packaging.py`](../rank/packaging.py)，
    **始终**要给）与 T1 的 `content` 前缀（下面的 `render_date`，**默认**不给）。
    **实现只此一处**——两处若各写一份格式，早晚会分叉。

    * `event_time` 是**该对首条消息**的 timestamp（§6.1），可空
    * **NULL 时发 `""`**（§11.3）：渲染代码是 `str(item.get("created_at") or "")`，
      假值会退化成 `- {text}`——那是一条**有定义的降级路径**
    * ⚠ **绝不拿 Add 的到达时间兜底**：那是"何时写入"而不是"何时发生"，
      会给模型**错误信息**
    * **只给到日粒度**：粒度会被模型看见，秒级会诱发它按秒级回答，从而踩中
      "粒度变细"那条判负规则（§11.3）
    * **时区固定 UTC、无旋钮**：契约没规定时区，而它必须与加载层合成 `event_time` 时的
      口径一致——不一致会让日期整体偏一天，**且不报错**
    """
    day = event_day(event_time)
    return "" if day is None else day.strftime(_DATE_FORMAT)


def render_date(event_time: int | None, *, inject_abs_time: bool) -> str:
    """正文前缀里的日期串（`packaging.inject_abs_time`，见 **D21**）。

    * `inject_abs_time=False` ⇒ `""`：正文里不带任何时间（旧口径，仍可作消融臂）
    * `True`（**2026-09-25 起默认**）⇒ **日粒度日期**，如 `2023-10-22`

    ⚠ **只给到"日"**：`created_at` 与它同口径（契约 §3 要求日粒度）；**秒级绝不许出现**
    （那会踩裁判的「粒度变细」，`preflight` 有检查钉着）。

    ⛔ **星期试过，更差**（2026-09-25）：`[2023-10-22 Sun]` 让整体 0.633 → 0.601、三段全降，
    而对"某星期几之前"那 29 道 gold **一道没救回来** ⇒ 已撤回。**不要再加回星期。**
    """
    return day_granularity(event_time) if inject_abs_time else ""


def template_version(*, inject_abs_time: bool) -> str:
    """缓存坐标里的模板标识（`embed/base.EmbeddingCoordinate.render_template`）。

    **改模板必须换版本号**（见 `TEMPLATE_VERSION` 的注释）。T1 的"带"臂改的正是正文
    ⇒ 它的坐标必须与"不带"臂**不同**，否则两臂的向量会互相静默复用——
    而"缓存命中了另一臂的向量"在结果里看不出来（§7.2 / §12.1 R1）。
    """
    return f"{TEMPLATE_VERSION}+abs_time" if inject_abs_time else TEMPLATE_VERSION


def render(question: str | None, answer: str | None, *, date: str = "") -> str:
    """把一个 QA 对渲染成最终文本。**这是唯一一处拼装。**

    规则（rank/CLAUDE.md §4 = 渲染规则的唯一出处）：

    | 输入 | 输出 |
    | --- | --- |
    | 都有 | `Q: {question}\\nA: {answer}` |
    | `question` 为空（§6.2 的"无问的对"） | 只输出 `A:` 那一行 |
    | `answer` 为空（块里没有非 user 消息） | 只输出 `Q:` 那一行 |

    返回值**首尾无空白**：AML 只做 `"\\n".join(...)` 拼接、不插分隔符，
    任何一项首尾留白都会让拼接处粘连（§11.3"content 必须自定界"）。

    ⚠ **默认不注入任何时间戳**（`date=""`，v1 定稿）：§11.3 有两条**互相独立**的机制
    都指向"不要注入绝对时间"（而答案 prompt 第 7 条**却要求**转换相对时间——所以加绝对
    时间戳可能反而有害）。理由与适用范围见 config-reference §7。时间信息由 `event_time`
    列负责筛选，正文只保留原始表述。

    `date` 非空时在**正文最前面**加 `[date] `（T1 的"带"臂），取不取由 `render_date()`
    决定，**三个调用点（索引 / 精排输入 / content）用同一个口径**——这正是不变式 I1：
    三处渲染不同就是"命中的是另一个版本"，而**这种漂移不报错**。

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

    ⚠ `content` 那一路**不走本函数**：它由 `rank/neighbor.py` 的 `_build_segment._text()`
    用 `render()` + `render_date()` 拼出（为了接上 `annotate()`）——那是 I1 的**一个声明式例外**，
    见 [`../rank/CLAUDE.md`](../rank/CLAUDE.md) §4。

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
