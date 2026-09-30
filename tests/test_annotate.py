"""相对时间注解 —— **只加不改**，且**碰不到索引**（不变式 I1 的声明式例外）。

对应 [`../src/tianximem/common/annotate.py`](../src/tianximem/common/annotate.py) 与
[`../src/tianximem/rank/neighbor.py`](../src/tianximem/rank/neighbor.py) 的 `_text()`。

这一层错了**不会报错**：注解只是往 `content` 里多写几个字，检索照常返回值、分数照常算。
所以这里钉三件事：

1. **只加不改**：相对表述原样保留（D21 的口径），注解跟在后面
2. **推不出就不动**：`a few years ago` 这类**一个字符都不改**——编一个日期比不换算更糟
3. **它碰不到索引**：开/关这个开关，被 embedding 的文本**逐字相同**（否则它与 T1 就没区别了，
   而"它不用重建索引"正是它存在的理由）
"""

from __future__ import annotations

import re
from datetime import date
from functools import partial

import pytest

from tianximem.common.annotate import annotate
from tianximem.common.render import render_pair

#: 2023-07-20 是**周四**——下面几条期望值全部锚在它上面。
_ANCHOR = date(2023, 7, 20)


def test_annotate_only_adds_and_never_replaces() -> None:
    """原文一字不动，注解跟在后面（D21："日期是额外锚点，不是替换"）。"""
    out = annotate("Hey Mel! I joined an activist group last Tues.", _ANCHOR)
    assert out == "Hey Mel! I joined an activist group last Tues (July 18, 2023)."


@pytest.mark.parametrize(
    "text",
    [
        "A few years ago I moved.",
        "Several weeks ago we spoke.",
        "Nothing temporal here.",
        "I'll see you soon and recently I've been busy.",
    ],
)
def test_annotate_refuses_to_guess(text: str) -> None:
    """**推不出就不动**——`a few` / `several` 没有数字，换算它等于编一个日期，
    而编错的日期比不换算更糟（模型会照抄，裁判是精确比值的）。"""
    assert annotate(text, _ANCHOR) == text


def test_annotate_week_and_weekend_start_on_monday() -> None:
    """周的起点是**周一**——这条是拿 gold 反推出来的，不是选出来的。

    `The weekend before 17 July 2023`（gold 并列 `From July 15, 2023 to July 16, 2023`）
    与 `two weekends before 17 July 2023`（`From July 8, 2023 to July 9, 2023`）都锚在
    2023-07-17（周一）上；**按周日为界这两条都会差一天**。
    """
    monday = date(2023, 7, 17)
    assert annotate("last weekend we camped", monday) == (
        "last weekend (July 15 to 16, 2023) we camped"
    )
    assert annotate("camping two weekends ago", monday) == (
        "camping two weekends ago (July 8 to 9, 2023)"
    )
    # 2023-06-09 是周五：`The week before 9 June 2023` 的 gold 并列形式是 5/29–6/4
    assert annotate("last week", date(2023, 6, 9)) == "last week (May 29 to June 4, 2023)"


def test_annotate_covers_month_year_and_duration() -> None:
    """月 / 年粒度与"持续了多久"（`for 7 years` → `since 2016`）各一例。"""
    assert annotate("I started last month", _ANCHOR) == "I started last month (June 2023)"
    assert annotate("I moved last year", _ANCHOR) == "I moved last year (2022)"
    assert annotate("I've been teaching for seven years", _ANCHOR) == (
        "I've been teaching for seven years (since 2016)"
    )


def test_annotate_mark_switch_only_changes_the_wrapper() -> None:
    """两种记号**只差外壳**：换算结果一字不改（否则它就不是单变量对照了）。"""
    paren = annotate("last night we talked", _ANCHOR, mark="paren")
    tag = annotate("last night we talked", _ANCHOR, mark="tag")

    assert paren == "last night (July 19, 2023) we talked"
    assert tag == "last night [= July 19, 2023] we talked"
    with pytest.raises(ValueError, match="未知的注解记号"):
        annotate("x", _ANCHOR, mark="bogus")


# ── 两份实现的等价性（**重复的是逻辑，所以要用用例钉住**）────────────────────
_CASES: tuple[tuple[str, date], ...] = (
    ("I joined last Tues.", _ANCHOR),
    ("last weekend we camped", date(2023, 7, 17)),
    ("camping two weekends ago", date(2023, 7, 17)),
    ("last week I was in Berlin", date(2023, 6, 9)),
    ("I started last month and moved last year", _ANCHOR),
    ("I've been teaching for seven years", _ANCHOR),
    ("A few years ago I moved.", _ANCHOR),
    ("3 days ago and 5 months ago", date(2023, 3, 3)),
    ("yesterday, today, tomorrow, last night", date(2023, 12, 31)),
    ("nothing temporal", _ANCHOR),
    ("", _ANCHOR),
)


@pytest.mark.parametrize(("text", "anchor"), _CASES)
def test_annotate_matches_the_harness_copy(text: str, anchor: date) -> None:
    """[`eval/harness/annotate.py`](../eval/harness/annotate.py) 与 `src/` 这份**必须逐字一致**。

    harness **不许 import `src/`**（[`README`](./CLAUDE.md) §五 的 AST 断言钉着），
    而那次注解实验（run record 里的 `--memory-date annotate`）是在 harness 侧跑的——
    **两份都得留着**，所以重复的是**逻辑**而不是常量。

    ⇒ 常量可以靠"抄一遍 + 断言相等"守（`DATE_PREFIX` 就是这么办的），**逻辑只能靠用例**：
    同一张表喂给两份实现，输出必须逐字相同。**改任何一份，这条立刻红。**
    """
    from eval.harness.annotate import annotate as harness_annotate

    assert harness_annotate(text, anchor) == annotate(text, anchor)


# ── 开关纯度（§13）：两臂**只该差正文**，而且**索引那一侧一个字都不该变** ──────
#: 注解的形状（用来把注解从 `content` 里剥掉，验证"原文一字未改"）。
_ANNOTATION = re.compile(
    r" \((?:"
    r"[A-Z][a-z]+ \d{1,2}(?: to (?:[A-Z][a-z]+ )?\d{1,2})?, \d{4}"
    r"|[A-Z][a-z]+ \d{4}"
    r"|since \d{4}"
    r"|\d{4}"
    r")\)"
)


def _strip_annotations(text: str) -> str:
    return _ANNOTATION.sub("", text)


#: 三条**互不相邻**的位置（相邻会被扩窗合进同一段，段数断言就验不出东西）。
#: ⚠ D25 起 `seq` 由**已有的行**现算 ⇒ 跳号会挨在一起，
#: 所以要落满中间那些位置（`conftest.seed_line`）。
_PAIRS: tuple[tuple[int, str, str], ...] = (
    (0, "What's new?", "I joined a new activist group last Tues."),
    (2, "How was your weekend?", "Last weekend our city held a pride parade!"),
    (4, "Seen any old friends?", "A few years ago I moved away."),
)
_EVENT_TIME = 1689811200000  # 2023-07-20T00:00:00Z（周四）


def _dress(chain) -> list[str]:
    """灌三条带相对表达的 `pair`，返回被索引的 id。"""
    ids: list[str] = []
    pairs = []
    for ordinal, question, answer in _PAIRS:
        with chain.store.transaction() as conn:
            # 每次 Add 一块（D28：位置 = `(request_id, local_index)`，彼此不相邻）
            pair = chain.store.insert_pair(
                conn,
                user_id="u1",
                session_id="s1",
                request_id=f"r-annotate-{ordinal}",
                local_index=0,
                prev_memory_id=None,
                next_memory_id=None,
                question=question,
                answer=answer,
                status="complete",
                event_time=_EVENT_TIME,
            )
        ids.append(pair.id)
        pairs.append(pair)
    # ⚠ 索引侧走**生产路径的同一个渲染函数**（`index_pairs` 的 `renderer`）——
    #   若这里改成 `partial(render_pair, ...)` 之外的东西，"注解不进索引"就验不到了。
    chain.qdrant.index_pairs(pairs, chain.embedder, renderer=partial(render_pair))
    return ids


def test_annotate_switch_only_touches_the_text(wired, wired_annotated) -> None:
    """开关必须只影响它命名的那一件事：**正文**。名次 / `id` / `created_at` / `score` 全不变。"""
    for chain in (wired, wired_annotated):
        chain.qdrant.by_user["u1"] = _dress(chain)

    plain = wired.search(user_id="u1", query="q", top_k=5)
    annotated = wired_annotated.search(user_id="u1", query="q", top_k=5)

    assert len(plain.items) == len(annotated.items) > 0
    assert [i.id for i in annotated.items] == [i.id for i in plain.items]
    assert [i.created_at for i in annotated.items] == [i.created_at for i in plain.items]
    assert [i.score for i in annotated.items] == [i.score for i in plain.items]


def test_annotate_switch_leaves_the_index_untouched(wired, wired_annotated) -> None:
    """**开这个开关不改变被 embedding 的文本**——它存在的全部理由就是这一条。

    与 T1 的 `inject_abs_time` 正相反：那个改的正是索引的输入（⇒ 要重建索引、要分集合）。
    这里若哪天有人把注解挪到 `render_pair` 里去，这条会立刻红——
    而**在真环境里它表现为"分数没变好，但索引白重建了一次"**，很难归因。
    """
    for chain in (wired, wired_annotated):
        ids = _dress(chain)
        chain.qdrant.by_user["u1"] = ids

    plain = wired.search(user_id="u1", query="q", top_k=5)
    annotated = wired_annotated.search(user_id="u1", query="q", top_k=5)

    # 假 Qdrant 存的就是**被索引的文本**：两臂必须逐字相同
    assert wired.qdrant.points == wired_annotated.qdrant.points
    assert all("(" not in text for text in wired_annotated.qdrant.points.values()), (
        "索引里出现了括号 —— 注解漏进索引侧了"
    )

    # 而 `content` 那一侧**确实变了**（否则上面那条断言是空过的）
    assert any("(" in item.content for item in annotated.items)
    assert not any("(" in item.content for item in plain.items)


def test_annotated_content_is_the_indexed_text_plus_annotations(wired, wired_annotated) -> None:
    """**投影性质**（不变式 I1 的那个例外，写死在这里）：剥掉注解之后逐字相同。

    这条是"例外"这个词的全部含义——不是"两份渲染各写各的"，而是
    **一份渲染 + 一层可逆的确定性后处理**。
    """
    for chain in (wired, wired_annotated):
        chain.qdrant.by_user["u1"] = _dress(chain)

    plain = wired.search(user_id="u1", query="q", top_k=5)
    annotated = wired_annotated.search(user_id="u1", query="q", top_k=5)

    for item_plain, item_annotated in zip(plain.items, annotated.items, strict=True):
        assert _strip_annotations(item_annotated.content) == item_plain.content

    # 具体到语义：2023-07-20 是周四 ⇒ `last Tues` = 2023-07-18
    haystack = "\n".join(item.content for item in annotated.items)
    assert "last Tues (July 18, 2023)" in haystack
    # 「a few years ago」推不出 ⇒ **原样留着**（推不出就不动）
    assert "A few years ago I moved away." in haystack
