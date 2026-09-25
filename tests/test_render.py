"""渲染 —— **唯一实现**（不变式 I1）的测试。

对应 [`../tests/CLAUDE.md`](../tests/CLAUDE.md) 四、同一份渲染。
⚠ 这一层错了不会报错：两处渲染一旦漂移，"检索命中的是什么"与"模型读到的是什么"
就会不一致，而检索照常返回值、分数照常算。
"""

from __future__ import annotations

from tianxi_am.common.render import (
    TEMPLATE_VERSION,
    render,
)


def test_both_sides_present() -> None:
    assert render("Q1", "[assistant] A1") == "Q: Q1\nA: [assistant] A1"


def test_question_empty_only_answer_line() -> None:
    """`question` 为空的对（§6.2 的"无问的对"）**只输出 `A:` 那一行**。"""
    assert render(None, "[assistant] 先交代背景") == "A: [assistant] 先交代背景"
    assert render("", "[assistant] x") == "A: [assistant] x"


def test_answer_empty_only_question_line() -> None:
    """`answer` 暂缺的 `pending` 对**只输出 `Q:` 那一行**。"""
    assert render("Q1", None) == "Q: Q1"
    assert render("Q1", "") == "Q: Q1"


def test_both_empty_is_empty_string() -> None:
    """两边都空只会出现在"还没落到任何消息的对"，不应产生 `Q:` 之类的空壳。"""
    assert render(None, None) == ""


def test_multiline_answer_kept_verbatim() -> None:
    """多条非 user 消息的 role 标记**已经在 `answer` 里**（① 的决定 A1），
    渲染只负责加 `A:` 前缀——不重新拼装、也不加任何额外标记。"""
    answer = '[assistant] 我查一下。\n[system] {"tool": "x"}\n[assistant] 09:42。'
    assert render("火车几点开？", answer) == f"Q: 火车几点开？\nA: {answer}"
    assert render("火车几点开？", answer).count("\n") == 3


def test_output_has_no_surrounding_whitespace() -> None:
    """**首尾无空白**：AML 只做 `"\\n".join(...)` 拼接、不插分隔符，
    任何一项首尾留白都会让拼接处粘连（§11.3）。"""
    for q, a in [("Q1", "A1"), (None, "A1"), ("Q1", None), ("  Q1  ", "  A1  ")]:
        out = render(q, a)
        assert out == out.strip(), repr(out)


def test_render_is_pure_and_deterministic() -> None:
    """同一输入必须给出逐字相同的输出——它是 embedding 的输入，也是缓存键的来源。"""
    assert render("a", "b") == render("a", "b")
    assert render("a", "b") != render("a", "c")


def test_render_injects_no_timestamp() -> None:
    """**不注入任何时间戳前缀。**

    §11.3 有两条**互相独立**的规则都指向"不要注入绝对时间"（粒度变细、相对↔绝对互转），
    而答案 prompt 第 7 条却要求转换相对时间——所以加绝对时间戳可能反而有害。
    时间信息由 `event_time` 列负责筛选，正文只保留原始表述。
    """
    out = render("昨天发生了什么？", "[assistant] 你去了公园。")
    assert "2026" not in out
    assert "T00:00" not in out
    # 原始的相对表述必须原样保留
    assert "昨天" in out


def test_template_version_is_declared() -> None:
    """模板版本必须存在且非空——它进 embedding 缓存的坐标系。

    §11.3 明确"改模板 = 改变 embedding 输入 = 整个向量索引要重建"，
    版本号就是让"旧缓存静默命中"变成不可能的那把锁。
    """
    assert TEMPLATE_VERSION
    assert isinstance(TEMPLATE_VERSION, str)


def test_template_separator_is_a_single_constant() -> None:
    """Q 与 A 之间的分隔只有一处定义——改模板时必须改这一个地方。

    ⚠ PRD §11.3 与 rank/README §4 之间有一处**自相矛盾**：模板块是一个换行，
    同节的示例画成了空行。这里取模板块的字面（规范表述优先于示意）。
    """
    from tianxi_am.common.render import QUESTION_ANSWER_SEP

    assert QUESTION_ANSWER_SEP == "\n"
    assert render("q", "a").count(QUESTION_ANSWER_SEP) == 1


# ── T1 的渲染变体（§11.3 / §13）────────────────────────────────────────────
_MAY_8_2023_MS = 1683504000000  # 2023-05-08T00:00:00Z


def test_render_date_is_the_switch_and_nothing_else() -> None:
    """**开关只决定"取不取那个串"，不碰日期口径**（口径只有 `day_granularity` 一处）。

    两臂若连格式也不同，T1 就不是一个单变量对照了。
    """
    from tianxi_am.common.render import day_granularity, render_date

    assert render_date(_MAY_8_2023_MS, inject_abs_time=False) == ""
    # ⛔ 星期试过更差（见 `render_date` 的注释）⇒ 前缀**只到日粒度**
    assert render_date(_MAY_8_2023_MS, inject_abs_time=True) == "2023-05-08"
    # 时间缺失时**不是"1970-01-01"**，而是与 created_at 同一条降级路径：空串（§11.3）
    assert render_date(None, inject_abs_time=True) == ""
    assert day_granularity(None) == ""


def test_date_prefix_goes_in_front_and_stays_trimmed() -> None:
    """`date` 非空 ⇒ 正文最前面加 `[date] `；首尾仍然无空白（§11.3 自定界）。"""
    out = render("Q1", "A1", date="2023-05-08")
    assert out == "[2023-05-08] Q: Q1\nA: A1"
    assert out == out.strip()
    assert render("Q1", "A1", date="") == "Q: Q1\nA: A1"  # 不传 date ⇒ 一字不变


def test_empty_pair_gets_no_date_prefix() -> None:
    """**空文本不加前缀**：一个只有日期的 `content` 会让模型读到一条不存在的记忆。"""
    assert render(None, None, date="2023-05-08") == ""


def test_template_version_separates_the_two_arms() -> None:
    """两臂的**缓存坐标必须不同**——否则"带"臂会静默复用"不带"臂的向量（§7.2）。"""
    from tianxi_am.common.render import template_version

    assert template_version(inject_abs_time=False) == TEMPLATE_VERSION
    assert template_version(inject_abs_time=True) != TEMPLATE_VERSION


def test_render_pair_takes_the_date_from_the_pair_itself() -> None:
    """`render_pair` 是**三个调用点唯一的入口**——日期只能来自对自身的 `event_time`。"""
    from tianxi_am.common.render import render_pair

    class _Pair:
        question = "Q1"
        answer = "A1"
        event_time = _MAY_8_2023_MS

    assert render_pair(_Pair()) == "Q: Q1\nA: A1"
    assert render_pair(_Pair(), inject_abs_time=True) == "[2023-05-08] Q: Q1\nA: A1"

    class _Undated:
        question = "Q1"
        answer = "A1"
        event_time = None

    assert render_pair(_Undated(), inject_abs_time=True) == "Q: Q1\nA: A1"


# ── T1 的开关纯度（§13）：两臂**只该差正文**────────────────────────────────
def _dress(chain, *, inject_abs_time: bool) -> list[str]:
    """灌三条**相隔的** `pair_idx`（0/2/4）、每条都带 `event_time`，返回被索引的 id。

    ⚠ 三条不是相邻的：相邻会被扩窗合进同一段，那样"段数没变"这条断言就验不出东西
    （[`../tests/CLAUDE.md`](../tests/CLAUDE.md) §三 的"一个候选 ≠ 一项"）。
    ⚠ 用 `insert_pair` 而不是走 Add：这里要的是**确定的** `pair_idx` 与 `event_time`
    （Add 路径的 `pair_idx` 由配对逻辑给，`event_time` 取消息上的 timestamp）。
    索引侧仍然走**生产路径的同一个渲染函数**（`index_pairs` 的 `renderer`）。
    """
    from functools import partial

    from tianxi_am.common.render import render_pair

    ids: list[str] = []
    pairs = []
    for pair_idx in (0, 2, 4):
        with chain.store.transaction() as conn:
            pair = chain.store.insert_pair(
                conn,
                user_id="u1",
                session_id="s1",
                pair_idx=pair_idx,
                question=f"Q{pair_idx}",
                answer=f"A{pair_idx}",
                status="complete",
                event_time=_MAY_8_2023_MS,
                request_id="seed",
            )
        ids.append(pair.id)
        pairs.append(pair)
    chain.qdrant.index_pairs(
        pairs,
        chain.embedder,
        renderer=partial(render_pair, inject_abs_time=inject_abs_time),
    )
    return ids


def test_t1_switch_only_changes_the_text(wired, wired_dated) -> None:
    """同一个库、同一次检索、两臂逐项比：**只有 `content` 该变。**

    §13 的纯度规则（[`../tests/CLAUDE.md`](../tests/CLAUDE.md) §五）：开关必须只影响它
    命名的那一件事。这里被"命名"的是**正文里带不带日期**，所以名次、`id`、`created_at`、
    `score`、段数**一个都不许动**——动了就说明这个对照同时在测两件事，结论不可归因。
    """
    from tianxi_am.common.render import day_granularity

    prefix = f"[{day_granularity(_MAY_8_2023_MS)}] "
    for chain, inject in ((wired, False), (wired_dated, True)):
        chain.qdrant.by_user["u1"] = _dress(chain, inject_abs_time=inject)

    plain = wired.search(user_id="u1", query="q", top_k=5)
    dated = wired_dated.search(user_id="u1", query="q", top_k=5)

    assert len(plain.items) == len(dated.items) > 0
    assert [item.id for item in dated.items] == [item.id for item in plain.items]
    assert [item.created_at for item in dated.items] == [item.created_at for item in plain.items]
    assert [item.score for item in dated.items] == [item.score for item in plain.items]

    assert all(prefix in item.content for item in dated.items)
    assert not any(prefix in item.content for item in plain.items)
    # 去掉前缀之后应当**逐字相同**——两臂差的正好只有那一段
    for item_plain, item_dated in zip(plain.items, dated.items, strict=True):
        assert item_dated.content.replace(prefix, "") == item_plain.content


def test_t1_arm_renders_index_and_content_identically(wired_dated) -> None:
    """**I1 在 T1 的"带"臂上仍然成立**：被索引的文本 == content 里的那一对文本。

    这是这个开关最危险的地方——正文一改，索引侧与返回侧**必须同时改**。
    只改一边不会报错，只会让"检索命中的是什么"与"模型读到的是什么"悄悄分叉（§7.2）。
    """
    ids = _dress(wired_dated, inject_abs_time=True)
    wired_dated.qdrant.by_user["u1"] = ids

    response = wired_dated.search(user_id="u1", query="q", top_k=5)

    # 段的 `content` 是逐对渲染后用 `\n` 连起来的 ⇒ 每一对都必须**逐字**出现在某段里
    haystack = "\n".join(item.content for item in response.items)
    for memory_id in ids:
        text = wired_dated.qdrant.points[memory_id]
        assert text.startswith("[2023-05-08] "), text
        assert text in haystack, text
