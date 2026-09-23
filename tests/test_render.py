"""渲染 —— **唯一实现**（不变式 I1）的测试。

对应 [`../tests/README.md`](../tests/README.md) 四、同一份渲染。
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
    渲染只负责加 `A:` 前缀，不再重新拼装、也不加任何额外标记。"""
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
