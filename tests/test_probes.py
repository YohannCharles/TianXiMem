"""两个探针工具的核心逻辑（`tools/truncation_probe.py` · `tools/reorder_probe.py`）。

⚠ 它们都是**"只改注入、重跑判分"**的因果探针（不投喂、不检索），所以核心逻辑
（怎么截、怎么排）是纯函数——**这几条用例钉的就是那几个纯函数**。
判分那一步不在本文件里（它要真网关）。
"""

from __future__ import annotations

from tools.reorder_probe import _stem, reorder
from tools.truncation_probe import truncate_prefix


# ── 截断：按行边界切 ────────────────────────────────────────────────────
def test_truncate_prefix_keeps_whole_lines():
    """**行内不切**——与 `packaging` 的"段是原子单位"同一条理由。"""
    text = "\n".join(f"line{i}" for i in range(10))
    out = truncate_prefix(text, 0.5)
    assert out.split("\n") == [f"line{i}" for i in range(5)]
    assert truncate_prefix(text, 1.0) == text  # 100% ⇒ 一字不动
    assert truncate_prefix(text, 0.01) == "line0"  # 再小也留一行


# ── 重排：三个策略都**只改顺序** ─────────────────────────────────────────
def test_reorder_control_is_identity():
    segments = ["b", "a", "c"]
    assert reorder(segments, "query", mode="control") == segments


def test_reorder_keyword_puts_hits_first_and_keeps_the_rest_in_order():
    """命中的提到最前；**同分的保持服务给的原序**（`sorted` 稳定）。"""
    segments = ["nothing here", "the dog barked", "no match", "a dog again"]
    assert reorder(segments, "dog", mode="keyword") == [
        "the dog barked",
        "a dog again",
        "nothing here",
        "no match",
    ]


def test_reorder_keyword_is_case_insensitive_and_word_bounded():
    assert reorder(["CAT", "cat."], "cat", mode="keyword") == ["CAT", "cat."]
    # 词边界：`category` 不该被当作 `cat` 命中
    assert reorder(["category", "a cat here"], "cat", mode="keyword") == ["a cat here", "category"]


def test_stem_matches_plural_and_gerund():
    """`stem` 臂回答的是"**小狗/狗算不算同一个关键词**"——用最小词干化。"""
    assert _stem("dogs") == "dog"
    assert _stem("running") == "run"
    assert _stem("cats") == "cat"
    assert _stem("is") == "is"  # 去完要至少留 3 个字符
    order = reorder(["nothing", "two dogs played"], "dog", mode="stem")
    assert order == ["two dogs played", "nothing"]


def test_reorder_rejects_an_unknown_mode():
    """**未知策略要响亮**——静默退化成 control 会让整个对照失去意义。"""
    import pytest

    with pytest.raises(ValueError, match="未知重排策略"):
        reorder(["a"], "q", mode="semantic-magic")
