"""三个探针工具的核心逻辑（`tools/truncation_probe.py` · `tools/reorder_probe.py` ·
`tools/coq_decompose_probe.py`）。

⚠ 前两个是**"只改注入、重跑判分"**的因果探针（不投喂、不检索），所以核心逻辑
（怎么截、怎么排）是纯函数——**这几条用例钉的就是那几个纯函数**。
判分那一步不在本文件里（它要真网关）。

⚠ 第三个是**分解能力探针**：它的"判定"**不由模型做**（理由见那支工具的 docstring：
模型裁判至少错判 4/25，而且是事实性错误），而是读
[`eval/coq-decompose-labels.jsonl`](../eval/coq-decompose-labels.jsonl) 里**人填的 `defect`**。
⇒ 下面钉的是那份标签的**读入与汇总**，以及"标签本身合法"这件事。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.coq_decompose_probe import (
    DEFECTS,
    _summarize,
    load_labels,
    render_markdown,
    score_labels,
)
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


# ── 分解探针：读人的标签（⛔ 判定不由模型做）────────────────────────────


def _write(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_labels_are_read_and_comments_skipped(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "# 表头，读进来时跳过",
        json.dumps({"question_id": "q1", "defect": "none", "note": ""}),
        "",
        json.dumps({"question_id": "q2", "defect": "bundled", "note": "并了两跳"}),
    )
    assert load_labels(path) == {"q1": "none", "q2": "bundled"}


def test_an_unknown_defect_token_is_a_loud_failure(tmp_path: Path) -> None:
    """自由发挥的词表会让"缺陷分布"每次重判都换一套说法。"""
    path = _write(tmp_path, json.dumps({"question_id": "q1", "defect": "拆得有点乱"}))
    with pytest.raises(SystemExit, match="不在词表里"):
        load_labels(path)


def test_a_question_judged_twice_is_a_loud_failure(tmp_path: Path) -> None:
    """两份判读冲突时**挑哪一份都是替人做决定**（机器代填的另一种形态）。"""
    path = _write(
        tmp_path,
        json.dumps({"question_id": "q1", "defect": "none"}),
        json.dumps({"question_id": "q1", "defect": "bundled"}),
    )
    with pytest.raises(SystemExit, match="判了两次"):
        load_labels(path)


def test_a_broken_line_is_not_silently_skipped(tmp_path: Path) -> None:
    """⚠ 少一条 = 分母变小 = 分数悄悄变高，而屏幕上完全正常。"""
    path = _write(tmp_path, json.dumps({"question_id": "q1", "defect": "none"}), "{不是 JSON")
    with pytest.raises(SystemExit, match="不是 JSON"):
        load_labels(path)


def test_the_verdict_is_derived_from_the_defect() -> None:
    """⭐ **只有一个字段**：`none` ⇒ SAME，其余 ⇒ DIFFERENT。

    分成两个字段就会出现"defect 说有问题、verdict 说没问题"这种自相矛盾的行，
    而它**不会报错**——只会让两条数各说各话。
    """
    got = score_labels(["q1", "q2", "q3"], {"q1": "none", "q2": "restated", "q3": "oversplit"})
    assert got["same"] == 1
    assert got["same_rate"] == 0.333
    assert got["defect_counts"] == {"none": 1, "oversplit": 1, "restated": 1}
    assert got["wrong_ids"] == ["q2", "q3"]


def test_unjudged_questions_are_neither_right_nor_wrong() -> None:
    """未判的题**不进分母**，但要**点名报出来**——把它当对是虚高，当错是虚低。"""
    got = score_labels(["q1", "q2", "q3"], {"q1": "none"})
    assert (got["judged"], got["unjudged"], got["same_rate"]) == (1, 2, 1.0)
    assert got["unjudged_ids"] == ["q2", "q3"]


def test_the_report_carries_no_model_verdict_field() -> None:
    """⛔ **回归护栏**：`judge_*` 那一套（模型当裁判）已作废，不许再回到汇总里。

    它当时给出的 9/25 是**事实性错判**（连金标有几跳都数错），若哪天有人"顺手加回来"，
    两个数字会并列出现在报告里，而其中一个是不该被采信的。
    """
    report = _summarize(
        [
            {
                "question_id": "q1",
                "status": "ok",
                "shape": {
                    "steps": 4,
                    "gold_steps": 4,
                    "hop_count_match": True,
                    "back_ref_steps": 3,
                    "back_ref_rate": 1.0,
                    "final_covers_gold": True,
                },
            }
        ]
    )
    assert not [key for key in report if "judge" in key]
    assert report["per_question"]["q1"] == {"status": "ok", "steps": 4, "gold_steps": 4}


# ── 人读的那一份（原问题 / 模型拆解 / 金标拆解）──────────────────────────


def _row(qid: str, question: str, candidate: list[str], gold: list[str]) -> dict:
    return {
        "question_id": qid,
        "status": "ok",
        "question": question,
        "gold_hops": gold,
        "candidate_hops": candidate,
    }


def test_the_markdown_dump_puts_the_three_readings_side_by_side() -> None:
    """⭐ 它存在的理由就是**让人能直接复制着逐条复核**：原问题 / 模型拆解 / 金标拆解各一块。"""
    text = render_markdown(
        [_row("q1", "Q one?", ["c1", "c2"], ["g1", "g2"])],
        {"q1": {"defect": "none", "note": ""}},
        {"q1": ["a1", "a2"]},
    )
    assert "> Q one?" in text  # 原问题用引用块，方便复制
    assert "1. c1" in text and "2. c2" in text  # 模型的步编号
    assert "1. g1　→　`a1`" in text  # 金标的每跳带上那一跳的答案
    assert "✅" in text and "合计：结构同构 1/1" in text


def test_the_markdown_dump_marks_unjudged_apart_from_wrong() -> None:
    """未判 ≠ 判错：一个 `❓`、一个 `❌`——**两者的补救动作完全不同**（去判 / 去修）。"""
    text = render_markdown(
        [_row("q1", "Q?", ["c"], ["g"]), _row("q2", "Q?", ["c"], ["g"])],
        {"q1": {"defect": "bundled", "note": "并了两跳"}},
        {},
    )
    assert "❌ 判定：`bundled`" in text and "**判据**：并了两跳" in text
    assert "❓ 判定：`未判`" in text
    assert "合计：结构同构 0/2" in text
    assert "→" not in text.split("## 1.")[-1].split("---")[0]  # 拿不到金标答案就不硬编一列


def test_the_markdown_dump_never_claims_a_verdict_it_does_not_have() -> None:
    """模型没按格式答时**不许省略那一段**——省掉会让"没解析出来"看起来像"这道题不存在"。"""
    text = render_markdown([_row("q1", "Q?", [], ["g1"])], {}, {})
    assert "（没解析出来）" in text


# ── 交付的那份标签本身（不读数据集的题面）──────────────────────────────


def test_the_shipped_labels_cover_the_manifest_exactly() -> None:
    """⚠ **覆盖率对不上就是错的**：少一题 → 分母变小；多一题 → 那道题没在清单里。

    `coq-musique.jsonl` 全是 4 跳题（`hops == 4`），所以金标步数**没有变化**，
    清单里那 25 条 id 就是这份标签的全部应判集合。
    """
    manifest = [
        json.loads(line)["question_id"]
        for line in Path("eval/coq-musique.jsonl").read_text(encoding="utf-8").split("\n")
        if line.strip() and not line.lstrip().startswith("#")
    ]
    labels = load_labels(Path("eval/coq-decompose-labels.jsonl"))
    assert set(labels) == set(manifest)
    assert len(labels) == 25
    # 每一类都真出现过（否则词表里那条是死代码）
    assert set(labels.values()) <= set(DEFECTS)


def test_the_shipped_labels_store_no_question_text() -> None:
    """⛔ 与 `eval/coq-musique.jsonl` 同一条口径：只留 id 与判据，**题面/答案一字不进 git**。"""
    rows = [
        json.loads(line)
        for line in Path("eval/coq-decompose-labels.jsonl").read_text(encoding="utf-8").split("\n")
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert len(rows) == 25
    for row in rows:
        assert set(row) == {"question_id", "defect", "note"}, row
        assert row["defect"] in DEFECTS
