"""`tools/plan_decompose_eval.py` 的**纯函数**部分——不发请求、不读数据集。

## 为什么这个文件值当

拆解那一层是 Plan 模式的**上游瓶颈**，
而整轮跑批里它被检索、答案、判分三层盖住。这个工具把它单独拎出来量，
⇒ 它自己的**聚合口径**就成了结论的地基：

* **"少拆 / 相等 / 多拆"是按逐题差算的**，不是拿两个均值比——
  均值会把"3 跳拆 2 步"和"3 跳拆 4 步"混成一个数（两者都等于"均值相等"）；
* **解析失败（`plan_steps is None`）不许静默算成 0 步**：它会让"少拆"多一份，
  而那份根本不是"模型拆短了"，是**这一次调用没产出计划**。
"""

from __future__ import annotations

from tools.plan_decompose_eval import summarize


def _row(qid: str, gold: int, plan: int | None, error: str | None = None) -> dict:
    return {"qid": qid, "gold_hops": gold, "plan_steps": plan, "error": error, "steps": []}


def test_the_verdicts_are_per_question_not_two_means() -> None:
    """⭐ "少拆 / 相等 / 多拆"**逐题算**：一道 3→2 与一道 3→4 是"各偏一边"，
    而它们的**均值完全相同**。"""
    rows = [_row("a", 3, 2), _row("b", 3, 4), _row("c", 4, 4)]
    summary = summarize(rows)
    assert (summary["under"], summary["over"], summary["exact"]) == (1, 1, 1)
    assert summary["mean_gold"] == summary["mean_plan"]  # 3.33 vs 3.33 —— 均值看不出任何东西


def test_a_failed_parse_is_counted_apart_and_never_as_zero_steps() -> None:
    """⛔ 解析失败**不是 0 步**：把它算成 0 会让"少拆"多一份，
    而那一份根本不是"模型拆短了"，是**这一次调用没产出计划**。"""
    rows = [
        _row("a", 3, 2),
        _row("b", 3, None, error="InvalidPlanOutput: 字段不匹配"),
    ]
    summary = summarize(rows)
    assert summary["invalid"] == 1
    assert summary["under"] == 1  # 只有 a 算"少拆"
    assert summary["n"] == 2
    assert summary["mean_plan"] == 2.0  # 失败那条不进均值
    assert list(summary["errors"]) == ["InvalidPlanOutput: 字段不匹配"]


def test_everything_failed_is_not_a_zero_mean() -> None:
    """⚠ 全失败时均值是 `None` 而**不是 0.0**：`0.0` 会被读成"全都拆成 0 步"。"""
    summary = summarize([_row("a", 3, None, error="LlmUnavailable")])
    assert summary["mean_plan"] is None and summary["mean_gold"] is None
    assert summary["pairing"] == {}


def test_the_pairing_table_is_the_readable_form() -> None:
    rows = [_row("a", 4, 2), _row("b", 4, 2), _row("c", 3, 2), _row("d", 3, 3)]
    assert summarize(rows)["pairing"] == {(3, 2): 1, (3, 3): 1, (4, 2): 2}
