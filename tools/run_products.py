"""读 `eval/reports/runs/<run_id>/<样本目录>/<产物>.jsonl` 的共用入口。

## ⚠ 键是 `(样本目录, 题号)`，**不能只用题号**

产物按 `sample.user_id` 分目录，而题号只在**一个样本内**保证唯一。
`medmemorybench` 的题号**跨 persona 全局撞**（`session_10_eem_1` 在 20 个 persona 上各出现
一次）⇒ 按题号配对的读法把 172 行压成 20 个键，报出来的"答案只有 20/172 题、跑批中断？"
**是工具自己造的**。**别假定题号全局唯一**——这里以目录为准。

## 为什么单独一个模块

`diagnose_run` 与 `ab_answer_prompt` 各写过一份，两份只在**题号取哪个键**上不同
（归档产物用 `id`，早期的一批用 `idx`）。那份差别做成显式参数，
**`(目录, 题号)` 这条判据只留一份**——它是被实测事故换来的。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from eval.jsonl_io import read_jsonl


def _by_id(row: dict) -> str:
    return str(row["id"])


def rows_by_sample(
    run_dir: Path,
    filename: str,
    *,
    key: Callable[[dict], str] = _by_id,
) -> dict[tuple[str, str], dict]:
    """读某个产物文件的两层键：`(样本目录名, 题号)`。目录下没有该文件时返回空表。"""
    rows: dict[tuple[str, str], dict] = {}
    for path in sorted(run_dir.glob(f"*/{filename}")):
        for row in read_jsonl(path):
            rows[(path.parent.name, key(row))] = row
    return rows


def by_id_or_idx(row: dict) -> str:
    """题号取 `id`，缺了退回 `idx`——**早期那一批产物的形状**（见模块 docstring）。"""
    return str(row.get("id") or row.get("idx"))


def categories_of(
    dataset: str, *, limit: int | None = None, spread: bool = False
) -> dict[str, str]:
    """`qid → 类`——工具**不猜类别**，一律回数据集问。

    （归档 pipeline 的产物里没有分类：`labels.jsonl` 只写
    id/label/is_correct/judge_response，`input.jsonl` 的字段由 `judge.build_input_items`
    决定，也没有分类。所以这里回加载层要。）

    ⚠ `limit` / `spread` 是给 LongMemEval 的：它的文件**按 `question_type` 分块**，
    部分跑要 `--spread` 才跨类。**传不传都不影响这张映射表的正确性**（题号全局唯一，
    全量加载得到的是超集），传它们只是让加载量和当初那一轮对齐。
    """
    from eval.datasets import benchmark_dir, load_locomo, load_longmemeval

    bench = benchmark_dir()
    if dataset == "longmemeval-s":
        samples = load_longmemeval(bench, limit=limit, spread=spread)
    else:
        samples = load_locomo(bench)
    return {q.qid: str(q.category) for s in samples for q in s.questions}
