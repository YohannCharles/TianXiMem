"""T2 的**人那半边**：把标注表汇总成分布 + 判读（§13 / §附录 A）。

```text
tools/t2_retrieval_dump.py     机器：133 道题的纯 BM25 名次 → 待填表
      ↓  ← 人在这里填 label（三类 + ok）
本文件                          汇总分布 + 判读它决定什么
```

**它不产生任何数字之外的东西**：分布要落 [`../reports/ledger.md`](../reports/ledger.md)，
协议结论落 [`../../docs/experiments.md`](../../docs/experiments.md) 的结论列。

## 它决定什么（§13 / 附录 A）

| T2 分布的主因 | 下一步 |
| --- | --- |
| **措辞不同导致漏召** | **别名归并值得做**（v2 的实体层） |
| **召回但被截断 / 排序靠后** | **实体层解决的不是本项目的瓶颈**——v1 不做 |

⚠ 12 道拒答题（`_abs` 结尾）**必须单列**：LongMemEval 的拒答是**横切标记不是第 7 类**，
它们的行为与其他题不同（§12.3 第 6 条），混进分布会让三类占比失真。

## 用法

```bash
uv run python tools/t2_retrieval_dump.py   # 生成待填表
$EDITOR eval/experiments/t2_annotations.jsonl   # 人填 label
make t2                                     # 汇总
```
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Final

EXIT_OK: Final[int] = 0
EXIT_FAILED: Final[int] = 1
EXIT_PRECONDITION_FAILED: Final[int] = 2

DEFAULT_IN: Final[Path] = Path("eval/experiments/t2_annotations.jsonl")

#: 文档口径（[`../datasets/CLAUDE.md`](../datasets/CLAUDE.md)）：multi-session 133 道，
#: 其中 `_abs` 拒答 12 道。**对不上就说明筛错了题**——而"筛错了"与"分数变了"看起来一样。
EXPECTED_TOTAL: Final[int] = 133
EXPECTED_ABSTENTION: Final[int] = 12

#: `label` 的取值域（与 `tools/t2_retrieval_dump.py` 的 `LABELS` 同一套词表）。
LABELS: Final[tuple[str, ...]] = ("not_retrieved", "truncated", "ranked_low", "ok")

_VERDICTS: Final[dict[str, str]] = {
    "not_retrieved": "**漏召**：措辞不同导致检索根本够不着 ⇒ 别名归并（实体层）**值得做**",
    "truncated": "**召回但被 top-100 截断** ⇒ 瓶颈在预算/名次，**不是实体层**",
    "ranked_low": "**在里面但排序靠后** ⇒ 瓶颈在排序（§11 主线）",
    "ok": "这一题在纯 BM25 下就已经能拿到证据——它失败时的问题不在检索",
}


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict]) -> tuple[Counter, Counter, list[str]]:
    """返回 `(全部, 拒答, 未填的 qid)`。

    ⚠ **拒答题的分布单算**，但**也留在整体里**——两条都要看：只看整体会让 12 道的行为
    稀释掉；只看拒答会失去"全部题里主因是什么"这个结论本身。
    """
    total: Counter = Counter()
    abstention: Counter = Counter()
    unlabeled: list[str] = []
    for row in rows:
        label = str(row.get("label") or "").strip()
        if label not in LABELS:
            unlabeled.append(str(row["qid"]))
            continue
        total[label] += 1
        if row.get("is_abstention"):
            abstention[label] += 1
    return total, abstention, unlabeled


def _distribution(counter: Counter, n: int) -> str:
    if not n:
        return "（空）"
    return " · ".join(
        f"{label}={counter.get(label, 0)}（{counter.get(label, 0) / n:.0%}）" for label in LABELS
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="t2_cross_session.py", description="§13 的 T2 汇总与判读")
    parser.add_argument("--in", dest="in_path", default=str(DEFAULT_IN))
    args = parser.parse_args(argv)

    path = Path(args.in_path)
    if not path.exists():
        print(f"{path} 不存在——先跑 `uv run python tools/t2_retrieval_dump.py`", file=sys.stderr)
        return EXIT_PRECONDITION_FAILED

    rows = load_rows(path)
    if len(rows) != EXPECTED_TOTAL:
        print(
            f"⚠ 表里有 {len(rows)} 道题，文档口径是 {EXPECTED_TOTAL} 道 multi-session——"
            "筛错题与「分数变了」看起来一样，先查清楚再往下走",
            file=sys.stderr,
        )
        return EXIT_FAILED

    total, abstention, unlabeled = summarize(rows)
    n_abstention = sum(1 for row in rows if row.get("is_abstention"))
    if n_abstention != EXPECTED_ABSTENTION:
        print(
            f"⚠ 拒答题 {n_abstention} 道，文档口径是 {EXPECTED_ABSTENTION} 道"
            "（§12.3 第 6 条：拒答是横切标记，不是第 7 类）",
            file=sys.stderr,
        )

    print(f"T2 标注表：{path}（{len(rows)} 道，其中拒答 {n_abstention} 道）")
    if unlabeled:
        print(f"\n还差 {len(unlabeled)} 道没填（label 必须是 {' / '.join(LABELS)} 之一）：")
        print("  " + "、".join(unlabeled[:10]) + (" …" if len(unlabeled) > 10 else ""))
        print("\n**没填完就不出结论**：半张表的分布比没有分布更危险——它看起来像个结果。")
        return EXIT_FAILED

    print(f"\n全部：{_distribution(total, len(rows))}")
    print(f"拒答（单列）：{_distribution(abstention, n_abstention)}")

    print("\n判读（§13 / 附录 A）：")
    for label in LABELS:
        if total.get(label):
            print(f"  · {label}：{_VERDICTS[label]}")
    primary = max(LABELS, key=lambda label: total.get(label, 0))
    print(f"\n主因（占比最大）是 **{primary}** ⇒ {_VERDICTS[primary]}")

    print(
        "\n落点：分布进 `eval/reports/ledger.md`，结论写进 `docs/experiments.md` 的结论列"
        "（要求写清「因为 X 所以 Y」，不是打勾）。"
    )
    print("⚠ T2 **在 Step 5 切换后要重跑**（§12.1 R1 对冲 2）——它对外部模型依赖最小。")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
