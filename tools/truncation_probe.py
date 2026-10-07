"""把某轮的注入**从后往前截断**，看判分会不会翻正——一次**逐题因果探针**。

## 它回答什么

某一轮跑完、某几类题错了，问题可能是"**给太多了**"（长上下文稀释注意力），
也可能是"**给多少都答不出来**"（答案模型能力）。总体相关（对题 vs 错题的上下文长度）
**区分不了这两者**——实测在本项目的数据上它们几乎一样。

⇒ 这个探针换一种问法：**把同一道题的注入按比例截短，看判分在哪一档翻正**。

| 观察 | 读法 |
| --- | --- |
| 某题在 50% 处翻正 | **上下文确实在拖累它**（可以据此设计打包策略） |
| 所有档位都不翻正 | **不是上下文长度的问题**（该动的是答案模型或检索内容） |
| 翻正又翻回去 | 单点噪声，别当结论 |

## 用法

```bash
uv run --env-file .env python -m tools.truncation_probe --run ours-lme60 \\
    --category temporal-reasoning --category multi-session \\
    --fractions 0.75,0.5,0.25,0.1
```

⚠ **截断按行边界切**（注入文本里一行是一对的一行，行内不切）——
与 `packaging` 的"段是原子单位"同一条理由：切一半的句子会让模型读到缺一环的上下文。

⚠ 它**不打服务**（只重跑判分那两步）：题面与注入都从既有 run 的 `input.jsonl` 里读。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.datasets import benchmark_dir  # noqa: E402
from eval.harness import pipeline_for, run_judge  # noqa: E402
from tools.run_products import categories_of  # noqa: E402,F401 —— 转出给 reorder_probe

__all__ = ["categories_of", "load_run", "truncate_prefix"]


def load_run(run_id: str, reports_dir: Path) -> tuple[dict[str, dict], dict[str, bool]]:
    """读一轮的逐题产物：`(items, 判定)`。`items` 是**注入给判分的那一份**。"""
    items: dict[str, dict] = {}
    verdicts: dict[str, bool] = {}
    for sample_dir in sorted((reports_dir / "runs" / run_id).iterdir()):
        input_path, labels_path = sample_dir / "input.jsonl", sample_dir / "labels.jsonl"
        if input_path.exists():
            for line in input_path.read_text(encoding="utf-8").split("\n"):
                if line.strip():
                    row = json.loads(line)
                    items[row["id"]] = row
        if labels_path.exists():
            for line in labels_path.read_text(encoding="utf-8").split("\n"):
                if line.strip():
                    row = json.loads(line)
                    verdicts[row["id"]] = bool(row["is_correct"])
    return items, verdicts


def truncate_prefix(text: str, fraction: float) -> str:
    """保留**前 `fraction`** 的文本，**按行边界切**（行内不切）。"""
    if fraction >= 1.0:
        return text
    lines = text.split("\n")
    keep = max(1, int(len(lines) * fraction))
    return "\n".join(lines[:keep])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="truncation_probe", description=__doc__)
    parser.add_argument("--run", required=True, help="run_id")
    parser.add_argument("--dataset", default="longmemeval-s")
    parser.add_argument("--category", action="append", default=None, help="只看这几类（可重复）")
    parser.add_argument("--fractions", default="0.75,0.5,0.25,0.1")
    parser.add_argument("--limit", type=int, default=None, help="要与产生该 run 时一致")
    parser.add_argument("--spread", action="store_true", help="同上")
    parser.add_argument("--reports-dir", default="eval/reports")
    args = parser.parse_args(argv)

    reports = Path(args.reports_dir)
    items, verdicts = load_run(args.run, reports)
    categories = categories_of(args.dataset, limit=args.limit, spread=args.spread)
    wanted = set(args.category or [])
    failing = [
        qid
        for qid, ok in verdicts.items()
        if not ok and (not wanted or categories.get(qid) in wanted)
    ]
    fractions = [float(x) for x in args.fractions.split(",")]
    print(f"{args.run}：错题 {len(failing)} 道（过滤后）；档位 {fractions}\n")

    probe_dir = reports / "runs" / f"{args.run}__trunc"
    pipeline = pipeline_for(benchmark_dir(), args.dataset)
    flips: dict[str, list[float]] = {qid: [] for qid in failing}
    for fraction in fractions:
        batch = []
        for qid in failing:
            item = dict(items[qid])
            item["speaker_1_memories"] = truncate_prefix(
                str(items[qid]["speaker_1_memories"]), fraction
            )
            batch.append(item)
        results = run_judge(
            pipeline, batch, probe_dir / f"f{fraction}", dataset=args.dataset, max_tokens=256
        )
        got = {r.qid: r.is_correct for r in results}
        n_ok = sum(got.values())
        print(f"  截到 {fraction:>4.0%}：{n_ok}/{len(failing)} 判对   ", end="")
        print("翻正：" + (", ".join(q for q in failing if got.get(q)) or "（无）"))
        for qid in failing:
            if got.get(qid):
                flips[qid].append(fraction)

    print("\n各题的翻正点（空 = 所有档位都没翻正）：")
    for qid in failing:
        pts = flips[qid]
        print(f"   {qid}：{'、'.join(f'{p:.0%}' for p in pts) if pts else '——'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
