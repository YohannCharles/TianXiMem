"""错误案例归因 —— 读一轮 run 的逐题产物，回答"**错在哪一环**"。

```text
eval/reports/runs/<run_id>/<user_id>/
    input.jsonl     题目 + gold + **实际注入模型的那段记忆**（§11.3 的 content 拼起来的东西）
    answers.jsonl   模型生成的答案
    labels.jsonl    裁判判语（CORRECT / WRONG + 一句话理由）
```

## 它把失败分到三类，对应三条不同的修法

| 桶 | 判据（启发式，**不是判决**） | 该动哪儿 |
| --- | --- | --- |
| **证据不在上下文里** | gold 的内容词几乎不在记忆里 | 检索 / 打包（§7、§10、§11） |
| **证据在、但没答对** | gold 的词在记忆里，答案却错/漏 | 答案侧：注入形状、顺序（§11.3） |
| **答对了但被判错** | 答案与 gold 语义一致、判语却挑别处 | 裁判契约 / 渲染（§11.3） |

⚠ **它是启发式**：词面命中不等于"模型真读到了"，判语也可能误判。它的用途是**把 138 题分成几堆**，
好让人集中看那几堆，而不是替代人读。**结论要落到具体案例上。**

## 用法

```bash
uv run python -m tools.error_report --run smoke-locomo-1
uv run python -m tools.error_report --run a3-off --category 2 --sample 8
```

（`tools/` 可以 import `tianxi_am` 与 `eval.datasets`；
本脚本只读 JSONL，不碰网络、不碰 Qdrant。）

⚠ **用 `-m` 跑**（`python -m tools.error_report`）：直接给路径时仓库根不在 `sys.path` 上，
`import eval.datasets` 会失败——那几个 CLI 工具只 import `tianxi_am`
（editable 安装）所以没这个问题。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Final

EXIT_OK: Final[int] = 0
EXIT_PRECONDITION_FAILED: Final[int] = 2

RUNS_DIR: Final[Path] = Path("eval/reports/runs")

#: LoCoMo 的分类 ID（**实测映射**，见 `eval/datasets/CLAUDE.md`——与论文顺序不同）。
CATEGORY_NAMES: Final[dict[str, str]] = {
    "1": "multi-hop",
    "2": "temporal",
    "3": "open-domain",
    "4": "single-hop",
    "5": "adversarial",
}

#: 判语里出现这些词 ⇒ 失败**可能**来自时间表述（§11.3 的两条规则）。
_TIME_HINTS: Final[tuple[str, ...]] = (
    "granularity",
    "relative",
    "absolute",
    "date",
    "day",
    "month",
    "year",
    "time",
)

#: 判语里出现这些 ⇒ 记忆里**没有**那件事（检索/打包侧）。
_ABSENT_HINTS: Final[tuple[str, ...]] = (
    "no information",
    "does not contain",
    "not mentioned",
    "no mention",
    "cannot be determined",
    "no evidence",
)

_WORD = re.compile(r"[a-z0-9]+")
_STOP: Final[frozenset[str]] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "this",
        "that",
        "he",
        "she",
        "they",
        "them",
        "his",
        "her",
        "their",
        "i",
        "you",
        "we",
        "my",
        "your",
        "our",
        "as",
        "from",
        "not",
        "no",
        "do",
        "did",
        "does",
        "have",
        "has",
        "had",
        "what",
        "when",
        "where",
        "who",
        "how",
        "why",
        "which",
    }
)


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOP}


def _gold_strings(gold: object) -> list[str]:
    """LoCoMo 的 gold 是 `list[str]`（少数情况下是裸字符串 / 数字）。"""
    if isinstance(gold, list):
        return [str(g) for g in gold]
    return [str(gold)]


def _evidence_in_context(gold: object, memories: str) -> tuple[bool, float]:
    """gold 的内容词有多大比例出现在注入的记忆里。

    ⚠ 词面命中是**下界**：同义改写会让它漏判（把"在上下文里"误判成"不在"）。
    所以只有**很低**的比例才值得判成"证据不在上下文里"。
    """
    mem_words = _words(memories)
    gold_words = set()
    for text in _gold_strings(gold):
        gold_words |= _words(text)
    if not gold_words:
        return True, 1.0
    hit = len(gold_words & mem_words) / len(gold_words)
    return hit >= 0.5, hit


def _judge_hints(judge_response: str) -> list[str]:
    text = judge_response.lower()
    hints = []
    if any(word in text for word in _ABSENT_HINTS):
        hints.append("判语说记忆里没有")
    if any(word in text for word in _TIME_HINTS):
        hints.append("判语提到时间")
    return hints


def _category_map(dataset: str) -> dict[str, str]:
    """`qid → category`——**归档 pipeline 的产物里没有分类**，得回数据集取。

    （`labels.jsonl` 只写 id/label/is_correct/judge_response；`input.jsonl` 的字段由
    `judge.build_input_items` 决定，也没有分类。所以这里回加载层要。）
    """
    from eval.datasets import benchmark_dir, load_locomo, load_longmemeval

    bench = benchmark_dir()
    samples = load_locomo(bench) if dataset == "locomo-refined" else load_longmemeval(bench)
    return {q.qid: str(q.category) for sample in samples for q in sample.questions}


def load(run_id: str, *, dataset: str = "locomo-refined") -> list[dict]:
    """把一轮 run 的三份产物按 `id` 拼起来（目录里每个 user 一份）。"""
    root = RUNS_DIR / run_id
    categories = _category_map(dataset)
    rows: list[dict] = []
    for user_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        inputs = {r["id"]: r for r in _rows(user_dir / "input.jsonl")}
        answers = {r["id"]: r for r in _rows(user_dir / "answers.jsonl")}
        for label_row in _rows(user_dir / "labels.jsonl"):
            item = inputs.get(label_row["id"], {})
            rows.append(
                {
                    "qid": label_row["id"],
                    "user": user_dir.name,
                    "is_correct": bool(label_row["is_correct"]),
                    "judge_response": label_row.get("judge_response", ""),
                    "question": item.get("question", ""),
                    "gold": item.get("gold_answer"),
                    "memory_field": item.get("speaker_1_memories", ""),
                    "answer": answers.get(label_row["id"], {}).get("generated_answer", ""),
                    "category": str(label_row.get("category") or "")
                    or categories.get(label_row["id"], "?"),
                }
            )
    return rows


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def classify(row: dict) -> tuple[str, float]:
    """返回 `(桶, gold 词命中率)`。"""
    present, ratio = _evidence_in_context(row["gold"], row["memory_field"])
    if not present:
        return "证据不在上下文里", ratio
    hints = _judge_hints(row["judge_response"])
    if any("时间" in h for h in hints) and _mentions_time(row["gold"]):
        return "答了但时间表述被挑", ratio
    return "证据在、答案没答对", ratio


def _mentions_time(gold: object) -> bool:
    text = " ".join(_gold_strings(gold)).lower()
    return any(word in text for word in _TIME_HINTS)


def report(run_id: str, *, category: str | None, sample: int, dataset: str) -> int:
    root = RUNS_DIR / run_id
    if not root.exists():
        print(f"找不到 {root}——run_id 对吗？", file=sys.stderr)
        return EXIT_PRECONDITION_FAILED
    rows = load(run_id, dataset=dataset)
    if category:
        rows = [r for r in rows if r["category"] == category]
    if not rows:
        print("没有可分析的题（分类过滤后为空？）", file=sys.stderr)
        return EXIT_PRECONDITION_FAILED

    wrong = [r for r in rows if not r["is_correct"]]
    print(f"{run_id}：{len(rows)} 题，错 {len(wrong)} 题（{len(wrong) / len(rows):.1%}）")
    if category:
        print(f"（只看分类 {category} = {CATEGORY_NAMES.get(category, '?')}）")

    by_category: dict[str, list[int]] = {}
    for row in rows:
        bucket = by_category.setdefault(row["category"], [0, 0])
        bucket[0] += 1
        bucket[1] += 0 if row["is_correct"] else 1
    for cat, (total, bad) in sorted(by_category.items()):
        name = CATEGORY_NAMES.get(cat, "?")
        print(f"  分类 {cat}（{name}）：{total - bad}/{total} 正确")

    buckets: Counter[str] = Counter()
    hints: Counter[str] = Counter()
    for row in wrong:
        bucket, ratio = classify(row)
        row["_bucket"] = bucket
        row["_ratio"] = ratio
        buckets[bucket] += 1
        for hint in _judge_hints(row["judge_response"]):
            hints[hint] += 1

    print("\n失败分桶（启发式）：")
    for bucket, count in buckets.most_common():
        print(f"  {bucket}：{count} 题")
    print("\n判语关键词：")
    for hint, count in hints.most_common():
        print(f"  {hint}：{count} 题")

    for shown, row in enumerate(sorted(wrong, key=lambda r: r["_ratio"])):
        if shown >= sample:
            break
        print(
            f"\n── [{row['category']}] {row['qid']}"
            f"（{row['_bucket']}，gold 词命中 {row['_ratio']:.0%}）"
        )
        print(f"  问：{row['question'][:160]}")
        print(f"  gold：{json.dumps(row['gold'], ensure_ascii=False)[:200]}")
        print(f"  答：{row['answer'][:200]}")
        print(f"  判：{row['judge_response'][:260]}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="error_report.py", description="错误案例归因")
    parser.add_argument("--run", required=True, help="run_id（eval/reports/runs/<run_id>/）")
    parser.add_argument("--category", default=None, help="只看某个分类（如 2 = temporal）")
    parser.add_argument("--sample", type=int, default=6, help="列出几条失败案例")
    parser.add_argument(
        "--dataset", default="locomo-refined", choices=["locomo-refined", "longmemeval-s"]
    )
    args = parser.parse_args(argv)
    return report(args.run, category=args.category, sample=args.sample, dataset=args.dataset)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
