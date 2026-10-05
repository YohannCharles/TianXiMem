"""答案 prompt 的 A/B —— **只重答拒答的那些题，其余不动**。

```bash
uv run --env-file .env python tools/ab_answer_prompt.py --run-id tr-splitfix --dataset tempreason
```

## 它为什么存在

「这份数据集该用哪份答案 prompt」**已经连续三次是低分的直接原因**：

| 数据集 | 通用 `ANSWER_PROMPT` 的后果 | 换掉之后 |
| --- | --- | ---: |
| `memtrapbench` | 250 题里 **160 道逐字拒答、通过 0** | 0.144 → **0.424** |
| `mquake` | 冲突就拒答（旧值与替换值并存）+ 禁掉它**设计上要求**的世界知识 | 0.400 → **0.540** |
| `tempreason` | 93/332 逐字拒答（答案要落在**区间**上，条款却按字面匹配） | 0.638 → **0.795** |

而**通用那份不能改**：它还被 corporatebench / medmemorybench 用着，动它会连带动那几份的
基线（[`../eval/harness/extra_pipeline.py`](../eval/harness/extra_pipeline.py) 里
`MQUAKE_ANSWER_PROMPT` 的 docstring 已经把这条写死了）。

⇒ 每次都要"新加一份专用 prompt"，而**决定用哪一份不能靠猜**——2026-10-03 的 mquake
就是四个候选同条件跑出来的（B0 不含拒答条款改动 0 条新对、B1 66 条、B2 88 条、B3 93 条）。

## 方法（照 mquake 那次）

```text
输入：某个已跑完的 run 的产物（`input.jsonl` 里的 `retrieved_context` + `question`）
取子集：**当前逐字拒答的那些题**（其余题的答案与 prompt 无关，重答是白花时间）
对每个候选：同一份上下文、同一道题，只换 prompt 重答一次
判分：用该数据集**自己的判分口径**（纯函数那几条不发请求）
输出：每个候选"新答对多少条" + 折算后的总分
```

⚠ **新答对的题是加在旧分上的**：旧分里那几道是错的，现在对了 ⇒ 新总分 =
`(旧正确数 + 新对) / 总题数`。**不改判分口径**（改口径 = 改基线，那是另一件事）。

## 三条纪律

1. **串行**：网关在 Cloudflare 后面（源站 ~125 秒 ⇒ 524）。跑之前确认没有别的链在跑。
2. **样本要够**：几十条拒答题上的差可能是噪声。mquake 那次是 201 条、temperason 是 93 条。
3. **选了谁就把谁搬进 `extra_pipeline.py`** 并加一条数据集分派——**别把候选留在这里**。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根（`eval.*` 要它）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "harness"))  # `api_config`

from eval.harness import extra_pipeline as ep  # noqa: E402

#: `input.jsonl` 里给模型的那份记忆叫什么（逐数据集不同，见 `eval/CLAUDE.md` 的契约表）。
MEMORY_FIELD: Final[dict[str, str]] = {
    "tempreason": "retrieved_context",
    "mquake-remastered": "retrieved_context",
    "memtrapbench": "retrieved_context",
    "medmemorybench": "retrieved_context",
    "corporatebench": "retrieved_context",
}

#: 判定"这句话是不是拒答"——**与 prompt 里让模型回的那句一致**（改了 prompt 记得一起改）。
REFUSALS: Final[tuple[str, ...]] = ("cannot determine from the memories",)


def _read(run_dir: Path, filename: str) -> dict[tuple[str, str], dict]:
    """键是 `(批目录, 题号)`——**不能只用题号**（见 [`diagnose_run.py`](./diagnose_run.py)）。"""
    rows: dict[tuple[str, str], dict] = {}
    for path in sorted(run_dir.glob(f"*/{filename}")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[(path.parent.name, str(row["id"]))] = row
    return rows


def _judge(dataset: str, item: dict, generated: str) -> bool:
    """用数据集**自己的**判分口径。⚠ 只覆盖纯函数那几条——其余的要发请求，不在本工具范围。"""
    gold = item.get("gold_answer")
    if dataset == "tempreason":
        return bool(ep.judge_tempreason(generated, gold)[0])
    raise SystemExit(f"{dataset} 的判分不是纯函数，本工具还没接——见模块 docstring 的纪律 3")


#: 候选 prompt。**每个数据集一组**，名字要能自解释（写进报告里）。
#:
#: ⚠ 它们**不是运行时配置**——选完就把赢家搬进 `extra_pipeline.py` 当 `<数据集>_ANSWER_PROMPT`，
#: 这里只留"当时比过哪几个"的记录。
CANDIDATES: Final[dict[str, dict[str, str]]] = {
    "tempreason": {
        # B1：只删通用那份的**拒答条款**。
        # 通用那份的第 2 条是"If the memories do not contain the answer, reply exactly:
        # Cannot determine from the memories."——而 TempReason 的答案**从来不在记忆里
        # 字面出现**：记忆给的是"X plays for Y from Jan, 1991 to Jan, 1992"，
        # 问题问"May, 1991 他在哪支队"，要**读完区间再落到点上**。
        # 9B 的模型把"contain"读成字面匹配 ⇒ 一律拒答。
        "B1 只删拒答条款": (
            "You are an assistant that answers a question using the memories provided below.\n"
            "\n"
            "Rules:\n"
            "1. Use only the information in the memories. Do not use outside knowledge.\n"
            "2. Answer concisely — one short sentence or the value itself.\n"
            "\n"
            "Memories:\n"
            "{memories}\n"
            "\n"
            "Question: {question}\n"
            "\n"
            "Answer:"
        ),
        # B2：B1 + 明说"区间里的每个月都成立"。**单题探针上它反而答错了**（Real Madrid），
        # 所以它进 A/B 是为了**在 93 条上验一次**，不是因为它看起来更好。
        "B2 B1+区间说明": (
            "You are an assistant that answers a question using the memories provided below.\n"
            "\n"
            "The memories state facts as **time intervals**: a statement like\n"
            '"X plays for Y from Jan, 1991 to Jan, 1992" means it held for that whole span.\n'
            "\n"
            "Rules:\n"
            "1. Use the memories' time intervals to decide what was true **at the asked date**.\n"
            "   Do not use outside knowledge.\n"
            "2. If the memories do not contain the answer, reply exactly: Cannot determine from "
            "the memories.\n"
            "3. Answer concisely — one short sentence or the value itself.\n"
            "\n"
            "Memories:\n"
            "{memories}\n"
            "\n"
            "Question: {question}\n"
            "\n"
            "Answer:"
        ),
        # B3：B1 + 必须给一个答案（不许弃权）。测"拒答条款删干净之后，模型会不会靠瞎猜拿分"。
        "B3 B1+必须作答": (
            "You are an assistant that answers a question using the memories provided below.\n"
            "\n"
            "Rules:\n"
            "1. Use only the information in the memories. Do not use outside knowledge.\n"
            "2. The question is always answerable from the memories — commit to the best\n"
            "   answer they support, even if you must reason across time intervals.\n"
            "3. Answer concisely — one short sentence or the value itself.\n"
            "\n"
            "Memories:\n"
            "{memories}\n"
            "\n"
            "Question: {question}\n"
            "\n"
            "Answer:"
        ),
    },
}


def run_ab(run_id: str, dataset: str, reports_dir: Path, sample: int | None) -> int:
    run_dir = reports_dir / "runs" / run_id
    if not run_dir.is_dir():
        print(f"⛔ 没有这个 run 目录：{run_dir}", file=sys.stderr)
        return 2
    field = MEMORY_FIELD.get(dataset)
    candidates = CANDIDATES.get(dataset)
    if not field or not candidates:
        print(f"⛔ 还没为 {dataset} 准备候选（见本文件 CANDIDATES）", file=sys.stderr)
        return 2

    inputs = _read(run_dir, "input.jsonl")
    answers = _read(run_dir, "answers.jsonl")
    labels = _read(run_dir, "labels.jsonl")

    total = correct = 0
    refused: list[tuple[tuple[str, str], dict]] = []
    for key, item in inputs.items():
        label = labels.get(key)
        if label is None or "ERROR" in str(label.get("label")).upper():
            continue
        total += 1
        correct += int(bool(label.get("is_correct")))
        text = str((answers.get(key) or {}).get("generated_answer") or "")
        if any(phrase in text.lower() for phrase in REFUSALS):
            refused.append((key, item))
    if sample:
        refused = refused[:sample]
    base = correct / total if total else 0.0
    print(f"\n═══ {run_id} 的 A/B（{dataset}）═══")
    print(f"  判过的 {total} 题、对 {correct} ⇒ 旧总分 **{base:.4f}**")
    print(f"  逐字拒答的 **{len(refused)}** 题要被重答 × {len(candidates)} 个候选")

    judge = ep._config("answer")
    report: dict[str, Any] = {}
    for name, template in candidates.items():
        won = 0
        for _key, item in refused:
            prompt = template.format(
                memories=str(item.get(field) or ""), question=str(item.get("question") or "")
            )
            try:
                out = ep._chat(judge[0], judge[1], judge[2], prompt, max_tokens=256, timeout=180.0)
            except Exception as exc:  # noqa: BLE001 —— 单条失败不打死整轮，与 `_call_judge` 同一取向
                out = f"AB CALL FAILED: {type(exc).__name__}: {exc}"
            won += int(_judge(dataset, item, out.strip()))
        score = (correct + won) / total if total else 0.0
        report[name] = (won, score)
        print(
            f"    {name:22s} 新答对 **{won:4d}/{len(refused)}**  ⇒ 总分 {score:.4f}"
            f"  （{score - base:+.4f}）"
        )

    if report:
        best = max(report, key=lambda key: report[key][1])
        print(f"\n  ⇒ 赢家：**{best}**（{report[best][1]:.4f}）")
        print(
            "  ⚠ 选完把它搬进 `eval/harness/extra_pipeline.py` 当 `<数据集>_ANSWER_PROMPT`"
            " 并按 `dataset` 分派——**别让候选留在这里当运行时配置**。"
        )
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="答案 prompt 的 A/B（只重答拒答题）")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--reports-dir", default="eval/reports")
    parser.add_argument(
        "--sample", type=int, default=None, help="只试前 N 条（冒烟用，别用来下结论）"
    )
    args = parser.parse_args(argv)
    return run_ab(args.run_id, args.dataset, Path(args.reports_dir), args.sample)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
