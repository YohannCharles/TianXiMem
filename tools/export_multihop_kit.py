"""把**链式多跳题集**打成压缩包，交给另一个项目用（`make export-multihop`）。

```bash
uv run python tools/export_multihop_kit.py            # → var/exports/musique-multihop-<日期>/
```

## 它导出什么、为什么是这个形状

另一个项目要「测多跳」，需要三样东西，少一样那份实验就不成立：

| 文件 | 是什么 | 为什么必需 |
| --- | --- | --- |
| `questions.jsonl` | 每题一行：**题面 + 金标答案 + 逐跳分解 + 该题的候选段落** | 题目、答案、以及"哪一段是证据" |
| `haystack-corpus.jsonl` | **一个共享语料池**（50 题的段落按 `title` 去重） | ⭐ **这是多跳能不能被测出来的关键** |
| `README.md` | 口径、许可、怎么用 | 免得别人拿它当"随便一份 QA" |

### ⚠ 为什么必须给那一整个池子，而不是每题那 20 段

MuSiQue 原生**每题只有 20 个候选段落**。而单条题自己的 20 段里**已经包含它全部 4 跳的
支持段落** ⇒ 只要把那 20 段喂进去，答案模型**永远看得见全部证据**，
**检索无事可做、链式依赖自动消失**——那样测出来的只是"多跑一轮会不会把答案搅坏"。
（本仓 2026-10-10 就是这么栽的，见 `eval/reports/coq-musique-20261011.md`。）

⇒ 必须把 N 道题的段落**并成一个池子**，让目标题的证据被**其余题的段落埋起来**。
本仓 25 题那版并出来是 **427 段**；50 题这版见下面打印的数字。

## 它保证什么（打印出来，不靠嘴说）

对每一道题逐条验：

1. `answerable == true`（⛔ **没有拒答题**）；
2. **严格链**：hop1 无 `#` 引用、其余每一跳都引用前面某一跳；
3. ⭐ **每一跳的答案（含末答）都能在该题的支持段落正文里找到**（**词边界**匹配，
   不是子串——`NES` 不该被 `geNESis` 命中）；
4. 末答非空；题面不泄中间跳的答案。

任一条不过就**点名报出来并以非零码退出**——这种包交出去比不交更糟。

## 许可

MuSiQue 的数据随包转发，**CC BY-NC 4.0（非商业）**，且**不得用于训练模型**。
`README.md` 里逐字写着这两条。
"""  # noqa: E501

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: 导出落点。⚠ 产物**含数据集原文** ⇒ 只进 `var/`（gitignored），⛔ 绝不进 git。
DEFAULT_OUT = Path("var/exports")

LICENSE_NOTE = """## 许可与使用限制

* 数据来自 **MuSiQue**（`musique_full_v1.0_dev.jsonl`，随本包转发的是**子集**）。
* 许可：**CC BY-NC 4.0** ⇒ **非商业**用途。若下游要商用，这份数据的评测结果**引用不了**。
* ⛔ **不得用于训练任何模型**（AML 数据条款 + MuSiQue 许可双重约束）。
* 本包**不含**任何模型、权重或本仓代码——只是一份题集 + 语料。
"""


def _raw_rows(bench_dir: Path, wanted: list[str]) -> dict[str, dict]:
    """只取**可答**那条（同一个 id 在文件里出现两次：可答 / 不可答变体）。"""
    from eval.datasets.musique import DATA_DIR, JSONL

    wanted_set = set(wanted)
    out: dict[str, dict] = {}
    for line in (bench_dir / DATA_DIR / JSONL).open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("answerable") and str(row["id"]) in wanted_set:
            out[str(row["id"])] = row
    missing = [qid for qid in wanted if qid not in out]
    if missing:
        raise SystemExit(f"✗ 这些 id 不在可答题里：{missing[:3]}（共 {len(missing)} 个）")
    return out


def verify(wanted: list[str], rows: dict[str, dict]) -> list[str]:
    """逐条验那四条保证，返回**问题清单**（空 = 全部通过）。"""
    from eval.datasets.musique import (
        _appears_as_a_word,
        _does_not_leak_the_intermediate_entity,
        _is_strict_chain,
    )

    problems: list[str] = []
    for qid in wanted:
        row = rows[qid]
        hops = row["question_decomposition"]
        if not row.get("answerable"):
            problems.append(f"{qid}: ⛔ 不可答题")
        if not _is_strict_chain(hops):
            problems.append(f"{qid}: 不是严格链")
        if not str(row["answer"]).strip():
            problems.append(f"{qid}: 末答为空")
        if not _does_not_leak_the_intermediate_entity(row):
            problems.append(f"{qid}: 题面泄了中间跳的答案")
        evidence = "\n".join(
            str(p["paragraph_text"]) for p in row["paragraphs"] if p.get("is_supporting")
        )
        if not evidence:
            problems.append(f"{qid}: 没有支持段落")
            continue
        for index, hop in enumerate(hops, 1):
            if not _appears_as_a_word(evidence, str(hop["answer"])):
                problems.append(f"{qid}: 第 {index} 跳的答案不在支持段落里（{hop['answer']!r}）")
        # ⚠ 末答通常**就是**最后一跳的答案 ⇒ 那一条上面已经查过，别报两遍
        #   （同一个事实报两次会让"问题有几处"这个数虚高）。
        if str(row["answer"]) != str(hops[-1]["answer"]) and not _appears_as_a_word(
            evidence, str(row["answer"])
        ):
            problems.append(f"{qid}: 末答不在支持段落里（{row['answer']!r}）")
    return problems


def render_readme(count: int, corpus_size: int, hop_counts: dict[int, int]) -> str:
    hops = " · ".join(f"{n} 跳 {c} 道" for n, c in sorted(hop_counts.items(), reverse=True))
    return f"""# 链式多跳题集（MuSiQue 子集）— {count} 道

**来源**：MuSiQue `musique_full_v1.0_dev.jsonl` 的子集，按「严格链」规则挑出。
**规模**：{count} 道（{hops}）· 共享语料池 **{corpus_size}** 段。

## 文件

| 文件 | 说明 |
| --- | --- |
| `questions.jsonl` | 每题一行：`question_id` / `question` / `answer` / `answer_aliases` / `question_decomposition`（逐跳问句 + 该跳答案）/ `paragraphs`（`title` / `idx` / `paragraph_text` / `is_supporting`） |
| `haystack-corpus.jsonl` | **一个共享语料池**：{count} 道题的段落按 `title` 去重后的全部段落（`title` / `text`） |

## ⚠ 用法：**必须把这些段落并成一个池子**

```text
把 haystack-corpus.jsonl 的每一行当作一条记忆喂进记忆系统（role=user），
50 道题共用这一个池子；然后逐题检索 + 作答。
```

⛔ **不要**只把目标题自己的 `paragraphs` 喂进去——那 20 段里**已经包含它全部跳的支持段落**，
答案模型会永远看得见全部证据 ⇒ **检索无事可做、链式依赖自动消失**，
测出来的只是"多跑一轮会不会把答案搅坏"。
（{corpus_size} 段 ≫ 单题 20 段，这一条是这份题集能测多跳的**前提**。）

## 每道题都保证（已逐条脚本校验）

1. **可答**：`answerable == true`——⛔ **没有一道拒答题**；
2. **严格链**：第 1 跳不含引用，其余每一跳都引用前面某一跳（不是"两个子问题最后合并"）；
3. ⭐ **答案齐全**：**每一跳的答案（含末答）都能在该题的支持段落正文里找到**
   （词边界匹配，不是子串）；
4. 题面**不泄露**中间跳的答案（否则就退化成"一次查找"）。

## 判分建议

金标是**短答案串**。建议：把生成答案与 `answer` / `answer_aliases` **归一化后比对**
（大小写、空白、标点）；⛔ 不建议用 LLM 当裁判——本仓实测模型裁判连"金标有几跳"都会数错。

{LICENSE_NOTE}
"""  # noqa: E501


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="导出链式多跳题集（含共享语料池）")
    parser.add_argument("--manifest", default="eval/coq-musique.jsonl")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--bench-dir", default=None)
    parser.add_argument("--limit", type=int, default=None, help="只导前 N 题（冒烟）")
    args = parser.parse_args(argv)

    from eval.datasets.musique import HAYSTACK_USER, load_musique_haystack

    bench_dir = Path(args.bench_dir or os.environ.get("TIANXIMEM_BENCHMARK_DIR", "dataset"))
    wanted = [
        json.loads(line)["question_id"]
        for line in Path(args.manifest).read_text(encoding="utf-8").split("\n")
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if args.limit is not None:
        wanted = wanted[: args.limit]

    rows = _raw_rows(bench_dir, wanted)
    problems = verify(wanted, rows)
    if problems:
        print("✗ 校验没过——**这份包不能交出去**：")
        for item in problems[:20]:
            print("   ", item)
        return 1

    # 语料池走**评测用的那条加载器**，保证导出的是同一个池子（不是另拼一份）
    sample = load_musique_haystack(bench_dir, wanted)
    corpus = [m.content for session in sample.sessions for m in session.messages]

    stamp = date.today().strftime("%Y%m%d")
    out_dir = Path(args.out) / f"musique-multihop-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "questions.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "question_id": qid,
                    "question": rows[qid]["question"],
                    "answer": rows[qid]["answer"],
                    "answer_aliases": rows[qid]["answer_aliases"],
                    "answerable": True,
                    "hops": len(rows[qid]["question_decomposition"]),
                    "question_decomposition": rows[qid]["question_decomposition"],
                    "paragraphs": rows[qid]["paragraphs"],
                },
                ensure_ascii=False,
            )
            + "\n"
            for qid in wanted
        ),
        encoding="utf-8",
    )
    (out_dir / "haystack-corpus.jsonl").write_text(
        "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in corpus),
        encoding="utf-8",
    )
    hop_counts: dict[int, int] = {}
    for qid in wanted:
        hop_counts[len(rows[qid]["question_decomposition"])] = (
            hop_counts.get(len(rows[qid]["question_decomposition"]), 0) + 1
        )
    (out_dir / "README.md").write_text(
        render_readme(len(wanted), len(corpus), hop_counts), encoding="utf-8"
    )

    archive = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in ("questions.jsonl", "haystack-corpus.jsonl", "README.md"):
            bundle.write(out_dir / name, arcname=f"{out_dir.name}/{name}")

    print(f"✅ 校验通过：{len(wanted)} 道题逐条过（可答 / 严格链 / 每跳答案在支持段落里 / 不泄题）")
    print(f"   跳数分布：{dict(sorted(hop_counts.items(), reverse=True))}")
    print(f"   共享语料池：{len(corpus)} 段（user_id 口径 {HAYSTACK_USER}）")
    print(f"   目录：{out_dir}")
    print(f"   压缩包：{archive}  （{archive.stat().st_size / 1024:.0f} KB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
