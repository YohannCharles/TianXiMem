#!/usr/bin/env python3
"""给官方采集的 search 补 **CL-Bench** 金标——`official-gold-extra-clbench.jsonl`。

## 为什么要有它

CL-Bench 没有单独的 `question` 字段：题目指令写在 user 消息（整份文档）的**末尾**，
金标是 `rubrics`（**严格全有全无**）。官方采集（`official-eval-questions.jsonl`，
2026-09-29 那一轮）里有一簇 search 的 query 与 `clbench.jsonl` 某条 user 消息
**逐字相同**（归一化后）——本脚本把它们接上 rubrics，
供 `tools/build_official_kit.py` 合成评测套件（它收 `official-gold-extra*.jsonl`）。

## 匹配口径（独立重写；与旧试验脚本对拍过条数）

- 归一化：NFKC + 引号/破折号统一 + 小写 + 空白折叠（两侧同一套）;
- 采集侧形态：全文 / 末段（最后一个空行之后）/ 最后一行 / 采集自带 `question` 字段。
  ⚠ 本轮的 238 条命中**全部来自全文形态**，其余三种是给下一轮采集留的护栏;
- 题库侧索引：每条 user 消息的**全文**;
- **定行（本脚本与旧脚本的唯一实质差异）**：同一段文本会同时挂在同 `context_id` 的
  多个 task 行里（前序轮当历史），但只有"该文本恰是本行**最后一条** user 消息"的那一行
  在回答这个问题——rubrics 判的就是那一轮的回复。实测 238 条的 owner 行**唯一**，
  而"文件序先到先得"有 2 条会定错行：
  seq 41270（应取 1822——rubrics 全是 VLV-R 阀门那道题，而不是 28 行更后的轮次）、
  seq 41283（应取 1730——rubrics 是"拒答并要求上传文档"，而不是 22 行更后的轮次）;
- 结果：**238 条**（56 个 user、141 个 clbench 行），条数与旧脚本一致。

## 产物字段

`gold_kind=rubric`、`gold_answers=[]`、`gold_rubric` 是 **rubrics 列表本体**
（不 JSON 编码成串——与 `official-gold-extra.jsonl` 里 BEAM 那条先例一致），
`gold_native={rubrics, task_id, context_id, system_prompt}`，
`gold_judging` 写清判分口径（`clb_pipeline.py#rubric_judge_prompt` 的逐字规则）。

## 来源与重跑

- 题库：HF `tencent/CL-bench`，revision `b28a5832a09b0d96c0cf4c22e90d7c60ede25b80`
  的 `CL-bench.jsonl`（本地名 `dataset/clbench/clbench.jsonl`，`make fetch-data` 取回）。
  URL: https://huggingface.co/datasets/tencent/CL-bench/resolve/
       b28a5832a09b0d96c0cf4c22e90d7c60ede25b80/CL-bench.jsonl
  sha256 = `d5fc88d4b2eea75c61dd40862021b6ae2fba26bd21b58e8c5e18377a763943be`
  （= HF 的 LFS oid，实测相等；本地未改）。
- 采集：`official-dataset-2026-09-29/official-eval-questions.jsonl`（本地导出，无 URL）。
  本轮 sha256 = `5d0bbf4aaacfb10bc466b3ef6a475e3e73b577b402707f0b81a80ecd1864e78f`；
  ⚠ 换一轮采集这份哈希就会变——重跑对照时先比它。
- 重跑：`uv run python tools/recover_clbench.py`
  （默认取 `benchmark_dir()` / `capture_dir()`；`--dataset-dir` / `--benchmark-dir` /
  `--capture` / `--out` 可改）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ⚠ 直接跑脚本时 `sys.path[0]` 是 `tools/`，仓库根不在上面；而 `eval` 不是已安装的包
#   （只有 `src/` 打了包）⇒ 补一条路径（与 `tools/recover_official_gold.py` 同一处置）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.datasets.clbench import CLBENCH_JSONL  # noqa: E402
from eval.datasets.registry import benchmark_dir, capture_dir  # noqa: E402
from eval.jsonl_io import jsonl_line  # noqa: E402

__all__ = ["main", "norm", "load_clbench", "match_record"]

#: 仓库根——只用来把 registry 的**相对**缺省（`benchmark_data`）钉死，与调用 CWD 无关。
_REPO_ROOT = Path(__file__).resolve().parents[1]

#: 引号 / 破折号统一（CL-Bench 文档里这两种变体很多，两侧必须同一套）。
_QUOTE_MAP = {
    0x2018: "'",
    0x2019: "'",
    0x201C: '"',
    0x201D: '"',
    0x2013: "-",
    0x2014: "-",
    0x00A0: " ",
}

#: 太短的等价串（"OK" 之流）不建索引——误配的代价远大于漏配。
_MIN_LEN = 8

#: 判分口径的**逐字引用**（`dataset/.upstream/aml/clb_pipeline.py#rubric_judge_prompt`）。
_STRICT_RULE = (
    "To receive a score of 1, the student's answer must perfectly satisfy every "
    "single requirement listed in the 【Rubrics】. If even one requirement is not "
    "fully met, the final score will be 0."
)


def norm(text: Any) -> str:
    """题面归一化：NFKC + 引号/破折号统一 + 小写 + 空白折叠。两侧必须同一套。"""
    out = unicodedata.normalize("NFKC", str(text or ""))
    out = out.translate(_QUOTE_MAP)
    return re.sub(r"\s+", " ", out).strip().lower()


def load_clbench(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[tuple[int, int]]]]:
    """读题库并建「归一化 user 消息全文 → [(行号, 消息号)]」索引。

    ⚠ **按 `"\\n"` 切**（`for line in open(...)` 即如此），不要 `splitlines()`：
    这份文件里真的有 `U+2028`，`splitlines()` 会把一条记录劈成两半。
    """
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    index: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, row in enumerate(rows):
        for j, message in enumerate(row.get("messages") or []):
            if message.get("role") != "user":
                continue
            key = norm(message.get("content"))
            if len(key) >= _MIN_LEN:
                index[key].append((i, j))
    return rows, index


def _last_user_index(row: dict[str, Any]) -> int:
    """本行最后一条 user 消息的下标（rubrics 判的就是它的回复）；没有则 -1。"""
    indexes = [j for j, m in enumerate(row.get("messages") or []) if m.get("role") == "user"]
    return indexes[-1] if indexes else -1


def query_variants(record: dict[str, Any]) -> list[tuple[str, str]]:
    """采集行 → [(形态名, 归一化文本)]：全文 / 末段 / 最后一行 / 采集自带 question。"""
    raw = str(record.get("query") or "")
    variants: list[tuple[str, str]] = [("full", norm(raw))]
    chunks = [c.strip() for c in re.split(r"\n\s*\n", raw) if c.strip()]
    if len(chunks) > 1:
        variants.append(("tail_after_last_blank", norm(chunks[-1])))
    lines = [line for line in raw.splitlines() if line.strip()]
    if lines:
        variants.append(("last_line", norm(lines[-1])))
    question = norm(record.get("question"))
    if question and question != norm(raw):
        variants.append(("capture_question_field", question))
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for name, text in variants:
        if text and text not in seen:
            seen.add(text)
            out.append((name, text))
    return out


def match_record(
    record: dict[str, Any], index: dict[str, list[tuple[int, int]]], rows: list[dict[str, Any]]
) -> tuple[int, int, str] | None:
    """采集行 → (行号, 消息号, 命中形态)；不中返回 None。

    先按形态顺序收集去重候选，再**定行**：能当 owner 的候选（消息恰是该行最后一条
    user 消息）唯一时取它，否则退回第一个候选（调用方会把这个回退计数报出来）。
    """
    candidates: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for variant, text in query_variants(record):
        for i, j in index.get(text, []):
            if (i, j) in seen:
                continue
            seen.add((i, j))
            candidates.append((i, j, variant))
    if not candidates:
        return None
    owners = [c for c in candidates if c[1] == _last_user_index(rows[c[0]])]
    return owners[0] if len(owners) == 1 else candidates[0]


def build_row(
    record: dict[str, Any], row: dict[str, Any], row_index: int, msg_index: int, file_name: str
) -> dict[str, Any]:
    """一行产物：采集原文 + 该 clbench 行的 rubric 金标与判分口径。"""
    meta = row.get("metadata") or {}
    rubrics = list(row.get("rubrics") or [])
    messages = row.get("messages") or []
    system_prompt = str((messages[0] if messages else {}).get("content") or "")
    gold_judging = (
        f"CL-Bench：采集 query 与 {file_name} 第 {row_index + 1} 行的第 {msg_index} 条 "
        f"user 消息逐字相同（归一化后），且它正是该行最后一条 user 消息——rubrics 判的"
        f"就是这条查询的回复。task_id={meta.get('task_id')}，"
        f"context_id={meta.get('context_id')}，"
        f"类别={meta.get('context_category')}/{meta.get('sub_category')}，"
        f"共 {len(rubrics)} 条 rubric。官方判分口径 = "
        f"dataset/.upstream/aml/clb_pipeline.py#rubric_judge_prompt：严格全有全无，逐字规则"
        f"「{_STRICT_RULE}」；另按逐条 requirement 状态给 requirement_ratio 作部分分；"
        f"API/JSON 失败一律记 0。作答 prompt = clb_pipeline.py#build_answer_prompt，"
        f"需要 system_prompt / qa_type / options（system_prompt 在 gold_native；"
        f"qa_type/options 取采集行自身字段，本批 task={record.get('task')!r} / "
        f"options={len(record.get('options') or [])} 项）。"
    )
    return {
        "seq": record.get("seq"),
        "ts": record.get("ts"),
        "user_id": record.get("user_id"),
        "question": record.get("question"),
        "source_dataset": "clbench",
        "source_locator": {
            "file": file_name,
            "line": row_index + 1,
            "row_index": row_index,
            "message_index": msg_index,
        },
        "gold_kind": "rubric",
        "gold_answers": [],
        # rubrics 是**列表本体**（不 JSON 编码成串）：判分逐条吃它，BEAM 那条先例同型。
        "gold_rubric": rubrics,
        "gold_native": {
            "rubrics": rubrics,
            "task_id": meta.get("task_id"),
            "context_id": meta.get("context_id"),
            "system_prompt": system_prompt,
        },
        "gold_judging": gold_judging,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset-dir", default=None, help="采集导出目录（默认 TIANXIMEM_CAPTURE_DIR）"
    )
    parser.add_argument(
        "--capture",
        default="official-eval-questions.jsonl",
        help="采集文件名（它里面的 search 行）",
    )
    parser.add_argument(
        "--benchmark-dir", default=None, help="题库目录（默认 TIANXIMEM_BENCHMARK_DIR）"
    )
    parser.add_argument(
        "--out", default="official-gold-extra-clbench.jsonl", help="输出文件名（写在采集目录下）"
    )
    parser.add_argument("--offline", action="store_true", help="禁止下载缺失的题库材料")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else capture_dir()
    bench = Path(args.benchmark_dir) if args.benchmark_dir else benchmark_dir()
    if not bench.is_absolute():  # registry 的缺省是相对路径 ⇒ 钉到仓库根，与调用 CWD 无关
        bench = _REPO_ROOT / bench
    capture_path = dataset_dir / args.capture
    from eval.datasets.layout import archive_file
    from eval.datasets.prepare import ensure_dataset

    ensure_dataset("clbench", bench, offline=args.offline)
    clbench_path = archive_file(bench, CLBENCH_JSONL)
    for path in (capture_path, clbench_path):
        if not path.exists():
            print(f"✗ 找不到 {path}", file=sys.stderr)
            return 1

    rows, index = load_clbench(clbench_path)
    print(f"题库 {clbench_path}：{len(rows)} 行，user 消息索引 {len(index)} 个键")

    out_rows: list[dict[str, Any]] = []
    users: set[str] = set()
    matched_rows: set[int] = set()
    variants: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    fallback = 0
    for line in capture_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        hit = match_record(record, index, rows)
        if hit is None:
            continue
        i, j, variant = hit
        variants[variant] += 1
        if j != _last_user_index(rows[i]):
            fallback += 1
            print(f"  ⚠ seq {record.get('seq')}: owner 不唯一，回退候选 row {i}", file=sys.stderr)
        users.add(str(record.get("user_id")))
        matched_rows.add(i)
        categories[str((rows[i].get("metadata") or {}).get("context_category"))] += 1
        out_rows.append(build_row(record, rows[i], i, j, clbench_path.name))

    out_path = dataset_dir / args.out
    with out_path.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(jsonl_line(row))

    usable_sys = sum(1 for row in out_rows if str(row["gold_native"]["system_prompt"]).strip())
    print(f"命中 {len(out_rows)} 条 search → {out_path}")
    print(f"  distinct user {len(users)} / clbench 行 {len(matched_rows)}")
    print(f"  命中形态 {dict(variants)}；回退定行 {fallback} 条")
    print(f"  可用 system_prompt：按 search {usable_sys}/{len(out_rows)}")
    print(f"  context_category（按 search）{dict(categories)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
