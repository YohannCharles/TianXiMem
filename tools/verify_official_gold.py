#!/usr/bin/env python3
"""只读审计 `official-eval-kit.jsonl`：**每条金标能不能追回它的来源行，且被忠实搬运**。

## 它证明什么、不证明什么

| 能证明 | 不能证明 |
| --- | --- |
| 每个可评分行的金标来自 `official-eval-kit` 声明的来源（`official-gold-extra*` 或 `public_gold`） | 公开题库/上游 rubric 本身没有错 |
| 搬进套件的 `gold_answers` / `gold_rubric` / `gold_native` 与来源行**逐字段相同** | 模型按这个金标判分时一定判对 |
| 每行至少有一种金标，`judge_kind` 与金标形状一致 | LLM 裁判的语义判断与人工判断一致 |
| 每个家族抽 N 条给出 `source_locator`，供人工复核 | 那 8,783 条**全部**已经人工读过 |

## 用法

```bash
uv run python tools/verify_official_gold.py
uv run python tools/verify_official_gold.py --sample-per-family 10 \
  --out eval/reports/runs/official-gold-audit.md
```

⛔ 它只读采集目录，不发任何 Add/Search、不调模型、不改套件。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.build_official_kit import _extra_payload, _public_gold_payload  # noqa: E402

KIT = "official-eval-kit.jsonl"
QUESTIONS = "official-eval-questions.jsonl"
EXTRA_FILES = (
    "official-gold-extra.jsonl",
    "official-gold-extra-docpp.jsonl",
    "official-gold-extra-clbench.jsonl",
    "official-gold-extra-halumem.jsonl",
    "official-gold-extra-corpusqa.jsonl",
)

#: 这些家族的 `gold_native` 是裁判分支要的原生载荷；缺了会**静默走错分支**。
_NEEDS_NATIVE = {
    "beam",
    "clbench",
    "corporatebench",
    "docpp",
    "feverous",
    "halumem",
    "hybridqa",
    "medmemorybench",
    "memtrapbench",
    "musique",
}
_RUBRIC_FAMILIES = {"beam", "clbench", "docpp", "memtrapbench"}
_EXACT_FAMILIES = {
    "corporatebench",
    "feverous",
    "halumem",
    "hybridqa",
    "medmemorybench",
    "mquake-remastered",
    "musique",
    "personamem-v2",
    "scriptmem",
    "locomo-refined",
}

#: 已声明、**不是**上游原样口径的地方（写在这里是为了不被当成"已经逐条核对过"）。
KNOWN_DEVIATIONS = {
    "personamem-v2": (
        "**走官方 narrow 口径**：`preference` 从 `benchmark.csv` 按 query 逐字匹配补上"
        "（114/114 命中），prompt 与 `extract_boxed_score` 直接用归档那份；"
        "**二值化阈值 0.5 是我们定的**（官方只回 0–1 连续分）。"
    ),
    "feverous": "只判 label；不判证据 id 的 F1（答案 prompt 不要求输出证据）。",
    "beam": "二值化口径是我们定的：逐条 rubric 三点制均分 == 1.0 才算对。",
    "memtrapbench": "通过阈值是我们定的（四维均分）；原始分留在 judge_response。",
    "medmemorybench": "判分照上游 `metrics/`；作答 prompt 是我们写的。",
    "musique": "可答题按答案串/别名规则判；拒答变体恒走精确匹配。",
    "hybridqa": "按归一化答案串 EM 判；不聚合官方 F1。",
}


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()
    ]


def _load_extra(capture: Path) -> tuple[dict[int, dict], list[str]]:
    index: dict[int, dict] = {}
    duplicates: list[str] = []
    for name in EXTRA_FILES:
        path = capture / name
        if not path.is_file():
            duplicates.append(f"缺文件：{name}")
            continue
        for row in _read_jsonl(path):
            seq = int(row["seq"])
            if seq in index:
                duplicates.append(f"seq={seq} 同时出现在 {name} 与另一份 extra 文件")
                continue
            index[seq] = row
    return index, duplicates


def _has_answers(row: dict) -> bool:
    return any(str(value).strip() for value in (row.get("gold_answers") or []))


def _compare_fields(kit: dict, expected: dict, fields: tuple[str, ...]) -> list[str]:
    diffs = []
    for field in fields:
        if kit.get(field) != expected.get(field):
            diffs.append(field)
    return diffs


def audit(capture: Path, sample_per_family: int) -> tuple[dict, list[str]]:
    kit = _read_jsonl(capture / KIT)
    questions = {int(row["seq"]): row for row in _read_jsonl(capture / QUESTIONS)}
    extra, extra_errors = _load_extra(capture)
    scorable = [row for row in kit if row.get("judge_kind") != "none"]

    hard: list[str] = list(extra_errors)
    stats: dict[str, Counter] = defaultdict(Counter)
    samples: dict[str, list[dict]] = defaultdict(list)
    mismatches: list[str] = []

    for row in scorable:
        seq, family = int(row["seq"]), str(row["dataset"])
        counter = stats[family]
        counter["rows"] += 1
        question = questions.get(seq) or {}
        public = question.get("public_gold")

        if seq in extra:
            expected = _extra_payload(extra[seq])
            counter["provenance_extra"] += 1
            diffs = _compare_fields(
                row,
                expected,
                (
                    "gold_answers",
                    "gold_rubric",
                    "gold_native",
                    "gold_alternative_answers",
                    "gold_variant",
                ),
            )
            if str(row.get("gold_judging") or "") != str(extra[seq].get("gold_judging") or ""):
                diffs.append("gold_judging")
            if diffs:
                counter["mismatch"] += 1
                mismatches.append(f"seq={seq} {family}: 与 extra 来源不一致 {diffs}")
        elif isinstance(public, dict):
            expected = _public_gold_payload(public, family)
            counter["provenance_public"] += 1
            diffs = _compare_fields(row, expected, ("gold_answers", "gold_rubric"))
            if diffs:
                counter["mismatch"] += 1
                mismatches.append(f"seq={seq} {family}: 与 public_gold 不一致 {diffs}")
        else:
            counter["no_provenance"] += 1
            hard.append(f"seq={seq} {family}: 既不在 extra，也没有 public_gold")

        if _has_answers(row):
            counter["answers"] += 1
        if row.get("gold_rubric"):
            counter["rubric"] += 1
        if not _has_answers(row) and not row.get("gold_rubric"):
            counter["no_gold"] += 1
            hard.append(f"seq={seq} {family}: 没有任何金标")
        if family in _NEEDS_NATIVE and row.get("gold_native") is None:
            counter["missing_native"] += 1
            hard.append(f"seq={seq} {family}: 裁判需要 gold_native，但套件里是 null")

        if sample_per_family and len(samples[family]) < sample_per_family:
            samples[family].append(
                {
                    "seq": seq,
                    "question": str(row.get("query") or "")[:300],
                    "gold_answers": row.get("gold_answers"),
                    "gold_rubric": row.get("gold_rubric"),
                    "gold_variant": row.get("gold_variant"),
                    "source": (extra.get(seq) or {}).get("source_locator")
                    or (public if isinstance(public, dict) else None),
                }
            )
    hard.extend(mismatches)
    return {"stats": stats, "samples": samples, "scorable": len(scorable)}, hard


def _render(result: dict, hard: list[str]) -> str:
    lines = [
        "# 官方采集金标审计",
        "",
        f"> 生成：{datetime.now(tz=UTC).isoformat(timespec='seconds')}；只读，不调模型。",
        "",
        "## 逐族",
        "",
        "| 家族 | 行数 | extra 来源 | public_gold 来源 | 无来源 | 答案串 | rubric | 无金标 | 缺 native | 不一致 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family, counter in sorted(result["stats"].items()):
        lines.append(
            f"| {family} | {counter['rows']} | {counter['provenance_extra']} | "
            f"{counter['provenance_public']} | {counter['no_provenance']} | "
            f"{counter['answers']} | {counter['rubric']} | {counter['no_gold']} | "
            f"{counter['missing_native']} | {counter['mismatch']} |"
        )
    lines += [
        "",
        f"可评分总行数：**{result['scorable']}**。审计硬失败：**{len(hard)}**。",
        "",
        "## 已声明的口径偏离（不是上游原样）",
        "",
    ]
    for family, note in sorted(KNOWN_DEVIATIONS.items()):
        lines.append(f"- **{family}**：{note}")
    lines += ["", "## 抽样（人工复核用）", ""]
    for family, rows in sorted(result["samples"].items()):
        lines.append(f"### {family}")
        lines.append("")
        for row in rows:
            gold = row["gold_answers"] or row["gold_rubric"]
            lines.append(
                f"- `seq={row['seq']}` 题：{row['question']!r}\n"
                f"  金标：`{json.dumps(gold, ensure_ascii=False)[:300]}`；"
                f"来源：`{json.dumps(row['source'], ensure_ascii=False)[:300]}`"
            )
        lines.append("")
    if hard:
        lines += ["## 硬失败明细", ""]
        lines += [f"- {item}" for item in hard[:200]]
        if len(hard) > 200:
            lines.append(f"- … 另有 {len(hard) - 200} 条")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--capture-dir", default="official-dataset-2026-09-29")
    parser.add_argument("--sample-per-family", type=int, default=3)
    parser.add_argument("--out", default="eval/reports/runs/official-gold-audit.md")
    args = parser.parse_args(argv)

    capture = Path(args.capture_dir)
    result, hard = audit(capture, args.sample_per_family)
    report = _render(result, hard)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n→ {output}")
    return 2 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
