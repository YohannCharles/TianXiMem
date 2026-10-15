"""固定官方采集题目清单的逐数据集进度、准确率与两版对比。"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from eval.datasets.official_groups import data_identity, write_json
from eval.jsonl_io import read_jsonl

DISPLAY_NAMES = {
    "beam": "BEAM",
    "clbench": "CL-Bench",
    "corporatebench": "CorporateBench",
    "docpp": "Doc-PP",
    "feverous": "FEVEROUS",
    "halumem": "HaluMem",
    "hybridqa": "HybridQA",
    "locomo-refined": "LoCoMo-Refined",
    "medmemorybench": "MedMemoryBench 中文",
    "memtrapbench": "MemTrapBench",
    "mquake-remastered": "MQuAKE-Remastered",
    "musique": "MuSiQue",
    "personamem-v2": "PersonaMem-v2",
    "scriptmem": "ScriptMem",
    "tempreason": "TempReason",
    "other": "other",
}


def coverage_report(manifest: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 官方采集 Search 数据集归类",
        "",
        "按 Search 记录计数，以 seq 匹配套件；未知归属进入 other，不凭语料形态猜测。",
        "",
        "| 数据集 | Search 条数 | 有答案文本/选项 | 仅评分要点 | 可评分 | 本次所选 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    counts = manifest["counts"]
    for dataset, row in sorted(counts.items(), key=lambda item: (-item[1]["searches"], item[0])):
        lines.append(
            f"| {DISPLAY_NAMES.get(dataset, dataset)} | {row['searches']} | {row['answers']} | "
            f"{row['rubric_only']} | {row['scorable']} | {row['selected']} |"
        )
    totals = {key: sum(r[key] for r in counts.values()) for key in next(iter(counts.values()))}
    columns = ("searches", "answers", "rubric_only", "scorable", "selected")
    lines.append("| 合计 | " + " | ".join(str(totals[k]) for k in columns) + " |")
    lines += ["", "other 保留供归属分析；缺少可靠金标的记录不进入正确率分母。", ""]
    (output / "coverage.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "coverage.json", {"counts": counts, "totals": totals})


def summarize_run(run_dir: Path) -> dict:
    runtime = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "selection-manifest.json").read_text(encoding="utf-8"))
    if data_identity(manifest) != runtime["data_identity"]:
        raise ValueError("运行目录的题目清单已变化，不能汇总")
    wanted = {str(row["seq"]): row for row in manifest["questions"]}
    labels = {}
    expected_by_user = Counter(row["user_id"] for row in manifest["questions"])
    for source_user in expected_by_user:
        wire_user = runtime["users"][source_user]
        for row in read_jsonl(run_dir / wire_user / "labels.jsonl"):
            qid = str(row["id"])
            if qid not in wanted or wanted[qid]["user_id"] != source_user:
                raise ValueError(f"成绩包含清单外或跨用户题目：{qid}")
            if qid in labels:
                raise ValueError(f"成绩包含重复题目：{qid}")
            if type(row.get("is_correct")) is not bool:
                raise ValueError(f"成绩 is_correct 必须为布尔值：{qid}")
            if str(row.get("label", "")).endswith("_ERROR") and row["is_correct"]:
                raise ValueError(f"技术错误不能计为正确：{qid}")
            labels[qid] = row
    groups = {}
    for dataset, coverage in manifest["counts"].items():
        ids = [qid for qid, row in wanted.items() if row["dataset"] == dataset]
        found = [labels[qid] for qid in ids if qid in labels]
        correct = sum(row["is_correct"] for row in found)
        errors = Counter(
            str(row.get("label")) for row in found if str(row.get("label", "")).endswith("_ERROR")
        )
        groups[dataset] = {
            "searches": coverage["searches"],
            "eligible": coverage["scorable"],
            "expected": len(ids),
            "completed": len(found),
            "missing": len(ids) - len(found),
            "correct": correct,
            "errors": dict(errors),
            "accuracy": correct / len(ids) if ids and len(found) == len(ids) else None,
        }
    expected = len(wanted)
    completed = len(labels)
    correct = sum(row["is_correct"] for row in labels.values())
    errors = Counter(
        str(row.get("label"))
        for row in labels.values()
        if str(row.get("label", "")).endswith("_ERROR")
    )
    return {
        "run_id": runtime["run_id"],
        "service_revision": runtime["service_revision"],
        "data_identity": runtime["data_identity"],
        "evaluation": runtime["evaluation"],
        "expected": expected,
        "completed": completed,
        "missing": expected - completed,
        "correct": correct,
        "errors": dict(errors),
        "accuracy": correct / expected if expected and completed == expected else None,
        "datasets": groups,
    }


def _cell(row: dict) -> str:
    if not row["expected"]:
        return "—"
    if row["accuracy"] is None:
        return f"{row['correct']}/{row['expected']}（已判 {row['completed']}，待完成）"
    return f"{row['correct']}/{row['expected']}（{row['accuracy'] * 100:.2f}%）"


def write_summary(run_dir: Path) -> dict:
    summary = summarize_run(run_dir)
    write_json(run_dir / "summary.json", summary)
    lines = [
        "# 官方采集评测结果",
        "",
        "| 数据集 | 正确/题数（正确率） | 缺题 | 技术错误 |",
        "|---|---:|---:|---|",
    ]
    for dataset, row in sorted(summary["datasets"].items()):
        error_text = json.dumps(row["errors"], ensure_ascii=False) if row["errors"] else "0"
        lines.append(
            f"| {DISPLAY_NAMES.get(dataset, dataset)} | {_cell(row)} | "
            f"{row['missing']} | {error_text} |"
        )
    lines.append(
        f"| 合计 | {_cell(summary)} | {summary['missing']} | {sum(summary['errors'].values())} |"
    )
    lines += [
        "",
        f"完成 {summary['completed']}/{summary['expected']}；other 无金标，不参与准确率。",
        "",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def compare_runs(left: Path, right: Path, output: Path, left_name: str, right_name: str) -> dict:
    first, second = summarize_run(left), summarize_run(right)
    if first["data_identity"] != second["data_identity"]:
        raise ValueError("两次运行的题目清单或源数据指纹不同，不能比较")
    if first["evaluation"] != second["evaluation"]:
        raise ValueError("两次运行的答案/裁判模型、端点、输出预算或评分源码不同，不能比较")
    table = []
    for dataset, a in first["datasets"].items():
        b = second["datasets"][dataset]
        delta = (
            (b["accuracy"] - a["accuracy"]) * 100
            if a["accuracy"] is not None and b["accuracy"] is not None
            else None
        )
        table.append({"dataset": dataset, "left": a, "right": b, "delta_pp": delta})
    table.sort(key=lambda r: (r["delta_pp"] is None, -(r["delta_pp"] or 0), r["dataset"]))
    output.mkdir(parents=True, exist_ok=True)
    result = {"left": first, "right": second, "rows": table}
    write_json(output / "comparison.json", result)
    lines = [
        "# 官方采集版本对比",
        "",
        f"| 数据集 | {left_name} 正确/题数（正确率） | "
        f"{right_name} 正确/题数（正确率） | 变化（百分点） |",
        "|---|---:|---:|---:|",
    ]
    for row in table:
        delta = "—" if row["delta_pp"] is None else f"{row['delta_pp']:+.2f}"
        lines.append(
            f"| {DISPLAY_NAMES.get(row['dataset'], row['dataset'])} | {_cell(row['left'])} | "
            f"{_cell(row['right'])} | {delta} |"
        )
    total_delta = (
        f"{(second['accuracy'] - first['accuracy']) * 100:+.2f}"
        if first["accuracy"] is not None and second["accuracy"] is not None
        else "—"
    )
    lines.append(f"| 合计（加权） | {_cell(first)} | {_cell(second)} | {total_delta} |")
    lines += [
        "",
        f"完成：{left_name} {first['completed']}/{first['expected']}；"
        f"{right_name} {second['completed']}/{second['expected']}。",
        "",
        "技术错误保留在分母中，单独列示；未完成时不输出准确率或百分点变化。"
        "other 无金标，不计为错误。",
        "",
        "| 数据集 | 左侧缺题/技术错误 | 右侧缺题/技术错误 |",
        "|---|---|---|",
    ]
    for row in table:
        a, b = row["left"], row["right"]
        lines.append(
            f"| {DISPLAY_NAMES.get(row['dataset'], row['dataset'])} | "
            f"{a['missing']} / {sum(a['errors'].values())} | "
            f"{b['missing']} / {sum(b['errors'].values())} |"
        )
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "数据集",
                f"{left_name} 正确/题数（正确率）",
                f"{right_name} 正确/题数（正确率）",
                "变化（百分点）",
            ]
        )
        for row in table:
            writer.writerow(
                [
                    DISPLAY_NAMES.get(row["dataset"], row["dataset"]),
                    _cell(row["left"]),
                    _cell(row["right"]),
                    "" if row["delta_pp"] is None else f"{row['delta_pp']:.2f}",
                ]
            )
        writer.writerow(["合计（加权）", _cell(first), _cell(second), total_delta])
    return result
