"""Build an auditable, offline human rescore for the E0/E2 bundle.

This script never calls an API and never edits the original run artifacts.
It records a verdict for every row whose original label is incorrect.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FAMILIES = (
    "halumem",
    "hybridqa",
    "locomo-refined",
    "longmemeval-s",
    "mquake-remastered",
    "musique",
    "personamem-v2",
)
ARMS = ("e0", "e2")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_gold(value: Any) -> Any:
    if isinstance(value, dict) and "answer" in value:
        return value["answer"]
    return value


def compact(value: Any, limit: int = 2500) -> Any:
    if isinstance(value, str):
        return value[:limit]
    text = json.dumps(value, ensure_ascii=False)
    return text[:limit]


def load_arm(root: Path, arm: str) -> dict[str, dict[str, Any]]:
    labels = read_json(root / f"{arm}-labels.json")
    rows: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        family_root = root / arm / family
        inputs: dict[str, dict[str, Any]] = {}
        answers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for input_path in family_root.rglob("input.jsonl"):
            for row in read_jsonl(input_path):
                inputs[str(row["id"])] = row
            answer_path = input_path.parent / "answers.jsonl"
            if answer_path.exists():
                for row in read_jsonl(answer_path):
                    answers[str(row["id"])].append(row)
        for qid, label in labels.items():
            if label["family"] != family or label["is_correct"]:
                continue
            answer_rows = answers.get(qid, [])
            answer = answer_rows[-1] if answer_rows else None
            inp = inputs.get(qid)
            rows[qid] = {
                "family": family,
                "id": qid,
                "label": label,
                "input": inp,
                "answer": answer,
                "answer_row_count": len(answer_rows),
            }
    return rows


def manual_verdict(arm: str, family: str, qid: str, row: dict[str, Any]) -> dict[str, Any]:
    # Confirmed semantic judge errors. These are the only rows flipped by the
    # primary human score.
    judge_errors = {
        ("hybridqa", "memory500-0090"): (
            "The answer is a more specific, non-contradictory form of the gold answer: "
            "hysterical blindness contains blindness."
        ),
        ("longmemeval-s", "memory500-0196"): (
            "The memory explicitly identifies Rhythm Central as the music shop on Main St.; "
            "the answer is more specific and preserves the requested fact."
        ),
    }
    if (family, qid) in judge_errors:
        return {
            "human_correct": True,
            "issue": "judge_error",
            "verdict": "confirmed_flip",
            "confidence": "high",
            "reason": judge_errors[(family, qid)],
        }
    if family == "hybridqa" and qid == "memory500-0089" and arm == "e2":
        return {
            "human_correct": True,
            "issue": "judge_error",
            "verdict": "confirmed_flip",
            "confidence": "high",
            "reason": (
                "The answer and gold identify the same entity; the only difference is "
                "apostrophe/spacing normalization in The Devil's Advocate."
            ),
        }

    # Gold/contract ambiguities are retained as uncertain and do not alter the
    # primary score.
    if family == "hybridqa" and qid in {"memory100-mh-0089", "memory100-mh-0081"}:
        return {
            "human_correct": None,
            "issue": "gold_issue",
            "verdict": "uncertain",
            "confidence": "medium",
            "reason": (
                "The answer is semantically plausible (more specific entity or numeric "
                "equivalent), but the benchmark gold uses a truncated/strict spelling form."
            ),
        }
    if family == "musique" and qid in {"memory500-0009", "memory500-0020"}:
        return {
            "human_correct": None,
            "issue": "gold_issue",
            "verdict": "uncertain",
            "confidence": "low",
            "reason": (
                "The question chain and synthetic gold appear inconsistent or ambiguous; "
                "the row is excluded from confirmed score flips."
            ),
        }
    if family == "locomo-refined" and qid == "memory500-0274":
        return {
            "human_correct": None,
            "issue": "uncertain",
            "verdict": "uncertain",
            "confidence": "medium",
            "reason": (
                "The answer gives the absolute date corresponding to the relative gold "
                "expression, but this benchmark's strict time contract may require the "
                "relative form."
            ),
        }
    if family == "longmemeval-s" and qid == "memory500-0372":
        return {
            "human_correct": None,
            "issue": "uncertain",
            "verdict": "uncertain",
            "confidence": "medium",
            "reason": (
                "The calendar date matches the gold, but the gold includes a timestamp "
                "while the answer has day-level precision."
            ),
        }

    if family == "personamem-v2":
        label = str(row["label"].get("label", ""))
        if label.startswith("letter=/"):
            return {
                "human_correct": False,
                "issue": "parser_error",
                "verdict": "confirmed_still_wrong",
                "confidence": "high",
                "reason": (
                    "The official PersonaMem scorer requires an explicit final option "
                    "letter; the answer row has no parseable predicted letter."
                ),
            }
        return {
            "human_correct": False,
            "issue": "model_error",
            "verdict": "confirmed_still_wrong",
            "confidence": "high",
            "reason": "The parsed option letter differs from the official gold letter.",
        }

    if family == "musique" and row["answer"] is None:
        return {
            "human_correct": False,
            "issue": "mapping_error",
            "verdict": "not_scorable",
            "confidence": "high",
            "reason": "No answer row with this question id exists in the arm artifact.",
        }

    # Dataset-specific strict contracts make these rows genuine model failures:
    # unsupported extras, relative/absolute time mismatches, synthetic aliases,
    # and wrong entities are not silently converted to correct.
    return {
        "human_correct": False,
        "issue": "model_error",
        "verdict": "confirmed_still_wrong",
        "confidence": "high",
        "reason": (
            "The answer does not satisfy the dataset's gold and scoring contract; "
            "the mismatch is in the model output rather than the judge."
        ),
    }


def source_paths(root: Path, arm: str, family: str, qid: str) -> dict[str, str | None]:
    for input_path in (root / arm / family).rglob("input.jsonl"):
        ids = {str(row["id"]) for row in read_jsonl(input_path)}
        if qid in ids:
            answer_path = input_path.parent / "answers.jsonl"
            label_path = input_path.parent / "labels.jsonl"
            return {
                "input": str(input_path),
                "answer": str(answer_path) if answer_path.exists() else None,
                "label": str(label_path) if label_path.exists() else None,
            }
    return {"input": None, "answer": None, "label": None}


def make_record(root: Path, arm: str, qid: str, row: dict[str, Any]) -> dict[str, Any]:
    family = row["family"]
    verdict = manual_verdict(arm, family, qid, row)
    inp = row["input"] or {}
    answer = row["answer"] or {}
    return {
        "arm": arm,
        "family": family,
        "id": qid,
        "original_correct": bool(row["label"]["is_correct"]),
        "original_label": row["label"].get("label"),
        "original_judge_response": row["label"].get("judge_response"),
        "question": inp.get("question"),
        "gold": normalize_gold(inp.get("gold_answer")),
        "generated_answer": answer.get("generated_answer"),
        "answer_row_count_for_id": row["answer_row_count"],
        **verdict,
        "source_paths": source_paths(root, arm, family, qid),
    }


def artifact_findings(root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for arm in ARMS:
        path = root / arm / "musique" / "selected-candidate-haystack" / "answers.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        ids = [str(row["id"]) for row in rows]
        extra = [qid for qid in ids if qid.startswith("memory100-")]
        if extra:
            findings.append(
                {
                    "arm": arm,
                    "family": "musique",
                    "issue": "mapping_error",
                    "path": str(path),
                    "detail": (
                        f"{len(extra)} answer rows use memory100 ids, while this run's "
                        "MuSiQue input/labels contain only memory500 ids. They are excluded "
                        "from the 600-question score."
                    ),
                    "ids": extra,
                }
            )
    return findings


def summarize(records: list[dict[str, Any]], labels: dict[str, Any]) -> dict[str, Any]:
    total = len(labels)
    original_correct = sum(bool(row["is_correct"]) for row in labels.values())
    summary: dict[str, Any] = {
        "questions": total,
        "original_correct": original_correct,
        "original_accuracy": original_correct / total if total else None,
        "confirmed_human_correct_additions": 0,
        "human_correct_including_uncertain": 0,
        "by_issue": dict(Counter(row["issue"] for row in records)),
        "by_verdict": dict(Counter(row["verdict"] for row in records)),
        "artifact_issue_counts": {},
        "per_family": {},
    }
    for family in FAMILIES:
        family_labels = [row for row in labels.values() if row["family"] == family]
        family_records = [row for row in records if row["family"] == family]
        confirmed = sum(row["human_correct"] is True for row in family_records)
        uncertain = sum(row["human_correct"] is None for row in family_records)
        orig = sum(bool(row["is_correct"]) for row in family_labels)
        summary["confirmed_human_correct_additions"] += confirmed
        summary["human_correct_including_uncertain"] += confirmed + uncertain
        summary["per_family"][family] = {
            "questions": len(family_labels),
            "original_correct": orig,
            "original_accuracy": orig / len(family_labels) if family_labels else None,
            "confirmed_human_correct": orig + confirmed,
            "confirmed_accuracy": (
                (orig + confirmed) / len(family_labels) if family_labels else None
            ),
            "uncertain_rows": uncertain,
            "issue_counts": dict(Counter(row["issue"] for row in family_records)),
        }
    summary["confirmed_human_correct"] = (
        original_correct + summary["confirmed_human_correct_additions"]
    )
    summary["confirmed_human_accuracy"] = summary["confirmed_human_correct"] / total
    summary["confirmed_delta"] = summary["confirmed_human_correct"] - original_correct
    summary["uncertain_upper_bound_correct"] = (
        summary["confirmed_human_correct"] + sum(row["human_correct"] is None for row in records)
    )
    summary["uncertain_upper_bound_accuracy"] = (
        summary["uncertain_upper_bound_correct"] / total
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()

    all_records: list[dict[str, Any]] = []
    arm_summaries: dict[str, Any] = {}
    for arm in ARMS:
        labels = read_json(root / f"{arm}-labels.json")
        rows = load_arm(root, arm)
        records = [make_record(root, arm, qid, row) for qid, row in rows.items()]
        all_records.extend(records)
        arm_summaries[arm] = summarize(records, labels)

    payload = {
        "run_dir": str(root),
        "source_policy": {
            "api_calls": False,
            "original_artifacts_modified": False,
            "scope": "Every originally incorrect row in E0 and E2, recorded per arm.",
        },
        "records": all_records,
        "artifact_findings": artifact_findings(root),
        "arms": arm_summaries,
    }
    for finding in payload["artifact_findings"]:
        arm = finding["arm"]
        issue = finding["issue"]
        arm_summaries[arm]["artifact_issue_counts"][issue] = (
            arm_summaries[arm]["artifact_issue_counts"].get(issue, 0)
            + len(finding.get("ids", []))
        )
    (root / "human-rescore.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "human-rescore-summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(root),
                "arms": arm_summaries,
                "artifact_findings": payload["artifact_findings"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Human Rescore: E0 vs E2",
        "",
        f"Run: `{root}`",
        "",
        "This is an offline review of every originally incorrect arm-level row. "
        "Original labels and answer artifacts are unchanged.",
        "",
        "## Summary",
        "",
        "| Arm | Original | Confirmed human | Uncertain upper bound | Judge errors | Model errors | Parser errors | Gold/uncertain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        s = arm_summaries[arm]
        issues = s["by_issue"]
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    f'{s["original_correct"]}/{s["questions"]} ({s["original_accuracy"]:.2%})',
                    f'{s["confirmed_human_correct"]}/{s["questions"]} ({s["confirmed_human_accuracy"]:.2%})',
                    f'{s["uncertain_upper_bound_correct"]}/{s["questions"]} ({s["uncertain_upper_bound_accuracy"]:.2%})',
                    str(issues.get("judge_error", 0)),
                    str(issues.get("model_error", 0)),
                    str(issues.get("parser_error", 0)),
                    str(issues.get("gold_issue", 0) + issues.get("uncertain", 0)),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Confirmed flips are deliberately conservative. Uncertain rows are not added to "
        "the primary human score.",
        "",
        "## Confirmed Judge Errors",
        "",
    ]
    for row in all_records:
        if row["verdict"] == "confirmed_flip":
            lines.append(
                f'- `{row["arm"]}` `{row["family"]}` `{row["id"]}`: {row["reason"]}'
            )
    lines += [
        "",
        "## Artifact Findings",
        "",
    ]
    for finding in payload["artifact_findings"]:
        lines.append(f'- `{finding["arm"]}` `{finding["family"]}`: {finding["detail"]}')
    lines += [
        "",
        "The complete row-level audit, including gold, answer, original judge response, "
        "verdict, reason, and source paths, is in `human-rescore.json`.",
        "",
    ]
    (root / "human-rescore.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"records": len(all_records), "arms": arm_summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
