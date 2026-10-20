"""Offline rescoring for an existing official-capture run.

Only saved ``input.jsonl``, ``answers.jsonl``, and ``labels.jsonl`` are used.
This module never calls Add, Search, an answer model, or a judge model.
LLM-judged families remain unchanged because their verdict cannot be
reconstructed deterministically without another judge call.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from eval.harness.corpusqa_pipeline import score_answer
from eval.harness.extra_musique import judge_musique

__all__ = ["rescore_run"]

# Families whose saved answers can be rescored by a deterministic contract.
RESCORED_FAMILIES = frozenset({"hybridqa", "musique"})
LLM_FAMILIES = frozenset(
    {
        "beam",
        "clbench",
        "docpp",
        "halumem",
        "locomo-refined",
        "medmemorybench",
        "memtrapbench",
        "personamem-v2",
    }
)
DEDICATED_UNCHANGED = frozenset({"corporatebench", "feverous", "mquake-remastered", "scriptmem"})


def _read_one(path: Path, key: str) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one row, found {len(rows)}")
    if str(rows[0].get("id", rows[0].get("seq"))) != key:
        raise ValueError(f"{path}: id/seq does not match directory {key}")
    return rows[0]


def _load_questions(run_dir: Path) -> dict[int, dict[str, Any]]:
    questions: dict[int, dict[str, Any]] = {}
    for directory in sorted(run_dir.glob("*/questions/*")):
        if not directory.is_dir():
            continue
        key = directory.name
        if not key.isdigit():
            raise ValueError(f"unexpected question directory: {directory}")
        seq = int(key)
        if seq in questions:
            raise ValueError(f"duplicate question seq {seq}")
        input_row = _read_one(directory / "input.jsonl", key)
        answer_row = _read_one(directory / "answers.jsonl", key)
        label_row = _read_one(directory / "labels.jsonl", key)
        questions[seq] = {
            "input": input_row,
            "answer": str(answer_row.get("generated_answer") or ""),
            "label": label_row,
        }
    if not questions:
        raise ValueError(f"no question checkpoints under {run_dir}")
    return questions


def _hybridqa_score(input_row: dict[str, Any], generated: str) -> dict[str, Any]:
    item = dict(input_row)
    gold = dict(item.get("gold_answer") or {})
    if "answer" not in gold:
        gold["answer"] = gold.get("answer-text")
    item["gold_answer"] = gold
    return score_answer(item, generated)


def _rescored_result(
    dataset: str, input_row: dict[str, Any], generated: str
) -> tuple[bool, str, str] | None:
    if dataset == "musique":
        ok, reason = judge_musique(generated, input_row.get("gold_answer"))
        return ok, "CORRECT" if ok else "WRONG", reason
    if dataset == "hybridqa":
        result = _hybridqa_score(input_row, generated)
        ok = bool(result["is_correct"])
        return ok, "CORRECT" if ok else "WRONG", str(result.get("judge_response") or "")
    return None


def rescore_run(run_dir: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    """Write a non-destructive corrected label snapshot and summary."""
    run_dir = Path(run_dir)
    questions = _load_questions(run_dir)
    output = Path(output_dir) if output_dir else run_dir / "offline-rescore"
    if output == run_dir:
        raise ValueError("output directory must differ from the source run")
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    by_dataset: dict[str, Counter] = {}
    original_total = corrected_total = 0
    for seq, payload in sorted(questions.items()):
        input_row = payload["input"]
        original = payload["label"]
        dataset = str(input_row.get("dataset") or "unknown")
        original_ok = bool(original.get("is_correct"))
        result = _rescored_result(dataset, input_row, payload["answer"])
        if result is None:
            correct = original_ok
            label = str(original.get("label") or "")
            reason = str(original.get("judge_response") or "")
            changed = False
        else:
            correct, label, reason = result
            changed = correct != original_ok or label != original.get("label")
        row = {
            "id": str(seq),
            "seq": seq,
            "dataset": dataset,
            "is_correct": correct,
            "label": label,
            "judge_response": reason,
            "original_is_correct": original_ok,
            "original_label": original.get("label"),
            "changed": changed,
            "policy": (
                f"offline-deterministic:{dataset}" if result is not None else "unchanged"
            ),
        }
        rows.append(row)
        if changed:
            changes.append(row)
        counts = by_dataset.setdefault(dataset, Counter())
        counts["total"] += 1
        counts["original_correct"] += int(original_ok)
        counts["corrected_correct"] += int(correct)
        counts["changed"] += int(changed)
        original_total += int(original_ok)
        corrected_total += int(correct)

    total = len(rows)
    summary = {
        "policy": "offline-contract-rescore-v1",
        "network_calls": False,
        "source_run": str(run_dir),
        "rescored_families": sorted(RESCORED_FAMILIES),
        "unchanged_llm_families": sorted(set(by_dataset) & LLM_FAMILIES),
        "unchanged_dedicated_families": sorted(set(by_dataset) & DEDICATED_UNCHANGED),
        "original": {
            "correct": original_total,
            "total": total,
            "accuracy": round(original_total / total, 6),
        },
        "corrected": {
            "correct": corrected_total,
            "total": total,
            "accuracy": round(corrected_total / total, 6),
        },
        "changed_rows": len(changes),
        "by_dataset": {
            dataset: {
                "total": counts["total"],
                "original_correct": counts["original_correct"],
                "corrected_correct": counts["corrected_correct"],
                "changed": counts["changed"],
                "status": (
                    "rescored"
                    if dataset in RESCORED_FAMILIES
                    else "unchanged-llm"
                    if dataset in LLM_FAMILIES
                    else "unchanged-dedicated"
                    if dataset in DEDICATED_UNCHANGED
                    else "unchanged-unknown"
                ),
            }
            for dataset, counts in sorted(by_dataset.items())
        },
        "unresolved": [
            "LLM-judged families were not rejudged; that would require judge-only calls.",
            "Wrong answer prompts cannot be repaired without regenerating answers.",
            "MuSiQue is contract-rescored on the saved answers, not a post-prompt-fix score.",
        ],
    }

    _write_jsonl(output / "labels.jsonl", rows)
    _write_jsonl(output / "changes.jsonl", changes)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _summary_markdown(summary: dict[str, Any]) -> str:
    original = summary["original"]
    corrected = summary["corrected"]
    lines = [
        "# Offline corrected scores",
        "",
        "No Add, Search, answer, or judge request was sent.",
        "",
        "| Scope | Correct/Total | Accuracy |",
        "|---|---:|---:|",
        f"| Original | {original['correct']}/{original['total']} | {original['accuracy']:.2%} |",
        f"| Corrected | {corrected['correct']}/{corrected['total']} | {corrected['accuracy']:.2%} |",
        "",
        "| Dataset | Original | Corrected | Changed | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset, row in summary["by_dataset"].items():
        lines.append(
            f"| {dataset} | {row['original_correct']}/{row['total']} "
            f"| {row['corrected_correct']}/{row['total']} | {row['changed']} | {row['status']} |"
        )
    lines += ["", "## Unresolved", ""]
    lines.extend(f"- {item}" for item in summary["unresolved"])
    return "\n".join(lines) + "\n"
