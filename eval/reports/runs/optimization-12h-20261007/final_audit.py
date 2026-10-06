"""Audit all frozen native questions and report historical comparisons honestly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from compare import technical_errors
from experiment import NAME, OUT, check_arm, truth_hash, write


def records(run_id: str, kind: str) -> dict[str, dict]:
    result = {}
    for path in (OUT.parent / run_id).glob(f"*/{kind}.jsonl"):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["id"])
            if qid in result:
                raise RuntimeError(f"Duplicate {kind} qid: {run_id}/{qid}")
            result[qid] = row
    return result


def fingerprint(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def paired_inputs(current: dict, previous: dict) -> dict:
    if current.keys() != previous.keys():
        raise RuntimeError("Historical and current full question sets differ")
    fixed = ("id", "dataset", "question", "gold_answer", "category")
    for qid, row in current.items():
        if any(row.get(k) != previous[qid].get(k) for k in fixed):
            raise RuntimeError(f"Historical scoring input changed: {qid}")
    return {
        "changed_answer_inputs": sum(current[qid] != previous[qid] for qid in current),
        "current_input_sha256": fingerprint(current),
        "historical_input_sha256": fingerprint(previous),
    }


def correctness_changes(current: dict, previous: dict, inputs: dict, previous_inputs: dict) -> dict:
    """Separate score flips with changed evidence from flips on identical inputs."""
    if not (current.keys() == previous.keys() == inputs.keys() == previous_inputs.keys()):
        raise RuntimeError("Correctness comparison question sets differ")
    result = dict(wins=0, losses=0, changed_input_wins=0, changed_input_losses=0, cases=[])
    for qid, row in current.items():
        before, after = bool(previous[qid].get("is_correct")), bool(row.get("is_correct"))
        if before == after:
            continue
        direction = "wins" if after else "losses"
        result[direction] += 1
        same = inputs[qid] == previous_inputs[qid]
        if not same:
            result[f"changed_input_{direction}"] += 1
        result["cases"].append(
            dict(
                qid=qid,
                question=inputs[qid]["question"],
                before_correct=before,
                after_correct=after,
                identical_input=same,
                before_label=previous[qid].get("label"),
                after_label=row.get("label"),
            )
        )
    result["identical_input_wins"] = result["wins"] - result["changed_input_wins"]
    result["identical_input_losses"] = result["losses"] - result["changed_input_losses"]
    return result


def audit(arm: str, *, historical: bool = False, allow_partial: bool = False) -> dict:
    plans = json.loads((OUT / "plans.json").read_text())
    manifest = json.loads((OUT / f"arm-{arm}.json").read_text())
    check_arm(manifest)
    truth = {}
    for group, metadata in manifest["groups"].items():
        actual = truth_hash(Path(metadata["db"]))
        if actual != metadata["truth_sha256"]:
            raise RuntimeError(f"Truth rows changed: {arm}/{group}")
        truth[group] = actual
    summary = {
        "arm": arm,
        "complete": True,
        "scope": "11 frozen native pipelines; no AML Full or cross-dataset average",
        "historical_comparison_note": (
            "Historical answers were generated in an earlier run; score changes include model "
            "and API variation. Same-round small/stage comparisons establish candidate gates."
        ),
        "truth_sha256": truth,
        "source_commit_at_freeze": manifest["commit"],
        "datasets": [],
    }
    total = 0
    for meta in plans["datasets"]:
        dataset = meta["dataset"]
        run_id = meta["prior_run"] if historical else f"{NAME}-{arm}-full-{dataset}"
        inputs = records(run_id, "input")
        answers = records(run_id, "answers")
        labels = records(run_id, "labels")
        expected = set(meta["qids"])
        if not (inputs.keys() == answers.keys() == labels.keys() == expected):
            summary["complete"] = False
            if not allow_partial:
                raise RuntimeError(f"Full question coverage incomplete: {run_id}")
            summary["datasets"].append(
                dict(
                    dataset=dataset,
                    complete=False,
                    expected_n=len(expected),
                    n_inputs=len(inputs),
                    n_answers=len(answers),
                    n_labels=len(labels),
                )
            )
            continue
        record = json.loads((OUT.parent / f"{run_id}.json").read_text())
        n, correct = len(labels), sum(bool(r.get("is_correct")) for r in labels.values())
        if abs(record["scores"]["overall"] - correct / n) > 1e-6:
            raise RuntimeError(f"Stored overall does not match original denominator: {run_id}")
        errors = technical_errors(labels, answers)
        entry = dict(
            dataset=dataset,
            run_id=run_id,
            complete=True,
            n=n,
            correct=correct,
            overall=record["scores"]["overall"],
            dataset_score=record["scores"].get("dataset_score"),
            errors=errors,
            models=record["models"],
            execution_backends=record.get("execution_backends", []),
            execution_transport=record.get("execution_transport"),
            input_sha256=fingerprint(inputs),
        )
        if not historical:
            previous_inputs = records(meta["prior_run"], "input")
            previous_labels = records(meta["prior_run"], "labels")
            previous_answers = records(meta["prior_run"], "answers")
            prior = json.loads((OUT.parent / f"{meta['prior_run']}.json").read_text())
            entry.update(paired_inputs(inputs, previous_inputs))
            entry["correctness_changes"] = correctness_changes(
                labels, previous_labels, inputs, previous_inputs
            )
            entry.update(
                historical_run_id=meta["prior_run"],
                historical_overall=prior["scores"]["overall"],
                delta_percentage_points=100
                * (record["scores"]["overall"] - prior["scores"]["overall"]),
                historical_errors=technical_errors(previous_labels, previous_answers),
                historical_dataset_score=prior["scores"].get("dataset_score"),
            )
        summary["datasets"].append(entry)
        total += n
        print(f"{dataset}: {correct}/{n}; technical errors {len(errors)}", flush=True)
    summary["complete_question_count"] = total
    if summary["complete"] and total != 3001:
        raise RuntimeError(f"Expected 3001 total frozen questions, got {total}")
    suffix = "historical" if historical else arm
    write(OUT / f"audit-full-{suffix}.json", summary)
    print("complete:", summary["complete"], "questions:", total, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm")
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    audit(args.arm, historical=args.historical, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
