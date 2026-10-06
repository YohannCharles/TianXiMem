"""Compare paired native runs, retaining errors and exact context fingerprints."""

from __future__ import annotations

import argparse
import hashlib
import json

from experiment import NAME, OUT, SETS, write


def records(arm: str, phase: str, dataset: str, kind: str) -> dict[str, dict]:
    directory = OUT.parent / f"{NAME}-{arm}-{phase}-{dataset}"
    result = {}
    for path in directory.glob(f"*/{kind}.jsonl"):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["id"])
            if qid in result:
                raise RuntimeError(f"Duplicate {kind} id in {directory}: {qid}")
            result[qid] = row
    return result


def technical_errors(labels: dict[str, dict], answers: dict[str, dict]) -> list[str]:
    failures = []
    for qid, row in labels.items():
        answer = answers.get(qid, {})
        if (
            row.get("label") in {"API_ERROR", "ERROR", "ANSWER_ERROR", "JUDGE_ERROR"}
            or answer.get("error")
            or row.get("error")
            or str(row.get("label", "")).startswith("ERROR")
        ):
            failures.append(qid)
    return failures


def compare(before: str, after: str, phase: str, datasets: list[str]) -> dict:
    result = {"before": before, "after": after, "phase": phase, "datasets": []}
    for dataset in datasets:
        a = records(before, phase, dataset, "labels")
        b = records(after, phase, dataset, "labels")
        ai = records(before, phase, dataset, "input")
        bi = records(after, phase, dataset, "input")
        aa = records(before, phase, dataset, "answers")
        ba = records(after, phase, dataset, "answers")
        if not a or not b:
            raise RuntimeError(f"Missing scored run: {dataset}")
        if not (a.keys() == b.keys() == ai.keys() == bi.keys() == aa.keys() == ba.keys()):
            raise RuntimeError(f"Paired question ids differ: {dataset}")
        immutable_fields = {"id", "dataset", "question", "gold_answer", "category"}
        for qid in a:
            if any(ai[qid].get(k) != bi[qid].get(k) for k in immutable_fields):
                raise RuntimeError(f"Frozen scoring input changed: {dataset}/{qid}")
        cases = []
        changed = 0
        for qid in a:
            equal = ai[qid] == bi[qid]
            changed += not equal
            old_correct = bool(a[qid].get("is_correct"))
            new_correct = bool(b[qid].get("is_correct"))
            if old_correct != new_correct:
                cases.append(
                    {
                        "qid": qid,
                        "question": ai[qid]["question"],
                        "before_correct": old_correct,
                        "after_correct": new_correct,
                        "identical_input": equal,
                        "before_answer": aa[qid].get("generated_answer"),
                        "after_answer": ba[qid].get("generated_answer"),
                    }
                )
        fingerprints = {
            arm: hashlib.sha256(
                json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            for arm, inputs in ((before, ai), (after, bi))
        }
        entry = {
            "dataset": dataset,
            "n": len(a),
            "before_correct": sum(bool(r.get("is_correct")) for r in a.values()),
            "after_correct": sum(bool(r.get("is_correct")) for r in b.values()),
            "wins": sum(c["after_correct"] for c in cases),
            "losses": sum(c["before_correct"] for c in cases),
            "changed_inputs": changed,
            "input_sha256": fingerprints,
            "before_errors": technical_errors(a, aa),
            "after_errors": technical_errors(b, ba),
            "changed_correctness": cases,
        }
        result["datasets"].append(entry)
        print(
            f"{dataset}: {entry['before_correct']}/{len(a)} -> "
            f"{entry['after_correct']}/{len(b)}; inputs changed {changed}; "
            f"win/loss {entry['wins']}/{entry['losses']}; "
            f"errors {len(entry['before_errors'])}/{len(entry['after_errors'])}"
        )
    write(OUT / f"compare-{before}-{after}-{phase}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("phase", help="Frozen small/stage/full or a named targeted selection")
    parser.add_argument("--datasets", nargs="+", choices=SETS, default=list(SETS))
    args = parser.parse_args()
    compare(args.before, args.after, args.phase, args.datasets)


if __name__ == "__main__":
    main()
