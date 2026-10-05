"""把实际返回事实手工改成候选表示，验证通过后才能改产品。"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
OLD = OUT.parent / "generic-evidence-20261005"
sys.path.insert(0, str(ROOT))

from eval.datasets import benchmark_dir  # noqa: E402
from eval.harness import pipeline_for, run_judge  # noqa: E402
from eval.harness.judge import EXTRA_DATASETS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--mode", choices=("fields", "plain", "readable"), required=True)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    folder = OUT / args.phase
    folder.mkdir(exist_ok=True)
    items = {
        row["id"]: row
        for row in map(json.loads, (OLD / "regression-input.jsonl").read_text().splitlines())
    }
    rows = json.loads((OUT / "candidate-final/regression/results.json").read_text())
    selected = {"kb_qa-3", "kb_qa-7", "conv-41#q0006"}
    if args.all:
        selected = {row["qid"] for row in json.loads((OUT / "preview-contexts.json").read_text())}
    conn = sqlite3.connect(
        f"file:{ROOT / 'var/unified-evidence-20261005/candidate/tianxi.db'}?mode=ro", uri=True
    )
    conn.row_factory = sqlite3.Row
    results = []
    for row in rows:
        if row["qid"] not in selected:
            continue
        fragments = []
        hits = row["hits"]
        if args.mode == "readable":
            # 手工排列已取到的独立事实，先验证直述/引用的共同排序及来源措辞。
            if row["qid"] == "kb_qa-3":
                hits = sorted(hits, key=lambda h: h["content"].split("Statement: ")[1])
            elif row["qid"] == "kb_qa-7":
                hits = sorted(hits, key=lambda h: "orientation schedule" in h["content"])
            elif row["qid"] == "conv-41#q0006":
                hits = next(
                    x["fragments"]
                    for x in json.loads((OUT / "preview-contexts.json").read_text())
                    if x["qid"] == row["qid"]
                )
                assert "fellow volunteers" in hits[0]
                hits = [{"id": "manual", "content": text} for text in hits]
        for hit in hits:
            fact = (
                dict(conn.execute("SELECT * FROM memory_facts WHERE id=?", (hit["id"],)).fetchone())
                if hit["id"] != "manual"
                else {}
            )
            text = hit["content"]
            if args.mode == "fields":
                text = text.replace(
                    "Memory fact\n",
                    f"Memory fact\nSubject: {fact['subject']}\n"
                    f"Relation: {fact['relation']}\nObject: {fact['object']}\n",
                    1,
                )
            elif args.mode == "plain":
                text = text.replace("Memory fact\nStatement: ", "", 1)
            elif args.mode == "readable" and fact:
                if fact["relation"] == "role commencement":
                    statement = (
                        f"{fact['subject']} reports having commenced "
                        f"their role at {fact['object']}."
                    )
                    text = text.replace(fact["statement"], statement, 1)
                elif fact["relation"] == "orientation schedule":
                    statement = (
                        f"{fact['subject']} reports going through "
                        f"{fact['object']}'s orientation schedule."
                    )
                    text = text.replace(fact["statement"], statement, 1)
            fragments.append(text)
        item = dict(items[row["qid"]])
        item["retrieved_context" if row["dataset"] in EXTRA_DATASETS else "speaker_1_memories"] = (
            "\n".join(fragments)
        )
        if row["dataset"] not in EXTRA_DATASETS:
            item["speaker_2_memories"] = ""
        result = run_judge(
            pipeline_for(benchmark_dir(), row["dataset"]),
            [item],
            folder / row["qid"],
            dataset=row["dataset"],
        )[0]
        results.append({"dataset": row["dataset"], "fragments": fragments, **asdict(result)})
        (folder / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(row["qid"], result.is_correct, result.generated_answer, flush=True)
    conn.close()
    assert len(results) == len(selected) and all(row["is_correct"] for row in results)


if __name__ == "__main__":
    main()
