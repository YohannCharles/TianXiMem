"""手工验证共同数量字段，容量和数量分列；不改产品、不求总数。"""

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
    parser.add_argument("--mode", choices=("fields", "json", "quote"), default="fields")
    args = parser.parse_args()
    output = OUT / (
        "quantity-preview" if args.mode == "fields" else f"quantity-preview-{args.mode}"
    )
    rows = []
    for scope, arm in (("regression", "candidate"), ("generalization", "generalization")):
        conn = sqlite3.connect(
            f"file:{ROOT / 'var/unified-evidence-20261005' / arm / 'tianxi.db'}?mode=ro", uri=True
        )
        conn.row_factory = sqlite3.Row
        sources = (
            ["regression-input.jsonl"]
            if scope == "regression"
            else ["input.jsonl", "heldout-input.jsonl"]
        )
        items = {
            row["id"]: row
            for name in sources
            for row in map(json.loads, (OLD / name).read_text().splitlines())
        }
        results = json.loads((OUT / "candidate-repaired" / scope / "results.json").read_text())
        for row in results:
            if scope == "regression" and row["qid"] != "eeda8a6d":
                continue
            if scope == "generalization" and not any(
                x in row["qid"] for x in ("backpack", "toolbox", "crate")
            ):
                continue
            fragments = []
            facts = [
                dict(conn.execute("SELECT * FROM memory_facts WHERE id=?", (hit["id"],)).fetchone())
                for hit in row["hits"]
            ]
            used = set()
            for hit in row["hits"]:
                fact = dict(
                    conn.execute("SELECT * FROM memory_facts WHERE id=?", (hit["id"],)).fetchone()
                )
                qualifiers = json.loads(fact["qualifiers"])
                assert "quantity" in qualifiers
                if args.mode == "quote":
                    identity = (fact["parent_memory_id"], fact["source_side"], fact["source_quote"])
                    if identity in used:
                        continue
                    used.add(identity)
                    related = [
                        f
                        for f in facts
                        if (f["parent_memory_id"], f["source_side"], f["source_quote"]) == identity
                    ]
                    if len(related) > 1:
                        fragments.append(
                            f"Memory source\nSpeaker: {fact['subject']}\n"
                            f"Source record date: {fact['source_date']}\n"
                            f"Source quotation: {fact['source_quote']}"
                        )
                    else:
                        fragments.append(hit["content"])
                    continue
                fields = {
                    "Subject": fact["subject"],
                    "Relation": fact["relation"],
                    "Object": fact["object"],
                    "Container": qualifiers["container"],
                    "Quantity": qualifiers["quantity"],
                }
                if qualifiers.get("name"):
                    fields["Name"] = qualifiers["name"]
                fields["Source record date"] = fact["source_date"]
                if args.mode == "json":
                    payload = {
                        "subject": fact["subject"],
                        "relation": fact["relation"],
                        "object": fact["object"],
                        "qualifiers": {
                            "container": qualifiers["container"],
                            "quantity": qualifiers["quantity"],
                        },
                        "source_record_date": fact["source_date"],
                    }
                    if qualifiers.get("name"):
                        payload["qualifiers"]["name"] = qualifiers["name"]
                    fragments.append("Memory fact\n" + json.dumps(payload, ensure_ascii=False))
                else:
                    fragments.append(
                        "Memory fact\n" + "\n".join(f"{k}: {v}" for k, v in fields.items())
                    )
            item = dict(items[row["qid"]])
            item[
                "retrieved_context" if row["dataset"] in EXTRA_DATASETS else "speaker_1_memories"
            ] = "\n".join(fragments)
            if row["dataset"] not in EXTRA_DATASETS:
                item["speaker_2_memories"] = ""
            result = run_judge(
                pipeline_for(benchmark_dir(), row["dataset"]),
                [item],
                output / row["qid"],
                dataset=row["dataset"],
            )[0]
            rows.append({"dataset": row["dataset"], "fragments": fragments, **asdict(result)})
            output.with_name(output.name + "-results.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
            )
            print(row["qid"], result.is_correct, result.generated_answer, flush=True)
        conn.close()
    assert len(rows) == 13 and all(row["is_correct"] for row in rows)


if __name__ == "__main__":
    main()
