"""先手工把旧成功案例的证据改成共同字段；不改产品或答案链。"""

from __future__ import annotations

import json
import re
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


def fact(subject: str, relation: str, obj: str, quote: str, date: str, **attrs: object) -> str:
    statement = attrs.pop("statement", f"{subject} — {relation} — {obj}")
    lines = ["Memory fact", f"Statement: {statement}"]
    # 数量引句另存于实验审计与产品事实索引，不在正文重复同一项数量。
    lines.append(f"Source record date: {date}")
    if "quantity" not in attrs:
        lines.append(f"Source quotation: {quote}")
    return "\n".join(lines)


def main() -> None:
    selected = {
        "kb_qa-3": "corporatebench",
        "kb_qa-7": "corporatebench",
        "eeda8a6d": "longmemeval-s",
        "conv-26#q0011": "locomo-refined",
        "conv-30#q0018": "locomo-refined",
        "conv-41#q0006": "locomo-refined",
        "conv-30#q0012": "locomo-refined",
        "conv-30#q0017": "locomo-refined",
    }
    items = {
        x["id"]: x
        for x in map(json.loads, (OLD / "regression-input.jsonl").read_text().splitlines())
    }
    db = sqlite3.connect(
        f"file:{ROOT / 'var/generic-evidence-20261005/regression/tianxi-snapshot.db'}?mode=ro",
        uri=True,
    )
    db.row_factory = sqlite3.Row
    results, previews = [], []
    for qid, dataset in selected.items():
        path = next((OLD / "regression").glob(f"*/{qid}/search.json"))
        hits = json.loads(path.read_text())["hits"]
        fragments = []
        for hit in hits:
            text = hit["content"]
            fields = dict(re.findall(r"^([A-Za-z ]+): (.*)$", text, re.M))
            date = fields.get("Source record date", fields.get("Document date", ""))
            if text.startswith("Employment record"):
                row = db.execute(
                    "SELECT q.question FROM employment_facts f "
                    "JOIN qa_pairs q ON q.id=f.parent_memory_id WHERE f.id=?",
                    (hit["id"],),
                ).fetchone()
                assert row is not None
                signature = re.search(
                    r"(?m)^"
                    + re.escape(fields["Person"])
                    + r"\s*\n(?:[^\n]+\n){0,2}[^\n]*"
                    + re.escape(fields["Employer"]),
                    row["question"],
                )
                quote = signature[0].strip() if signature else fields["Source claim"]
                assert quote in row["question"]
                fragments.append(
                    fact(
                        fields["Person"],
                        "employee of",
                        fields["Employer"],
                        quote,
                        date,
                        statement=f"{fields['Person']} is an employee of {fields['Employer']}.",
                    )
                )
            elif text.startswith(("Self-reported role", "Orientation observation")):
                relation = (
                    "role commencement"
                    if text.startswith("Self-reported")
                    else "orientation schedule"
                )
                fragments.append(
                    fact(
                        fields["Person"],
                        relation,
                        fields["Organization mentioned"],
                        fields["Source claim"],
                        date,
                        statement=(
                            f"{fields['Person']} reports {relation} "
                            f"at {fields['Organization mentioned']}."
                        ),
                    )
                )
            elif text.startswith("Aquarium inventory"):
                name = f" named {fields['Name']}" if "Name" in fields else ""
                fragments.append(
                    fact(
                        "I",
                        "contains",
                        fields["Item"],
                        fields["Source claim"],
                        date,
                        quantity=int(fields["Quantity"]),
                        statement=(
                            f"My {fields['Aquarium']} contains "
                            f"{fields['Quantity']} {fields['Item']}{name}."
                        ),
                    )
                )
            elif text.startswith("Conversation statement"):
                subject, quote = fields["Speaker"], fields["Statement"]
                if qid.endswith("q0018"):
                    relation, obj = (
                        "participated in",
                        "a fair" if "went to a fair" in quote else "networking events",
                    )
                elif qid.endswith("q0012"):
                    relation, obj = "is open", "online clothes store"
                else:
                    relation, obj = "accepted for", "fashion internship"
                statement = {
                    "participated in": f"{subject} participated in {obj}.",
                    "is open": f"{subject}'s {obj} is open.",
                    "accepted for": f"{subject} just got accepted for a {obj}.",
                }[relation]
                fragments.append(fact(subject, relation, obj, quote, date, statement=statement))
            elif qid == "conv-26#q0011":
                relation, obj = (
                    ("home country", "Sweden")
                    if "home country is" in text
                    else ("moved from", "my home country")
                )
                statement = text.split("\nSource record date:", 1)[0]
                fragments.append(
                    fact(
                        "Caroline",
                        relation,
                        obj,
                        fields["Source quotation"],
                        date,
                        statement=statement,
                    )
                )
            else:
                if "fellow volunteers" in text:
                    relation, obj = "friends with", "fellow volunteers"
                elif "volunteers at" in text:
                    relation, obj = "volunteer at", "a homeless shelter"
                else:
                    relation, obj = (
                        "friends from",
                        "church" if "church friends" in text else "the gym",
                    )
                fragments.append(
                    fact(
                        "Maria",
                        relation,
                        obj,
                        fields["Source quotation"],
                        date,
                        statement=text.split("\nSource record date:", 1)[0],
                    )
                )
        context = "\n".join(fragments)
        previews.append({"dataset": dataset, "qid": qid, "fragments": fragments})
        (OUT / "preview-contexts.json").write_text(
            json.dumps(previews, ensure_ascii=False, indent=2) + "\n"
        )
        item = dict(items[qid])
        item["retrieved_context" if dataset in EXTRA_DATASETS else "speaker_1_memories"] = context
        if dataset not in EXTRA_DATASETS:
            item["speaker_2_memories"] = ""
        result = run_judge(
            pipeline_for(benchmark_dir(), dataset), [item], OUT / "preview" / qid, dataset=dataset
        )[0]
        results.append({"dataset": dataset, **asdict(result)})
        (OUT / "preview-results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n"
        )
        print(qid, result.is_correct, result.generated_answer, flush=True)
    db.close()
    assert len(results) == 8 and all(row["is_correct"] for row in results)


if __name__ == "__main__":
    main()
