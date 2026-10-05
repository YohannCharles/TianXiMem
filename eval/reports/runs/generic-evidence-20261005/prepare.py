"""通用关系/库存事实格式的手工预检；先冻结，再修改产品。"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
BASE = OUT.parent / "generalization-20261005"
sys.path.insert(0, str(ROOT))

from eval.datasets import benchmark_dir  # noqa: E402
from eval.harness import pipeline_for, run_judge  # noqa: E402
from tools.targeted_eval import freeze  # noqa: E402


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def fact(subject: str, relation: str, obj: str, quote: str, **qualifiers: object) -> str:
    lines = ["Memory fact", f"Subject: {subject}", f"Relation: {relation}", f"Object: {obj}"]
    lines.extend(f"{key}: {value}" for key, value in qualifiers.items())
    lines.extend(["Source record date: 2025-09-01", f"Source quotation: {quote}"])
    return "\n".join(lines)


def main() -> None:
    sources = json.loads((BASE / "sources.json").read_text())
    items = list(map(json.loads, (BASE / "input.jsonl").read_text().splitlines()))
    people = ["Maya Chen", "Noah Reed", "Inez Patel"]
    contexts = {}
    for domain, relation, target in (
        ("company", "employee of", "Arden Labs"),
        ("school", "student at", "Cedar School"),
        ("volunteers", "volunteer at", "Harbor Shelter"),
    ):
        contexts[domain] = [
            fact(person, relation, target, sources[domain][index])
            for index, person in enumerate(people)
        ]
    contexts["backpack"] = [
        fact(
            people[0],
            "contains",
            item,
            sources["backpack"][0],
            Container="blue backpack",
            Quantity=quantity,
        )
        for item, quantity in (("notebooks", 2), ("pens", 3), ("ruler", 1))
    ]
    sources["club"] = [f"{person}: I am a member of Ridge Hiking Club." for person in people] + [
        "Maya Chen: I am still a member of Ridge Hiking Club. This is the same membership.",
        "Leah Stone: I hope to become a member of Ridge Hiking Club. I have not joined yet.",
        "Owen Bell: I visited Ridge Hiking Club as a guest. I am not a member.",
        "Inez Patel: A fictional example said 'Owen Bell belongs to Ridge Hiking Club.' "
        "It is not a report about Owen.",
    ]
    contexts["club"] = [
        fact(person, "member of", "Ridge Hiking Club", sources["club"][index])
        for index, person in enumerate(people)
    ]
    sources["toolbox"] = [
        "Noah Reed: My green toolbox currently contains four bolts, two nuts, and three washers.",
        "Noah Reed: The same green toolbox still contains four bolts, two nuts, and three washers.",
        "Noah Reed: I plan to add five more bolts next week. I have not bought them.",
        "Maya Chen: My green toolbox contains eleven bolts. My toolbox is separate from Noah's.",
        "Noah Reed: My red drawer contains seven washers, not my green toolbox.",
        "Noah Reed: I did not buy the hammer mentioned in my old shopping list.",
        "Inez Patel: A fictional example said 'Noah Reed keeps eight nuts in a green toolbox.' "
        "It is not a report about Noah's real belongings.",
    ]
    contexts["toolbox"] = [
        fact(
            people[1],
            "contains",
            item,
            sources["toolbox"][0],
            Container="green toolbox",
            Quantity=quantity,
        )
        for item, quantity in (("bolts", 4), ("nuts", 2), ("washers", 3))
    ]
    for domain, questions, gold in (
        (
            "club",
            [
                "Who are the members of Ridge Hiking Club?",
                "How many members are there in Ridge Hiking Club?",
                "Who belongs to Ridge Hiking Club?",
                "What is the total number of people who belong to Ridge Hiking Club?",
            ],
            [people, 3, people, 3],
        ),
        (
            "toolbox",
            [
                "What types of items are in Noah Reed's green toolbox?",
                "How many items are in Noah Reed's green toolbox in total?",
                "List the kinds of belongings Noah Reed currently keeps in the green toolbox.",
                "What is the total number of items currently inside the green toolbox "
                "owned by Noah Reed?",
            ],
            [["bolts", "nuts", "washers"], 9, ["bolts", "nuts", "washers"], 9],
        ),
    ):
        for variant, question, answer in zip(
            ("list-canonical", "count-canonical", "list-paraphrase", "count-paraphrase"),
            questions,
            gold,
            strict=True,
        ):
            items.append(
                {
                    "id": f"generalization-{domain}-{variant}",
                    "dataset": "corporatebench",
                    "question": question,
                    "gold_answer": {
                        "answer": answer,
                        "answer_type": "int" if isinstance(answer, int) else "List[str]",
                    },
                    "category": "synthetic-generalization",
                }
            )
    groups = [
        {
            "dataset": "corporatebench",
            "user_id": f"generic-evidence-20261005-{domain}",
            "source": str((OUT / "input.jsonl").relative_to(ROOT)),
            "group": domain,
            "qids": [
                item["id"] for item in items if item["id"].startswith(f"generalization-{domain}-")
            ],
        }
        for domain in sources
    ]
    encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in items)
    path = OUT / "input.jsonl"
    if path.exists():
        assert path.read_text() == encoded
    path.write_text(encoded)
    write("sources.json", sources)
    write("manual-contexts.json", contexts)
    manifest = {"top_k": 100, "groups": groups}
    write("manifest.json", manifest)
    frozen = freeze(manifest)
    path = OUT / "frozen.json"
    if path.exists():
        assert json.loads(path.read_text()) == frozen
    write("frozen.json", frozen)
    results = []
    for item in items:
        domain = item["id"].split("-")[1]
        item = {**item, "retrieved_context": "\n".join(contexts[domain])}
        result = run_judge(
            pipeline_for(benchmark_dir(), "corporatebench"),
            [item],
            OUT / "manual" / item["id"],
            dataset="corporatebench",
        )[0]
        results.append(asdict(result))
        write("manual-results.json", results)
        print(item["id"], result.is_correct, result.generated_answer, flush=True)


if __name__ == "__main__":
    main()
