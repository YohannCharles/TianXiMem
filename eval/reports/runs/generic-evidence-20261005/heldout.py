"""实现冻结后增加诊所患者/带容量的箱子检查；先手工片段，再原文 HTTP Add。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval.datasets import benchmark_dir  # noqa: E402
from eval.harness import ServiceClient, pipeline_for, run_judge  # noqa: E402
from tools.targeted_eval import freeze  # noqa: E402

PEOPLE = ["Rosa Quinn", "Evan Cole", "Tessa Roy"]
SOURCES = {
    "clinic": [
        "Rosa Quinn: I'm a patient at Aspen Clinic.",
        "Evan Cole: I am currently a patient at Aspen Clinic.",
        "Tessa Roy: I am a patient at Aspen Clinic.",
        "Rosa Quinn: I am still a patient at Aspen Clinic. This is the same patient relationship.",
        "Leah Stone: I hope to become a patient at Aspen Clinic next month. I have not registered.",
        "Owen Bell: I visited Aspen Clinic to deliver a parcel. I am not a patient.",
        "Tessa Roy: A fictional example said 'Owen Bell is a patient at Aspen Clinic.' "
        "It is not a report about Owen.",
    ],
    "crate": [
        "Rosa Quinn: My 10-liter gray crate currently contains four cups, "
        "two plates, and one kettle.",
        "Rosa Quinn: The same 10-liter gray crate still contains four cups, "
        "two plates, and one kettle.",
        "Rosa Quinn: I plan to add five cups to the 10-liter gray crate next week. "
        "I have not bought them.",
        "Evan Cole: My 10-liter gray crate contains nine cups. It is separate from Rosa's.",
        "Rosa Quinn: My red shelf contains twelve plates, not my 10-liter gray crate.",
        "Rosa Quinn: I did not buy the fork mentioned in my shopping list.",
        "Tessa Roy: A fictional example said "
        "'Rosa Quinn has eight kettles in a 10-liter gray crate.' "
        "It is not a report about Rosa.",
    ],
}
QUESTIONS = {
    "clinic": [
        "Who are the patients at Aspen Clinic?",
        "How many patients are there at Aspen Clinic?",
        "List all patients at Aspen Clinic.",
        "What is the total number of patients at Aspen Clinic?",
    ],
    "crate": [
        "What types of items are in Rosa Quinn's 10-liter gray crate?",
        "How many items are in Rosa Quinn's 10-liter gray crate in total?",
        "List the kinds of belongings Rosa Quinn currently keeps in the 10-liter gray crate.",
        "What is the total number of items currently inside the 10-liter gray crate "
        "owned by Rosa Quinn?",
    ],
}


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def code_fingerprint() -> dict:
    paths = [
        "facts/evidence.py",
        "facts/index.py",
        "common/render.py",
        "common/config.py",
        "store/sqlite_store.py",
        "store/schema.sql",
        "pairing/apply.py",
        "service/pipeline.py",
        "service/app.py",
    ]
    return {
        path: hashlib.sha256((ROOT / "src/tianximem" / path).read_bytes()).hexdigest()
        for path in paths
    }


def prepare() -> tuple[list[dict], dict]:
    contexts = {
        "clinic": [
            "\n".join(
                [
                    "Memory fact",
                    f"Subject: {person}",
                    "Relation: patient at",
                    "Object: Aspen Clinic",
                    "Source record date: 2025-09-01",
                    "Source quotation: " + SOURCES["clinic"][index],
                ]
            )
            for index, person in enumerate(PEOPLE)
        ],
        "crate": [
            "\n".join(
                [
                    "Memory fact",
                    "Subject: Rosa Quinn",
                    "Relation: contains",
                    f"Object: {item}",
                    "Container: 10-liter gray crate",
                    f"Quantity: {quantity}",
                    "Source record date: 2025-09-01",
                    "Source quotation: " + SOURCES["crate"][0],
                ]
            )
            for item, quantity in [("cups", 4), ("plates", 2), ("kettle", 1)]
        ],
    }
    items = []
    for domain, questions in QUESTIONS.items():
        names = PEOPLE if domain == "clinic" else ["cups", "plates", "kettle"]
        count = 3 if domain == "clinic" else 7
        for variant, question, answer in zip(
            ("list-canonical", "count-canonical", "list-paraphrase", "count-paraphrase"),
            questions,
            [names, count, names, count],
            strict=True,
        ):
            items.append(
                {
                    "id": f"heldout-{domain}-{variant}",
                    "dataset": "corporatebench",
                    "question": question,
                    "gold_answer": {
                        "answer": answer,
                        "answer_type": "int" if isinstance(answer, int) else "List[str]",
                    },
                }
            )
    path = OUT / "heldout-input.jsonl"
    encoded = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
    if path.exists():
        assert path.read_text() == encoded
    path.write_text(encoded)
    manifest = {
        "top_k": 100,
        "groups": [
            {
                "dataset": "corporatebench",
                "user_id": f"generic-evidence-20261005-heldout-{domain}",
                "group": domain,
                "source": str(path.relative_to(ROOT)),
                "qids": [
                    item["id"] for item in items if item["id"].startswith(f"heldout-{domain}-")
                ],
            }
            for domain in SOURCES
        ],
    }
    write("heldout-sources.json", SOURCES)
    write("heldout-manifest.json", manifest)
    frozen = freeze(manifest)
    path = OUT / "heldout-frozen.json"
    if path.exists():
        assert json.loads(path.read_text()) == frozen
    write("heldout-frozen.json", frozen)
    path = OUT / "heldout-code.json"
    if path.exists():
        assert json.loads(path.read_text()) == code_fingerprint(), (
            "code changed after heldout freeze"
        )
    write("heldout-code.json", code_fingerprint())
    return items, contexts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "manual", "ingest"))
    parser.add_argument("--base-url")
    args = parser.parse_args()
    items, contexts = prepare()
    if args.stage == "manual":
        results = []
        for item in items:
            domain = item["id"].split("-")[1]
            item = {**item, "retrieved_context": "\n".join(contexts[domain])}
            result = run_judge(
                pipeline_for(benchmark_dir(), "corporatebench"),
                [item],
                OUT / "heldout-manual" / item["id"],
                dataset="corporatebench",
            )[0]
            results.append(asdict(result))
            write("heldout-manual-results.json", results)
            print(item["id"], result.is_correct, result.generated_answer, flush=True)
    elif args.stage == "ingest":
        assert args.base_url is not None
        manual = json.loads((OUT / "heldout-manual-results.json").read_text())
        assert len(manual) == 8 and all(row["is_correct"] for row in manual)
        with ServiceClient(args.base_url, timeout=180) as client:
            for domain, texts in SOURCES.items():
                client.add(
                    request_id=f"generic-heldout-{domain}",
                    user_id=f"generic-evidence-20261005-heldout-{domain}",
                    session_id=f"original-{domain}",
                    messages=[
                        {"role": "user", "content": text, "timestamp": 1756684800000}
                        for text in texts
                    ],
                )
                print(domain, "original Add done", flush=True)


if __name__ == "__main__":
    main()
