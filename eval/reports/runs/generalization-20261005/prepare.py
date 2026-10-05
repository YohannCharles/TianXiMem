"""冻结跨领域合成诊断；原 Add 语料与手工事实对照分开，不修改产品。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
DATE = "2025-09-01"
TIMESTAMP = 1756684800000
PEOPLE = ["Maya Chen", "Noah Reed", "Inez Patel"]


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def frozen_text(path: Path, text: str) -> None:
    if path.exists() and path.read_text() != text:
        raise RuntimeError(f"cannot mutate frozen input/config: {path}")
    path.write_text(text)


def email(person: str, role: str, body: str, index: int) -> str:
    address = person.lower().replace(" ", ".")
    return (
        f"document: Message-ID: generalization-{index}\n"
        f"From: {person} <{address}@arden.example>\nDate: {DATE}\n"
        f"Subject: Weekly technical notes {index}\n\n{body}\n\n"
        f"{person}\n{role}\nArden Labs"
    )


sources = {
    "company": [
        email(PEOPLE[0], "Software Engineer", "Here are my weekly technical notes.", 0),
        email(PEOPLE[1], "Data Analyst", "The data review is ready.", 1),
        email(PEOPLE[2], "Marketing Specialist", "The campaign review is ready.", 2),
        email(PEOPLE[0], "Software Engineer", "A second update from me this week.", 3),
        "Leah Stone: I hope to start a role at Arden Labs next spring. I have not joined yet.",
        "Owen Bell: I visited Arden Labs once as an external supplier. I am not an employee there.",
        "Leah Stone: A fictional example said 'Owen Bell is an engineer at Arden Labs.' "
        "That is a made-up example, not a report about Owen.",
    ],
    "school": [
        "Maya Chen: I am currently enrolled as a student at Cedar School.",
        "Noah Reed: I am currently enrolled as a student at Cedar School.",
        "Inez Patel: I am currently enrolled as a student at Cedar School.",
        "Maya Chen: I am still enrolled at Cedar School. "
        "This is an update about the same enrollment.",
        "Leah Stone: I hope to enroll at Cedar School next term. I have not enrolled yet.",
        "Owen Bell: I visited Cedar School once as a guest. I do not study there.",
        "Inez Patel: A fictional example in class said 'Owen Bell studies at Cedar School.' "
        "This example is not a report about Owen.",
    ],
    "volunteers": [
        "Maya Chen: I volunteer at Harbor Shelter.",
        "Noah Reed: I volunteer at Harbor Shelter.",
        "Inez Patel: I volunteer at Harbor Shelter.",
        "Maya Chen: I volunteer at Harbor Shelter. "
        "This is another report about my same volunteer role.",
        "Leah Stone: I plan to volunteer at Harbor Shelter next month. I have not started yet.",
        "Owen Bell: I visited Harbor Shelter once to deliver a parcel. I am not a volunteer there.",
        "Inez Patel: A fictional example said 'Owen Bell volunteers at Harbor Shelter.' "
        "This example is not a report about Owen.",
    ],
    "backpack": [
        "Maya Chen: My blue backpack currently has two notebooks, three pens, and one ruler.",
        "Maya Chen: The same blue backpack still has two notebooks, three pens, and one ruler.",
        "Maya Chen: I plan to add four more pens to the blue backpack next week. "
        "I have not bought them.",
        "Owen Bell: My blue backpack currently has nine pens. My backpack is separate from Maya's.",
        "Noah Reed: I keep twelve pens in my red desk drawer, not in a backpack.",
        "Maya Chen: My old shopping list mentioned a calculator, "
        "but I did not buy it or put it in my backpack.",
        "Inez Patel: A fictional example said 'Maya Chen has eight rulers in a blue backpack.' "
        "This is not a report about Maya's real belongings.",
    ],
}

facts: dict[str, list[dict]] = {}
for domain, relation, value in [
    ("company", "works at", "Arden Labs"),
    ("school", "is enrolled as a student at", "Cedar School"),
    ("volunteers", "volunteers at", "Harbor Shelter"),
]:
    facts[domain] = []
    for slot, person in enumerate(PEOPLE):
        quote = sources[domain][slot]
        if domain == "company":
            quote = quote[quote.rfind("\n\n") + 2 :]
        facts[domain].append(
            {
                "source_slot": slot,
                "quote": quote,
                "content": (
                    f"{person} {relation} {value}.\n"
                    f"Source record date: {DATE}\nSource quotation: {quote}"
                ),
            }
        )

facts["backpack"] = []
for item, quantity, quote in [
    ("notebooks", 2, "two notebooks"),
    ("pens", 3, "three pens"),
    ("ruler", 1, "one ruler"),
]:
    facts["backpack"].append(
        {
            "source_slot": 0,
            "quote": quote,
            "content": (
                f"Belongings record\nOwner: Maya Chen\nContainer: blue backpack\n"
                f"Item: {item}\nQuantity: {quantity}\nSource record date: {DATE}\n"
                f"Source quotation: {sources['backpack'][0]}"
            ),
        }
    )

questions = {
    "company": [
        ("list-canonical", "Who are the employees of Arden Labs?", "List[str]", PEOPLE),
        (
            "count-canonical",
            "What is the total number of employees mentioned in the corpus?",
            "int",
            3,
        ),
        ("list-paraphrase", "Who works at Arden Labs?", "List[str]", PEOPLE),
        ("count-paraphrase", "How many people currently work for Arden Labs?", "int", 3),
    ],
    "school": [
        ("list-canonical", "Who are the students at Cedar School?", "List[str]", PEOPLE),
        ("count-canonical", "How many students are studying at Cedar School?", "int", 3),
        ("list-paraphrase", "List everyone studying at Cedar School.", "List[str]", PEOPLE),
        ("count-paraphrase", "What is the total number of students at Cedar School?", "int", 3),
    ],
    "volunteers": [
        ("list-canonical", "Who volunteers at Harbor Shelter?", "List[str]", PEOPLE),
        ("count-canonical", "How many people volunteer at Harbor Shelter?", "int", 3),
        (
            "list-paraphrase",
            "List the people who volunteer at Harbor Shelter.",
            "List[str]",
            PEOPLE,
        ),
        ("count-paraphrase", "What is the number of volunteers at Harbor Shelter?", "int", 3),
    ],
    "backpack": [
        (
            "list-canonical",
            "What types of items are in Maya Chen's blue backpack?",
            "List[str]",
            ["notebooks", "pens", "ruler"],
        ),
        ("count-canonical", "How many items are in Maya Chen's blue backpack in total?", "int", 6),
        (
            "list-paraphrase",
            "List the kinds of belongings Maya Chen currently keeps in the blue backpack.",
            "List[str]",
            ["notebooks", "pens", "ruler"],
        ),
        (
            "count-paraphrase",
            "What is the total number of items currently inside the blue backpack "
            "owned by Maya Chen?",
            "int",
            6,
        ),
    ],
}

items, manual_items, groups = [], [], []
for domain, rows in questions.items():
    qids = []
    for variant, question, answer_type, answer in rows:
        qid = f"generalization-{domain}-{variant}"
        item = {
            "id": qid,
            "dataset": "corporatebench",
            "question": question,
            "gold_answer": {"answer": answer, "answer_type": answer_type},
            "category": "synthetic-generalization",
        }
        items.append(item)
        manual_items.append(
            {**item, "retrieved_context": "\n".join(f["content"] for f in facts[domain])}
        )
        qids.append(qid)
    groups.append(
        {
            "dataset": "corporatebench",
            "user_id": f"generalization-20261005-{domain}",
            "source": str((OUT / "input.jsonl").relative_to(ROOT)),
            "qids": qids,
            "group": domain,
        }
    )

for domain, records in facts.items():
    for record in records:
        assert record["quote"] in sources[domain][record["source_slot"]]
assert len(items) == 16
for name, rows in [("input.jsonl", items), ("manual-input.jsonl", manual_items)]:
    path = OUT / name
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    frozen_text(path, text)
write("sources.json", sources)
write("manual-facts.json", facts)
write("manifest.json", {"top_k": 100, "groups": groups})
write(
    "protocol.json",
    {
        "scope": "synthetic transfer diagnostic, not benchmark questions or score",
        "domains": list(sources),
        "questions": len(items),
        "question_hash": hashlib.sha256((OUT / "input.jsonl").read_bytes()).hexdigest(),
        "manual_facts_are_diagnostic_only": True,
        "http_add_contains_only_sources": True,
        "answer_model_prompt_scorer_unchanged": True,
        "timestamp": TIMESTAMP,
        "date": DATE,
        "decoys": [
            "repeated source",
            "future plan",
            "non-member",
            "fictional quote",
            "different owner/container",
        ],
        "return_boundary": "original memory or source-supported fact; "
        "no cross-source answer list or aggregate count",
    },
)
config = ROOT / "configs/runs/generalization-20261005"
config.mkdir(parents=True, exist_ok=True)
(ROOT / "var/generalization-20261005").mkdir(parents=True, exist_ok=True)
frozen_text(config / "default.yaml", (ROOT / "configs/default.yaml").read_text())
local = yaml.safe_load((ROOT / "configs/local.yaml").read_text())
local["storage"]["qdrant"]["collection"] = "memories_generalization_20261005"
local["capture"] = {"enabled": False}
frozen_text(config / "local.yaml", yaml.safe_dump(local, sort_keys=False))
print("prepared 4 synthetic domains, 16 frozen questions, 28 original source messages")
