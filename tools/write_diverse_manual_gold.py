"""Write the hand-reviewed gold annotations for the diversity fixture.

The mapping below is intentionally explicit and small: every relation was
reviewed against the source text, and ambiguous conversational material is
left unannotated rather than guessed.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "eval/facts/common-facts-diverse-sample.jsonl"
OUTPUT = ROOT / "eval/facts/common-facts-diverse-gold.jsonl"


def rel(subject: str, relation: str, object_: str) -> dict[str, str]:
    return {"subject": subject, "relation": relation, "object": object_}


ANNOTATIONS: dict[str, list[dict[str, str]]] = {
    "diverse-0001": [rel("Caroline", "attended", "a LGBTQ support group")],
    "diverse-0002": [],
    "diverse-0003": [rel("Melanie", "participated in", "a charity race")],
    "diverse-0004": [rel("Caroline", "gave a speech at", "my school event")],
    "diverse-0005": [rel("Caroline", "attended", "a LGBTQ support group")],
    "diverse-0006": [],
    "diverse-0007": [rel("Melanie", "participated in", "a charity race")],
    "diverse-0008": [rel("Caroline", "gave a speech at", "my school event")],
    "diverse-0009": [
        rel("Jon", "occupation", "banker"),
        rel("Gina", "employer", "Door Dash"),
        rel("Jon", "started", "a dance studio"),
    ],
    "diverse-0010": [
        rel("Jon", "passionate about", "dancing"),
        rel("Jon", "started", "a dance studio"),
    ],
    "diverse-0011": [rel("Gina", "launched", "an ad campaign")],
    "diverse-0012": [
        rel("Jon", "plans to open", "dance studio"),
        rel("Gina", "plans to expand", "my clothing store"),
    ],
    "diverse-0013": [rel("I", "use", "a planner")],
    "diverse-0014": [rel("I", "wore", "the navy blue blazer")],
    "diverse-0015": [rel("I", "use", "Adobe Premiere Pro")],
    "diverse-0016": [rel("I", "visited", "the Museum of Modern Art")],
    "diverse-0017": [rel("I", "play soccer", "with my coworkers")],
    "diverse-0018": [],
    "diverse-0019": [],
    "diverse-0020": [],
    "diverse-0021": [],
    "diverse-0022": [],
    "diverse-0023": [],
    "diverse-0024": [],
    "diverse-0025": [],
    "diverse-0026": [],
    "diverse-0027": [rel("Aramais Yepiskoposyan", "played for", "FC Ararat Yerevan")],
    "diverse-0028": [rel("The Discoverie of Witchcraft", "dedicated to", "Sir Roger Manwood")],
    "diverse-0029": [],
    "diverse-0030": [],
    "diverse-0031": [],
    "diverse-0032": [],
    "diverse-0033": [
        rel("我", "有", "糖尿病病史"),
        rel("我", "觉得", "眼睛总是时不时看不清"),
        rel("我", "特别渴", "夜里"),
        rel("体重", "下降", "大概5公斤"),
    ],
    "diverse-0034": [rel("我", "是", "早期的2型糖尿病")],
    "diverse-0035": [
        rel("我", "服用", "二甲双胍"),
        rel("我", "过敏", "头孢呋辛"),
    ],
    "diverse-0036": [rel("我", "住在", "上海")],
    "diverse-0037": [],
    "diverse-0038": [],
    "diverse-0039": [
        rel("Ellie Kemper", "country of citizenship", "United States of America"),
        rel("United States of America", "head of state", "Donald Trump"),
        rel("Ellie Kemper", "country of citizenship", "Croatia"),
        rel("Croatia", "head of state", "Kolinda Grabar-Kitarović"),
    ],
    "diverse-0040": [
        rel("Misery", "author", "Stephen King"),
        rel("Stephen King", "country of citizenship", "United States of America"),
        rel("Misery", "author", "Richard Dawkins"),
    ],
    "diverse-0041": [
        rel("Michael Feinstein", "country of citizenship", "United States of America"),
        rel("United States of America", "capital", "Washington, D.C."),
    ],
    "diverse-0042": [
        rel("Lou Pearlman", "country of citizenship", "United States of America"),
        rel("Lou Pearlman", "country of citizenship", "India"),
        rel("India", "capital", "Taloga"),
    ],
    "diverse-0043": [
        rel("Jaroslav Pelikan", "employer", "Concordia Seminary"),
        rel("Jaroslav Pelikan", "employer", "University of Chicago"),
        rel("Jaroslav Pelikan", "employer", "Valparaiso University"),
        rel("Jaroslav Pelikan", "employer", "Yale University"),
    ],
    "diverse-0044": [
        rel("Jaroslav Pelikan", "employer", "Yale University"),
        rel("Jaroslav Pelikan", "employer", "University of Chicago"),
        rel("Jaroslav Pelikan", "employer", "Valparaiso University"),
    ],
    "diverse-0045": [
        rel("Charlotte Dowerg", "organized", "Charlotte Dowerg <> Jared Chartier 1:1"),
        rel("Charlotte Dowerg", "attendee", "Jared Chartier"),
    ],
    "diverse-0046": [rel("Emilia Anglada", "employee", "Zenith Labs")],
    "diverse-0047": [
        rel("Emilia Anglada", "employer", "Zenith Labs"),
        rel("Albert Garcia", "employer", "Zenith Labs"),
    ],
    "diverse-0048": [
        rel("Name: Craig Baker", "profession", "Profession: Colour Technologist"),
        rel("Name: Craig Baker", "lives in", "Location: Port Michael, Bosnia and Herzegovina"),
    ],
    "diverse-0049": [
        rel("Name: Christina Baker", "profession", "Profession: Colour Technologist"),
        rel("Name: Christina Baker", "lives in", "Location: Port Michael, Bosnia and Herzegovina"),
    ],
    "diverse-0050": [],
    "diverse-0051": [],
    "diverse-0052": [],
}


def main() -> None:
    rows = []
    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            relations = ANNOTATIONS.get(row["id"], [])
            entities = []
            for item in relations:
                for value in (item["subject"], item["object"]):
                    if value not in entities:
                        entities.append(value)
            row["entities"] = entities
            row["relations"] = relations
            row["annotation_status"] = "manual_gold_v1"
            rows.append(row)
    if set(ANNOTATIONS) != {row["id"] for row in rows}:
        raise RuntimeError("manual annotation IDs do not match the source sample")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} manually annotated rows to {OUTPUT}")


if __name__ == "__main__":
    main()
