"""Select a fixed set where deterministic fact rules miss gold relations.

This is an evaluation-only selector.  It does not change the production fact
extractor and only keeps rows with at least one manually annotated relation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from fact_entity_smoke import _fact_triples, _gold_triples, load_cases

from tianximem.facts.grammar import extract_evidence


def _category(text: str) -> str:
    if text.startswith("Corpus:"):
        return "corpus"
    if text.startswith("source:"):
        return "source"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "chinese"
    if re.search(r"(?m)^(?:user|assistant|[A-Z][\w .'’-]+):", text):
        return "dialogue"
    if any(marker in text for marker in ("From:", "To:", "Organizer:", "Attendees:", "Name:", "Profession:", "Location:")):
        return "structured"
    if len(text) > 250:
        return "long"
    return "other"


def _candidate(row: dict[str, Any], index: int, origin: str) -> dict[str, Any] | None:
    gold = _gold_triples(row)
    if not gold:
        return None
    found = _fact_triples(
        extract_evidence(
            parent_memory_id=f"llm-target-{origin}-{index}",
            user_id="gold",
            question=row["text"],
            answer=None,
            event_time=None,
        )
    )
    missed = gold - found
    if not missed:
        return None
    return {
        "id": row["id"],
        "text": row["text"],
        "entities": list(row.get("entities", ())),
        "relations": [
            {"subject": s, "relation": r, "object": o}
            for s, r, o in row.get("relations", ())
        ],
        "selection_category": _category(row["text"]),
        "selection_origin": origin,
        "rule_false_negatives": [
            {"subject": s, "relation": r, "object": o}
            for s, r, o in sorted(missed)
        ],
    }


def _round_robin(rows: list[dict[str, Any]], count: int, categories: tuple[str, ...]) -> list[dict[str, Any]]:
    buckets = {category: [] for category in categories}
    for row in rows:
        buckets.setdefault(row["selection_category"], []).append(row)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    while len(selected) < count:
        progressed = False
        for category in categories:
            for row in buckets.get(category, []):
                if row["id"] in used:
                    continue
                selected.append(row)
                used.add(row["id"])
                progressed = True
                break
            if len(selected) >= count:
                break
        if not progressed:
            break
    return selected


def _dedupe_text(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep the first annotated candidate for each exact source text."""
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = 0
    for row in rows:
        text = row["text"]
        if text in seen:
            duplicates += 1
            continue
        seen.add(text)
        selected.append(row)
    return selected, duplicates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diverse", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--diverse-limit",
        type=int,
        default=None,
        help="maximum number of rows taken from the diverse source before supplementing gold",
    )
    parser.add_argument(
        "--dedupe-text",
        action="store_true",
        help="deduplicate exact source texts within each source before selection",
    )
    args = parser.parse_args()

    diverse = [
        candidate
        for index, row in enumerate(load_cases(args.diverse))
        if (candidate := _candidate(row, index, "diverse")) is not None
    ]
    original = [
        candidate
        for index, row in enumerate(load_cases(args.gold))
        if (candidate := _candidate(row, index, "gold")) is not None
    ]

    duplicate_counts = {"diverse": 0, "gold": 0}
    if args.dedupe_text:
        diverse, duplicate_counts["diverse"] = _dedupe_text(diverse)
        original, duplicate_counts["gold"] = _dedupe_text(original)

    diverse_limit = args.limit if args.diverse_limit is None else min(args.diverse_limit, args.limit)
    selected = _round_robin(
        diverse,
        diverse_limit,
        ("dialogue", "chinese", "corpus", "long", "source", "structured", "other"),
    )
    seen_text = {row["text"] for row in selected}
    remaining = [row for row in original if row["text"] not in seen_text]
    need = max(0, args.limit - len(selected))
    selected.extend(
        _round_robin(
            remaining,
            need,
            ("corpus", "source", "other", "structured", "long", "chinese", "dialogue"),
        )
    )
    if len(selected) < args.limit:
        raise SystemExit(f"only selected {len(selected)} rows, need {args.limit}")
    selected = selected[: args.limit]
    for index, row in enumerate(selected, 1):
        row["selection_rank"] = index

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    category_counts: dict[str, int] = {}
    origin_counts: dict[str, int] = {}
    for row in selected:
        category_counts[row["selection_category"]] = category_counts.get(row["selection_category"], 0) + 1
        origin_counts[row["selection_origin"]] = origin_counts.get(row["selection_origin"], 0) + 1
    report = {
        "cases": len(selected),
        "source_candidates": {"diverse": len(diverse), "gold": len(original)},
        "duplicate_texts_removed": duplicate_counts,
        "selected_by_origin": origin_counts,
        "selected_by_category": category_counts,
        "ids": [row["id"] for row in selected],
        "rule_false_negative_relations": sum(len(row["rule_false_negatives"]) for row in selected),
        "selection_policy": (
            "deduplicated diverse rule-FN cases selected by category, then "
            "round-robin original gold rule-FN candidates"
            if args.dedupe_text
            else "all diverse rule-FN cases, then round-robin original gold rule-FN candidates"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
